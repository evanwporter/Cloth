#include <nanobind/nanobind.h>
#include <nanobind/numpy.h>
#include <nanobind/stl.h>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <numeric>
#include <Eigen/Dense>
#include "lib/robinhood.h"

namespace nb = nanobind;
using MatrixXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
struct slice {
    T start, stop;
    int step;

    slice(T start_, T stop_, int step_) : start(start_), stop(stop_), step(step_) {
        if (step == 0) throw std::invalid_argument("Step cannot be zero");
    }

    void normalize(Eigen::Index length) {
        if (start < 0) start += length;
        if (stop < 0) stop += length;
        if (start < 0) start = 0;
        if (stop > length) stop = length;
    }

    Eigen::Index length() const {
        return (step > 0) ? (stop - start + step - 1) / step : (start - stop - step + 1) / (-step);
    }
    
    T get_start() const { return start; }
    T get_stop() const { return stop; }
    int get_step() const { return step; }

    std::string repr() const {
        std::ostringstream oss;
        oss << "slice(" << start << ", " << stop << ", " << step << ")";
        return oss.str();
    }
};

slice<int> combine_slices(const slice<int>& mask, const slice<int>& overlay, Eigen::Index length_mask) {
    int start = mask.start + (overlay.start * mask.step);
    int stop = mask.start + (overlay.stop * mask.step);
    int step = mask.step * overlay.step;
    return slice<int>(start, stop, step);
}

// Helper function to convert std::vector<T> to nb::ndarray
template <typename T>
nb::ndarray<T> vector_to_numpy(const std::vector<T>& vec) {
    return nb::ndarray<T>(vec.size(), vec.data());
}

template <typename T>
MatrixXdRowMajor load_csv(const std::string &fname, const std::string &path, const bool quiet=true) {
    std::vector<T> values;

    std::ifstream indata(path + fname + ".csv");
    if (!indata.is_open()) {
        throw std::runtime_error("Unable to open file: " + path + fname + ".csv");
    }

    std::string line;

    // HEADERS
    std::getline(indata, line);
    std::stringstream lineStream(line);
    std::vector<std::string> headers;
    std::string header;
    while (getline(lineStream, header, ',')) {
        headers.push_back(header);
    }

    std::size_t data_width = headers.size();
    unsigned int rows = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // DATA
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;

        for (unsigned int i = 0; i < data_width; ++i) {
            std::getline(lineStream, cell, ',');
            if (i > 0) { // Skip index for values
                values.push_back(std::stod(cell)); // TODO: Accept other types
            }
        }
        ++rows;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    if (!quiet) {
        std::cout << "Loaded " << fname << ". Time taken: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                  << " microseconds." << std::endl;
    }

    return Eigen::Map<MatrixXdRowMajor>(values.data(), rows, data_width - 1);
};

class DataFrame;
class Series;

class Index_ {
public:
    virtual ~Index_() = default;
};

class ObjectIndex : public Index_ {
public:
    robin_hood::unordered_map<std::string, int> index_;
    std::vector<std::string> keys_;
    std::shared_ptr<slice<int>> mask_;

    ObjectIndex(robin_hood::unordered_map<std::string, int> index, std::vector<std::string> keys)
        : index_(index), keys_(keys), mask_(std::make_shared<slice<int>>(0, static_cast<int>(keys.size()), 1)) {}

    std::shared_ptr<ObjectIndex> fast_init(std::shared_ptr<slice<int>> mask) const {
        auto new_index = std::make_shared<ObjectIndex>(*this);
        new_index->mask_ = mask;
        return new_index;
    }

    std::vector<std::string> keys() const {
        std::vector<std::string> result;
        for (Eigen::Index i = mask_->start; i < mask_->stop; i += mask_->step) {
            result.push_back(keys_[i]);
        }
        return result;
    }

    std::shared_ptr<slice<int>> get_mask() const {
        return mask_;
    }
};

typedef ObjectIndex ColumnIndex;

class SeriesView {
public:
    const Eigen::VectorXd& values_;
    std::shared_ptr<ObjectIndex> index_;
    std::shared_ptr<slice<int>> mask_;

    SeriesView(const Eigen::VectorXd& values,
               std::shared_ptr<ObjectIndex> index,
               std::shared_ptr<slice<int>> mask)
        : values_(values), index_(index), mask_(mask) {}

    std::string repr() const {
        std::ostringstream oss;
        oss << "SeriesView, Length: " << mask_->length() << "\nValues:\n";
        for (Eigen::Index i = mask_->start; i < mask_->stop; i += mask_->step) {
            oss << values_(i) << "\n";
        }
        return oss.str();
    }

    Eigen::Index size() const {
        return mask_->length();
    }

    double sum() const {
        double result = 0.0;
        for (Eigen::Index i = mask_->start; i < mask_->stop; i += mask_->step) {
            result += values_(i);
        }
        return result;
    }

    double mean() const {
        return sum() / size();
    }

    double min() const {
        double min_value = values_(mask_->start);
        for (Eigen::Index i = mask_->start + mask_->step; i < mask_->stop; i += mask_->step) {
            if (values_(i) < min_value) {
                min_value = values_(i);
            }
        }
        return min_value;
    }

    double max() const {
        double max_value = values_(mask_->start);
        for (Eigen::Index i = mask_->start + mask_->step; i < mask_->stop; i += mask_->step) {
            if (values_(i) > max_value) {
                max_value = values_(i);
            }
        }
        return max_value;
    }

    std::shared_ptr<Series> to_series() const {
        // Apply mask to values_
        Eigen::Index num_values = mask_->length();
        Eigen::VectorXd filtered_values(num_values);
        for (Eigen::Index i = 0, mask_idx = mask_->start; i < num_values; ++i, mask_idx += mask_->step) {
            filtered_values(i) = values_(mask_idx);
        }

        // Apply mask to index_
        std::vector<std::string> filtered_keys;
        filtered_keys.reserve(num_values);
        for (Eigen::Index i = mask_->start; i < mask_->stop; i += mask_->step) {
            filtered_keys.push_back(index_->keys_[i]);
        }

        // Create a new index map
        robin_hood::unordered_map<std::string, int> filtered_index_map;
        for (size_t i = 0; i < filtered_keys.size(); ++i) {
            filtered_index_map[filtered_keys[i]] = static_cast<int>(i);
        }

        auto filtered_index = std::make_shared<ObjectIndex>(std::move(filtered_index_map), std::move(filtered_keys));

        return std::make_shared<Series>(std::move(filtered_values), std::move(filtered_index));
    }

    double get_row(const std::string& arg) const {
        if (index_->index_.find(arg) == index_->index_.end()) {
            throw std::out_of_range("Key '" + arg + "' not found in the SeriesView index.");
        }
        int idx = index_->index_.at(arg);
        if (idx < mask_->start || idx >= mask_->stop || (idx - mask_->start) % mask_->step != 0) {
            throw std::out_of_range("Key '" + arg + "' not accessible due to current slicing.");
        }
        return values_(idx);
    }

    std::vector<std::string> get_index() const {
        std::vector<std::string> result;
        for (Eigen::Index i = mask_->start; i < mask_->stop; i += mask_->step) {
            result.push_back(index_->keys_[i]);
        }
        return result;
    }

    class IlocProxy {
    private:
        SeriesView& parent_;

    public:
        IlocProxy(SeriesView& parent) : parent_(parent) {}

        std::shared_ptr<SeriesView> operator[](Eigen::Index idx) const {
            auto combined_slice = combine_slices(*parent_.mask_, slice<int>(idx, idx + 1, 1), parent_.values_.size());
            return parent_.create_view(std::make_shared<slice<int>>(combined_slice));
        }

        std::shared_ptr<SeriesView> operator[](const slice<int>& arg) const {
            auto combined_slice = combine_slices(*parent_.mask_, arg, parent_.values_.size());
            return parent_.create_view(std::make_shared<slice<int>>(combined_slice));
        }

        std::shared_ptr<SeriesView> operator[](const nb::slice& nbSlice) const {
            nb::ssize_t start, stop, step, slicelength;
            if (!nbSlice.compute(parent_.values_.size(), &start, &stop, &step, &slicelength)) {
                throw nb::error_already_set();
            }
            slice<int> arg(static_cast<int>(start), static_cast<int>(stop), static_cast<int>(step));
            return (*this)[arg];
        }
    };

    std::shared_ptr<SeriesView> create_view(std::shared_ptr<slice<int>> mask) const {
        return std::make_shared<SeriesView>(values_, index_->fast_init(mask), mask);
    }

    IlocProxy iloc() {
        return IlocProxy(*this);
    }
};

class Series {
public:
    Eigen::VectorXd values_;
    std::shared_ptr<ObjectIndex> index_;
    std::string name_;

    Series(Eigen::VectorXd values, std::shared_ptr<ObjectIndex> index)
        : values_(std::move(values)), index_(std::move(index)) {}

    Series(nb::ndarray<double> values, nb::ndarray<nb::object> index) {
        auto buf = values.view();
        this->values_.resize(buf.shape(0));
        for (std::ptrdiff_t i = 0; i < buf.shape(0); ++i) {
            this->values_(i) = buf(i);
        }
        std::vector<std::string> keys = nb::cast<std::vector<std::string>>(index);
        robin_hood::unordered_map<std::string, int> index_map;
        for (size_t i = 0; i < keys.size(); ++i) {
            index_map[keys[i]] = static_cast<Eigen::Index>(i);
        }
        this->index_ = std::make_shared<ObjectIndex>(std::move(index_map), keys);
    }

    std::string repr() const {
        std::ostringstream oss;
        oss << "Series, Length: " << values_.size() << "\nValues:\n";
        for (Eigen::Index i = 0; i < values_.size(); ++i) {
            oss << values_(i) << "\n";
        }
        return oss.str();
    }

    double sum() const {
        return values_.sum();
    }

    double mean() const {
        return values_.mean();
    }

    double min() const {
        return values_.minCoeff();
    }

    double max() const {
        return values_.maxCoeff();
    }

    double get_row(const std::string& arg) const {
        if (index_->index_.find(arg) == index_->index_.end()) {
            throw std::out_of_range("Key '" + arg + "' not found in the Series index.");
        }
        int idx = index_->index_.at(arg);
        return values_(idx);
    }

    std::vector<std::string> get_index() const {
        return index_->keys();
    }

    class IlocProxy {
    private:
        Series& parent_;

    public:
        IlocProxy(Series& parent) : parent_(parent) {}

        double operator[](Eigen::Index idx) const {
            if (idx < 0 || idx >= parent_.values_.size()) {
                throw std::out_of_range("Index out of range");
            }
            return parent_.values_(idx);
        }
    };

    IlocProxy iloc() {
        return IlocProxy(*this);
    }
};

class DataFrameView {
public:
    const MatrixXdRowMajor& values_;
    std::shared_ptr<ObjectIndex> index_;
    std::shared_ptr<ColumnIndex> columns_;
    std::shared_ptr<slice<int>> mask_;

    DataFrameView(const MatrixXdRowMajor& values,
                  std::shared_ptr<ObjectIndex> index,
                  std::shared_ptr<ColumnIndex> columns,
                  std::shared_ptr<slice<int>> mask)
        : values_(values), index_(index), columns_(columns), mask_(mask) {}

    std::string repr() const {
        std::ostringstream oss;
        oss << "DataFrameView, Rows: " << mask_->length() << ", Columns: " << values_.cols() << "\nValues:\n";
        for (Eigen::Index i = mask_->start; i < mask_->stop; i += mask_->step) {
            for (Eigen::Index j = 0; j < values_.cols(); ++j) {
                oss << values_(i, j) << " ";
            }
            oss << "\n";
        }
        return oss.str();
    }

    std::pair<Eigen::Index, Eigen::Index> shape() const {
        return {mask_->length(), values_.cols()};
    }

    nb::ndarray<double> get_col(const std::string& col) const {
        if (columns_->index_.find(col) == columns_->index_.end()) {
            throw std::invalid_argument("Column name not found");
        }

        int col_index = columns_->index_.at(col);
        nb::ndarray<double> column(mask_->length());
        auto buf = column.request();
        double* ptr = static_cast<double*>(buf.ptr);

        for (Eigen::Index i = mask_->start, idx = 0; i < mask_->stop; i += mask_->step, ++idx) {
            ptr[idx] = values_(i, col_index);
        }

        return column;
    }

    std::vector<std::string> get_index() const {
        std::vector<std::string> row_indices;
        for (Eigen::Index i = mask_->start; i < mask_->stop; i += mask_->step) {
            row_indices.push_back(index_->keys_[i]);
        }
        return row_indices;
    }

    nb::ndarray<double> values() const {
        Eigen::Index num_rows = mask_->length();
        Eigen::Index num_cols = values_.cols();

        nb::ndarray<double> result({num_rows, num_cols});
        auto buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);

        for (Eigen::Index i = 0, mask_row = mask_->start; i < num_rows; ++i, mask_row += mask_->step) {
            Eigen::VectorXd row = values_.row(mask_row);
            std::copy(row.data(), row.data() + num_cols, ptr + i * num_cols);
        }

        return result;
    }

    std::shared_ptr<DataFrame> to_dataframe() const {
        // Apply mask to values
        Eigen::Index num_rows = mask_->length();
        Eigen::Index num_cols = values_.cols();

        // Create a matrix to hold the filtered rows
        MatrixXdRowMajor filtered_values(num_rows, num_cols);
        Eigen::Index current_row = 0;
        for (Eigen::Index i = mask_->start; i < mask_->stop; i += mask_->step) {
            filtered_values.row(current_row++) = values_.row(i);
        }

        // Apply mask to index_
        std::vector<std::string> filtered_keys;
        filtered_keys.reserve(num_rows);
        for (Eigen::Index i = mask_->start; i < mask_->stop; i += mask_->step) {
            filtered_keys.push_back(index_->keys_[i]);
        }

        // Create new index map
        robin_hood::unordered_map<std::string, int> filtered_index_map;
        for (size_t i = 0; i < filtered_keys.size(); ++i) {
            filtered_index_map[filtered_keys[i]] = static_cast<int>(i);
        }

        auto filtered_index = std::make_shared<ObjectIndex>(std::move(filtered_index_map), std::move(filtered_keys));

        return std::make_shared<DataFrame>(std::move(filtered_values), std::move(filtered_index), columns_);
        
    }

    // DataFrameView::Proxy classes have to take into account the mask, while 
    //   DataFrame::Proxy classes do not
    // This is the main difference between the two.
    class LocProxy {
    private:
        DataFrameView& parent_;

    public:
        LocProxy(DataFrameView& parent) : parent_(parent) {}

        std::shared_ptr<Series> get(const std::string& idx) const {
            auto& values_ = parent_.values_;
            auto index_ = parent_.index_;
            auto columns_ = parent_.columns_;

            if (index_->index_.find(idx) == index_->index_.end()) {
                throw std::out_of_range("Key '" + idx + "' not found in the DataFrameView index.");
            }

            int actual_row = index_->index_.at(idx);

            Eigen::VectorXd row_values = values_.row(actual_row);
            auto series_index = std::make_shared<ColumnIndex>(*columns_);

            return std::make_shared<Series>(std::move(row_values), std::move(series_index));
        }

        std::shared_ptr<DataFrameView> get(const slice<std::string>& arg) const {
            int start = parent_.index_->index_.at(arg.start);
            int stop = parent_.index_->index_.at(arg.stop);
            auto new_arg = slice<int>(start, stop, arg.step);

            // Adjust for step
            auto combined_slice = combine_slices(*parent_.mask_, new_arg, parent_.values_.rows());

            // Return new DataFrameView with combined slice
            return parent_.create_view(std::make_shared<slice<int>>(combined_slice));
        }

        std::shared_ptr<DataFrameView> get(const nb::slice& nbSlice) const {
            nb::object nb_start = nbSlice.attr("start");
            nb::object nb_stop = nbSlice.attr("stop");
            nb::object nb_step = nbSlice.attr("step");

            std::string start = nb::isinstance<nb::none>(nb_start) ? "" : nb::cast<std::string>(nb_start);
            std::string stop = nb::isinstance<nb::none>(nb_stop) ? "" : nb::cast<std::string>(nb_stop);
            int step = nb::isinstance<nb::none>(nb_step) ? 1 : nb::cast<int>(nb_step);

            slice<std::string> arg(start, stop, step);

            // Call DataFrameView::LocProxy.get(slice<std::string>)
            return get(arg);
        }
    };

    class IlocProxy {
    private:
        DataFrameView& parent_;

    public:
        IlocProxy(DataFrameView& parent) : parent_(parent) {}

        std::shared_ptr<Series> get(int arg) const {
            auto& values_ = parent_.values_;
            auto& columns_ = parent_.columns_;

            if (arg < 0 || arg >= parent_.mask_->length()) {
                throw std::out_of_range("Index out of range");
            }

            int actual_row = parent_.mask_->start + arg * parent_.mask_->step;
            Eigen::VectorXd row_values = values_.row(actual_row);

            auto series_index = std::make_shared<ColumnIndex>(*columns_);
            return std::make_shared<Series>(std::move(row_values), std::move(series_index));
        }


        std::shared_ptr<DataFrameView> get(const slice<int>& arg) const {
            auto combined_slice = combine_slices(*parent_.mask_, arg, parent_.values_.rows());
            return parent_.create_view(std::make_shared<slice<int>>(combined_slice));
        }

        std::shared_ptr<DataFrameView> get(const nb::slice& nbSlice) const {
            nb::ssize_t start, stop, step, slicelength;
            if (!nbSlice.compute(parent_.values_.rows(), &start, &stop, &step, &slicelength)) {
                throw nb::error_already_set();
            }
            slice<int> arg(static_cast<int>(start), static_cast<int>(stop), static_cast<int>(step));
            return get(arg);
        }
    };

    LocProxy loc() {
        return LocProxy(*this);
    }

    IlocProxy iloc() {
        return IlocProxy(*this);
    }

    std::shared_ptr<DataFrameView> create_view(std::shared_ptr<slice<int>> mask) const {
        return std::make_shared<DataFrameView>(values_, index_->fast_init(mask), columns_, mask);
    }
};


class DataFrame : public std::enable_shared_from_this<DataFrame> {
public:
    MatrixXdRowMajor values_;
    std::shared_ptr<ObjectIndex> index_;
    std::shared_ptr<ColumnIndex> columns_;

    DataFrame(MatrixXdRowMajor values, std::shared_ptr<ObjectIndex> index, std::shared_ptr<ColumnIndex> columns)
        : values_(std::move(values)), index_(std::move(index)), columns_(std::move(columns)) {}

    DataFrame(nb::list values, nb::list index, nb::list columns) {
        Eigen::Index rows = static_cast<Eigen::Index>(values.size());
        Eigen::Index cols = static_cast<Eigen::Index>(nb::len(values[0]));
        values_ = MatrixXdRowMajor(rows, cols);
        for (Eigen::Index i = 0; i < rows; ++i) {
            auto row = nb::cast<nb::list>(values[i]);
            for (Eigen::Index j = 0; j < cols; ++j) {
                values_(i, j) = nb::cast<double>(row[j]);
            }
        }
        robin_hood::unordered_map<std::string, int> index_map;
        std::vector<std::string> index_keys;
        for (nb::ssize_t i = 0; i < index.size(); ++i) {
            std::string key = nb::cast<std::string>(index[i]);
            index_map[key] = static_cast<Eigen::Index>(i);
            index_keys.push_back(key);
        }
        robin_hood::unordered_map<std::string, int> column_map;
        std::vector<std::string> column_keys;
        for (nb::ssize_t i = 0; i < columns.size(); ++i) {
            std::string key = nb::cast<std::string>(columns[i]);
            column_map[key] = static_cast<Eigen::Index>(i);
            column_keys.push_back(key);
        }
        index_ = std::make_shared<ObjectIndex>(std::move(index_map), std::move(index_keys));
        columns_ = std::make_shared<ColumnIndex>(std::move(column_map), std::move(column_keys));
    }

    DataFrame(nb::ndarray<double> values, nb::ndarray<nb::object> index, nb::ndarray<nb::object> columns) {
        
        auto buf = values.view();
        Eigen::Index rows = static_cast<Eigen::Index>(buf.shape(0));
        Eigen::Index cols = static_cast<Eigen::Index>(buf.shape(1));
        values_ = MatrixXdRowMajor(rows, cols);
        for (Eigen::Index i = 0; i < rows; ++i) {
            for (Eigen::Index j = 0; j < cols; ++j) {
                values_(i, j) = buf(i, j);
            }
        }
        robin_hood::unordered_map<std::string, int> index_map;
        std::vector<std::string> index_keys;
        auto index_array = index.cast<nb::list>();
        auto columns_array = columns.cast<nb::list>();
        for (nb::ssize_t i = 0; i < index_array.size(); ++i) {
            std::string key = nb::cast<std::string>(index_array[i]);
            index_map[key] = static_cast<Eigen::Index>(i);
            index_keys.push_back(key);
        }
        robin_hood::unordered_map<std::string, int> column_map;
        std::vector<std::string> column_keys;
        for (nb::ssize_t i = 0; i < columns_array.size(); ++i) {
            std::string key = nb::cast<std::string>(columns_array[i]);
            column_map[key] = static_cast<Eigen::Index>(i);
            column_keys.push_back(key);
        }
        index_ = std::make_shared<ObjectIndex>(std::move(index_map), std::move(index_keys));
        columns_ = std::make_shared<ColumnIndex>(std::move(column_map), std::move(column_keys));
    }

    std::shared_ptr<DataFrameView> create_view(std::shared_ptr<slice<int>> mask) const {
        return std::make_shared<DataFrameView>(values_, index_->fast_init(mask), columns_, mask);
    }

    std::string repr() const {
        std::ostringstream oss;
        oss << "Columns: " << columns_->keys().size() << ", Rows: " << values_.rows() << "\nValues:\n";
        for (Eigen::Index i = 0; i < values_.rows(); ++i) {
            for (Eigen::Index j = 0; j < values_.cols(); ++j) {
                oss << values_(i, j) << " ";
            }
            oss << "\n";
        }
        return oss.str();
    }

    std::pair<Eigen::Index, Eigen::Index> shape() const {
        return {values_.rows(), values_.cols()};
    }

    nb::ndarray<double> get_col(const std::string& col) const {
        if (columns_->index_.find(col) == columns_->index_.end()) {
            throw std::invalid_argument("Column name not found");
        }

        int col_index = columns_->index_.at(col);
        nb::ndarray<double> column(values_.rows());
        auto buf = column.request();
        double* ptr = static_cast<double*>(buf.ptr);

        for (Eigen::Index i = 0; i < values_.rows(); ++i) {
            ptr[i] = values_(i, col_index);
        }

        return column;
    }

    std::vector<std::string> get_index() const {
        return index_->keys();
    }

    nb::ndarray<double> values() const {
        Eigen::Index num_rows = values_.rows();
        Eigen::Index num_cols = values_.cols();

        nb::ndarray<double> result({num_rows, num_cols});
        auto buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);

        for (Eigen::Index i = 0; i < num_rows; ++i) {
            Eigen::VectorXd row = values_.row(i);
            std::copy(row.data(), row.data() + num_cols, ptr + i * num_cols);
        }

        return result;
    }

    std::vector<double> sum(int axis) const {
        if (axis == 0) {
            Eigen::VectorXd colSum = values_.colwise().sum();
            return std::vector<double>(colSum.data(), colSum.data() + colSum.size());
        } else if (axis == 1) {
            Eigen::VectorXd rowSum = values_.rowwise().sum();
            return std::vector<double>(rowSum.data(), rowSum.data() + rowSum.size());
        } else {
            throw std::invalid_argument("Invalid axis value. Use 0 for columns and 1 for rows.");
        }
    }

    std::vector<double> mean(int axis) const {
        if (axis == 0) {
            Eigen::VectorXd colMean = values_.colwise().mean();
            return std::vector<double>(colMean.data(), colMean.data() + colMean.size());
        } else if (axis == 1) {
            Eigen::VectorXd rowMean = values_.rowwise().mean();
            return std::vector<double>(rowMean.data(), rowMean.data() + rowMean.size());
        } else {
            throw std::invalid_argument("Invalid axis value. Use 0 for columns and 1 for rows.");
        }
    }

    std::vector<double> min(int axis) const {
        if (axis == 0) {
            Eigen::VectorXd colMin = values_.colwise().minCoeff();
            return std::vector<double>(colMin.data(), colMin.data() + colMin.size());
        } else if (axis == 1) {
            Eigen::VectorXd rowMin = values_.rowwise().minCoeff();
            return std::vector<double>(rowMin.data(), rowMin.data() + rowMin.size());
        } else {
            throw std::invalid_argument("Invalid axis value. Use 0 for columns and 1 for rows.");
        }
    }

    std::vector<double> max(int axis) const {
        if (axis == 0) {
            Eigen::VectorXd colMax = values_.colwise().maxCoeff();
            return std::vector<double>(colMax.data(), colMax.data() + colMax.size());
        } else if (axis == 1) {
            Eigen::VectorXd rowMax = values_.rowwise().maxCoeff();
            return std::vector<double>(rowMax.data(), rowMax.data() + rowMax.size());
        } else {
            throw std::invalid_argument("Invalid axis value. Use 0 for columns and 1 for rows.");
        }
    }

    class LocProxy {
    public:
        DataFrame* frame_;

        LocProxy(DataFrame* frame) : frame_(frame) {}

        std::shared_ptr<Series> get(const std::string& idx) const {
            auto& values_ = frame_->values_;
            auto index_ = frame_->index_;  // Access the index from DataFrameView
            auto columns_ = frame_->columns_;  // Access the columns from DataFrameView

            if (index_->index_.find(idx) == index_->index_.end()) {
                throw std::out_of_range("Key '" + idx + "' not found in the DataFrameView index.");
            }

            Eigen::Index row = index_->index_.at(idx);
            Eigen::VectorXd row_values = values_.row(row);

            // Use the existing ColumnIndex from DataFrameView for the new Series
            auto series_index = std::make_shared<ColumnIndex>(*columns_);

            // Return Series with row values and columns as index
            return std::make_shared<Series>(std::move(row_values), std::move(series_index));
    }

        std::shared_ptr<DataFrameView> get(const slice<std::string>& arg) const {
            auto index_ = frame_->index_;
            auto start = index_->index_.at(arg.start);
            auto stop = index_->index_.at(arg.stop);
            auto new_arg = slice<int>(start, stop, arg.step);
            auto combined_slice = combine_slices(
                slice<int>(0, frame_->values_.rows(), 1), 
                new_arg, frame_->values_.rows()
            );
            return frame_->create_view(std::make_shared<slice<int>>(combined_slice));
        }

        std::shared_ptr<DataFrameView> get(const nb::slice& nbSlice) const {
            nb::object nb_start = nbSlice.attr("start");
            nb::object nb_stop = nbSlice.attr("stop");
            nb::object nb_step = nbSlice.attr("step");

            std::string start = nb::isinstance<nb::none>(nb_start) ? "" : nb::cast<std::string>(nb_start);
            std::string stop = nb::isinstance<nb::none>(nb_stop) ? "" : nb::cast<std::string>(nb_stop);
            int step = nb::isinstance<nb::none>(nb_step) ? 1 : nb::cast<int>(nb_step);

            slice<std::string> arg(start, stop, step);
            return get(arg);
        }
    };

    class IlocProxy {
    public:
        DataFrame* frame_;

        IlocProxy(DataFrame* frame) : frame_(frame) {}

        std::shared_ptr<DataFrameView> get(const slice<int>& arg) const {
            auto& values_ = frame_->values_;
            auto combined_slice = combine_slices(slice<int>(0, values_.rows(), 1), arg, values_.rows());
            return frame_->create_view(std::make_shared<slice<int>>(combined_slice));
        }

        std::shared_ptr<DataFrameView> get(const nb::slice& nbSlice) const {
            nb::ssize_t start, stop, step, slicelength;
            if (!nbSlice.compute(frame_->values_.rows(), &start, &stop, &step, &slicelength)) {
                throw nb::error_already_set();
            }

            slice<int> arg(static_cast<int>(start), static_cast<int>(stop), static_cast<int>(step));
            return get(arg);
        }

        std::shared_ptr<Series> get(int arg) const {
            auto& values_ = frame_->values_;
            auto& columns_ = frame_->columns_;

            if (arg < 0 || arg >= values_.rows()) {
                throw std::out_of_range("Index out of range");
            }
            Eigen::VectorXd row_values = values_.row(arg);
            std::string idx_str = std::to_string(arg);

            // Use the existing ObjectIndex/ColumnsIndex for columns
            auto series_index = std::make_shared<ObjectIndex>(*frame_->columns_);

            // Return Series with row values and columns as index
            return std::make_shared<Series>(std::move(row_values), std::move(series_index));
        }
    };

    LocProxy loc() {
        return LocProxy(this);
    }

    IlocProxy iloc() {
        return IlocProxy(this);
    }
};


NB_MODULE(cloth, m) {
    nb::class_<slice<int>>(m, "slice")
        .def(nb::init<int, int, int>())
        .def("normalize", &slice<int>::normalize)
        .def("length", &slice<int>::length);

    nb::class_<Index_, std::shared_ptr<Index_>>(m, "Index_");

    nb::class_<ObjectIndex, Index_, std::shared_ptr<ObjectIndex>>(m, "ObjectIndex")
        .def(nb::init<robin_hood::unordered_map<std::string, int>, std::vector<std::string>>())
        .def("keys", &ObjectIndex::keys)
        .def("get_mask", &ObjectIndex::get_mask);

    nb::class_<DataFrame, std::shared_ptr<DataFrame>>(m, "DataFrame")
        .def(nb::init<MatrixXdRowMajor, std::shared_ptr<ObjectIndex>, std::shared_ptr<ColumnIndex>>())
        .def(nb::init<nb::list, nb::list, nb::list>())
        .def(nb::init<nb::ndarray<double>, nb::ndarray<nb::object>, nb::ndarray<nb::object>>())
        .def("__repr__", &DataFrame::repr)
        .def_property_readonly("shape", &DataFrame::shape)
        .def("__getitem__", &DataFrame::get_col)
        .def_property_readonly("values", &DataFrame::values)
        .def_property_readonly("loc", &DataFrame::loc)
        .def_property_readonly("iloc", &DataFrame::iloc)
        .def("sum", &DataFrame::sum)
        .def("mean", &DataFrame::mean)
        .def("min", &DataFrame::min)
        .def("max", &DataFrame::max)
        .def("rows", &DataFrame::rows)
        .def("cols", &DataFrame::cols)
        .def_property_readonly("index", &DataFrame::get_index);

    nb::class_<DataFrame::LocProxy>(m, "DataFrameLocProxy")
        .def(nb::init<DataFrame*>())
        .def("__getitem__", (std::shared_ptr<Series> (DataFrame::LocProxy::*)(const std::string&) const) &DataFrame::LocProxy::get)
        .def("__getitem__", (std::shared_ptr<DataFrameView> (DataFrame::LocProxy::*)(const slice<std::string>&) const) &DataFrame::LocProxy::get)
        .def("__getitem__", (std::shared_ptr<DataFrameView> (DataFrame::LocProxy::*)(const nb::slice&) const) &DataFrame::LocProxy::get);

    nb::class_<DataFrame::IlocProxy>(m, "DataFrameIlocProxy")
        .def(nb::init<DataFrame*>())
        .def("__getitem__", (std::shared_ptr<Series> (DataFrame::IlocProxy::*)(int) const) &DataFrame::IlocProxy::get)
        .def("__getitem__", (std::shared_ptr<DataFrameView> (DataFrame::IlocProxy::*)(const slice<int>&) const) &DataFrame::IlocProxy::get)
        .def("__getitem__", (std::shared_ptr<DataFrameView> (DataFrame::IlocProxy::*)(const nb::slice&) const) &DataFrame::IlocProxy::get);

    nb::class_<DataFrameView, std::shared_ptr<DataFrameView>>(m, "DataFrameView")
        .def("__repr__", &DataFrameView::repr)
        .def_property_readonly("shape", &DataFrameView::shape)
        .def("__getitem__", &DataFrameView::get_col)
        .def_property_readonly("values", &DataFrameView::values)
        .def("to_dataframe", &DataFrameView::to_dataframe)
        .def_property_readonly("loc", &DataFrameView::loc)
        .def_property_readonly("iloc", &DataFrameView::iloc)
        .def_property_readonly("index", &DataFrameView::get_index);

    nb::class_<DataFrameView::LocProxy>(m, "DataFrameViewLocProxy")
        .def("__getitem__", (std::shared_ptr<Series> (DataFrameView::LocProxy::*)(const std::string&) const) &DataFrameView::LocProxy::get)
        .def("__getitem__", (std::shared_ptr<DataFrameView> (DataFrameView::LocProxy::*)(const slice<std::string>&) const) &DataFrameView::LocProxy::get)
        .def("__getitem__", (std::shared_ptr<DataFrameView> (DataFrameView::LocProxy::*)(const nb::slice&) const) &DataFrameView::LocProxy::get);

    nb::class_<DataFrameView::IlocProxy>(m, "DataFrameViewIlocProxy")
        .def("__getitem__", (std::shared_ptr<Series> (DataFrameView::IlocProxy::*)(int) const) &DataFrameView::IlocProxy::get)
        .def("__getitem__", (std::shared_ptr<DataFrameView> (DataFrameView::IlocProxy::*)(const slice<int>&) const) &DataFrameView::IlocProxy::get)
        .def("__getitem__", (std::shared_ptr<DataFrameView> (DataFrameView::IlocProxy::*)(const nb::slice&) const) &DataFrameView::IlocProxy::get);

    nb::class_<Series, std::shared_ptr<Series>>(m, "Series")
        .def(nb::init<Eigen::VectorXd, std::shared_ptr<ObjectIndex>>())
        .def(nb::init<nb::ndarray<double>, nb::ndarray<nb::object>>())
        .def("sum", &Series::sum)
        .def("mean", &Series::mean)
        .def("min", &Series::min)
        .def("max", &Series::max)
        .def("__repr__", &Series::repr)
        .def_property_readonly("iloc", &Series::iloc)
        .def("__getitem__", &Series::get_row)
        .def_property_readonly("index", &Series::get_index);


    nb::class_<Series::IlocProxy>(m, "SeriesIlocProxy")
        .def("__getitem__", &Series::IlocProxy::operator[], nb::is_operator());

    nb::class_<SeriesView, std::shared_ptr<SeriesView>>(m, "SeriesView")
        .def("__repr__", &SeriesView::repr)
        .def_property_readonly("size", &SeriesView::size)
        .def("sum", &SeriesView::sum)
        .def("mean", &SeriesView::mean)
        .def("min", &SeriesView::min)
        .def("max", &SeriesView::max)
        .def_property_readonly("iloc", &SeriesView::iloc)
        .def("__getitem__", &SeriesView::get_row)
        .def_property_readonly("index", &SeriesView::get_index);

}