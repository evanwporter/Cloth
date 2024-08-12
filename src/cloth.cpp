#define LOGGING_ENABLED 1

#if LOGGING_ENABLED
    #define LOG(msg) std::cout << msg << std::endl;
#else
    #define LOG(msg)
#endif


#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>


#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <numeric>
#include <iomanip>

#include <iostream>

#include <Eigen/Dense>
#include "../lib/robinhood.h"

namespace nb = nanobind;
using MatrixXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T = Eigen::Index>
struct slice {
    T start, stop;
    int step;

    // Don't normalize
    slice(T start_, T stop_, int step_) : start(start_), stop(stop_), step(step_) {
        LOG("Creating slice with start=" << start << ", stop=" << stop << ", step=" << step);
        if (step == 0) throw std::invalid_argument("Step cannot be zero");
    }

    // If length is passed, then normalize
    slice(T start_, T stop_, int step_, Eigen::Index length) : start(start_), stop(stop_), step(step_) {
        LOG("Creating slice with start=" << start << ", stop=" << stop << ", step=" << step);
        if (step == 0) throw std::invalid_argument("Step cannot be zero");
        normalize(length);
    }

    void normalize(Eigen::Index length) {
        LOG("Normalizing slice with original start=" << start << ", stop=" << stop << ", step=" << step);
        if (start < 0) start += length;
        if (stop < 0) stop += length;
        if (start < 0) start = 0;
        if (stop > length) stop = length;
        LOG("Normalized slice to start=" << start << ", stop=" << stop << ", step=" << step);
    }

    Eigen::Index length() const {
        LOG("Calculating length of slice with start=" << start << ", stop=" << stop << ", step=" << step);
        Eigen::Index len = (step > 0) ? (stop - start + step - 1) / step : (start - stop - step + 1) / (-step);
        LOG("Calculated length=" << len);
        return len;
    }
    
    T get_start() const { return start; }
    T get_stop() const { return stop; }
    int get_step() const { return step; }

    friend std::ostream& operator<<(std::ostream& os, const slice& sl) {
        os << "slice(" << sl.start << ", " << sl.stop << ", " << sl.step << ")";
        return os;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << *this;
        return oss.str();
    }
};

template <typename T = Eigen::Index>
slice<T> combine_slices(const slice<T>& mask, const slice<T>& overlay) {
    LOG("Combining slices: mask(" << mask.start << ", " << mask.stop << ", " << mask.step << "), "
        << "overlay(" << overlay.start << ", " << overlay.stop << ", " << overlay.step << ")");

    if (overlay.start < 0 || overlay.stop < 0) {
        throw std::out_of_range("Overlay slice is out of bounds of the mask slice. Be sure to normalize slice first.");
    }

    int start = mask.start + (overlay.start * mask.step);
    int stop = mask.start + (overlay.stop * mask.step);
    int step = mask.step * overlay.step;

    LOG("Resulting combined slice: start=" << start << ", stop=" << stop << ", step=" << step);

    return slice<T>(start, stop, step);
}

template <typename T = Eigen::Index>
T combine_slice_with_index(const slice<T>& mask, T index) {
    LOG("Combining slice with index: mask(" << mask.start << ", " << mask.stop << ", " << mask.step << "), index=" << index);

    if (index < 0 || index >= mask.length()) {
        throw std::out_of_range("Index is out of bounds of the mask slice. Be sure to normalize slice first.");
    }

    T combined_index = mask.start + (index * mask.step);

    LOG("Resulting combined index: " << combined_index);

    return combined_index;
}

typedef Eigen::Index index_t;
typedef slice<index_t> mask_t;

class Index_ {
public:
    virtual ~Index_() = default;
};

class ObjectIndex : public Index_ {
public:
    robin_hood::unordered_map<std::string, int> index_;
    std::vector<std::string> keys_;
    std::shared_ptr<mask_t> mask_;


    ObjectIndex(robin_hood::unordered_map<std::string, int> index, std::vector<std::string> keys)
        : index_(index), keys_(keys), mask_(std::make_shared<slice<Eigen::Index>>(0, static_cast<int>(keys.size()), 1)) {}

    ObjectIndex(std::vector<std::string> keys)
        : keys_(keys)
    {
        for (size_t i = 0; i < keys.size(); ++i) {
            this->index_[keys[i]] = static_cast<int>(i);
        }
        this->mask_ = std::make_shared<slice<Eigen::Index>>(0, static_cast<int>(this->keys_.size()), 1);
    }

    friend std::ostream& operator<<(std::ostream& os, const ObjectIndex& objIndex) {
        auto k = objIndex.keys();
        os << "[";
        for (size_t i = 0; i < k.size(); ++i) {
            os << k[i];
            if (i < k.size() - 1) {
                os << ", ";
            }
        }
        os << "]";
        return os;
    }

    ObjectIndex(nb::list keys) {
        keys_.reserve(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            keys_.push_back(std::move(nb::cast<std::string>(keys[i])));
            index_[keys_.back()] = static_cast<int>(i);
        }
        this->mask_ = std::make_shared<slice<Eigen::Index>>(0, static_cast<int>(keys_.size()), 1);
    }

    std::shared_ptr<ObjectIndex> fast_init(std::shared_ptr<slice<Eigen::Index>> mask) const {
        auto new_index = std::make_shared<ObjectIndex>(*this);
        new_index->mask_ = mask;
        std::cout << "ST" << *new_index << "\n";
        return new_index;
    }

    virtual Eigen::Index length() const {
        return mask_->length();
    }

    std::vector<std::string> keys() const {
        std::vector<std::string> result;
        // result.reserve(length());
        for (int i = mask_->start; i < mask_->stop; i += mask_->step) {
            result.push_back(keys_[i]);
        }
        return result;
    }

    std::string operator[](Eigen::Index idx) const {
        if (idx < 0 || idx >= keys_.size()) {
            throw std::out_of_range("Index out of range");
        }
        return keys_[combine_slices<Eigen::Index>(*mask_, slice<Eigen::Index>(idx, idx + 1, 1, length())).start];
    }

    int operator[](const std::string& key) const {
        return index_.at(key);
    }
};

typedef ObjectIndex ColumnIndex;

template <typename Derived>
class IlocBase {
public:
    virtual ~IlocBase() = default;
    IlocBase() = default;

    virtual Derived& parent() = 0;

    Derived operator[](slice<Eigen::Index>& overlay) {
        overlay.normalize(parent().length());
        auto combined_mask = std::make_shared<slice<Eigen::Index>>(combine_slices(*parent().mask_, overlay));
        return Derived(parent(), combined_mask);
    }

    Derived operator[](nb::slice& nbSlice) {
        auto [start, stop, step, slice_length] = nbSlice.compute(parent().length());
        auto overlay = std::make_shared<slice<Eigen::Index>>(static_cast<Eigen::Index>(start), static_cast<Eigen::Index>(stop), static_cast<Eigen::Index>(step), parent().length());
        auto combined_mask = std::make_shared<slice<Eigen::Index>>(combine_slices(*parent().mask_, *overlay));
        return Derived(parent(), combined_mask);
    }
};

template <typename Derived>
class LocBase {
public:
    virtual ~LocBase() = default;
    LocBase() = default;

    virtual Derived& parent() = 0;

    Derived operator[](nb::object& nbSlice) {
        nb::object start, stop, step;
        std::string start_str, stop_str;
        Eigen::Index step_int = 1;

        if (nb::hasattr(nbSlice, "start")) {
            start = nb::getattr(nbSlice, "start");
            if (!start.is_none()) {
                start_str = nb::cast<std::string>(start);
            }
        }

        if (nb::hasattr(nbSlice, "stop")) {
            stop = nb::getattr(nbSlice, "stop");
            if (!stop.is_none()) {
                stop_str = nb::cast<std::string>(stop);
            }
        }

        if (nb::hasattr(nbSlice, "step")) {
            step = nb::getattr(nbSlice, "step");
            if (!step.is_none()) {
                step_int = nb::cast<Eigen::Index>(step);
            }
        }

        Eigen::Index start_ = parent().index_->operator[](start_str);
        Eigen::Index stop_ = parent().index_->operator[](stop_str);

        auto overlay = slice<Eigen::Index>(start_, stop_, step_int, parent().length());

        return parent().iloc()[overlay];
    }
};

template <typename Derived>
class Frame {
public:
    virtual ~Frame() = default;
    Frame() = default;

    virtual Eigen::Index length() const {
        return mask()->length();
    }

    virtual std::shared_ptr<slice<Eigen::Index>> mask() const = 0;

    Derived head(Eigen::Index n) const {
        return Derived(static_cast<const Derived&>(*this), 
                       std::make_shared<slice<Eigen::Index>>(0, std::min(n, length()), 1));
    }

    Derived tail(Eigen::Index n) const {
        return Derived(static_cast<const Derived&>(*this), 
                       std::make_shared<slice<Eigen::Index>>(length() - std::min(n, length()), length(), 1));
    }
};

class Series : public Frame<Series> {
public:
    std::shared_ptr<Eigen::VectorXd> values_;
    std::shared_ptr<ObjectIndex> index_;
    std::string name_;
    std::shared_ptr<slice<Eigen::Index>> mask_;
          
    Series(const Series& other,  std::shared_ptr<slice<Eigen::Index>> mask)
        : values_(other.values_),
          index_(other.index_->fast_init(mask)),
          name_(other.name_),
          mask_(mask) {}

    Series(std::shared_ptr<Eigen::VectorXd> values, std::shared_ptr<ObjectIndex> index)
        : values_(std::move(values)), 
          index_(std::move(index)) 
    {
        mask_ = std::make_shared<slice<Eigen::Index>>(0, values_->size(), 1); 
    }

    Series(const Eigen::VectorXd &values, std::shared_ptr<ObjectIndex> index)
        : values_(std::make_shared<Eigen::VectorXd>(values)), 
        index_(std::move(index)) 
    {
        mask_ = std::make_shared<slice<Eigen::Index>>(0, values_->size(), 1); 
    }

    Series(nb::ndarray<double> values, nb::list index)
        : values_(std::make_shared<Eigen::VectorXd>(Eigen::Map<const Eigen::VectorXd>(values.data(), values.size()))),
        index_(std::make_shared<ObjectIndex>(index))
    {
        mask_ = std::make_shared<slice<Eigen::Index>>(0, values_->size(), 1);
    }

    friend std::ostream& operator<<(std::ostream& os, const Series& series) {
        try {
            Eigen::Index len = series.length();
            for (Eigen::Index i = 0; i < len; ++i) {
                std::string index_value = series.index_->keys_[series.mask_->start + i * series.mask_->step];
                double series_value = series.values()(series.mask_->start + i * series.mask_->step);
                os << index_value << "    " << series_value << "\n";
            }

            if (!series.name_.empty()) {
                os << "Name: " << series.name_ << "\n";
            }

        } catch (const std::exception &e) {
            os << "Error during stream output: " << e.what();
        }

        return os;
    }

    std::shared_ptr<slice<Eigen::Index>> mask() const {
        return mask_;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << *this;
        return oss.str();
    }
    
    Eigen::Index size() const {
        return mask_->length();
    }

    double sum() const {
        return values().sum();
    }

    double mean() const {
        return values().mean();
    }

    double min() const {
        return values().minCoeff();
    }

    double max() const {
        return values().maxCoeff();
    }

    std::vector<std::string> get_index() const {
        return index_->keys();
    }

    const Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> values() const {
        return Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(
            values_->data() + mask_->start, 
            mask_->length(), 
            Eigen::Stride<Eigen::Dynamic, 1>(1, 1)
        );
    }

    Eigen::VectorXd py_val() const {
        return values();
    }

    virtual const ObjectIndex& index() const {
        return *index_;
    }

    class IlocProxy : public IlocBase<Series> {
    private:
        Series& parent_;

    public:
        IlocProxy(Series& parent_) : parent_(parent_) {}

        Series& parent() {
            return parent_;
        }

        using IlocBase<Series>::operator[];

        double operator[](Eigen::Index idx) const {
            Eigen::Index combined_index = combine_slice_with_index(*parent_.mask_, idx);
            return (*parent_.values_)(combined_index);
        }
    };

    IlocProxy iloc() {
        return IlocProxy(*this);
    }

    class LocProxy : public LocBase<Series> {
    private:
        Series& parent_;

    public:
        LocProxy(Series& parent_) : parent_(parent_) {}

        Series& parent() {
            return parent_;
        }

        using LocBase<Series>::operator[];

        double operator[](const std::string& key) const {
            int idx = parent_.index_->operator[](key);
            return parent_.iloc()[idx];
        }
    };

    LocProxy loc() {
        return LocProxy(*this);
    }
};

class DataFrame : public Frame<DataFrame> {
public:
    std::shared_ptr<MatrixXdRowMajor> values_;
    std::shared_ptr<ObjectIndex> index_;
    std::shared_ptr<ColumnIndex> columns_;
    std::shared_ptr<slice<Eigen::Index>> mask_;

    DataFrame(const DataFrame& other,  std::shared_ptr<slice<Eigen::Index>> mask)
        : values_(other.values_),
          index_(other.index_->fast_init(mask)),
          columns_(other.columns_),
          mask_(mask) {}

    DataFrame(std::shared_ptr<MatrixXdRowMajor> values, std::shared_ptr<ObjectIndex> index, std::shared_ptr<ColumnIndex> columns)
        : values_(std::move(values)), 
          index_(std::move(index)), 
          columns_(std::move(columns)), 
          mask_(std::make_shared<slice<Eigen::Index>>(0, values_->rows(), 1)) {}

    DataFrame(nb::ndarray<> values, nb::list index, nb::list columns)
        : index_(std::make_shared<ObjectIndex>(index)),
          columns_(std::make_shared<ColumnIndex>(columns)) {
        Eigen::Index rows = static_cast<Eigen::Index>(values.shape(0));
        Eigen::Index cols = static_cast<Eigen::Index>(values.shape(1));
        values_ = std::make_shared<MatrixXdRowMajor>(Eigen::Map<MatrixXdRowMajor>(static_cast<double*>(values.data()), rows, cols));
    
        mask_ = std::make_shared<slice<Eigen::Index>>(0, values_->rows(), 1);
    }

    std::shared_ptr<slice<Eigen::Index>> mask() const {
        return mask_;
    }

    class IlocProxy : IlocBase<DataFrame> {
    private:
        DataFrame& parent_;

    public:
        IlocProxy(DataFrame& parent_) : parent_(parent_) {}

        DataFrame& parent() {
            return parent_;
        }

        using IlocBase<DataFrame>::operator[];

        Series operator[](Eigen::Index idx) const {
            Eigen::Index combined_index = combine_slice_with_index(*parent_.mask_, idx);
            auto row = std::make_shared<Eigen::VectorXd>(parent_.values_->row(combined_index));
            return Series(row, parent_.columns_);
        }
    };

    IlocProxy iloc() {
        return IlocProxy(*this);
    }

    class LocProxy : public LocBase<DataFrame> {
    private:
        DataFrame& parent_;

    public:
        LocProxy(DataFrame& parent_) : parent_(parent_) {}

        DataFrame& parent() {
            return parent_;
        }

        using LocBase<DataFrame>::operator[];

        Series operator[](const std::string& key) {
            int idx = parent_.index_->operator[](key);
            return parent().iloc()[idx];
        }
    };

    LocProxy loc() {
        return LocProxy(*this);
    }

    Eigen::Index rows() const {
        return length();
    }

    Eigen::Index cols() const {
        return values_->cols();
    }

    virtual const Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> values() const {
        return Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(
            values_->data() + mask_->start, 
            mask_->length(), 
            Eigen::Stride<Eigen::Dynamic, 1>(1, 1)
        );
    }


    virtual const ObjectIndex& index() const {
        return *index_;
    }

    Series operator[](const std::string& colName) const {
        return Series(std::make_shared<Eigen::VectorXd>(values_->col(columns_->index_.at(colName))), index_);
    }

    friend std::ostream& operator<<(std::ostream& os, const DataFrame& df) {
        std::vector<std::string> rowNames = df.index_->keys();
        std::vector<std::string> colNames = df.columns_->keys();

        // Max width of each column
        std::vector<size_t> colWidths(colNames.size(), 0);

        // Calculate the max width for each column based on column names
        for (size_t j = 0; j < colNames.size(); ++j) {
            colWidths[j] = colNames[j].length();
        }

        // Calculate the max width for each column based on data
        for (Eigen::Index i = df.mask_->start; i < df.mask_->stop; i += df.mask_->step) {
            for (Eigen::Index j = 0; j < df.cols(); ++j) {
                std::ostringstream temp_oss;
                temp_oss << std::fixed << std::setprecision(2) << (*df.values_)(i, j);
                if (temp_oss.str().length() > colWidths[j]) {
                    colWidths[j] = temp_oss.str().length();
                }
            }
        }

        // Determine the max width of the row names
        size_t rowNameWidth = 0;
        for (const auto& rowName : rowNames) {
            if (rowName.length() > rowNameWidth) {
                rowNameWidth = rowName.length();
            }
        }

        // Format the column headers with appropriate padding for the row names
        os << std::setw(static_cast<int>(rowNameWidth) + 2) << " "; // Adjust space for row names
        for (size_t j = 0; j < colNames.size(); ++j) {
            os << std::setw(colWidths[j] + 2) << colNames[j];
        }
        os << std::endl;

        for (Eigen::Index i = df.mask_->start; i < df.mask_->stop; i += df.mask_->step) {
            os << std::setw(static_cast<int>(rowNameWidth) + 2) << rowNames[i]; // Align row names
            for (Eigen::Index j = 0; j < df.cols(); ++j) {
                os << std::setw(colWidths[j] + 2) 
                << std::fixed << std::setprecision(2) << (*df.values_)(i, j);
            }
            os << std::endl;
        }

        return os;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << *this;
        return oss.str();
    }
};


NB_MODULE(cloth, m) {
    nb::class_<slice<Eigen::Index>>(m, "slice")
        .def(nb::init<int, int, int>())
        .def("normalize", &slice<Eigen::Index>::normalize)
        .def_prop_ro("start", &slice<Eigen::Index>::get_start)
        .def_prop_ro("stop", &slice<Eigen::Index>::get_stop)
        .def_prop_ro("step", &slice<Eigen::Index>::get_step)
        .def_prop_ro("length", &slice<Eigen::Index>::length)
        .def("__repr__", &slice<Eigen::Index>::to_string);

    nb::class_<Index_>(m, "Index_");

    nb::class_<ObjectIndex, Index_>(m, "ObjectIndex")
        .def(nb::init<std::vector<std::string>>()) 
        .def("fast_init", &ObjectIndex::fast_init)
        .def("keys", &ObjectIndex::keys)
        // .def("get_mask", &ObjectIndex::get_mask)
        .def_rw("index", &ObjectIndex::keys_)
        // .def_rw("mask", &ObjectIndex::mask_);
        .def("__getitem__", [](ObjectIndex& self, std::string& key) {
            return self[key];
        }, nb::is_operator());

    m.attr("ColumnIndex") = m.attr("ObjectIndex");

    nb::class_<Series::IlocProxy>(m, "SeriesIlocProxy")
        .def("__getitem__", [](Series::IlocProxy& self, Eigen::Index idx) {
            return self[idx];
        }, nb::is_operator())
        .def("__getitem__", [](Series::IlocProxy& self, slice<Eigen::Index>& overlay) {
            return self[overlay];
        }, nb::is_operator())
        .def("__getitem__", [](Series::IlocProxy& self, nb::slice& nbSlice) {
            return self[nbSlice];
        }, nb::is_operator());

    nb::class_<Series::LocProxy>(m, "SeriesLocProxy")
        .def("__getitem__", [](Series::LocProxy& self, const std::string& key) {
            return self[key];
        }, nb::is_operator())
        .def("__getitem__", [](Series::LocProxy& self, nb::object& nbSlice) {
            return self[nbSlice];
        }, nb::is_operator());

    nb::class_<Series>(m, "Series")
        .def(nb::init<Eigen::VectorXd, std::shared_ptr<ObjectIndex>>())
        .def(nb::init<nb::ndarray<double>, nb::list>())
        .def("__repr__", &Series::to_string)
        .def("sum", &Series::sum)
        .def("mean", &Series::mean)
        .def("min", &Series::min)
        .def("max", &Series::max)
        .def("get_index", &Series::get_index)
        .def_prop_ro("values", &Series::values)
        .def("length", &Series::length)
        .def_prop_ro("mask", [](const Series &series) {
            return *series.mask_;
        })
        .def_prop_ro("iloc", &Series::iloc)
        .def_prop_ro("loc", &Series::loc)
        .def("head", &Series::head)
        .def("tail", &Series::tail);

    nb::class_<DataFrame::IlocProxy>(m, "DataFrameIlocProxy")
        .def("__getitem__", [](DataFrame::IlocProxy& self, Eigen::Index idx) {
            return self[idx];
        }, nb::is_operator())
        .def("__getitem__", [](DataFrame::IlocProxy& self, slice<Eigen::Index>& overlay) {
            return self[overlay];
        }, nb::is_operator())
        .def("__getitem__", [](DataFrame::IlocProxy& self, nb::slice& nbSlice) {
            return self[nbSlice];
        }, nb::is_operator());

    nb::class_<DataFrame::LocProxy>(m, "DataFrameLocProxy")
        .def("__getitem__", [](DataFrame::LocProxy& self, const std::string& key) {
            return self[key];
        }, nb::is_operator())
        .def("__getitem__", [](DataFrame::LocProxy& self, nb::object& nbSlice) {
            return self[nbSlice];
        }, nb::is_operator());

    nb::class_<DataFrame>(m, "DataFrame")
        .def("__repr__", &DataFrame::to_string)
        .def(nb::init<nb::ndarray<>, nb::list, nb::list>())
        .def_prop_ro("values", &DataFrame::values)
        .def_prop_ro("iloc", &DataFrame::iloc)
        .def_prop_ro("loc", &DataFrame::loc)
        .def("__getitem__", &DataFrame::operator[], nb::is_operator())
        .def("__getattr__", &DataFrame::operator[], nb::is_operator())
        .def("length", &DataFrame::length)
        .def("rows", &DataFrame::rows)
        .def("cols", &DataFrame::cols)
        .def("__len__", &DataFrame::length)
        .def("head", &DataFrame::head)
        .def("tail", &DataFrame::tail);
}
