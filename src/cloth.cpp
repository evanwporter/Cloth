#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <numeric>

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
        if (step == 0) throw std::invalid_argument("Step cannot be zero");
    }

    // If length is passed, then normalize
    slice(T start_, T stop_, int step_, Eigen::Index length) : start(start_), stop(stop_), step(step_) {
        if (step == 0) throw std::invalid_argument("Step cannot be zero");
        normalize(length);
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

template <typename T = Eigen::Index>
slice<T> combine_slices(const slice<T>& mask, const slice<T>& overlay) { //, Eigen::Index length_mask) {
    
    // Ensure overlay fits within the mask
    if (overlay.start < 0 || overlay.stop < 0) {
        throw std::out_of_range("Overlay slice is out of bounds of the mask slice. Be sure to normalize slice first.");
    }
    int start = mask.start + (overlay.start * mask.step);
    int stop = mask.start + (overlay.stop * mask.step);
    int step = mask.step * overlay.step;
    return slice<T>(start, stop, step);
}

// template <typename T = double>
// std::vector<T> ndarray2vec(nb::ndarray<> arr) {
//     if (arr.ndim() != 1 || arr.dtype() != nb::dtype<T>()) {
//         throw std::invalid_argument("Input should be a 1D array of doubles");
//     }

//     std::vector<T> vec(arr.size());

//     // Copy data from the NumPy array to the std::vector
//     std::memcpy(vec.data(), arr.data(), arr.size() * sizeof(T));

//     return vec;
// }

class Index_ {
public:
    virtual ~Index_() = default;
};

class ObjectIndex : public Index_ {
public:
    robin_hood::unordered_map<std::string, int> index_;
    std::vector<std::string> keys_;
    std::shared_ptr<slice<Eigen::Index>> mask_;


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
        return new_index;
    }

    virtual Eigen::Index length() const {
        return mask_->length();
    }

    std::vector<std::string> keys() const {
        std::vector<std::string> result;
        // result.reserve(length());
        for (int i = mask_->start; i < mask_->stop; i += mask_->step) {
            // result.push_back(keys_[i]);
            std::cout << keys_[i];
        }
        return result;
    }

    std::string operator[](Eigen::Index idx) const {
        if (idx < 0 || idx >= keys_.size()) {
            throw std::out_of_range("Index out of range");
        }
        return keys_[combine_slices<Eigen::Index>(*mask_, slice<Eigen::Index>(idx, idx + 1, 1, length())).get_start()];
    }
};

typedef ObjectIndex ColumnIndex;

class Series {
public:
    Eigen::VectorXd values_;
    std::shared_ptr<ObjectIndex> index_;
    std::string name_;
    std::shared_ptr<slice<Eigen::Index>> mask_;

    Series(Eigen::VectorXd values, std::shared_ptr<ObjectIndex> index)
        : values_(std::move(values)), 
          index_(std::move(index)), 
          mask_(std::make_shared<slice<Eigen::Index>>(0, values_.size(), 1)) {}

    Series(nb::ndarray<double> values, nb::list index)
        : values_(Eigen::Map<const Eigen::VectorXd>(values.data(), values.size())),
          index_(std::make_shared<ObjectIndex>(index)),
          mask_(std::make_shared<slice<Eigen::Index>>(0, values_.size(), 1)) {}

    std::string repr() const {
        std::ostringstream oss;
        oss << "Series, Length: " << length() << "\nIndex and Values:\n";
        for (Eigen::Index i = 0; i < length(); ++i) {
            oss << index_->keys_[i] << ": " << values()(i) << "\n";
        }
        return oss.str();
    }

    Eigen::Index length() const {
        return mask_->length();
    }

    Eigen::Index size() const {
        return mask_->length();
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

    virtual const Eigen::VectorXd& values() const {
        return values_;
    }

    virtual const ObjectIndex& index() const {
        return *index_;
    }

    class view;


    class IlocProxy {
    private:
        const Series& parent;

    public:
        IlocProxy(const Series& parent) : parent(parent) {}

        view operator[](const nb::slice& nbSlice) const; // Declaration only

        double operator[](Eigen::Index idx) const {
            if (idx < 0 || idx >= parent.values_.size()) {
                throw std::out_of_range("Index out of range");
            }
            return parent.values_(combine_slices(*parent.mask_, slice<Eigen::Index>(idx, idx + 1, 1, parent.length())).start);
        }

        // std::shared_ptr<SeriesView> operator[](Eigen::Index idx) const {
        //     auto combined_slice = combine_slices(*parent.mask_, slice<Eigen::Index>(idx, idx + 1, 1), parent.values_.size());
        //     return parent.create_view(std::make_shared<slice<Eigen::Index>>(combined_slice));
        // }
    };

    IlocProxy iloc() {
        return IlocProxy(*this);
    }
};

class Series::view : public Series {
public:

    std::shared_ptr<ObjectIndex> index_;
    std::shared_ptr<slice<Eigen::Index>> mask_;

    view(const Series& parent, std::shared_ptr<slice<Eigen::Index>> mask)
        : Series(parent), mask_(std::move(mask)), index_(parent.index_->fast_init(mask)) {}

    // const Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> values() const override {
    const Eigen::VectorXd& values() const override {
        
        return Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(
            values_.data() + mask_->start, 
            mask_->length(), 
            Eigen::Stride<Eigen::Dynamic, 1>(mask_->step, 1)
        );
    }

    // const ObjectIndex& index() const override {
    //     std::vector<std::string> masked_keys;
    //     for (Eigen::Index i = mask_->start; i < mask_->stop; i += mask_->step) {
    //         masked_keys.push_back(index_->keys_[i]);
    //     }
    //     return ObjectIndex(masked_keys);
    // }
};

Series::view Series::IlocProxy::operator[](const nb::slice& nbSlice) const {
    auto [start, stop, step, slice_length] = nbSlice.compute(parent.values().size());
    auto overlay = std::make_shared<slice<Eigen::Index>>(static_cast<int>(start), static_cast<int>(stop), static_cast<int>(step), parent.size());
    auto combined_mask = std::make_shared<slice<Eigen::Index>>(combine_slices(*parent.mask_, *overlay));
    return Series::view(parent, combined_mask);
};

class DataFrame {
public:
    MatrixXdRowMajor values_;
    std::shared_ptr<ObjectIndex> index_;
    std::shared_ptr<ColumnIndex> columns_;
    std::shared_ptr<slice<Eigen::Index>> mask_;

    DataFrame(MatrixXdRowMajor values, std::shared_ptr<ObjectIndex> index, std::shared_ptr<ColumnIndex> columns)
        : values_(std::move(values)), 
          index_(std::move(index)), 
          columns_(std::move(columns)), 
          mask_(std::make_shared<slice<Eigen::Index>>(0, values_.rows(), 1)) {}

    DataFrame(nb::list values, nb::list index, nb::list columns)
        : index_(std::make_shared<ObjectIndex>(index)),
          columns_(std::make_shared<ColumnIndex>(columns)) {
        Eigen::Index rows = static_cast<Eigen::Index>(values.size());
        Eigen::Index cols = static_cast<Eigen::Index>(nb::cast<nb::list>(values[0]).size());

        values_ = MatrixXdRowMajor(rows, cols);
        for (Eigen::Index i = 0; i < rows; ++i) {
            auto row = nb::cast<nb::list>(values[i]);
            // Check that size row == cols
            for (Eigen::Index j = 0; j < cols; ++j) {
                values_(i, j) = nb::cast<double>(row[j]);
            }
        }

        mask_ = std::make_shared<slice<Eigen::Index>>(0, values_.rows(), 1);
    }

    DataFrame(nb::ndarray<> values, nb::list index, nb::list columns)
        : index_(std::make_shared<ObjectIndex>(index)),
          columns_(std::make_shared<ColumnIndex>(columns)) {
        Eigen::Index rows = static_cast<Eigen::Index>(values.shape(0));
        Eigen::Index cols = static_cast<Eigen::Index>(values.shape(1));
        values_ = Eigen::Map<MatrixXdRowMajor>(static_cast<double*>(values.data()), rows, cols);
    
        mask_ = std::make_shared<slice<Eigen::Index>>(0, values_.rows(), 1);
    }

    class view;

    // Corrected method declaration and definition
    class IlocProxy {
    private:
        const DataFrame& parent;

    public:
        IlocProxy(const DataFrame& parent) : parent(parent) {}

        view operator[](const nb::slice& nbSlice) const;
    };

    IlocProxy iloc() {
        return IlocProxy(*this);
    }

    Eigen::Index rows() const {
        return mask_->length();
    }

    Eigen::Index cols() const {
        return values_.cols();
    }

    virtual const MatrixXdRowMajor& values() const {
        return values_;
    }

    virtual const ObjectIndex& index() const {
        return *index_;
    }

    std::string repr() const {
        std::ostringstream oss;
        oss << "Columns: " << cols() << ", Rows: " << rows() << "\nValues:\n";
        for (Eigen::Index i = 0; i < rows(); ++i) {
            for (Eigen::Index j = 0; j < cols(); ++j) {
                oss << values_(i, j) << " ";
            }
            oss << "\n";
        }
        // oss << values_ << " " << rows() << values_.rows() << " " << mask_->start << " " << mask_->stop << " " << mask_->step;// << " " << cols();
        return oss.str();
    }
};

class DataFrame::view : public DataFrame {
public:
    std::shared_ptr<slice<Eigen::Index>> mask_;

    view(const DataFrame& parent, std::shared_ptr<slice<Eigen::Index>> mask)
        : DataFrame(parent), mask_(std::move(mask)) {
        // Re-initialize indices to reflect the mask
        this->index_ = parent.index_->fast_init(mask_);
    }

    const MatrixXdRowMajor& values() const override {
        return Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(
            values_.data() + mask_->start, 
            mask_->length(), 
            Eigen::Stride<Eigen::Dynamic, 1>(mask_->step, 1)
        );    
    }

    const ObjectIndex& index() const override {
        return *index_;
    }
};

DataFrame::view DataFrame::IlocProxy::operator[](const nb::slice& nbSlice) const {
    auto [start, stop, step, slice_length] = nbSlice.compute(parent.rows());
    auto overlay = std::make_shared<slice<Eigen::Index>>(static_cast<int>(start), static_cast<int>(stop), static_cast<int>(step), parent.rows());
    auto combined_mask = std::make_shared<slice<Eigen::Index>>(combine_slices(*parent.mask_, *overlay));
    return DataFrame::view(parent, combined_mask);
}


NB_MODULE(cloth, m) {
    nb::class_<slice<Eigen::Index>>(m, "slice")
        .def(nb::init<int, int, int>())
        .def("normalize", &slice<Eigen::Index>::normalize)
        .def_prop_ro("length", &slice<Eigen::Index>::length);

    nb::class_<Index_>(m, "Index_");

    nb::class_<ObjectIndex, Index_>(m, "ObjectIndex")
        .def(nb::init<std::vector<std::string>>()) 
        .def("fast_init", &ObjectIndex::fast_init)
        .def("keys", &ObjectIndex::keys)
        // .def("get_mask", &ObjectIndex::get_mask)
        .def_rw("index", &ObjectIndex::keys_);
        // .def_rw("mask", &ObjectIndex::mask_);

    m.attr("ColumnIndex") = m.attr("ObjectIndex");

    nb::class_<Series::IlocProxy>(m, "SeriesIlocProxy")
        .def("__getitem__", [](const Series::IlocProxy &self, const nb::slice &nbSlice) {
            return self[nbSlice];
        }, nb::is_operator());

    nb::class_<Series>(m, "Series")
        .def(nb::init<Eigen::VectorXd, std::shared_ptr<ObjectIndex>>())
        .def(nb::init<nb::ndarray<double>, nb::list>())
        .def("__repr__", &Series::repr)
        .def("sum", &Series::sum)
        .def("mean", &Series::mean)
        .def("min", &Series::min)
        .def("max", &Series::max)
        .def("get_row", &Series::get_row)
        .def("get_index", &Series::get_index)
        .def_prop_ro("iloc", &Series::iloc);

    nb::class_<Series::view, Series>(m, "view")
        .def("values", &Series::view::values)
        .def("index", &Series::view::index);

    nb::class_<DataFrame::IlocProxy>(m, "DataFrameIlocProxy")
        .def("__getitem__", [](const DataFrame::IlocProxy &self, const nb::slice &nbSlice) {
            return self[nbSlice];
        }, nb::is_operator());

    nb::class_<DataFrame>(m, "DataFrame")
        .def("__repr__", &DataFrame::repr)
        .def(nb::init<nb::list, nb::list, nb::list>())
        .def(nb::init<nb::ndarray<>, nb::list, nb::list>())
        .def_prop_ro("iloc", &DataFrame::iloc);
}
