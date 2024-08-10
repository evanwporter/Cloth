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

typedef slice<Eigen::Index> mask_t;

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

class Series {
public:
    std::shared_ptr<Eigen::VectorXd> values_;
    std::shared_ptr<ObjectIndex> index_;
    std::string name_;
    std::shared_ptr<slice<Eigen::Index>> mask_;
          
    Series(const Series& other,  std::shared_ptr<slice<Eigen::Index>> mask)
        : values_(other.values_),
          index_(other.index_),
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


    std::string to_string() const {
        std::ostringstream oss;
        oss << *this;
        return oss.str();
    }

    virtual Eigen::Index length() const {
        LOG("Calculating length of Series");
        return mask_->length();  // Use mask_ to determine length
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

    // double get_row(const std::string& arg) const {
    //     if (index_->index_.find(arg) == index_->index_.end()) {
    //         throw std::out_of_range("Key '" + arg + "' not found in the Series index.");
    //     }
    //     int idx = index_->index_.at(arg);
    //     return values_->(idx);
    // }

    std::vector<std::string> get_index() const {
        return index_->keys();
    }

    virtual const Eigen::VectorXd& values() const {
        return *values_;
    }

    Eigen::VectorXd py_val() const {
        return values();
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
            if (idx < 0 || idx >= parent.size()) {
                throw std::out_of_range("Index out of range");
            }
            return (*parent.values_)(combine_slices(*parent.mask_, slice<Eigen::Index>(idx, idx + 1, 1, parent.length())).start);
        }

        // std::shared_ptr<SeriesView> operator[](Eigen::Index idx) const {
        //     auto combined_slice = combine_slices(*parent.mask_, slice<Eigen::Index>(idx, idx + 1, 1), parent.values_.size());
        //     return parent.create_view(std::make_shared<slice<Eigen::Index>>(combined_slice));
        // }
    };

    IlocProxy iloc() const {
        return IlocProxy(*this);
    }
};

class Series::view : public Series {
public:
    std::shared_ptr<ObjectIndex> index_;
    std::shared_ptr<slice<Eigen::Index>> mask_;

    Series::view(const Series& parent, std::shared_ptr<slice<Eigen::Index>> mask)
        : Series(parent, mask), mask_(std::move(mask)) {
        LOG("Constructing Series::view with parent Series and mask(" << mask_->start << ", " << mask_->stop << ", " << mask_->step << ")");

        if (!mask_) {
            LOG("Error: Mask is null!");
            throw std::invalid_argument("Mask cannot be null");
        }

        // Re-initialize indices to reflect the mask
        this->index_ = parent.index_->fast_init(mask_);
        LOG("Series::view constructed successfully with index: " << *this->index_);
    }


    // const Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> values() const override {
    const Eigen::VectorXd& values() const override {
        
        std::cout << "MASK " << *mask_ << "\n";
        
        return Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(
            values_->data() + mask_->start, 
            mask_->length(), 
            Eigen::Stride<Eigen::Dynamic, 1>(mask_->step, 1)
        );
    }

    // Eigen::Index length() const override {
    //     return mask_->length();
    // }

    std::string to_string() const {
        std::ostringstream oss;
        oss << *this;
        return oss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const Series::view& view) {
        // Avoid directly invoking methods that might cause recursion
        try {
            // Print only essential information initially
            os << "Series::view with length: " << view.mask_->length() << "\n";

            // Output the mask, but ensure no recursive or circular dependencies are invoked
            os << "Mask start: " << view.mask_->start << ", stop: " << view.mask_->stop << "\n";

            // Avoid calling any overridden methods directly
            // os << "Values: " << view.values() << "\n"; // Potentially recursive
            os << "Index: " << *view.index_ << "\n"; // Ensure this does not cause recursion

        } catch (const std::exception &e) {
            os << "Error during stream output: " << e.what();
        }

        return os;
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

    std::cout << stop;
    auto overlay = std::make_shared<slice<Eigen::Index>>(static_cast<Eigen::Index>(start), static_cast<Eigen::Index>(stop), static_cast<Eigen::Index>(step), parent.size());
    auto combined_mask = std::make_shared<slice<Eigen::Index>>(combine_slices(*parent.mask_, *overlay));
    std::cout << "Combined Mask " << combined_mask->start << " " << combined_mask->stop << "\n";
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

    class IlocProxy {
    private:
        const DataFrame& parent;

    public:
        IlocProxy(const DataFrame& parent) : parent(parent) {}

        view operator[](const nb::slice& nbSlice) const;

        Series operator[](Eigen::Index idx) const {
            if (idx < 0 || idx >= parent.rows()) {
                throw std::out_of_range("Index out of range");
            }
            // Extract the correct row using the mask and the requested index
            auto row = std::make_shared<Eigen::VectorXd>(parent.values_.row(parent.mask_->start + idx * parent.mask_->step).transpose());
            return Series(row, parent.columns_);
        }
    };


    IlocProxy iloc() const {
        return IlocProxy(*this);
    }

    class LocProxy {
    private:
        const DataFrame& parent;

    public:
        LocProxy(const DataFrame& parent) : parent(parent) {}

        Series operator[](const std::string& key) const {
            int idx = parent.index_->operator[](key);  // Use the overloaded [] operator
            return parent.iloc()[idx];  // Access DataFrame's iloc, not ObjectIndex's
        }
    };

    LocProxy loc() const {
        return LocProxy(*this);
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

    friend std::ostream& operator<<(std::ostream& os, const DataFrame& df) {
        os << "DataFrame\nRows: " << df.values_.rows() << ", Columns: " << df.values_.cols() << "\nValues:\n";
        for (Eigen::Index i = 0; i < df.values_.rows(); ++i) {
            for (Eigen::Index j = 0; j < df.values_.cols(); ++j) {
                os << df.values_(i, j) << " ";
            }
            os << "\n";
        }
        return os;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << *this;
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
        .def_rw("index", &ObjectIndex::keys_);
        // .def_rw("mask", &ObjectIndex::mask_);

    m.attr("ColumnIndex") = m.attr("ObjectIndex");

    nb::class_<Series::IlocProxy>(m, "SeriesIlocProxy")
        .def("__getitem__", [](const Series::IlocProxy &self, const nb::slice &nbSlice) {
            return self[nbSlice];
        }, nb::is_operator())
        .def("__getitem__", [](const Series::IlocProxy &self, Eigen::Index idx) {
            return self[idx];
        }, nb::is_operator());

    nb::class_<Series>(m, "Series")
        .def(nb::init<Eigen::VectorXd, std::shared_ptr<ObjectIndex>>())
        .def(nb::init<nb::ndarray<double>, nb::list>())
        .def("__repr__", &Series::to_string)
        .def("sum", &Series::sum)
        .def("mean", &Series::mean)
        .def("min", &Series::min)
        .def("max", &Series::max)
        // .def("get_row", &Series::get_row)
        .def("get_index", &Series::get_index)
        .def_prop_ro("values", &Series::values)
        .def("length", &Series::length)
        .def_prop_ro("mask", [](const Series &series) {
            return *series.mask_;
        })
        .def_prop_ro("iloc", &Series::iloc);

    nb::class_<Series::view, Series>(m, "SeriesView")
        .def(nb::init<const Series&, std::shared_ptr<slice<Eigen::Index>>>())
        .def_prop_ro("mask", [](const Series::view &view) {
            return *view.mask_;
        })
        .def("__repr__", &Series::view::to_string);

        // .def("values", &Series::view::values)
        // .def("index", &Series::view::index);

    nb::class_<DataFrame::IlocProxy>(m, "DataFrameIlocProxy")
        .def("__getitem__", [](const DataFrame::IlocProxy &self, const nb::slice &nbSlice) {
            return self[nbSlice];
        }, nb::is_operator())
        .def("__getitem__", [](const DataFrame::IlocProxy &self, Eigen::Index idx) {
            return self[idx];
        }, nb::is_operator());

    nb::class_<DataFrame>(m, "DataFrame")
        .def("__repr__", &DataFrame::to_string)
        .def(nb::init<nb::list, nb::list, nb::list>())
        .def(nb::init<nb::ndarray<>, nb::list, nb::list>())
        .def_prop_ro("iloc", &DataFrame::iloc);
}
