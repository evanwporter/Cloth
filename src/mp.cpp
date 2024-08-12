#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <numeric>
#include <iomanip>
#include <Eigen/Dense>
#include "../lib/robinhood.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>

#define LOGGING_ENABLED 1

#if LOGGING_ENABLED
    #define LOG(msg) std::cout << msg << std::endl;
#else
    #define LOG(msg)
#endif

namespace nb = nanobind;
using MatrixXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T = Eigen::Index>
struct slice {
    T start, stop;
    int step;

    slice(T start_, T stop_, int step_) : start(start_), stop(stop_), step(step_) {
        LOG("Creating slice with start=" << start << ", stop=" << stop << ", step=" << step);
        if (step == 0) throw std::invalid_argument("Step cannot be zero");
    }

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
    
    T get_start() { return start; }
    T get_stop() { return stop; }
    int get_step() { return step; }

    friend std::ostream& operator<<(std::ostream& os, slice& sl) {
        os << "slice(" << sl.start << ", " << sl.stop << ", " << sl.step << ")";
        return os;
    }

    std::string to_string() {
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
slice<T> combine_slices(slice<T>&& mask, slice<T>&& overlay) {
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
T combine_slice_with_index(slice<T>& mask, T index) {
    LOG("Combining slice with index: mask(" << mask.start << ", " << mask.stop << ", " << mask.step << "), index=" << index);

    if (index < 0 || index >= mask.length()) {
        throw std::out_of_range("Index is out of bounds of the mask slice. Be sure to normalize slice first.");
    }

    T combined_index = mask.start + (index * mask.step);

    LOG("Resulting combined index: " << combined_index);

    return combined_index;
}

typedef Eigen::Index index_t;
typedef slice<Eigen::Index> mask_t;

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

    friend std::ostream& operator<<(std::ostream& os, ObjectIndex& objIndex) {
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

    std::shared_ptr<ObjectIndex> fast_init(std::shared_ptr<slice<Eigen::Index>> mask) {
        auto new_index = std::make_shared<ObjectIndex>(*this);
        new_index->mask_ = mask;
        std::cout << "ST" << *new_index << "\n";
        return new_index;
    }

    virtual Eigen::Index length() const {
        return mask_->length();
    }

    std::vector<std::string> keys() {
        std::vector<std::string> result;
        for (int i = mask_->start; i < mask_->stop; i += mask_->step) {
            result.push_back(keys_[i]);
        }
        return result;
    }

    std::string operator[](Eigen::Index idx) {
        if (idx < 0 || idx >= keys_.size()) {
            throw std::out_of_range("Index out of range");
        }
        return keys_[combine_slices<Eigen::Index>(*mask_, slice<Eigen::Index>(idx, idx + 1, 1, length())).start];
    }

    int operator[](std::string& key) {
        return index_.at(key);
    }
};

typedef ObjectIndex ColumnIndex;

template <typename Parent, typename Reduced>
class IlocProxy {
private:
    Parent& parent_;

public:
    IlocProxy(Parent& parent) : parent_(parent) {}

    Parent operator[](slice<Eigen::Index>& overlay) {
        overlay.normalize(parent.length());
        auto combined_mask = std::make_shared<slice<Eigen::Index>>(combine_slices(*parent.mask_, overlay));
        return Parent(parent.derived(), combined_mask);
    }

    Parent operator[](nb::slice& nbSlice) {
        auto [start, stop, step, slice_length] = nbSlice.compute(parent.length());
        auto overlay = std::make_shared<slice<Eigen::Index>>(static_cast<Eigen::Index>(start), static_cast<Eigen::Index>(stop), static_cast<Eigen::Index>(step), parent.length());
        auto combined_mask = std::make_shared<slice<Eigen::Index>>(combine_slices(*parent.mask_, *overlay));
        return Parent(parent.derived(), combined_mask);
    }

    Reduced operator[](Eigen::Index idx) {
        if (idx < 0 || idx >= parent.rows()) {
            throw std::out_of_range("Index out of range");
        }
        Eigen::Index combined_index = combine_slice_with_index(*parent.mask_, idx);
        auto row = std::make_shared<Eigen::VectorXd>(parent.values_->row(combined_index).transpose());
        return Reduced(row, parent.columns_);
    }
};

template <typename Parent, typename Reduced>
class LocProxy {
private:
    Parent& parent_;

public:
    LocProxy(Parent& parent) : parent_(parent) {}

    Parent operator[](nb::object& nbSlice) {
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

        Eigen::Index start_ = parent.index_->operator[](start_str);
        Eigen::Index stop_ = parent.index_->operator[](stop_str);

        auto overlay = slice<Eigen::Index>(start_, stop_, step_int, parent.length());

        return parent.iloc()[overlay];
    }
};

template <typename Derived>
class Frame {
public:
    Frame() = default;
    virtual ~Frame() = default;

    virtual std::string to_string() = 0;

    Eigen::Index length() {
        return mask()->length();
    }

    // virtual std::shared_ptr<MatrixXdRowMajor> values_f() = 0;
    virtual std::shared_ptr<ObjectIndex> index() = 0;
    virtual std::shared_ptr<slice<Eigen::Index>> mask() = 0;

    // virtual IlocProxy<DataFrame, Series>& iloc() = 0;
    // virtual LocProxy<DataFrame, Series>& loc() = 0;

    // Derived head(Eigen::Index n) {
    //     return iloc()[slice<Eigen::Index>(0, std::min(n, length()), 1)];
    // }

    // Derived tail(Eigen::Index n) {
    //     return iloc()[slice<Eigen::Index>(length() - std::min(n, length()), length(), 1)];
    // }

    friend std::ostream& operator<<(std::ostream& os, Frame& frame) {
        os << frame.to_string();
        return os;
    }
};


class Series : public Frame<Series> {
private:
    std::shared_ptr<Eigen::VectorXd> values_;
    std::shared_ptr<ObjectIndex> index_;
    std::shared_ptr<slice<Eigen::Index>> mask_;
    std::string name_;

    IlocProxy<Series, Series> iloc_;
    LocProxy<Series, Series> loc_;

public:
    Series(Series& other, std::shared_ptr<slice<Eigen::Index>> mask)
        : values_(other.values_),
          index_(other.index_),
          mask_(mask),
          name_(other.name_),
          iloc_(*this),
          loc_(*this) {}

    Series(std::shared_ptr<Eigen::VectorXd> values, std::shared_ptr<ObjectIndex> index)
        : values_(std::move(values)),
          index_(std::move(index)),
          mask_(std::make_shared<slice<Eigen::Index>>(0, values_->size(), 1)),
          iloc_(*this),
          loc_(*this) {}

    Series(Eigen::VectorXd &values, std::shared_ptr<ObjectIndex> index)
        : values_(std::make_shared<Eigen::VectorXd>(values)),
          index_(std::move(index)),
          mask_(std::make_shared<slice<Eigen::Index>>(0, values_->size(), 1)),
          iloc_(*this),
          loc_(*this) {}

    Series(nb::ndarray<double> values, nb::list index)
        : values_(std::make_shared<Eigen::VectorXd>(Eigen::Map<Eigen::VectorXd>(values.data(), values.size()))),
          index_(std::make_shared<ObjectIndex>(index)),
          mask_(std::make_shared<slice<Eigen::Index>>(0, values_->size(), 1)),
          iloc_(*this),
          loc_(*this) {}

    // std::shared_ptr<Eigen::VectorXd> values_f() override { return values_; }
    std::shared_ptr<ObjectIndex> index() override { return index_; }
    std::shared_ptr<slice<Eigen::Index>> mask() override { return mask_; }

    IlocProxy<Series, Series>& iloc() { return iloc_; }
    LocProxy<Series, Series>& loc() { return loc_; }

    std::string to_string() override {
        std::ostringstream oss;
        LOG("AT SERIES" << mask_->to_string() << mask_->length());
        try {
            Eigen::Index len = length();
            LOG("AT SERIES" << mask_->to_string() << mask_->length());
            for (Eigen::Index i = 0; i < len; ++i) {
                std::cout << i << " ";
                std::string index_value = index_->keys_[mask_->start + i * mask_->step];
                double series_value = values()(mask_->start + i * mask_->step);
                oss << index_value << "    " << series_value << "\n";
                std::cout << "Opera";
            }
            std::cout << "YOLO";

            if (!name_.empty()) {
                oss << "Name: " << name_ << "\n";
            }

        } catch (std::exception &e) {
            oss << "Error during stream output: " << e.what();
        }
        LOG(oss.str())
        return oss.str();
    }
    
    Eigen::Index size() {
        return mask_->length();
    }

    double sum() {
        return values().sum();
    }

    double mean() {
        return values().mean();
    }

    double min() {
        return values().minCoeff();
    }

    double max() {
        return values().maxCoeff();
    }

    std::vector<std::string> get_index() {
        return index_->keys();
    }

    const Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> values() {
        return Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(
            values_->data() + mask_->start, 
            mask_->length(), 
            Eigen::Stride<Eigen::Dynamic, 1>(1, 1)
        );
    }
};

class DataFrame : public Frame<DataFrame> {
private:
    std::shared_ptr<MatrixXdRowMajor> values_;
    std::shared_ptr<ObjectIndex> index_;
    std::shared_ptr<ColumnIndex> columns_;
    std::shared_ptr<slice<Eigen::Index>> mask_;

    IlocProxy<DataFrame, Series> iloc_;
    LocProxy<DataFrame, Series> loc_;

public:
    DataFrame(DataFrame& other, std::shared_ptr<slice<Eigen::Index>> mask)
        : values_(other.values_),
          index_(other.index_),
          columns_(other.columns_),
          mask_(mask),
          iloc_(*this),
          loc_(*this) {}

    DataFrame(std::shared_ptr<MatrixXdRowMajor> values, std::shared_ptr<ObjectIndex> index, std::shared_ptr<ColumnIndex> columns)
        : values_(std::move(values)),
          index_(std::move(index)),
          columns_(std::move(columns)),
          mask_(std::make_shared<slice<Eigen::Index>>(0, values_->rows(), 1)),
          iloc_(*this),
          loc_(*this) {}

    DataFrame(nb::ndarray<> values, nb::list index, nb::list columns)
        : index_(std::make_shared<ObjectIndex>(index)),
          columns_(std::make_shared<ColumnIndex>(columns)),
          mask_(std::make_shared<slice<Eigen::Index>>(0, values.shape(0), 1)),
          iloc_(*this),
          loc_(*this) {
        Eigen::Index rows = static_cast<Eigen::Index>(values.shape(0));
        Eigen::Index cols = static_cast<Eigen::Index>(values.shape(1));
        values_ = std::make_shared<MatrixXdRowMajor>(Eigen::Map<MatrixXdRowMajor>(static_cast<double*>(values.data()), rows, cols));
    }

    // Override the virtual functions
    // std::shared_ptr<MatrixXdRowMajor> values_f() override { return values_; }
    std::shared_ptr<ObjectIndex> index() override { return index_; }
    std::shared_ptr<slice<Eigen::Index>> mask() override { return mask_; }

    IlocProxy<DataFrame, Series>& iloc() { return iloc_; }
    LocProxy<DataFrame, Series>& loc() { return loc_; }

    Eigen::Index rows() {
        return length();
    }

    Eigen::Index cols() {
        return values_->cols();
    }

    virtual Eigen::Map<MatrixXdRowMajor, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> values() {
        return Eigen::Map<MatrixXdRowMajor, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>(
            values_->data(), 
            values_->rows(), 
            values_->cols(),
            Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(values_->cols(), 1)
        );
    }

    Series operator[](std::string& colName) {
        return Series(std::make_shared<Eigen::VectorXd>(values_->col(columns_->index_.at(colName))), index_);
    }

    std::string to_string() override {
        std::ostringstream oss;

        std::vector<std::string> rowNames = index_->keys();
        std::vector<std::string> colNames = columns_->keys();

        std::vector<size_t> colWidths(colNames.size(), 0);

        for (size_t j = 0; j < colNames.size(); ++j) {
            colWidths[j] = colNames[j].length();
        }

        for (Eigen::Index i = mask_->start; i < mask_->stop; i += mask_->step) {
            for (Eigen::Index j = 0; j < cols(); ++j) {
                std::ostringstream temp_oss;
                temp_oss << std::fixed << std::setprecision(2) << (*values_)(i, j);
                if (temp_oss.str().length() > colWidths[j]) {
                    colWidths[j] = temp_oss.str().length();
                }
            }
        }

        size_t rowNameWidth = 0;
        for (auto& rowName : rowNames) {
            if (rowName.length() > rowNameWidth) {
                rowNameWidth = rowName.length();
            }
        }

        oss << std::setw(static_cast<int>(rowNameWidth) + 2) << " ";
        for (size_t j = 0; j < colNames.size(); ++j) {
            oss << std::setw(colWidths[j] + 2) << colNames[j];
        }
        oss << std::endl;

        for (Eigen::Index i = mask_->start; i < mask_->stop; i += mask_->step) {
            oss << std::setw(static_cast<int>(rowNameWidth) + 2) << rowNames[i];
            for (Eigen::Index j = 0; j < cols(); ++j) {
                oss << std::setw(colWidths[j] + 2) 
                << std::fixed << std::setprecision(2) << (*values_)(i, j);
            }
            oss << std::endl;
        }

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
        .def("__getitem__", [](ObjectIndex &self, std::string &key) {
            return self[key];
        }, nb::is_operator())
        .def("fast_init", &ObjectIndex::fast_init)
        .def("keys", &ObjectIndex::keys)
        .def_rw("index", &ObjectIndex::keys_);

    m.attr("ColumnIndex") = m.attr("ObjectIndex");


    nb::class_<Series>(m, "Series")
        .def(nb::init<Eigen::VectorXd, std::shared_ptr<ObjectIndex>>())
        .def(nb::init<nb::ndarray<double>, nb::list>())
        .def("sum", &Series::sum)
        .def("mean", &Series::mean)
        .def("min", &Series::min)
        .def("max", &Series::max)
        .def("get_index", &Series::get_index)
        .def_prop_ro("values", &Series::values)
        .def_prop_ro("iloc", [](Series& self) -> IlocProxy<Series, Series>& {
            return self.iloc();
        })
        .def_prop_ro("loc", [](Series& self) -> LocProxy<Series, Series>& {
            return self.loc();
        })
        // .def("head", &Series::head)
        // .def("tail", &Series::tail)
        .def("__repr__", &Series::to_string);

    nb::class_<DataFrame>(m, "DataFrame")
        .def(nb::init<nb::ndarray<>, nb::list, nb::list>())
        .def_prop_ro("values", &DataFrame::values)
        .def_prop_ro("iloc", [](DataFrame& self) -> IlocProxy<DataFrame, Series>& {
            return self.iloc();
        })
        .def_prop_ro("loc", [](DataFrame& self) -> LocProxy<DataFrame, Series>& {
            return self.loc();
        })
        .def("rows", &DataFrame::rows)
        .def("cols", &DataFrame::cols)
        // .def("head", &DataFrame::head)
        // .def("tail", &DataFrame::tail)
        .def("__repr__", &DataFrame::to_string);
}
