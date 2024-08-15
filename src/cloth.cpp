#define LOGGING_ENABLED 0

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
#include <cmath>

#include <Eigen/Dense>
#include "../lib/robinhood.h"
#include "../include/read_csv.h"
#include "../include/dt.h"
#include "../include/slice.h"
#include "../include/digitize.h"


namespace nb = nanobind;
using MatrixXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

#ifndef INDEX_T
#define INDEX_T
using index_t = Eigen::Index;
#endif

using mask_t = slice<index_t>;

class Index_ {
public:
    virtual ~Index_() = default;
};

class StringIndex : public Index_ {
public:
    std::shared_ptr<robin_hood::unordered_map<std::string, int>> index_;
    std::shared_ptr<std::vector<std::string>> keys_;
    std::shared_ptr<mask_t> mask_;

    StringIndex(std::shared_ptr<robin_hood::unordered_map<std::string, int>> index, std::shared_ptr<std::vector<std::string>> keys)
        : index_(std::move(index)), keys_(std::move(keys)), 
          mask_(std::make_shared<mask_t>(0, static_cast<int>(this->keys_->size()), 1)) {}

    StringIndex(std::shared_ptr<StringIndex> other, std::shared_ptr<mask_t> mask)
        : index_(other->index_), keys_(other->keys_), mask_(std::move(mask)) {}

    explicit StringIndex(std::vector<std::string> keys)
        : keys_(std::make_shared<std::vector<std::string>>(std::move(keys))),
          index_(std::make_shared<robin_hood::unordered_map<std::string, int>>()) {
        for (size_t i = 0; i < this->keys_->size(); ++i) {
            (*index_)[(*this->keys_)[i]] = static_cast<int>(i);
        }
        mask_ = std::make_shared<mask_t>(0, static_cast<int>(this->keys_->size()), 1);
    }

    explicit StringIndex(nb::list keys)
        : keys_(std::make_shared<std::vector<std::string>>()), 
          index_(std::make_shared<robin_hood::unordered_map<std::string, int>>()) {
        keys_->reserve(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            keys_->push_back(std::move(nb::cast<std::string>(keys[i])));
            (*index_)[keys_->back()] = static_cast<int>(i);
        }
        mask_ = std::make_shared<mask_t>(0, static_cast<int>(keys_->size()), 1);
    }

    std::vector<std::string> keys() const {
        std::vector<std::string> result;
        result.reserve(length());
        for (int i = mask_->start; i < mask_->stop; i += mask_->step) {
            result.push_back((*keys_)[i]);
        }
        return result;
    }

    std::string operator[](Eigen::Index idx) const {
        if (idx < 0 || idx >= static_cast<Eigen::Index>(keys_->size())) {
            throw std::out_of_range("Index out of range");
        }
        return (*keys_)[combine_slice_with_index(*mask_, idx)];
    }

    int operator[](const std::string& key) const {
        return index_->at(key);
    }

    friend std::ostream& operator<<(std::ostream& os, const StringIndex& strIndex) {
        auto k = strIndex.keys();
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

    Eigen::Index length() const {
        return mask_->length();
    }
};

using ColumnIndex = StringIndex;

class DateTimeIndex : public Index_ {
public:
    std::shared_ptr<robin_hood::unordered_map<Datetime64, int>> index_;
    std::shared_ptr<std::vector<Datetime64>> keys_;
    std::shared_ptr<mask_t> mask_;

    
    DateTimeIndex(std::shared_ptr<robin_hood::unordered_map<Datetime64, int>> index, std::shared_ptr<std::vector<Datetime64>> keys)
        : index_(std::move(index)), keys_(std::move(keys)), 
          mask_(std::make_shared<mask_t>(0, static_cast<int>(this->keys_->size()), 1)) {}

    
    DateTimeIndex(std::shared_ptr<DateTimeIndex> other, std::shared_ptr<mask_t> mask)
        : index_(other->index_), keys_(other->keys_), mask_(std::move(mask)) {}

    
    explicit DateTimeIndex(std::vector<std::string> iso_keys)
        : keys_(std::make_shared<std::vector<Datetime64>>()),
          index_(std::make_shared<robin_hood::unordered_map<Datetime64, int>>()) {

        for (size_t i = 0; i < iso_keys.size(); ++i) {
            Datetime64 dt(iso_keys[i]);
            keys_->push_back(dt);
            (*index_)[dt] = static_cast<int>(i);
        }
        mask_ = std::make_shared<mask_t>(0, static_cast<int>(this->keys_->size()), 1);
    }

    
    explicit DateTimeIndex(nb::list iso_keys)
        : keys_(std::make_shared<std::vector<Datetime64>>()), 
          index_(std::make_shared<robin_hood::unordered_map<Datetime64, int>>()) {

        keys_->reserve(iso_keys.size());
        for (size_t i = 0; i < iso_keys.size(); ++i) {
            std::string iso_key = nb::cast<std::string>(iso_keys[i]);
            Datetime64 dt(iso_key);
            keys_->push_back(dt);
            (*index_)[dt] = static_cast<int>(i);
        }
        mask_ = std::make_shared<mask_t>(0, static_cast<int>(keys_->size()), 1);
    }

    
    std::vector<Datetime64> keys() const {
        std::vector<Datetime64> result;
        result.reserve(length());
        for (int i = mask_->start; i < mask_->stop; i += mask_->step) {
            result.push_back((*keys_)[i]);
        }
        return result;
    }

    
    Datetime64 operator[](Eigen::Index idx) const {
        if (idx < 0 || idx >= static_cast<Eigen::Index>(keys_->size())) {
            throw std::out_of_range("Index out of range");
        }
        return (*keys_)[combine_slice_with_index(*mask_, idx)];
    }

    
    int operator[](const Datetime64& key) const {
        return index_->at(key);
    }

    
    friend std::ostream& operator<<(std::ostream& os, const DateTimeIndex& dtIndex) {
        auto k = dtIndex.keys();
        os << "[";
        for (size_t i = 0; i < k.size(); ++i) {
            os << k[i].seconds();  
            if (i < k.size() - 1) {
                os << ", ";
            }
        }
        os << "]";
        return os;
    }

    
    Eigen::Index length() const {
        return mask_->length();
    }
};


template <typename Derived>
class IlocBase {
public:
    virtual ~IlocBase() = default;
    IlocBase() = default;

    virtual Derived& parent() = 0;

    Derived operator[](slice<Eigen::Index>& overlay) {
        overlay.normalize(parent().length());
        auto combined_mask = std::make_shared<slice<Eigen::Index>>(combine_slices(*parent().mask_, overlay));
        return Derived(parent(), std::move(combined_mask));
    }

    Derived operator[](nb::slice& nbSlice) {
        auto [start, stop, step, slice_length] = nbSlice.compute(parent().length());
        auto overlay = std::make_shared<slice<Eigen::Index>>(static_cast<Eigen::Index>(start), static_cast<Eigen::Index>(stop), static_cast<Eigen::Index>(step), parent().length());
        auto combined_mask = std::make_shared<slice<Eigen::Index>>(combine_slices(*parent().mask_, *overlay));
        return Derived(parent(), std::move(combined_mask));
    }
};

template <typename Derived>
class LocBase {
public:
    virtual ~LocBase() = default;
    LocBase() = default;

    virtual Derived& parent() = 0;

    Derived operator[](nb::object& nbSlice) {
        std::string start_str, stop_str;
        Eigen::Index step_int = 1;

        Eigen::Index start_ = 0;
        Eigen::Index stop_ = parent().length(); // default to length (exclusive)

        if (nb::hasattr(nbSlice, "start")) {
            nb::object start = nb::getattr(nbSlice, "start");
            if (!start.is_none()) {
                start_str = nb::cast<std::string>(start);
                start_ = parent().index_->operator[](start_str);
            }
        }

        if (nb::hasattr(nbSlice, "stop")) {
            nb::object stop = nb::getattr(nbSlice, "stop");
            if (!stop.is_none()) {
                stop_str = nb::cast<std::string>(stop);
                stop_ = parent().index_->operator[](stop_str);
            }
        }

        if (nb::hasattr(nbSlice, "step")) {
            nb::object step = nb::getattr(nbSlice, "step");
            if (!step.is_none()) {
                step_int = nb::cast<Eigen::Index>(step);
            }
        }

        stop_ += 1;

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
    std::shared_ptr<StringIndex> index_;
    std::string name_;
    std::shared_ptr<slice<Eigen::Index>> mask_;
          
    Series(const Series& other, std::shared_ptr<slice<Eigen::Index>> mask)
        : values_(other.values_),
        index_(std::make_shared<StringIndex>(other.index_, mask)),
        name_(other.name_),
        mask_(std::move(mask)) {}


    Series(std::shared_ptr<Eigen::VectorXd> values, std::shared_ptr<StringIndex> index)
        : values_(std::move(values)), 
          index_(std::move(index)) 
    {
        mask_ = std::make_shared<slice<Eigen::Index>>(0, values_->size(), 1); 
    }

    Series(const Eigen::VectorXd &values, std::shared_ptr<StringIndex> index)
        : values_(std::make_shared<Eigen::VectorXd>(values)), 
          index_(std::move(index)) 
    {
        mask_ = std::make_shared<slice<Eigen::Index>>(0, values_->size(), 1); 
    }

    Series(nb::ndarray<double> values, nb::list index)
        : values_(std::make_shared<Eigen::VectorXd>(Eigen::Map<const Eigen::VectorXd>(values.data(), values.size()))),
          index_(std::make_shared<StringIndex>(index))
    {
        mask_ = std::make_shared<slice<Eigen::Index>>(0, values_->size(), 1);
    }

    std::shared_ptr<slice<Eigen::Index>> mask() const override {
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

    const Eigen::Map<const Eigen::VectorXd> values() const {
        return Eigen::Map<const Eigen::VectorXd>(
            values_->data() + mask_->start, 
            mask_->length()
        );
    }

    Eigen::VectorXd py_val() const {
        return values();
    }

    const StringIndex& index() const {
        return *index_;
    }

    class IlocProxy : public IlocBase<Series> {
    private:
        Series& parent_;

    public:
        explicit IlocProxy(Series& parent_) : parent_(parent_) {}

        Series& parent() override {
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
        explicit LocProxy(Series& parent_) : parent_(parent_) {}

        Series& parent() override {
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

    friend std::ostream& operator<<(std::ostream& os, const Series& series) {
        try {
            Eigen::Index len = series.length();
            for (Eigen::Index i = 0; i < len; ++i) {
                std::string index_value = series.index_->keys_->operator[](series.mask_->start + i * series.mask_->step);
                double series_value = (*series.values_)(series.mask_->start + i * series.mask_->step);
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


};

class DataFrame : public Frame<DataFrame> {
public:
    std::shared_ptr<MatrixXdRowMajor> values_;
    std::shared_ptr<StringIndex> index_;
    std::shared_ptr<ColumnIndex> columns_;
    std::shared_ptr<slice<Eigen::Index>> mask_;

    DataFrame(const DataFrame& other, std::shared_ptr<slice<Eigen::Index>> mask)
        : values_(other.values_),
        index_(std::make_shared<StringIndex>(other.index_, mask)),
        columns_(other.columns_),
        mask_(std::move(mask)) {}


    DataFrame(std::shared_ptr<MatrixXdRowMajor> values, std::shared_ptr<StringIndex> index, std::shared_ptr<ColumnIndex> columns)
        : values_(std::move(values)), 
          index_(std::move(index)), 
          columns_(std::move(columns)), 
          mask_(std::make_shared<slice<Eigen::Index>>(0, values_->rows(), 1)) {}

    DataFrame(nb::ndarray<> values, nb::list index, nb::list columns)
        : index_(std::make_shared<StringIndex>(index)),
          columns_(std::make_shared<ColumnIndex>(columns)) {
        Eigen::Index rows = static_cast<Eigen::Index>(values.shape(0));
        Eigen::Index cols = static_cast<Eigen::Index>(values.shape(1));
        values_ = std::make_shared<MatrixXdRowMajor>(Eigen::Map<MatrixXdRowMajor>(static_cast<double*>(values.data()), rows, cols));
    
        mask_ = std::make_shared<slice<Eigen::Index>>(0, values_->rows(), 1);
    }

    std::shared_ptr<slice<Eigen::Index>> mask() const override {
        return mask_;
    }

    class IlocProxy : IlocBase<DataFrame> {
    private:
        DataFrame& parent_;

    public:
        explicit IlocProxy(DataFrame& parent_) : parent_(parent_) {}

        DataFrame& parent() override {
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
        explicit LocProxy(DataFrame& parent_) : parent_(parent_) {}

        DataFrame& parent() override {
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
    
    Series sum() const {
        Eigen::VectorXd colSums = values().colwise().sum();
        return Series(std::make_shared<Eigen::VectorXd>(colSums), columns_);
    }

    const Eigen::Map<const MatrixXdRowMajor> values() const {
        return Eigen::Map<const MatrixXdRowMajor>(
            values_->data() + mask_->start * values_->cols(), 
            mask_->length(), 
            values_->cols()
        );
    }

    const StringIndex& index() const {
        return *index_;
    }

    const StringIndex& columns() const {
        return *columns_;
    }

    Series operator[](const std::string& colName) const {
        int colIndex = columns_->index_->at(colName);        
        auto colValues = std::make_shared<Eigen::VectorXd>(values().col(colIndex));
        return Series(colValues, index_);
    }

    friend std::ostream& operator<<(std::ostream& os, const DataFrame& df) {
        auto &values = df.values();
        const auto& rowNames = df.index_->keys();
        const auto& colNames = df.columns_->keys();

        std::vector<size_t> colWidths(colNames.size());
        
        for (size_t i = 0; i < colNames.size(); ++i) {
            colWidths[i] = colNames[i].length();
            for (size_t j = 0; j < rowNames.size(); ++j) {
                std::stringstream temp_ss;
                temp_ss << values(j, i);
                colWidths[i] = std::max(colWidths[i], temp_ss.str().length());
            }
        }

        size_t rowNameWidth = 0;
        for (const auto &rowName : rowNames) {
            rowNameWidth = std::max(rowNameWidth, rowName.length());
        }

        os << std::setw(rowNameWidth) << " ";
        for (size_t i = 0; i < colNames.size(); ++i) {
            os << std::setw(colWidths[i] + 2) << colNames[i];
        }
        os << std::endl;

        for (size_t i = 0; i < values.rows(); ++i) {
            os << std::setw(rowNameWidth) << rowNames[i];
            for (size_t j = 0; j < values.cols(); ++j) {
                os << std::setw(colWidths[j] + 2) << values(i, j);
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

// class Resampler {
//     std::shared_ptr<DateTimeIndex> index_;
//     Timedelta64 freq;

//     void _resample() {
//         slice<Datetime64> bins(
//             std::ceil(index_->keys().front() / freq) * freq,
//             std::ceil(index_->keys().back() / freq) * freq,
//             freq
//         );
//         digitize(index_->keys(), bins)
//     };
// };

NB_MODULE(cloth, m) {
    m.def("read_csv", &read_csv);

    nb::class_<slice<Eigen::Index>>(m, "slice")
        .def(nb::init<int, int, int>())
        .def("normalize", &slice<Eigen::Index>::normalize)
        .def_prop_ro("start", &slice<Eigen::Index>::get_start)
        .def_prop_ro("stop", &slice<Eigen::Index>::get_stop)
        .def_prop_ro("step", &slice<Eigen::Index>::get_step)
        .def_prop_ro("length", &slice<Eigen::Index>::length)
        .def("__repr__", &slice<Eigen::Index>::to_string);

    nb::class_<Index_>(m, "Index_");

    nb::class_<StringIndex, Index_>(m, "StringIndex")
        .def(nb::init<std::vector<std::string>>()) 
        .def("keys", &StringIndex::keys)
        .def_rw("index", &StringIndex::keys_)
        .def("__getitem__", [](StringIndex& self, std::string& key) {
            return self[key];
        }, nb::is_operator())
        .def_prop_ro("mask", [](const DataFrame &df) {
            return *df.mask();
        });

    m.attr("ColumnIndex") = m.attr("StringIndex");

    nb::class_<Datetime64>(m, "Datetime64")
        // .def(nb::init<dtime_t>())  
        .def(nb::init<const std::string&>())  
        .def("__add__", &Datetime64::operator+)    
        .def("seconds", &Datetime64::seconds)  
        .def("minutes", &Datetime64::minutes)  
        .def("hours", &Datetime64::hours)  
        .def("days", &Datetime64::days)  
        .def("weeks", &Datetime64::weeks)  
        .def("years", &Datetime64::years)  
        .def("months", &Datetime64::months)  
        .def("__repr__", [](const Datetime64& self) {
            std::ostringstream oss;
            oss << "Datetime64(" << self.seconds() << ")";
            return oss.str();
        }); 

    
    // nb::class_<Timedelta64>(m, "Timedelta64")
    //     // .def(nb::init<dtime_t>())  
    //     .def("__add__", &Timedelta64::operator+)  
    //     .def("__sub__", &Timedelta64::operator-)  
    //     .def("__repr__", [](const Timedelta64& self) {
    //         std::ostringstream oss;
    //         oss << "Timedelta64(" << self.data_ << ")";
    //         return oss.str();
    //     });  

    
    nb::class_<DateTimeIndex, Index_>(m, "DateTimeIndex")
        .def(nb::init<std::vector<std::string>>())  
        .def(nb::init<nb::list>())  
        .def("__getitem__", [](DateTimeIndex& self, Eigen::Index idx) {
            return self[idx];
        }, nb::is_operator())  
        .def("__getitem__", [](DateTimeIndex& self, const Datetime64& key) {
            return self[key];
        }, nb::is_operator())  
        .def("keys", &DateTimeIndex::keys)  
        .def("__repr__", [](const DateTimeIndex& self) {
            std::ostringstream oss;
            oss << "DateTimeIndex(";
            auto keys = self.keys();
            for (size_t i = 0; i < keys.size(); ++i) {
                oss << keys[i].seconds();
                if (i < keys.size() - 1) oss << ", ";
            }
            oss << ")";
            return oss.str();
        });  

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
        .def(nb::init<Eigen::VectorXd, std::shared_ptr<StringIndex>>())
        .def(nb::init<nb::ndarray<double>, nb::list>())
        .def("__repr__", &Series::to_string)
        .def("sum", &Series::sum)
        .def("mean", &Series::mean)
        .def("min", &Series::min)
        .def("max", &Series::max)
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
        .def_prop_ro("mask", [](const DataFrame &df) {
            return *df.mask();
        })
        .def("__len__", &DataFrame::length)
        .def("head", &DataFrame::head)
        .def("tail", &DataFrame::tail)
        .def_prop_ro("index", &DataFrame::index)
        .def("sum", &DataFrame::sum)
        .def_prop_ro("columns", &DataFrame::columns);
}
