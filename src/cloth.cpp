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
#include "../lib/decimal.h"

#include "../include/read_csv.h"
#include "../include/dt.h"
#include "../include/slice.h"
#include "../include/index.h"
#include "../include/boolview.h"
#include "../include/deceig.h"
#include "../include/resampler.h"


#ifndef NB_T
#define NB_T
namespace nb = nanobind;
#endif

#ifndef MATRIX_RM_T
#define MATRIX_RM_T
using MatrixXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
#endif

#ifndef INDEX_T
#define INDEX_T
using index_t = Eigen::Index;
#endif

#ifndef MASK_T
#define MASK_T
using mask_t = slice<index_t>;
#endif

#ifndef DEC_T
#define DEC_T
constexpr int precision_ = 2;
using Decimal = dec::decimal<precision_>;
#endif

#ifndef MATRIX_DEC_RM_T
#define MATRIX_DEC_RM_T
using MatrixDecimalRowMajor = Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
#endif

#ifndef VECTOR_DEC_RM_T
#define VECTOR_DEC_RM_T
using VectorDecimal = Eigen::Vector<Decimal, Eigen::Dynamic>;
#endif

#ifndef COLUMN_INDEX_T
#define COLUMN_INDEX_T
using ColumnIndex = StringIndex;
#endif

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
protected:
    bool is_datetime(const nb::ndarray<>& array) const {
        nb::handle dtype = array.dtype();        
        if (nb::module_::import_("numpy").attr("datetime64")().get_type() == dtype) {
            return true;
        }
        return false;
    }
 
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

    auto resample(const timedelta& freq) const {
        return Resampler<Derived>(std::make_shared<Derived>(static_cast<const Derived&>(*this)), freq);
    }
};

class Series : public Frame<Series> {
public:
    std::shared_ptr<Eigen::VectorXd> values_;
    std::shared_ptr<Index_> index_; // Dynamic polymorphism
    std::string name_;
    std::shared_ptr<slice<Eigen::Index>> mask_;
          
    Series(const Series& other, std::shared_ptr<slice<Eigen::Index>> mask)
        : values_(other.values_),
        index_(other.index_->clone(mask)),
        name_(other.name_),
        mask_(std::move(mask)) {}


    Series(std::shared_ptr<Eigen::VectorXd> values, std::shared_ptr<Index_> index)
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

    // Series(nb::ndarray<double> values, nb::object index, nb::object name = nb::none()) {
    //     // Handle values
    //     Eigen::Index size = static_cast<Eigen::Index>(values.size());
    //     values_ = std::make_shared<Eigen::VectorXd>(Eigen::Map<const Eigen::VectorXd>(values.data(), size));

    //     // Handle index
    //     if (index.is_none()) {
    //         index_ = std::make_shared<RangeIndex>(0, size, 1);
    //     } else if (nb::isinstance<nb::list>(index)) {
    //         index_ = std::make_shared<StringIndex>(index);
    //     } else if (nb::isinstance<nb::ndarray<std::string>>(index)) {
    //         index_ = std::make_shared<StringIndex>(index);
    //     } else {
    //         throw std::invalid_argument("Invalid type for index");
    //     }

    //     // Handle name
    //     if (name.is_none()) {
    //         name_ = "";
    //     } else {
    //         name_ = nb::cast<std::string>(name);
    //     }

    //     mask_ = index_->mask();
    // }

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

    const Index_& index() const {
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

        double operator[](const datetime& key) const {
            int idx = parent_.index_->operator[](key);
            return parent_.iloc()[idx];
        }
    };

    LocProxy loc() {
        return LocProxy(*this);
    }

    BoolView operator>(double threshold) const {
        std::vector<bool> mask(values_->size(), false);
        int ones_count = 0;
        for (int i = 0; i < values_->size(); ++i) {
            if ((*values_)(i) > threshold) {
                mask[i] = true;
                ++ones_count;
            }
        }
        return BoolView(mask, ones_count);
    }

    BoolView operator<(double threshold) const {
        std::vector<bool> mask(values_->size(), false);
        int ones_count = 0;
        for (int i = 0; i < values_->size(); ++i) {
            if ((*values_)(i) < threshold) {
                mask[i] = true;
                ++ones_count;
            }
        }
        return BoolView(mask, ones_count);
    }

    Series where(const BoolView& view) const {
        Eigen::VectorXd filtered_values = view.apply(*values_);
        std::shared_ptr<Index_> filtered_index = index_->apply(view);
        return Series(std::make_shared<Eigen::VectorXd>(filtered_values), filtered_index);
    }

    friend std::ostream& operator<<(std::ostream& os, const Series& series) {
        try {
            Eigen::Index len = series.length();
            std::vector<std::string> keys = series.index().keys();
            for (Eigen::Index i = 0; i < len; ++i) {
                std::string index_value = keys[i];
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
    std::shared_ptr<Index_> index_;
    std::shared_ptr<ColumnIndex> columns_;
    std::shared_ptr<slice<Eigen::Index>> mask_;
    
public:
    DataFrame(const DataFrame& other, std::shared_ptr<slice<Eigen::Index>> mask)
        : values_(other.values_),
          index_(other.index_->clone(mask)),
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

    DataFrame(std::shared_ptr<MatrixXdRowMajor> values, std::shared_ptr<Index_> index, std::shared_ptr<ColumnIndex> columns)
        : values_(std::move(values)), 
          index_(std::move(index)), 
          columns_(std::move(columns)), 
          mask_(std::make_shared<slice<Eigen::Index>>(0, values_->rows(), 1)) {}

    // DataFrame(nb::ndarray<> values, nb::object index, nb::object columns) {
    //     // Handle values
    //     Eigen::Index rows = static_cast<Eigen::Index>(values.shape(0));
    //     Eigen::Index cols = static_cast<Eigen::Index>(values.shape(1));
    //     values_ = std::make_shared<MatrixXdRowMajor>(Eigen::Map<MatrixXdRowMajor>(static_cast<double*>(values.data()), rows, cols));

    //     // Handle index
    //     if (index.is_none()) {
    //         index_ = std::make_shared<RangeIndex>(0, rows, 1);
    //     } else if (nb::isinstance<nb::list>(index)) {
    //         index_ = std::make_shared<StringIndex>(nb::cast<nb::list>(index));
    //     } else if (nb::isinstance<nb::ndarray<std::string>>(index)) {
    //         index_ = std::make_shared<StringIndex>(index);
    //     } else {
    //         throw std::invalid_argument("Invalid type for index");
    //     }

    //     // Handle columns
    //     if (columns.is_none()) {
    //         columns_ = std::make_shared<ColumnIndex>(std::make_shared<RangeIndex>(0, cols, 1));
    //     } else if (nb::isinstance<nb::list>(columns)) {
    //         columns_ = std::make_shared<StringIndex>(nb::cast<nb::list>(columns));
    //     } else if (nb::isinstance<nb::ndarray<std::string>>(columns)) {
    //         columns_ = std::make_shared<StringIndex>(columns);
    //     } else {
    //         throw std::invalid_argument("Invalid type for columns");
    //     }

    //     mask_ = index_->mask();
    // }

    std::shared_ptr<mask_t> mask() const override {
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

    const Index_& index() const {
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

    DataFrame where(const BoolView& view) const {
        Eigen::MatrixXd filtered_values = view.apply(*values_);
        std::shared_ptr<Index_> filtered_index = index_->apply(view);
        return DataFrame(std::make_shared<MatrixXdRowMajor>(filtered_values), filtered_index, columns_);
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

class TimeSeries : public Frame<TimeSeries> {
public:
    std::shared_ptr<VectorDecimal> values_;
    std::shared_ptr<DateTimeIndex> index_;
    std::string name_;
    std::shared_ptr<slice<Eigen::Index>> mask_;

    TimeSeries(const TimeSeries& other, std::shared_ptr<slice<Eigen::Index>> mask)
        : values_(other.values_),
          index_(std::static_pointer_cast<DateTimeIndex>(other.index_->clone(mask))),
          name_(other.name_),
          mask_(std::move(mask)) {}

    TimeSeries(std::shared_ptr<VectorDecimal> values, std::shared_ptr<Index_> index)
        : values_(std::move(values)),
          index_(std::static_pointer_cast<DateTimeIndex>(index)) {
        mask_ = std::make_shared<slice<Eigen::Index>>(0, values_->size(), 1);
    }

    TimeSeries(nb::ndarray<double> values, nb::list index)
        : values_(std::make_shared<Eigen::Matrix<Decimal, Eigen::Dynamic, 1>>(values.size())),
          index_(std::make_shared<DateTimeIndex>(index)) {

        double* values_data = values.data();
        for (Eigen::Index i = 0; i < values.size(); ++i) {
            (*values_)(i) = Decimal(values_data[i]);
        }

        mask_ = std::make_shared<slice<Eigen::Index>>(0, values_->size(), 1);
    }

    BoolView operator>(const Decimal& threshold) const {
        std::vector<bool> mask(values_->size(), false);
        int ones_count = 0;
        for (int i = 0; i < values_->size(); ++i) {
            if ((*values_)(i) > threshold) {
                mask[i] = true;
                ++ones_count;
            }
        }
        return BoolView(mask, ones_count);
    }

    BoolView operator<(const Decimal& threshold) const {
        std::vector<bool> mask(values_->size(), false);
        int ones_count = 0;
        for (int i = 0; i < values_->size(); ++i) {
            if ((*values_)(i) < threshold) {
                mask[i] = true;
                ++ones_count;
            }
        }
        return BoolView(mask, ones_count);
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

    Decimal sum() const {
        return values().sum();
    }

    Decimal mean() const {
        return values().mean();
    }

    Decimal min() const {
        return values().minCoeff();
    }

    Decimal max() const {
        return values().maxCoeff();
    }

    std::vector<std::string> get_index() const {
        return index_->keys();
    }

    const Eigen::Map<const VectorDecimal> values() const {
        return Eigen::Map<const VectorDecimal>(
            values_->data() + mask_->start, 
            mask_->length()
        );
    }

    const DateTimeIndex& index() const {
        return *index_;
    }

    class IlocProxy : public IlocBase<TimeSeries> {
    private:
        TimeSeries& parent_;

    public:
        explicit IlocProxy(TimeSeries& parent_) : parent_(parent_) {}

        TimeSeries& parent() override {
            return parent_;
        }

        using IlocBase<TimeSeries>::operator[];

        Decimal operator[](Eigen::Index idx) const {
            Eigen::Index combined_index = combine_slice_with_index(*parent_.mask_, idx);
            return (*parent_.values_)(combined_index);
        }
    };

    IlocProxy iloc() {
        return IlocProxy(*this);
    }

    class LocProxy : public LocBase<TimeSeries> {
    private:
        TimeSeries& parent_;

    public:
        explicit LocProxy(TimeSeries& parent_) : parent_(parent_) {}

        TimeSeries& parent() override {
            return parent_;
        }

        using LocBase<TimeSeries>::operator[];

        Decimal operator[](const datetime& key) const {
            int idx = parent_.index_->operator[](key);
            return parent_.iloc()[idx];
        }
    };

    LocProxy loc() {
        return LocProxy(*this);
    }

    TimeSeries operator[](const BoolView& view) const {
        VectorDecimal filtered_values = view.apply(*values_);
        std::shared_ptr<DateTimeIndex> filtered_index = std::dynamic_pointer_cast<DateTimeIndex>(index_->apply(view));
        return TimeSeries(std::make_shared<VectorDecimal>(filtered_values), filtered_index);
    }

    friend std::ostream& operator<<(std::ostream& os, const TimeSeries& series) {
        try {
            Eigen::Index len = series.length();
            std::vector<std::string> keys = series.index().keys();
            for (Eigen::Index i = 0; i < len; ++i) {
                std::string index_value = keys[i];
                Decimal series_value = (*series.values_)(series.mask_->start + i * series.mask_->step);
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

class TimeFrame : public Frame<TimeFrame> {
public:
    std::shared_ptr<Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> values_;
    std::shared_ptr<DateTimeIndex> index_;
    std::shared_ptr<ColumnIndex> columns_;
    std::shared_ptr<slice<Eigen::Index>> mask_;

    TimeFrame(const TimeFrame& other, std::shared_ptr<slice<Eigen::Index>> mask)
        : values_(other.values_),
          index_(std::static_pointer_cast<DateTimeIndex>(other.index_->clone(mask))),
          columns_(other.columns_),
          mask_(std::move(mask)) {}

    TimeFrame(std::shared_ptr<Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> values, std::shared_ptr<DateTimeIndex> index, std::shared_ptr<ColumnIndex> columns)
        : values_(std::move(values)),
          index_(std::move(index)),
          columns_(std::move(columns)),
          mask_(std::make_shared<slice<Eigen::Index>>(0, values_->rows(), 1)) {}

    TimeFrame(nb::ndarray<double> values, nb::list index, nb::list columns)
        : index_(std::make_shared<DateTimeIndex>(index)),
          columns_(std::make_shared<ColumnIndex>(columns)) {

        Eigen::Index rows = static_cast<Eigen::Index>(values.shape(0));
        Eigen::Index cols = static_cast<Eigen::Index>(values.shape(1));

        values_ = std::make_shared<MatrixDecimalRowMajor>(rows, cols);

        double* values_data = values.data();
        for (Eigen::Index i = 0; i < rows; ++i) {
            for (Eigen::Index j = 0; j < cols; ++j) {
                (*values_)(i, j) = Decimal(values_data[i * cols + j]);
            }
        }

        mask_ = std::make_shared<slice<Eigen::Index>>(0, values_->rows(), 1);
    }


    std::shared_ptr<mask_t> mask() const override {
        return mask_;
    }

    class IlocProxy : IlocBase<TimeFrame> {
    private:
        TimeFrame& parent_;

    public:
        explicit IlocProxy(TimeFrame& parent_) : parent_(parent_) {}

        TimeFrame& parent() override {
            return parent_;
        }

        using IlocBase<TimeFrame>::operator[];

        TimeSeries operator[](Eigen::Index idx) const {
            Eigen::Index combined_index = combine_slice_with_index(*parent_.mask_, idx);
            auto row = std::make_shared<VectorDecimal>(parent_.values_->row(combined_index));
            return TimeSeries(row, parent_.columns_);
        }
    };

    IlocProxy iloc() {
        return IlocProxy(*this);
    }

    class LocProxy : public LocBase<TimeFrame> {
    private:
        TimeFrame& parent_;

    public:
        explicit LocProxy(TimeFrame& parent_) : parent_(parent_) {}

        TimeFrame& parent() override {
            return parent_;
        }

        using LocBase<TimeFrame>::operator[];

        TimeSeries operator[](const datetime& key) {
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

    TimeSeries sum() const {
        VectorDecimal colSums = values().colwise().sum();
        return TimeSeries(std::make_shared<VectorDecimal>(colSums), columns_);
    }

    const Eigen::Map<const Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> values() const {
        return Eigen::Map<const MatrixDecimalRowMajor>(
            values_->data() + mask_->start * values_->cols(),
            mask_->length(),
            values_->cols()
        );
    }

    const DateTimeIndex& index() const {
        return *index_;
    }

    const StringIndex& columns() const {
        return *columns_;
    }

    TimeSeries operator[](const std::string& colName) const {
        int colIndex = columns_->index_->at(colName);        
        auto colValues = std::make_shared<VectorDecimal>(values().col(colIndex));
        return TimeSeries(colValues, index_);
    }

    TimeFrame where(const BoolView& view) const {
        MatrixDecimalRowMajor filtered_values = view.apply(*values_);
        std::shared_ptr<DateTimeIndex> filtered_index = std::dynamic_pointer_cast<DateTimeIndex>(index_->apply(view));
        return TimeFrame(std::make_shared<MatrixDecimalRowMajor>(filtered_values), filtered_index, columns_);
    }

    friend std::ostream& operator<<(std::ostream& os, const TimeFrame& df) {
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

    nb::class_<Index_>(m, "Index_")
        .def("length", &Index_::length);
        // .def("mask", &Index_::mask);

    nb::class_<StringIndex, Index_>(m, "StringIndex")
        .def(nb::init<std::vector<std::string>>()) 
        .def("__getitem__", [](StringIndex& self, std::string& key) {
            return self[key];
        }, nb::is_operator())
        .def_prop_ro("mask", [](const DataFrame &df) {
            return *df.mask();
        })
        .def("__repr__", &StringIndex::to_string)
        .def_prop_ro("index", &StringIndex::keys);

    m.attr("ColumnIndex") = m.attr("StringIndex");

    nb::class_<DateTimeIndex, Index_>(m, "DateTimeIndex")
        .def(nb::init<std::vector<std::string>>())  
        .def(nb::init<nb::list>())    
        .def("__getitem__", [](DateTimeIndex& self, const datetime& key) {
            return self[key];
        }, nb::is_operator())  
        .def("__repr__", &DateTimeIndex::to_string)
        .def_prop_ro("index", &DateTimeIndex::keys);

    nb::class_<RangeIndex, Index_>(m, "RangeIndex") 
        .def("__getitem__", [](RangeIndex& self, const index_t& key) {
            return self[key];
        }, nb::is_operator())  
        .def("__repr__", &RangeIndex::to_string)
        .def_prop_ro("index", &RangeIndex::keys);

    nb::class_<datetime>(m, "datetime")
        .def(nb::init<dtime_t>())
        .def(nb::init<const std::string&>())
        .def("__add__", &datetime::operator+, nb::is_operator())
        .def("__sub__", [](const datetime &a, const timedelta &b) {
            return a - b;
        }, nb::is_operator())
        .def("__sub__", [](const datetime &a, const datetime &b) {
            return a - b;
        }, nb::is_operator())
        .def("floor", &datetime::floor)
        .def("ceil", &datetime::ceil)
        .def("seconds", &datetime::seconds)
        .def("minutes", &datetime::minutes)
        .def("hours", &datetime::hours)
        .def("days", &datetime::days)
        .def("weeks", &datetime::weeks)
        .def("years", &datetime::years)
        .def("months", &datetime::months)
        .def("__eq__", &datetime::operator==, nb::is_operator())
        .def("__ne__", &datetime::operator!=, nb::is_operator())
        .def("__lt__", &datetime::operator<, nb::is_operator())
        .def("__le__", &datetime::operator<=, nb::is_operator())
        .def("__gt__", &datetime::operator>, nb::is_operator())
        .def("__ge__", &datetime::operator>=, nb::is_operator())
        .def("__str__", [](const datetime &dt) {
            std::ostringstream oss;
            oss << "Datetime(" << dt.seconds() << " seconds since epoch)";
            return oss.str();
        })
        .def("data", &datetime::data);

    nb::class_<timedelta>(m, "timedelta")
        .def(nb::init<dtime_t>(), nb::arg("units") = 0)
        .def(nb::init<const std::string&>())
        .def("__add__", &timedelta::operator+, nb::is_operator())
        // .def("__sub__", &timedelta::operator-, nb::is_operator())
        .def("__mul__", &timedelta::operator*, nb::is_operator())
        .def("__str__", [](const timedelta &td) {
            std::ostringstream oss;
            oss << td.data() << " units";
            return oss.str();
        })
        .def("data", &timedelta::data);

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
        .def("tail", &Series::tail)
        .def_prop_ro("index", &Series::index)
        .def("__gt__", [](const Series &self, double other) {
            return self > other;
        }, nb::is_operator())
        .def("__lt__", [](const Series &self, double other) {
            return self < other;
        }, nb::is_operator())
        .def("where", &Series::where);

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
        .def("__getitem__", [](DataFrame& self, std::string& other) {
            return self[other];
        }, nb::is_operator())
        .def("where", &DataFrame::where)
        .def("__getattr__", [](DataFrame& self, std::string& other) {
            return self[other];
        }, nb::is_operator())        
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

    nb::class_<BoolView>(m, "BoolView")
        .def("__repr__", &BoolView::to_string);

    nb::class_<Decimal>(m, "decimal")
        .def(nb::init<double>())
        .def("__add__", [](const Decimal& a, const Decimal& b) { return a + b; }, nb::is_operator())
        .def("__sub__", [](const Decimal& a, const Decimal& b) { return a - b; }, nb::is_operator())
        .def("__mul__", [](const Decimal& a, const Decimal& b) { return a * b; }, nb::is_operator())
        .def("__truediv__", [](const Decimal& a, const Decimal& b) { return a / b; }, nb::is_operator())
        .def("__str__", [](const Decimal &dec) {
            std::ostringstream oss;
            oss << dec;
            return oss.str();
        });

    nb::class_<TimeSeries>(m, "TimeSeries")
        .def(nb::init<nb::ndarray<double>, nb::list>())
        .def("__repr__", &TimeSeries::to_string)
        .def("sum", &TimeSeries::sum)
        .def("mean", &TimeSeries::mean)
        .def("min", &TimeSeries::min)
        .def("max", &TimeSeries::max)
        .def_prop_ro("values", &TimeSeries::values)
        .def("length", &TimeSeries::length)
        .def_prop_ro("mask", [](const TimeSeries &series) {
            return *series.mask_;
        })
        .def_prop_ro("iloc", &TimeSeries::iloc)
        .def_prop_ro("loc", &TimeSeries::loc)
        .def("head", &TimeSeries::head)
        .def("tail", &TimeSeries::tail)
        .def_prop_ro("index", &TimeSeries::index)
        .def("__gt__", [](const TimeSeries &self, Decimal other) {
            return self > other;
        }, nb::is_operator())
        .def("__lt__", [](const TimeSeries &self, Decimal other) {
            return self < other;
        }, nb::is_operator())
        .def("__getitem__", [](const TimeSeries &self, const BoolView &view) {
            return self[view];
        }, nb::is_operator())
        .def("resample", &TimeSeries::resample);

    nb::class_<TimeFrame>(m, "TimeFrame")
        .def(nb::init<nb::ndarray<double>, nb::list, nb::list>())
        .def("__repr__", &TimeFrame::to_string)
        .def("sum", &TimeFrame::sum)
        .def("rows", &TimeFrame::rows)
        .def("cols", &TimeFrame::cols)
        .def_prop_ro("values", &TimeFrame::values)
        .def("length", &TimeFrame::length)
        .def_prop_ro("mask", [](const TimeFrame &df) {
            return *df.mask();
        })
        .def_prop_ro("iloc", &TimeFrame::iloc)
        .def_prop_ro("loc", &TimeFrame::loc)
        .def("head", &TimeFrame::head)
        .def("tail", &TimeFrame::tail)
        .def_prop_ro("index", &TimeFrame::index)
        .def_prop_ro("columns", &TimeFrame::columns)
        .def("__getitem__", [](TimeFrame& self, std::string& other) {
            return self[other];
        }, nb::is_operator())
        .def("where", &TimeFrame::where)
        .def("__getattr__", [](TimeFrame& self, std::string& other) {
            return self[other];
        }, nb::is_operator());

    // nb::class_<Resampler<Series>>(m, "SeriesResampler")
    //     .def(nb::init<std::shared_ptr<Series>, timedelta>())
    //     .def("__iter__", [](Resampler<Series>& self) -> Resampler<Series>& { return self.begin(); })
    //     .def("__next__", [](Resampler<Series>& self) -> Series {
    //         if (self != self.end()) {
    //             Series result = *self;
    //             ++self;
    //             return result;
    //         } else {
    //             throw nb::stop_iteration();
    //         }
    //     });

    nb::class_<Resampler<TimeSeries>>(m, "TimeSeriesResampler")
        .def("__iter__", [](Resampler<TimeSeries>& self) -> Resampler<TimeSeries>& {
            return self.begin();  // Ensure __iter__ returns self at the start of iteration
        })
        .def("__next__", [](Resampler<TimeSeries>& self) -> TimeSeries {
            if (self.has_next()) {
                return self.next();
            } else {
                throw nb::stop_iteration();  // Raise StopIteration to signal the end
            }
        })
        .def("bins", &Resampler<TimeSeries>::bins)
        .def("__repr__", &Resampler<TimeSeries>::to_string);

}
