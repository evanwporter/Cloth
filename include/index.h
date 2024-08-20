#ifndef INDEX_H
#define INDEX_H

#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <numeric>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <memory>

#include <Eigen/Dense>
#include "../lib/robinhood.h"
#include "../include/dt.h"
#include "../include/slice.h"
#include "../include/boolview.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>

#ifndef NB_T
#define NB_T
namespace nb = nanobind;
#endif

#ifndef MASK_T
#define MASK_T
using mask_t = slice<index_t>;
#endif

class Index_ {
public:
    Index_() = default;

    virtual ~Index_() = default;

    virtual std::shared_ptr<mask_t> mask() const = 0;

    index_t length() const {
        return mask()->length();
    }

    virtual index_t operator[](const datetime& index) const = 0;
    virtual index_t operator[](const std::string& key) const = 0;
    virtual index_t operator[](const index_t& index) const = 0;
    virtual std::vector<std::string> keys() const = 0;
    virtual std::shared_ptr<Index_> clone(const std::shared_ptr<mask_t> mask) const = 0;
    virtual std::shared_ptr<Index_> apply(const BoolView& view) const = 0;
};

class StringIndex : public Index_ {
private:
    std::shared_ptr<mask_t> mask_;

public:
    std::shared_ptr<robin_hood::unordered_map<std::string, int>> index_;
    std::shared_ptr<std::vector<std::string>> keys_;

public:
    StringIndex(std::shared_ptr<robin_hood::unordered_map<std::string, int>> index, std::shared_ptr<std::vector<std::string>> keys)
        : index_(std::move(index)), keys_(std::move(keys)), 
          mask_(std::make_shared<mask_t>(0, static_cast<int>(this->keys_->size()), 1)) {}

    StringIndex(const StringIndex& other, const std::shared_ptr<mask_t> mask)
        : mask_(mask), index_(other.index_), keys_(other.keys_) {} 

    std::shared_ptr<Index_> clone(const std::shared_ptr<mask_t> mask) const override {
        return std::make_shared<StringIndex>(*this, mask);
    };

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

    // std::string operator[](index_t idx) const {
    //     if (idx < 0 || idx >= static_cast<index_t>(keys_->size())) {
    //         throw std::out_of_range("Index out of range");
    //     }
    //     return (*keys_)[combine_slice_with_index(*mask_, idx)];
    // }

    index_t operator[](const datetime& index) const override {
        throw std::runtime_error("StringIndex does not support datetime indexing.");
    }

    index_t operator[](const index_t& index) const override {
        throw std::runtime_error("StringIndex does not support integer indexing.");
    }

    index_t operator[](const std::string& key) const {
        return index_->at(key);
    }

    std::shared_ptr<mask_t> mask() const override {
        return mask_;
    }

    std::shared_ptr<Index_> apply(const BoolView& view) const override {
        std::vector<std::string> filtered_keys;
        filtered_keys.reserve(view.ones_count_);
        for (int i = 0; i < keys_->size(); ++i) {
            if (view.mask_[i]) {
                filtered_keys.push_back((*keys_)[i]);
            }
        }
        return std::make_shared<StringIndex>(filtered_keys);
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "StringIndex(";
        std::vector<std::string> k = keys();
        for (size_t i = 0; i < k.size(); ++i) {
            oss << k[i];
            if (i < k.size() - 1) oss << ", ";
        }
        oss << ")";
        return oss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const StringIndex& other) {
        os << other.to_string();
        return os;
    }
};

class DateTimeIndex : public Index_ {
private:
    std::shared_ptr<mask_t> mask_;

public:
    std::shared_ptr<robin_hood::unordered_map<datetime, int>> index_;
    std::shared_ptr<std::vector<datetime>> keys_;
    
    DateTimeIndex(std::shared_ptr<robin_hood::unordered_map<datetime, int>> index, std::shared_ptr<std::vector<datetime>> keys)
        : index_(std::move(index)), keys_(std::move(keys)), 
          mask_(std::make_shared<mask_t>(0, static_cast<int>(this->keys_->size()), 1)) {}
    
    DateTimeIndex(const DateTimeIndex& other, std::shared_ptr<mask_t> mask)
        : index_(other.index_), keys_(other.keys_), mask_(std::move(mask)) {}

    explicit DateTimeIndex(std::vector<datetime> datetime_keys)
        : keys_(std::make_shared<std::vector<datetime>>(std::move(datetime_keys))),
          index_(std::make_shared<robin_hood::unordered_map<datetime, int>>()) {
        for (size_t i = 0; i < keys_->size(); ++i) {
            (*index_)[(*keys_)[i]] = static_cast<int>(i);
        }
        mask_ = std::make_shared<mask_t>(0, static_cast<int>(this->keys_->size()), 1);
    }

    explicit DateTimeIndex(std::vector<std::string> iso_keys)
        : keys_(std::make_shared<std::vector<datetime>>()),
          index_(std::make_shared<robin_hood::unordered_map<datetime, int>>()) {

        for (size_t i = 0; i < iso_keys.size(); ++i) {
            datetime dt(iso_keys[i]);
            keys_->push_back(dt);
            (*index_)[dt] = static_cast<int>(i);
        }
        mask_ = std::make_shared<mask_t>(0, static_cast<int>(this->keys_->size()), 1);
    }
    
    explicit DateTimeIndex(nb::list iso_keys)
        : keys_(std::make_shared<std::vector<datetime>>()), 
          index_(std::make_shared<robin_hood::unordered_map<datetime, int>>()) {

        keys_->reserve(iso_keys.size());
        for (size_t i = 0; i < iso_keys.size(); ++i) {
            std::string iso_key = nb::cast<std::string>(iso_keys[i]);
            datetime dt(iso_key);
            keys_->push_back(dt);
            (*index_)[dt] = static_cast<int>(i);
        }
        mask_ = std::make_shared<mask_t>(0, static_cast<int>(keys_->size()), 1);
    }

    // // TODO Keys should return an iterator
    std::vector<std::string> keys() const {
        std::vector<std::string> result;
        result.reserve(length());
        for (int i = mask_->start; i < mask_->stop; i += mask_->step) {
            result.push_back((*keys_)[i].to_iso());
        }
        return result;
    }

    // datetime operator[](Eigen::Index idx) const {
    //     if (idx < 0 || idx >= static_cast<Eigen::Index>(keys_->size())) {
    //         throw std::out_of_range("Index out of range");
    //     }
    //     return (*keys_)[combine_slice_with_index(*mask_, idx)];
    // }
    
    index_t operator[](const datetime& key) const override {
        return index_->at(key);
    }

    index_t operator[](const std::string &key) const override {
        return this->operator[](datetime(key));
    }

    index_t operator[](const index_t &key) const override {
        throw std::runtime_error("DateTimeIndex does not support integer indexing.");
    }

    std::shared_ptr<Index_> clone(const std::shared_ptr<mask_t> mask) const override {
        return std::make_shared<DateTimeIndex>(*this, mask);
    }

    std::shared_ptr<Index_> apply(const BoolView& view) const override {
        std::vector<datetime> filtered_keys;
        filtered_keys.reserve(view.ones_count_);
        for (int i = 0; i < keys_->size(); ++i) {
            if (view.mask_[i]) {
                filtered_keys.push_back((*keys_)[i]);
            }
        }
        return std::make_shared<DateTimeIndex>(filtered_keys);
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "DateTimeIndex(";
        std::vector<std::string> k = keys();
        for (size_t i = 0; i < k.size(); ++i) {
            oss << k[i];
            if (i < k.size() - 1) oss << ", ";
        }
        oss << ")";
        return oss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const DateTimeIndex& other) {
        os << other.to_string();
        return os;
    }

    std::shared_ptr<mask_t> mask() const override {
        return mask_;
    }
};

class RangeIndex : public Index_ {
private:
    std::shared_ptr<mask_t> mask_;

public:
    explicit RangeIndex(std::shared_ptr<mask_t> mask)
        : mask_(std::move(mask)) {}

    RangeIndex(index_t start, index_t stop, index_t step = 1)
        : mask_(std::make_shared<mask_t>(start, stop, step)) {}

    std::shared_ptr<Index_> clone(const std::shared_ptr<mask_t> mask) const override {
        return std::make_shared<RangeIndex>(mask);
    }

    index_t operator[](const index_t& index) const override {
        return combine_slice_with_index(*mask_, index);
    }

    index_t operator[](const datetime& index) const override {
        throw std::runtime_error("RangeIndex does not support datetime indexing.");
    }

    index_t operator[](const std::string& key) const override {
        return (*this)[std::stoi(key)];
    }

    std::vector<std::string> keys() const {
        std::vector<std::string> result({
            std::to_string(mask_->start), 
            std::to_string(mask_->stop), 
            std::to_string(mask_->step)
        });
        return result;
    }

    std::shared_ptr<mask_t> mask() const override {
        return mask_;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "RangeIndex(" << mask_->start << ", " << mask_->stop << ", " << mask_->step << ")";
        return oss.str();
    }

    std::shared_ptr<Index_> apply(const BoolView& view) const override {
        return std::make_shared<RangeIndex>(0, view.ones_count_, 1);
    }

    friend std::ostream& operator<<(std::ostream& os, const RangeIndex& other) {
        os << other.to_string();
        return os;
    }
};

#endif