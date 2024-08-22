#include "../include/index.h"
#include "../include/datetime.h"
#include "../include/digitize.h"
#include "../include/slice.h"

#include <vector>
#include <memory>

#ifndef INDEX_T
#define INDEX_T
#include <cstddef>
typedef std::ptrdiff_t index_t;
#endif

#ifndef MASK_T
#define MASK_T
using mask_t = slice<index_t>;
#endif

template <typename FrameType>
class Resampler {
private:
    std::shared_ptr<FrameType> frame_;
    timedelta freq_;
    size_t current_bin_;
    std::vector<index_t> bins_;

public:
    std::vector<index_t> bins() {
        return bins_;
    }

    Resampler(std::shared_ptr<FrameType> data, timedelta freq)
        : frame_(std::move(data)), freq_(freq), current_bin_(0) {
        resample();
    }

    // Iterator implementation
    Resampler& begin() {
        current_bin_ = 0;
        return *this;
    }

    Resampler& end() {
        current_bin_ = bins_.size() - 1;
        return *this;
    }

    bool operator!=(const Resampler& other) const {
        return current_bin_ != other.current_bin_;
    }

    void operator++() {
        if (current_bin_ < bins_.size() - 1) {
            ++current_bin_;
        }
    }

    FrameType operator*() const {
        return frame_->iloc()[mask_t(bins_[current_bin_], bins_[current_bin_ + 1], 1)];
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "Resampler at bin " << current_bin_ << ":\n";
        FrameType current_frame = frame_->iloc()[mask_t(bins_[current_bin_], bins_[current_bin_ + 1], 1)];
        oss << current_frame.to_string();  // Assuming FrameType has a to_string method
        return oss.str();
    }

    bool has_next() const {
        return current_bin_ < bins_.size() - 1;
    }

    FrameType next() {
        if (has_next()) {
            FrameType result = *(*this);
            this->operator++();
            return result;
        } else {
            throw std::out_of_range("End of resampler bins reached");
        }
    }

private:
    void resample() {
        // Dynamic cast throws std::bad_cast if its not a DateTimeIndex. For runtime safety
        const DateTimeIndex& datetime_index = dynamic_cast<const DateTimeIndex&>(frame_->index());
        slice<datetime, timedelta> bins(
            datetime_index.keys_->front().floor(freq_),
            datetime_index.keys_->back().ceil(freq_),
            freq_
        );
        bins_ = digitize<datetime, timedelta>(datetime_index.keys_, bins);
    }
};