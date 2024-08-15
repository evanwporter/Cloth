#ifndef SLICE_H
#define SLICE_H

#ifndef INDEX_T
#define INDEX_T
#include <cstddef>
typedef std::ptrdiff_t index_t;
#endif

template <typename T = index_t>
struct slice {
    T start, stop;
    int step;

    slice(T start_, T stop_, int step_) 
        : start(start_), stop(stop_), step(step_) {
        if (step == 0) throw std::invalid_argument("Step cannot be zero");
        LOG("Creating slice with start=" << start << ", stop=" << stop << ", step=" << step);
    }

    slice(T start_, T stop_, int step_, Eigen::Index length) 
        : slice(start_, stop_, step_) {
        normalize(length);
    }

    void normalize(Eigen::Index length) {
        if (start < 0) start += length;
        if (stop < 0) stop += length;
        if (start < 0) start = 0;
        if (stop > length) stop = length;
        LOG("Normalized slice to start=" << start << ", stop=" << stop << ", step=" << step);
    }

    Eigen::Index length() const {
        // Number elements
        return (step > 0) ? (stop - start + step - 1) / step : (start - stop - step + 1) / (-step);
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

template <typename T = index_t>
slice<T> combine_slices(const slice<T>& mask, const slice<T>& overlay) {
    int start = mask.start + (overlay.start * mask.step);
    int stop = mask.start + (overlay.stop * mask.step);
    int step = mask.step * overlay.step;
    return slice<T>(start, stop, step);
}

template <typename T = index_t>
T combine_slice_with_index(const slice<T>& mask, T index) {
    if (index < 0 || index >= mask.length()) {
        throw std::out_of_range("Index is out of bounds of the mask slice. Be sure to normalize slice first.");
    }
    return mask.start + (index * mask.step);
}

#endif