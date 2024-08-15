#ifndef DATETIME64_H
#define DATETIME64_H

#include <iostream>
#include <cstdint>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <functional>


//#define USE_NANOSECONDS
//#define USE_MICROSECONDS
//#define USE_MILLISECONDS
//#define USE_SECONDS
//#define USE_MINUTES
//#define USE_HOURS
//#define USE_DAYS

#ifndef INDEX_T
#define INDEX_T
#include <cstddef>
typedef std::ptrdiff_t index_t;
#endif

#ifndef USE_NANOSECONDS
#ifndef USE_MICROSECONDS
#ifndef USE_MILLISECONDS
#ifndef USE_SECONDS
#ifndef USE_MINUTES
#ifndef USE_HOURS
#ifndef USE_DAYS
    #define USE_SECONDS // Default macro
#endif
#endif
#endif
#endif
#endif
#endif
#endif

typedef uint64_t ns_time_t, us_time_t, ms_time_t;
typedef uint32_t s_time_t, minutes_time_t;
typedef uint16_t days_time_t, weeks_time_t, month_time_t;
typedef uint8_t years_time_t;

#ifdef USE_NANOSECONDS
    typedef uint64_t dtime_t;
    #define UNITS_PER_SECOND 1e9
    #define UNITS_PER_MINUTE (60 * UNITS_PER_SECOND)
    #define UNITS_PER_HOUR (60 * UNITS_PER_MINUTE)
    #define UNITS_PER_DAY (24 * UNITS_PER_HOUR)
    #define UNITS_PER_WEEK (7 * UNITS_PER_DAY)
#elif defined(USE_MICROSECONDS)
    typedef uint64_t dtime_t;
    #define UNITS_PER_SECOND 1e6
    #define UNITS_PER_MINUTE (60 * UNITS_PER_SECOND)
    #define UNITS_PER_HOUR (60 * UNITS_PER_MINUTE)
    #define UNITS_PER_DAY (24 * UNITS_PER_HOUR)
    #define UNITS_PER_WEEK (7 * UNITS_PER_DAY)
#elif defined(USE_MILLISECONDS)
    typedef uint64_t dtime_t;
    #define UNITS_PER_SECOND 1e3
    #define UNITS_PER_MINUTE (60 * UNITS_PER_SECOND)
    #define UNITS_PER_HOUR (60 * UNITS_PER_MINUTE)
    #define UNITS_PER_DAY (24 * UNITS_PER_HOUR)
    #define UNITS_PER_WEEK (7 * UNITS_PER_DAY)
#elif defined(USE_SECONDS)
    typedef uint32_t dtime_t;
    #define UNITS_PER_SECOND 1
    #define UNITS_PER_MINUTE (60 * UNITS_PER_SECOND)
    #define UNITS_PER_HOUR (60 * UNITS_PER_MINUTE)
    #define UNITS_PER_DAY (24 * UNITS_PER_HOUR)
    #define UNITS_PER_WEEK (7 * UNITS_PER_DAY)
#elif defined(USE_MINUTES)
    typedef uint32_t dtime_t;
    #define UNITS_PER_MINUTE 1
    #define UNITS_PER_HOUR (60 * UNITS_PER_MINUTE)
    #define UNITS_PER_DAY (24 * UNITS_PER_HOUR)
    #define UNITS_PER_WEEK (7 * UNITS_PER_DAY)
#elif defined(USE_HOURS)
    typedef uint32_t dtime_t;
    #define UNITS_PER_HOUR 1
    #define UNITS_PER_DAY (24 * UNITS_PER_HOUR)
    #define UNITS_PER_WEEK (7 * UNITS_PER_DAY)
#elif defined(USE_DAYS)
    typedef uint16_t dtime_t;
    #define UNITS_PER_DAY 1
    #define UNITS_PER_WEEK (7 * UNITS_PER_DAY)
#endif

namespace TimeConstants {
    constexpr int16_t DAYS_PER_COMMON_YEAR = 365;
    constexpr int16_t DAYS_PER_LEAP_YEAR = 366;

    // Days in each month for a common year and leap year
    constexpr int16_t DAYS_IN_MONTH_COMMON[] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
    constexpr int16_t DAYS_IN_MONTH_LEAP[] = { 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };

    constexpr dtime_t UNITS_PER_MONTH_COMMON(int month) {
        return DAYS_IN_MONTH_COMMON[month] * UNITS_PER_DAY;
    }

    constexpr dtime_t UNITS_PER_MONTH_LEAP(int month) {
        return DAYS_IN_MONTH_LEAP[month] * UNITS_PER_DAY;
    }

    constexpr bool is_leap_year(int32_t year) {
        return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    }

    constexpr int16_t days_in_year(int32_t year) {
        return is_leap_year(year) ? DAYS_PER_LEAP_YEAR : DAYS_PER_COMMON_YEAR;
    }
};

// Forward declaration for the iso_datetime64 function
dtime_t iso_to_time_units(const std::string& iso_time);

class Timedelta64 {
public:
    dtime_t data_;

    Timedelta64(dtime_t units = 0) : data_(units) {}

    Timedelta64 operator+(const Timedelta64 &other) const {
        return Timedelta64(data_ + other.data_);
    }

    Timedelta64 operator-(const Timedelta64 &other) const {
        return Timedelta64(data_ - other.data_);
    }

    Timedelta64 operator*(const index_t &other) const {
        return Timedelta64(data_ * other);
    }
};

class Datetime64 {
private:
    dtime_t data_;

public:

    Datetime64(dtime_t data) : data_(data) {}

    // ISO 8601 formatted string constructor
    Datetime64(const std::string& iso_time) : data_(iso_to_time_units(iso_time)) {}

    Datetime64 operator+(const Timedelta64 &delta) const;
    Datetime64 operator-(const Timedelta64 &delta) const;
    Timedelta64 operator-(const Datetime64 &other) const;

    // Floor the datetime to the nearest given Timedelta64
    Datetime64 floor(const Timedelta64 &delta) const {
        dtime_t floored_data = (data_ / delta.data_) * delta.data_;
        return Datetime64(floored_data);
    }

    // Ceil the datetime to the nearest given Timedelta64
    Datetime64 ceil(const Timedelta64 &delta) const {
        dtime_t floored_data = (data_ / delta.data_) * delta.data_;
        if (data_ % delta.data_ != 0) {
            floored_data += delta.data_;
        }
        return Datetime64(floored_data);
    }

    // Conversion methods
    int64_t seconds() const { return data_ / UNITS_PER_SECOND; }
    int64_t minutes() const { return data_ / UNITS_PER_MINUTE; }
    int64_t hours() const { return data_ / UNITS_PER_HOUR; }
    int64_t days() const { return data_ / UNITS_PER_DAY; }
    int64_t weeks() const { return data_ / UNITS_PER_WEEK; }

    int64_t years(int32_t start_year) const {
        dtime_t total_units = data_;
        int32_t year_counter = start_year;
        int64_t year_count = 0;

        while (total_units > 0) {
            dtime_t units_per_year = TimeConstants::is_leap_year(year_counter) ?
                                      TimeConstants::DAYS_PER_LEAP_YEAR * UNITS_PER_DAY :
                                      TimeConstants::DAYS_PER_COMMON_YEAR * UNITS_PER_DAY;

            if (total_units >= units_per_year) {
                total_units -= units_per_year;
                ++year_count;
                ++year_counter;
            } else {
                break;
            }
        }
        return year_count;
    }

    int64_t months(int32_t start_year, int16_t start_month) const {
        dtime_t total_units = data_;
        int32_t year_counter = start_year;
        int16_t month_counter = start_month;
        int64_t month_count = 0;

        while (total_units > 0) {
            dtime_t units_per_month = TimeConstants::is_leap_year(year_counter) ?
                                       TimeConstants::UNITS_PER_MONTH_LEAP(month_counter) :
                                       TimeConstants::UNITS_PER_MONTH_COMMON(month_counter);

            if (total_units >= units_per_month) {
                total_units -= units_per_month;
                ++month_count;
                ++month_counter;

                if (month_counter == 12) {
                    month_counter = 0;
                    ++year_counter;
                }
            } else {
                break;
            }
        }
        return month_count;
    }

    bool operator==(const Datetime64& other) const {
        return data_ == other.data_;
    }

    bool operator!=(const Datetime64& other) const {
        return data_ != other.data_;
    }

    bool operator<(const Datetime64& other) const {
        return data_ < other.data_;
    }

    bool operator<=(const Datetime64& other) const {
        return data_ <= other.data_;
    }

    bool operator>(const Datetime64& other) const {
        return data_ > other.data_;
    }

    bool operator>=(const Datetime64& other) const {
        return data_ >= other.data_;
    }
};

inline Datetime64 Datetime64::operator+(const Timedelta64 &delta) const {
    return Datetime64(data_ + delta.data_);
}

inline Datetime64 Datetime64::operator-(const Timedelta64 &delta) const {
    return Datetime64(data_ - delta.data_);
}

inline Timedelta64 Datetime64::operator-(const Datetime64 &other) const {
    return Timedelta64(data_ - other.data_);
};

dtime_t iso_to_time_units(const std::string& iso_time) {
    std::tm t = {};
    std::istringstream ss(iso_time);
    ss >> std::get_time(&t, "%Y-%m-%dT%H:%M:%S");

    if (ss.fail()) {
        throw std::invalid_argument("Invalid ISO format");
    }

    dtime_t total_units = 0;

    for (int32_t year = 1970; year < t.tm_year + 1900; ++year) {
        total_units += TimeConstants::days_in_year(year) * UNITS_PER_DAY;
    }

    for (int16_t month = 0; month < t.tm_mon; ++month) {
        if (TimeConstants::is_leap_year(t.tm_year + 1900)) {
            total_units += TimeConstants::UNITS_PER_MONTH_LEAP(month);
        } else {
            total_units += TimeConstants::UNITS_PER_MONTH_COMMON(month);
        }
    }

    total_units += (t.tm_mday - 1) * UNITS_PER_DAY;

    #ifdef USE_HOURS
    total_units += t.tm_hour * UNITS_PER_HOUR;
    #endif

    #ifdef USE_MINUTES
    total_units += t.tm_min * UNITS_PER_MINUTE;
    #endif

    #ifdef USE_SECONDS
    total_units += t.tm_sec * UNITS_PER_SECOND;
    #endif

    #ifdef USE_MILLISECONDS
    size_t dot_pos = iso_time.find('.');
    if (dot_pos != std::string::npos && iso_time.length() > dot_pos + 3) {
        int milliseconds = std::stoi(iso_time.substr(dot_pos + 1, 3));
        total_units += milliseconds * (UNITS_PER_SECOND / 1000);
    }
    #endif

    #ifdef USE_MICROSECONDS
    size_t micro_pos = iso_time.find_first_of("0123456789", dot_pos + 4);
    if (micro_pos != std::string::npos && iso_time.length() > micro_pos + 2) {
        int microseconds = std::stoi(iso_time.substr(micro_pos, 3));
        total_units += microseconds * (UNITS_PER_SECOND / 1000000);
    }
    #endif

    #ifdef USE_NANOSECONDS
    size_t nano_pos = iso_time.find_first_of("0123456789", micro_pos + 4);
    if (nano_pos != std::string::npos && iso_time.length() > nano_pos + 2) {
        int nanoseconds = std::stoi(iso_time.substr(nano_pos, 3));
        total_units += nanoseconds * (UNITS_PER_SECOND / 1000000000);
    }
    #endif

    return total_units;
}

namespace std {
    template<>
    struct hash<Datetime64> {
        std::size_t operator()(const Datetime64& dt) const noexcept {
            return std::hash<time_t>{}(dt.seconds());
        }
    };
}


#endif // DATETIME64_H
