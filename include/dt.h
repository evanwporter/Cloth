#ifndef DATETIME64_H
#define DATETIME64_H

#include <iostream>
#include <cstdint>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <functional>

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

class timedelta {
public:
    timedelta(dtime_t data = 0) : data_(data) {}

    // String constructor
    timedelta(const std::string& str) {
        data_ = parse_string(str);
    }

    dtime_t data() const { return data_; }

    timedelta operator+(const timedelta &other) const {
        return timedelta(data_ + other.data_);
    }

    timedelta operator-(const timedelta &other) const {
        return timedelta(data_ - other.data_);
    }

    timedelta operator*(const index_t &other) const {
        return timedelta(data_ * other);
    }

private:
    dtime_t data_;

    static dtime_t parse_string(const std::string& str) {
        if (str.empty()) {
            throw std::invalid_argument("Invalid input string");
        }

        size_t num_length = 0;
        while (num_length < str.size() && isdigit(str[num_length])) {
            num_length++;
        }

        if (num_length == 0) {
            throw std::invalid_argument("No numeric value found in string");
        }

        int64_t value = std::stoll(str.substr(0, num_length));
        std::string unit = str.substr(num_length);

        if (unit == "ns") {
            return value;
        } else if (unit == "us") {
            return value * (UNITS_PER_SECOND / 1e6);
        } else if (unit == "ms") {
            return value * (UNITS_PER_SECOND / 1e3);
        } else if (unit == "s") {
            return value * UNITS_PER_SECOND;
        } else if (unit == "m") {
            return value * UNITS_PER_MINUTE;
        } else if (unit == "h") {
            return value * UNITS_PER_HOUR;
        } else if (unit == "D") {
            return value * UNITS_PER_DAY;
        } else if (unit == "W") {
            return value * UNITS_PER_WEEK;
        } else if (unit == "Y") {
            // Assuming 1 year = 365 days
            return value * 365 * UNITS_PER_DAY;
        } else {
            throw std::invalid_argument("Unknown time unit: " + unit);
        }
    }
};

class datetime {
private:
    dtime_t data_;

public:

    datetime(dtime_t data) : data_(data) {}

    // ISO 8601 formatted string constructor
    datetime(const std::string& iso_time) : data_(iso_to_time_units(iso_time)) {}

    dtime_t data() const { return data_; }

    datetime operator+(const timedelta &delta) const;
    datetime operator-(const timedelta &delta) const;
    timedelta operator-(const datetime &other) const;

    // Floor the datetime to the nearest given timedelta
    datetime floor(const timedelta &delta) const {
        dtime_t floored_data = (data_ / delta.data()) * delta.data();
        return datetime(floored_data);
    }

    // Ceil the datetime to the nearest given timedelta
    datetime ceil(const timedelta &delta) const {
        dtime_t floored_data = (data_ / delta.data()) * delta.data();
        if (data_ % delta.data() != 0) {
            floored_data += delta.data();
        }
        return datetime(floored_data);
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

    bool operator==(const datetime& other) const {
        return data_ == other.data();
    }

    bool operator!=(const datetime& other) const {
        return data_ != other.data();
    }

    bool operator<(const datetime& other) const {
        return data_ < other.data();
    }

    bool operator<=(const datetime& other) const {
        return data_ <= other.data();
    }

    bool operator>(const datetime& other) const {
        return data_ > other.data();
    }

    bool operator>=(const datetime& other) const {
        return data_ >= other.data();
    }

    std::string to_iso() const {

        time_t total_seconds = seconds();

        std::tm t;

        // TODO: Allow other systems since this only works for windows
        _gmtime64_s(&t, &total_seconds);  // Use gmtime to get UTC time

        std::ostringstream oss;
        oss << std::put_time(&t, "%Y-%m-%dT%H:%M:%S");

        #ifdef USE_MILLISECONDS
        dtime_t remaining_units = data_ % UNITS_PER_SECOND;
        int milliseconds = static_cast<int>(remaining_units * 1000 / UNITS_PER_SECOND);
        if (milliseconds > 0) {
            oss << "." << std::setw(3) << std::setfill('0') << milliseconds;
        }
        #endif

        #ifdef USE_MICROSECONDS
        dtime_t remaining_units = data_ % UNITS_PER_SECOND;
        int microseconds = static_cast<int>(remaining_units * 1000000 / UNITS_PER_SECOND);
        if (microseconds > 0) {
            oss << "." << std::setw(6) << std::setfill('0') << microseconds;
        }
        #endif

        #ifdef USE_NANOSECONDS
        dtime_t remaining_units = data_ % UNITS_PER_SECOND;
        int nanoseconds = static_cast<int>(remaining_units * 1000000000 / UNITS_PER_SECOND);
        if (nanoseconds > 0) {
            oss << "." << std::setw(9) << std::setfill('0') << nanoseconds;
        }
        #endif

        return oss.str();
    }
};

inline datetime datetime::operator+(const timedelta &delta) const {
    return datetime(data_ + delta.data());
}

inline datetime datetime::operator-(const timedelta &delta) const {
    return datetime(data_ - delta.data());
}

inline timedelta datetime::operator-(const datetime &other) const {
    return timedelta(data_ - other.data());
};

dtime_t iso_to_time_units(const std::string& iso_time) {
    std::tm t = {};
    std::istringstream ss(iso_time);
    ss >> std::get_time(&t, "%Y-%m-%dT%H:%M:%S");

    if (ss.fail()) {
        throw std::invalid_argument("Invalid ISO format");
    }

    // Assume the input time is in GMT/UTC
    t.tm_isdst = -1;

    // Convert the tm structure to time_t which is the number of seconds since the Unix epoch
    time_t epochTime = _mkgmtime(&t);

    // Depending on the defined time unit (USE_SECONDS, USE_MILLISECONDS, etc.)
    dtime_t total_units = epochTime * UNITS_PER_SECOND;

    #ifdef USE_MILLISECONDS
    size_t dot_pos = iso_time.find('.');
    if (dot_pos != std::string::npos && iso_time.length() > dot_pos + 3) {
        int milliseconds = std::stoi(iso_time.substr(dot_pos + 1, 3));
        total_units += milliseconds * (UNITS_PER_SECOND / 1000);
    }
    #endif

    #ifdef USE_MICROSECONDS
    size_t dot_pos = iso_time.find('.');
    if (dot_pos != std::string::npos && iso_time.length() > dot_pos + 6) {
        int microseconds = std::stoi(iso_time.substr(dot_pos + 4, 3));
        total_units += microseconds * (UNITS_PER_SECOND / 1000000);
    }
    #endif

    #ifdef USE_NANOSECONDS
    size_t dot_pos = iso_time.find('.');
    if (dot_pos != std::string::npos && iso_time.length() > dot_pos + 9) {
        int nanoseconds = std::stoi(iso_time.substr(dot_pos + 7, 3));
        total_units += nanoseconds * (UNITS_PER_SECOND / 1000000000);
    }
    #endif

    return total_units;
}

namespace std {
    template<>
    struct hash<datetime> {
        std::size_t operator()(const datetime& dt) const noexcept {
            return std::hash<time_t>{}(dt.seconds());
        }
    };
}


#endif // DATETIME64_H
