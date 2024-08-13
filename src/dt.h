#ifndef DATETIME64_H
#define DATETIME64_H

#include <iostream>
#include <cstdint>

typedef int64_t ns_time_t;

namespace TimeConstants {
    constexpr ns_time_t NANOSECONDS_PER_SECOND = 1e9;
    constexpr ns_time_t NANOSECONDS_PER_MINUTE = 60 * NANOSECONDS_PER_SECOND;
    constexpr ns_time_t NANOSECONDS_PER_HOUR = 60 * NANOSECONDS_PER_MINUTE;
    constexpr ns_time_t NANOSECONDS_PER_DAY = 24 * NANOSECONDS_PER_HOUR;
    constexpr ns_time_t NANOSECONDS_PER_WEEK = 7 * NANOSECONDS_PER_DAY;

    constexpr int16_t DAYS_PER_COMMON_YEAR = 365;
    constexpr int16_t DAYS_PER_LEAP_YEAR = 366;

    constexpr ns_time_t NANOSECONDS_PER_COMMON_YEAR = DAYS_PER_COMMON_YEAR * NANOSECONDS_PER_DAY;
    constexpr ns_time_t NANOSECONDS_PER_LEAP_YEAR = DAYS_PER_LEAP_YEAR * NANOSECONDS_PER_DAY;

    // Days in each month for a common year and leap year
    constexpr int16_t DAYS_IN_MONTH_COMMON[] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
    constexpr int16_t DAYS_IN_MONTH_LEAP[] = { 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };

    constexpr ns_time_t NANOSECONDS_PER_MONTH_COMMON(int month) {
        return DAYS_IN_MONTH_COMMON[month] * NANOSECONDS_PER_DAY;
    }

    constexpr ns_time_t NANOSECONDS_PER_MONTH_LEAP(int month) {
        return DAYS_IN_MONTH_LEAP[month] * NANOSECONDS_PER_DAY;
    }

    constexpr bool is_leap_year(int32_t year) {
        return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    }

    constexpr int16_t days_in_year(int32_t year) {
        return is_leap_year(year) ? DAYS_PER_LEAP_YEAR : DAYS_PER_COMMON_YEAR;
    }
}

class Timedelta64;

class Datetime64 {
public:
    ns_time_t nanoseconds;

    Datetime64(ns_time_t ns = 0) : nanoseconds(ns) {}

    Datetime64 operator+(const Timedelta64 &delta) const;
    Datetime64 operator-(const Timedelta64 &delta) const;
    Timedelta64 operator-(const Datetime64 &other) const;

    // Conversion methods
    int64_t to_seconds() const { return nanoseconds / TimeConstants::NANOSECONDS_PER_SECOND; }
    int64_t to_minutes() const { return nanoseconds / TimeConstants::NANOSECONDS_PER_MINUTE; }
    int64_t to_hours() const { return nanoseconds / TimeConstants::NANOSECONDS_PER_HOUR; }
    int64_t to_days() const { return nanoseconds / TimeConstants::NANOSECONDS_PER_DAY; }
    int64_t to_weeks() const { return nanoseconds / TimeConstants::NANOSECONDS_PER_WEEK; }

    int64_t to_years(int32_t start_year) const {
        ns_time_t total_nanoseconds = nanoseconds;
        int32_t year_counter = start_year;
        int64_t year_count = 0;

        while (total_nanoseconds > 0) {
            ns_time_t ns_per_year = TimeConstants::is_leap_year(year_counter) ?
                                      TimeConstants::NANOSECONDS_PER_LEAP_YEAR :
                                      TimeConstants::NANOSECONDS_PER_COMMON_YEAR;

            if (total_nanoseconds >= ns_per_year) {
                total_nanoseconds -= ns_per_year;
                ++year_count;
                ++year_counter;
            } else {
                break;
            }
        }
        return year_count;
    }

    int64_t to_months(int32_t start_year, int16_t start_month) const {
        ns_time_t total_nanoseconds = nanoseconds;
        int32_t year_counter = start_year;
        int16_t month_counter = start_month;
        int64_t month_count = 0;

        while (total_nanoseconds > 0) {
            ns_time_t ns_per_month = TimeConstants::is_leap_year(year_counter) ?
                                       TimeConstants::NANOSECONDS_PER_MONTH_LEAP(month_counter) :
                                       TimeConstants::NANOSECONDS_PER_MONTH_COMMON(month_counter);

            if (total_nanoseconds >= ns_per_month) {
                total_nanoseconds -= ns_per_month;
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
};

class Timedelta64 {
public:
    ns_time_t nanoseconds;

    Timedelta64(ns_time_t ns = 0) : nanoseconds(ns) {}

    Timedelta64 operator+(const Timedelta64 &other) const {
        return Timedelta64(nanoseconds + other.nanoseconds);
    }

    Timedelta64 operator-(const Timedelta64 &other) const {
        return Timedelta64(nanoseconds - other.nanoseconds);
    }
};

// Definitions of the overloaded operators for Datetime64
inline Datetime64 Datetime64::operator+(const Timedelta64 &delta) const {
    return Datetime64(nanoseconds + delta.nanoseconds);
}

inline Datetime64 Datetime64::operator-(const Timedelta64 &delta) const {
    return Datetime64(nanoseconds - delta.nanoseconds);
}

inline Timedelta64 Datetime64::operator-(const Datetime64 &other) const {
    return Timedelta64(nanoseconds - other.nanoseconds);
}

#endif // DATETIME64_H
