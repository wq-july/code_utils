#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>

namespace Utils {

class TimeUtils {
public:
    enum TimeUnit { Milliseconds, Microseconds, Seconds };

    TimeUtils();
    void StartTimer(const std::string& description = "");
    void StopTimer();
    double GetElapsedTime(TimeUnit unit = Milliseconds);
    void PrintElapsedTime(TimeUnit unit = Milliseconds);

// static function
public:
    static std::string TimestampToReadable(long long timestamp);
    static double TimestampToDoubleSeconds(const uint64_t timestamp);

private:
    void CalculateElapsedTime();

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    std::string timer_description_;
    double elapsed_time_;
};

}