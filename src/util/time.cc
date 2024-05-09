#include "util/time.h"

#include <ctime>
#include <sstream>

namespace Utils {

Timer::Timer() : elapsed_time_(0) {}

void Timer::StartTimer(const std::string& description) {
  timer_description_ = description;
  if (!timer_description_.empty()) {
    std::cout << timer_description_ << " Start! " << std::endl;
  }
  start_time_ = std::chrono::high_resolution_clock::now();
}

void Timer::StopTimer() {
  end_time_ = std::chrono::high_resolution_clock::now();
  CalculateElapsedTime();
}

double Timer::GetElapsedTime(TimeUnit unit) {
  switch (unit) {
    case Microseconds:
      return std::chrono::duration<double, std::micro>(end_time_ - start_time_).count();
    case Seconds:
      return std::chrono::duration<double>(end_time_ - start_time_).count();
    case Milliseconds:
    default:
      return std::chrono::duration<double, std::milli>(end_time_ - start_time_).count();
  }
}

void Timer::PrintElapsedTime(TimeUnit unit) {
  std::cout << timer_description_ << " took " << GetElapsedTime(unit);
  switch (unit) {
    case Microseconds:
      std::cout << " us.";
      break;
    case Seconds:
      std::cout << " s.";
      break;
    case Milliseconds:
    default:
      std::cout << " ms.";
      break;
  }
  std::cout << std::endl;
}

void Timer::CalculateElapsedTime() {
  elapsed_time_ = std::chrono::duration<double, std::milli>(end_time_ - start_time_).count();
}

std::string Timer::TimestampToReadable(long long timestamp) {
  std::time_t t = timestamp;
  std::tm* tm = std::localtime(&t);
  std::stringstream ss;
  ss << std::put_time(tm, "%Y-%m-%d %H:%M:%S");
  return ss.str();
}

double Timer::TimestampToDoubleSeconds(const uint64_t timestamp) {
  return static_cast<double>(timestamp) / 1.0e9;
}

}  // namespace Utils
