
#include "util/logger.h"

namespace Utils {
Logger::Logger(const std::string& filename)
    : console_log_levels_(FATAL + 1, false), console_log_enabled_(true) {
  if (!filename.empty()) {
    file_stream_ = std::make_unique<std::ofstream>(filename, std::ios::app);
  }
  console_log_levels_[INFO] = true;  // 默认启用INFO级别
}

Logger::~Logger() {
  if (file_stream_) {
    file_stream_->close();
  }
}

void Logger::SetLogFile(const std::string& file_path) {
  if (!file_path.empty()) {
    file_stream_ = std::make_unique<std::ofstream>(file_path, std::ios::app);
  }
}

void Logger::Log(const std::string& message, LogLevel level) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::ostringstream log_stream;
  log_stream << GetTimestamp() << " [" << GetLevelString(level) << "] "
             << message << std::endl;
  std::string log_output = log_stream.str();

  if (console_log_enabled_ && IsLevelEnabledForConsole(level)) {
    std::cout << GetColorCode(level) << log_output << "\033[0m";
  }
  if (file_stream_ && file_stream_->is_open()) {
    *file_stream_ << log_output;
  }
}

void Logger::SetConsoleLevel(const std::string& level) {
  LogLevel logLevel = StringToLogLevel(level);
  std::fill(console_log_levels_.begin(), console_log_levels_.end(), false);
  if (logLevel == ALL) {
    std::fill(console_log_levels_.begin(), console_log_levels_.end(), true);
  } else {
    console_log_levels_[logLevel] = true;
  }
}

LogLevel Logger::StringToLogLevel(const std::string& levelStr) {
  static const std::unordered_map<std::string, LogLevel> levelMap = {
      {"ALL", ALL},   {"DEBUG", DEBUG}, {"INFO", INFO},
      {"WARN", WARN}, {"ERROR", ERROR}, {"FATAL", FATAL}};
  auto it = levelMap.find(levelStr);
  return it != levelMap.end() ? it->second
                              : INFO;  // Default to INFO if not found
}

void Logger::EnableConsoleLog(const bool enable) {
  console_log_enabled_ = enable;
}

std::string Logger::GetTimestamp() {
  std::time_t now = std::time(nullptr);
  char buf[20];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
  return std::string(buf);
}

std::string Logger::GetLevelString(LogLevel level) {
  switch (level) {
    case DEBUG:
      return "DEBUG";
    case INFO:
      return "INFO";
    case WARN:
      return "WARN";
    case ERROR:
      return "ERROR";
    case FATAL:
      return "FATAL";
    default:
      return "UNKNOWN";
  }
}

std::string Logger::GetColorCode(LogLevel level) {
  switch (level) {
    case DEBUG:
      return "\033[36m";  // Cyan
    case INFO:
      return "\033[32m";  // Green
    case WARN:
      return "\033[33m";  // Yellow
    case ERROR:
      return "\033[31m";  // Red
    case FATAL:
      return "\033[35m";  // Magenta
    default:
      return "\033[0m";  // Reset
  }
}

bool Logger::IsLevelEnabledForConsole(LogLevel level) {
  if (level >= -1 && level <= FATAL) {
    return console_log_levels_[level];
  }
  return false;
}

}  // namespace Utils