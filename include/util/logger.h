#pragma once

#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace Utils {

enum LogLevel { ALL = -1, DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3, FATAL = 4 };

class Logger {
 public:
  class LogStream {
   public:
    LogStream(Logger& logger, LogLevel level)
        : logger_(logger), level_(level) {}
    ~LogStream() {
      if (!buffer_.str().empty()) {  // Ensure there is something to log
        logger_.Log(buffer_.str(), level_);
      }
    }

    template <typename T>
    LogStream& operator<<(const T& message) {
      buffer_ << message;
      return *this;
    }

    // 删除复制构造函数和赋值运算符
    LogStream(const LogStream&) = delete;
    LogStream& operator=(const LogStream&) = delete;

    // 定义移动构造函数
    LogStream(LogStream&&) = default;
    LogStream& operator=(LogStream&&) = default;

   private:
    Logger& logger_;
    LogLevel level_;
    std::ostringstream buffer_;
  };

 public:
  explicit Logger(const std::string& filename = "");
  ~Logger();
  void SetConsoleLevel(const std::string& level);
  static LogLevel StringToLogLevel(const std::string& levelStr);
  void EnableConsoleLog(const bool enable);
  void SetLogFile(const std::string& file_path);
  LogStream Log(LogLevel level) { return LogStream(*this, level); }

 private:
  std::string GetTimestamp();
  std::string GetLevelString(LogLevel level);
  std::string GetColorCode(LogLevel level);
  bool IsLevelEnabledForConsole(LogLevel level);
  void Log(const std::string& message, LogLevel level);  // 保持为私有方法

 private:
  std::vector<bool> console_log_levels_;
  bool console_log_enabled_;
  std::unique_ptr<std::ofstream> file_stream_;
  std::mutex mutex_;
};

}  // namespace Utils
