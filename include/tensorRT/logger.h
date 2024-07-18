#pragma once

#include "tensorRT/logging.h"

class ErrorRecorder;
extern ErrorRecorder gRecorder;
namespace TensorRT {
extern Logger gLogger;
extern LogStreamConsumer gLogVerbose;
extern LogStreamConsumer gLogInfo;
extern LogStreamConsumer gLogWarning;
extern LogStreamConsumer gLogError;
extern LogStreamConsumer gLogFatal;

void SetReportableSeverity(Logger::Severity severity);
}  // namespace TensorRT
