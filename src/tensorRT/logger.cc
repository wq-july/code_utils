
#include "tensorRT/logger.h"

#include "tensorRT/error_recorder.h"
#include "tensorRT/logging.h"
using namespace nvinfer1;
ErrorRecorder gRecorder;
namespace TensorRT {
Logger gLogger{Logger::Severity::kINFO};
LogStreamConsumer gLogVerbose{LOG_VERBOSE(gLogger)};
LogStreamConsumer gLogInfo{LOG_INFO(gLogger)};
LogStreamConsumer gLogWarning{LOG_WARN(gLogger)};
LogStreamConsumer gLogError{LOG_ERROR(gLogger)};
LogStreamConsumer gLogFatal{LOG_FATAL(gLogger)};

void SetReportableSeverity(Logger::Severity severity) {
  gLogger.setReportableSeverity(severity);
  gLogVerbose.setReportableSeverity(severity);
  gLogInfo.setReportableSeverity(severity);
  gLogWarning.setReportableSeverity(severity);
  gLogError.setReportableSeverity(severity);
  gLogFatal.setReportableSeverity(severity);
}
}  // namespace TensorRT
