#include "sensor/sensor_base.h"

namespace Sensor{

SensorBase::SensorBase() {
    // 可以在这里设置默认的日志配置，如文件名或控制台输出启用
    logger_.EnableConsoleLog(true);  // 默认开启控制台日志
    logger_.SetConsoleLevel("INFO"); // 默认日志级别为 INFO
}

SensorBase::~SensorBase() {
    // 析构函数内容，如果有需要清理的资源
}
}



