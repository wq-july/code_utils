#include "common/state.h"

namespace Common {

SimpleState::SimpleState() = default;

SimpleState::SimpleState(double time, const Sophus::SO3d& rot,
                         const Eigen::Vector3d& trans,
                         const Eigen::Vector3d& vel, const Eigen::Vector3d& bg,
                         const Eigen::Vector3d& ba)
    : timestamp_(time), rot_(rot), trans_(trans), vel_(vel), bg_(bg), ba_(ba) {}

SimpleState::SimpleState(double time, const Sophus::SE3d& pose,
                         const Eigen::Vector3d& vel)
    : timestamp_(time),
      rot_(pose.so3()),
      trans_(pose.translation()),
      vel_(vel) {}

Sophus::SE3d SimpleState::GetSE3() const { return Sophus::SE3d(rot_, trans_); }

std::ostream& operator<<(std::ostream& os, const SimpleState& s) {
  os << "p: " << s.trans_.transpose() << "\n"
     << "v: " << s.vel_.transpose() << "\n"
     << "q: " << s.rot_.unit_quaternion().coeffs().transpose() << "\n"
     << "bg: " << s.bg_.transpose() << "\n"
     << "ba: " << s.ba_.transpose() << "\n";
  return os;
}

}  // namespace Common
