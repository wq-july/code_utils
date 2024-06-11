/****************************************************************************
 *
 * Copyright (c) 2023 ZelosTech.com, Inc. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file vg_icp.h
 **/

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "Eigen/Eigenvalues"
#include "pcl/common/transforms.h"

#include "localization/common/utils/math.h"
#include "localization/matching/registration/vg_icp/voxel_map/gaussian_voxel_map.h"
#include "localization/matching/registration/vg_icp/common/voxelnn.h"

#include "localization/matching/registration/proto/vgicp_config.pb.h"


namespace zelos {
namespace zoe {
namespace localization {

namespace {
using zelos::zoe::localization::GaussianVoxel;
using zelos::zoe::localization::GaussianVoxelMap;
using zelos::zoe::localization::GaussianPointCloud;
using zelos::zoe::localization::VoxelNN;
using zelos::zoe::localization::Boundary;
}  // namespace

class VGICP {
 public:
  VGICP(const ::zelos::zoe::localization::proto::VGICPConfig& config);

 private:

  std::shared_ptr<GaussianPointCloud> PreProcessPclCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud);

  void Align(const Eigen::Isometry3d& guess_matrix,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
    const std::shared_ptr<GaussianVoxelMap>& gaussian_map,
    Eigen::Isometry3d* const final_matrix);

 private:
  ::zelos::zoe::localization::proto::VGICPConfig config_;
  std::shared_ptr<GaussianVoxelMap> map_ = nullptr;
  std::shared_ptr<VoxelNN> voxelnn_ = nullptr;

  int32_t num_threads_ = 0;
  double rotation_eps_ = 0.0;
  double translation_eps_ = 0.0;
  double lambda_ = 0.0;
};

}  // namespace localization
}  // namespace zoe
}  // namespace zelos

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
