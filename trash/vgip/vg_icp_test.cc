/****************************************************************************
 *
 * Copyright (c) 2021 ZelosTech.com, Inc. All Rights Reserved
 *
 ***************************************************************************/
/**
 * @file ndt_test.cc
 **/

#include <memory>
#include <string>

#include "pcl/common/transforms.h"
#include "pcl/io/pcd_io.h"

#include "localization/common/transformation/transform3d.h"
#include "localization/matching/filter/random_sample_filter.h"

#include "gtest/gtest.h"

#define private public

#include "localization/matching/registration/vg_icp/vg_icp.h"

using ::zelos::zoe::localization::VGICP;
using ::zelos::zoe::localization::GaussianVoxelMap;
using PointType = pcl::PointXYZI;
using ::zelos::zoe::localization::Transform3d;
using ::zelos::zoe::localization::RandomSampleFilter;
using ::zelos::zoe::localization::GaussianPointCloud;

static constexpr char kLoadPcdFilePath[] = "localization/test_data/room_scan_1.pcd";
// Ndt can just get convergence in this level
// static constexpr double kMaxError = 1.0e-2;

class VGICPTest : public ::testing::Test {
 public:
  void SetUp() override {

  }

 public:
  bool debug_ = true;
};

// TEST_F(VGICPTest, AlignTest) {
//   if (!debug_) {
//     return;
//   }

// }

TEST_F(VGICPTest, PreProcessPclCloudTest) {
  if (!debug_) {
    return;
  }
  ::zelos::zoe::localization::proto::VGICPConfig vg_icp_config_;
  ::zelos::zoe::localization::proto::VGICPMapConfig map_config_;
  std::shared_ptr<VGICP> vg_icp_ = std::make_shared<VGICP>(vg_icp_config_);
  std::shared_ptr<GaussianVoxelMap> gaussian_voxel_map_ =
      std::make_shared<GaussianVoxelMap>(map_config_);
  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud_(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<PointType>::Ptr target_cloud_(new pcl::PointCloud<pcl::PointXYZI>);
  std::shared_ptr<RandomSampleFilter> random_sample_filter_ = std::make_shared<RandomSampleFilter>();

  pcl::PointCloud<PointType>::Ptr source_cloud(new pcl::PointCloud<PointType>());
  ZCHECK_NE(pcl::io::loadPCDFile<PointType>(kLoadPcdFilePath, *source_cloud), -1) << kLoadPcdFilePath;

  random_sample_filter_->Filter(0.0, 6000, source_cloud, filtered_cloud_.get());

  Eigen::Matrix3f init_rotation(Eigen::AngleAxisf(0.1, Eigen::Vector3f::UnitZ()));
  Eigen::Vector3f init_translation(0.3f, 0.4f, -0.4f);
  Eigen::Matrix4f init_matrix = Eigen::Matrix4f::Identity();
  init_matrix.topLeftCorner(3, 3) = init_rotation;
  init_matrix.topRightCorner(3, 1) = init_translation;
  std::cout << "init_matrix = \n" << init_matrix << "\n";
  target_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::transformPointCloud(*filtered_cloud_, *target_cloud_, init_matrix);
  auto map_cloud = vg_icp_->PreProcessPclCloud(target_cloud_);
  gaussian_voxel_map_->Insert(map_cloud);
  Eigen::Isometry3d final_pose = Eigen::Isometry3d::Identity();
  vg_icp_->Align(Eigen::Isometry3d::Identity(), filtered_cloud_, gaussian_voxel_map_, &final_pose);
  std::cout << "final pose: \n " << final_pose.matrix() << "\n";
}



int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
