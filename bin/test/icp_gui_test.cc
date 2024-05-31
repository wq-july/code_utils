#include <pangolin/pangolin.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <thread>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "pangolin/display/display.h"
#include "pangolin/display/view.h"
#include "pangolin/gl/gldraw.h"
#include "pangolin/handler/handler.h"
#include "pangolin/var/var.h"
#include "pcl/io/pcd_io.h"

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

DEFINE_string(scan_pcd_path, "../bin/data/lidar/scan.pcd", "scan点云路径");
DEFINE_string(map_pcd_path, "../bin/data/lidar/map.pcd", "地图点云路径");

class ICPGUITest : public testing::Test {
 public:
  void SetUp() override {
    scan_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::io::loadPCDFile<pcl::PointXYZ>(FLAGS_scan_pcd_path, *scan_);
    map_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::io::loadPCDFile<pcl::PointXYZ>(FLAGS_map_pcd_path, *map_);
  }

  // 执行ICP匹配的函数
  Eigen::Matrix4f PerformICP(const PointCloud::Ptr& source, const PointCloud::Ptr& target) {
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    PointCloud::Ptr aligned(new PointCloud);
    icp.align(*aligned);

    if (icp.hasConverged()) {
      LOG(INFO) << "ICP has converged, score: " << icp.getFitnessScore();
      return icp.getFinalTransformation();
    } else {
      LOG(FATAL) << "ICP did not converge.";
      return Eigen::Matrix4f::Identity();
    }
  }

  // 绘制坐标轴的函数
  void DrawAxis(const Eigen::Vector3f& origin) {
    const float axis_length = 5.0f;
    const float axis_line_width = 5.0f;

    glLineWidth(axis_line_width);
    glBegin(GL_LINES);

    // X axis - Red
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(origin.x(), origin.y(), origin.z());
    glVertex3f(origin.x() + axis_length, origin.y(), origin.z());

    // Y axis - Green
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(origin.x(), origin.y(), origin.z());
    glVertex3f(origin.x(), origin.y() + axis_length, origin.z());

    // Z axis - Blue
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(origin.x(), origin.y(), origin.z());
    glVertex3f(origin.x(), origin.y(), origin.z() + axis_length);

    glEnd();
  }

  Eigen::Vector3f ComputePointCloudCenter(const PointCloud::Ptr& cloud) {
    Eigen::Vector3f center(0.0f, 0.0f, 0.0f);
    for (const auto& point : cloud->points) {
      center += Eigen::Vector3f(point.x, point.y, point.z);
    }
    center /= static_cast<float>(cloud->points.size());
    return center;
  }

  PointCloud::Ptr scan_ = nullptr;
  PointCloud::Ptr map_ = nullptr;
  bool enable_test_ = true;
};

TEST_F(ICPGUITest, ICPGUITest) {
  int window_width = 1920;
  int window_height = 1080;
  pangolin::CreateWindowAndBind("PointCloud Viewer", window_width, window_height);
  glEnable(GL_DEPTH_TEST);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(
          window_width, window_height, 420, 420, window_width / 2, window_height / 2, 0.2, 1000),
      pangolin::ModelViewLookAt(1094, 8658, -14, 1094, 8678, 6, pangolin::AxisY));

  pangolin::View& d_cam =
      pangolin::CreateDisplay()
          .SetBounds(0.0, 1.0, 0.0, 1.0, -window_width / static_cast<float>(window_height))
          .SetHandler(new pangolin::Handler3D(s_cam));

  Eigen::Matrix4f scan_transform = Eigen::Matrix4f::Identity();
  scan_transform(0, 3) = 1094.f;
  scan_transform(1, 3) = 8678.f;
  scan_transform(2, 3) = 6.f;

  float step = 0.1f;
  float angle_step = 0.05f;

  // 创建控制面板
  pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(200));
  pangolin::Var<std::function<void()>> btn_w("ui.Move Up", [&scan_transform, step]() {
    scan_transform(1, 3) += step;
  });
  pangolin::Var<std::function<void()>> btn_s("ui.Move Down", [&scan_transform, step]() {
    scan_transform(1, 3) -= step;
  });
  pangolin::Var<std::function<void()>> btn_a("ui.Move Left", [&scan_transform, step]() {
    scan_transform(0, 3) -= step;
  });
  pangolin::Var<std::function<void()>> btn_d("ui.Move Right", [&scan_transform, step]() {
    scan_transform(0, 3) += step;
  });
  pangolin::Var<std::function<void()>> btn_q("ui.Move Up (Z)", [&scan_transform, step]() {
    scan_transform(2, 3) += step;
  });
  pangolin::Var<std::function<void()>> btn_e("ui.Move Down (Z)", [&scan_transform, step]() {
    scan_transform(2, 3) -= step;
  });

  // 注册键盘事件回调函数
  pangolin::RegisterKeyPressCallback('w', [&scan_transform, step]() {
    scan_transform(1, 3) += step;
  });
  pangolin::RegisterKeyPressCallback('s', [&scan_transform, step]() {
    scan_transform(1, 3) -= step;
  });
  pangolin::RegisterKeyPressCallback('a', [&scan_transform, step]() {
    scan_transform(0, 3) -= step;
  });
  pangolin::RegisterKeyPressCallback('d', [&scan_transform, step]() {
    scan_transform(0, 3) += step;
  });
  pangolin::RegisterKeyPressCallback('q', [&scan_transform, step]() {
    scan_transform(2, 3) += step;
  });
  pangolin::RegisterKeyPressCallback('e', [&scan_transform, step]() {
    scan_transform(2, 3) -= step;
  });

  pangolin::RegisterKeyPressCallback('r', [&scan_transform, angle_step]() {
    Eigen::Matrix4f roll = Eigen::Matrix4f::Identity();
    roll.block<3, 3>(0, 0) =
        Eigen::AngleAxisf(angle_step, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    scan_transform = scan_transform * roll;
  });
  pangolin::RegisterKeyPressCallback('p', [&scan_transform, angle_step]() {
    Eigen::Matrix4f pitch = Eigen::Matrix4f::Identity();
    pitch.block<3, 3>(0, 0) =
        Eigen::AngleAxisf(angle_step, Eigen::Vector3f::UnitX()).toRotationMatrix();
    scan_transform = scan_transform * pitch;
  });
  pangolin::RegisterKeyPressCallback('y', [&scan_transform, angle_step]() {
    Eigen::Matrix4f yaw = Eigen::Matrix4f::Identity();
    yaw.block<3, 3>(0, 0) =
        Eigen::AngleAxisf(angle_step, Eigen::Vector3f::UnitY()).toRotationMatrix();
    scan_transform = scan_transform * yaw;
  });

  pangolin::RegisterKeyPressCallback(' ', [this, &scan_transform]() {
    PointCloud::Ptr transformed_scan(new PointCloud);
    pcl::transformPointCloud(*this->scan_, *transformed_scan, scan_transform);
    Eigen::Matrix4f icp_transform = this->PerformICP(transformed_scan, this->map_);
    scan_transform = icp_transform * scan_transform;
  });

  Eigen::Vector3f map_center = ComputePointCloudCenter(map_);
  Eigen::Vector3f scan_center = ComputePointCloudCenter(scan_);

  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);

    // 绘制地图点云（灰白色）
    glColor3f(0.8, 0.8, 0.8);
    glPointSize(2);
    glBegin(GL_POINTS);
    for (const auto& point : map_->points) {
      glVertex3f(point.x, point.y, point.z);
    }
    glEnd();

    // 绘制扫描点云（青色），应用变换
    glPushMatrix();
    glMultMatrixf(scan_transform.data());

    glColor3f(0.0, 1.0, 1.0);
    glPointSize(2);
    glBegin(GL_POINTS);
    for (const auto& point : scan_->points) {
      glVertex3f(point.x, point.y, point.z);
    }
    glEnd();

    // 绘制与 scan_ 绑定的坐标轴
    DrawAxis(scan_center);

    glPopMatrix();

    // 绘制与 map_ 绑定的坐标轴
    DrawAxis(map_center);

    pangolin::FinishFrame();  // 完成当前帧的绘制
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;

  // Initialize Google Test framework
  ::testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);

  // Run tests
  return RUN_ALL_TESTS();
}
