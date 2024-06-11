#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <thread>

#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "pcl/filters/approximate_voxel_grid.h"
#include "pcl/io/pcd_io.h"
#include "pcl/point_types.h"
#include "pcl/registration/icp.h"

#include "glog/logging.h"
#include "lidar/match.h"
#include "pangolin/display/display.h"
#include "pangolin/display/view.h"
#include "pangolin/gl/gldraw.h"
#include "pangolin/handler/handler.h"
#include "pangolin/pangolin.h"
#include "pangolin/var/var.h"

typedef pcl::PointCloud<pcl::PointXYZ> PCLPointCloud;

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

  // 执行匹配的函数
  Eigen::Matrix4f MyMatcher(const PCLPointCloud::Ptr& source, const PCLPointCloud::Ptr& target) {
    PointCloudPtr pointcloud_scan_(new PointCloud);
    PointCloudPtr pointcloud_map_(new PointCloud);
    pointcloud_scan_->GetPointsFromPCL<pcl::PointXYZ>(source);
    pointcloud_map_->GetPointsFromPCL<pcl::PointXYZ>(target);
    Eigen::Isometry3d final_transform = Eigen::Isometry3d::Identity();
    Lidar::Matcher matcher(0.6,
                           100.0,
                           50,
                           Lidar::AlignMethod::NDT,
                           Lidar::SearchMethod::VOXEL_MAP,
                           50,
                           1.0e-6,
                           20.0,
                           50,
                           true);
    matcher.Align(
        Eigen::Isometry3d::Identity(), pointcloud_scan_, pointcloud_map_, &final_transform);
    return final_transform.matrix().cast<float>();
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

  Eigen::Vector3f ComputePointCloudCenter(const PCLPointCloud::Ptr& cloud) {
    Eigen::Vector3f center(0.0f, 0.0f, 0.0f);
    for (const auto& point : cloud->points) {
      center += Eigen::Vector3f(point.x, point.y, point.z);
    }
    center /= static_cast<float>(cloud->points.size());
    return center;
  }

  PCLPointCloud::Ptr scan_ = nullptr;
  PCLPointCloud::Ptr map_ = nullptr;

  bool enable_test_ = true;
};

TEST_F(ICPGUITest, ICPGUITest) {
  int window_width = 1280;
  int window_height = 720;
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
  pangolin::Var<std::function<void()>> btn_w("ui.Front", [&scan_transform, step]() {
    scan_transform(0, 3) += step;
  });
  pangolin::Var<std::function<void()>> btn_s("ui.Rear", [&scan_transform, step]() {
    scan_transform(0, 3) -= step;
  });
  pangolin::Var<std::function<void()>> btn_a("ui.Left", [&scan_transform, step]() {
    scan_transform(1, 3) += step;
  });
  pangolin::Var<std::function<void()>> btn_d("ui.Right", [&scan_transform, step]() {
    scan_transform(1, 3) -= step;
  });
  pangolin::Var<std::function<void()>> btn_q("ui.Up", [&scan_transform, step]() {
    scan_transform(2, 3) += step;
  });
  pangolin::Var<std::function<void()>> btn_e("ui.Down", [&scan_transform, step]() {
    scan_transform(2, 3) -= step;
  });

  pangolin::Var<std::function<void()>> btn_f("ui.Roll -", [&scan_transform, angle_step]() {
    Eigen::Matrix4f roll = Eigen::Matrix4f::Identity();
    roll.block<3, 3>(0, 0) =
        Eigen::AngleAxisf(-angle_step, Eigen::Vector3f::UnitX()).toRotationMatrix();
    scan_transform = scan_transform * roll;
  });

  pangolin::Var<std::function<void()>> btn_r("ui.Roll +", [&scan_transform, angle_step]() {
    Eigen::Matrix4f roll = Eigen::Matrix4f::Identity();
    roll.block<3, 3>(0, 0) =
        Eigen::AngleAxisf(angle_step, Eigen::Vector3f::UnitX()).toRotationMatrix();
    scan_transform = scan_transform * roll;
  });

  pangolin::Var<std::function<void()>> btn_l("ui.Pitch -", [&scan_transform, angle_step]() {
    Eigen::Matrix4f pitch = Eigen::Matrix4f::Identity();
    pitch.block<3, 3>(0, 0) =
        Eigen::AngleAxisf(-angle_step, Eigen::Vector3f::UnitY()).toRotationMatrix();
    scan_transform = scan_transform * pitch;
  });

  pangolin::Var<std::function<void()>> btn_p("ui.Pitch +", [&scan_transform, angle_step]() {
    Eigen::Matrix4f pitch = Eigen::Matrix4f::Identity();
    pitch.block<3, 3>(0, 0) =
        Eigen::AngleAxisf(angle_step, Eigen::Vector3f::UnitY()).toRotationMatrix();
    scan_transform = scan_transform * pitch;
  });

  pangolin::Var<std::function<void()>> btn_h("ui.yaw -", [&scan_transform, angle_step]() {
    Eigen::Matrix4f yaw = Eigen::Matrix4f::Identity();
    yaw.block<3, 3>(0, 0) =
        Eigen::AngleAxisf(-angle_step, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    scan_transform = scan_transform * yaw;
  });

  pangolin::Var<std::function<void()>> btn_y("ui.yaw +", [&scan_transform, angle_step]() {
    Eigen::Matrix4f yaw = Eigen::Matrix4f::Identity();
    yaw.block<3, 3>(0, 0) =
        Eigen::AngleAxisf(angle_step, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    scan_transform = scan_transform * yaw;
  });

  pangolin::Var<std::function<void()>> btn_space("ui.ICP", [this, &scan_transform]() {
    PCLPointCloud::Ptr transformed_scan(new PCLPointCloud);
    pcl::transformPointCloud(*this->scan_, *transformed_scan, scan_transform);
    Eigen::Matrix4f icp_transform = this->MyMatcher(transformed_scan, this->map_);
    scan_transform = icp_transform * scan_transform;
  });

  // 添加点云颜色选择器
  pangolin::Var<float> scan_r("ui.Scan R", 0.0f, 0.0f, 1.0f);
  pangolin::Var<float> scan_g("ui.Scan G", 1.0f, 0.0f, 1.0f);
  pangolin::Var<float> scan_b("ui.Scan B", 1.0f, 0.0f, 1.0f);
  pangolin::Var<float> map_r("ui.Map R", 0.8f, 0.0f, 1.0f);
  pangolin::Var<float> map_g("ui.Map G", 0.8f, 0.0f, 1.0f);
  pangolin::Var<float> map_b("ui.Map B", 0.8f, 0.0f, 1.0f);

  // 注册键盘事件回调函数
  pangolin::RegisterKeyPressCallback('w', [&scan_transform, step]() {
    scan_transform(0, 3) += step;
  });
  pangolin::RegisterKeyPressCallback('s', [&scan_transform, step]() {
    scan_transform(0, 3) -= step;
  });
  pangolin::RegisterKeyPressCallback('a', [&scan_transform, step]() {
    scan_transform(1, 3) += step;
  });
  pangolin::RegisterKeyPressCallback('d', [&scan_transform, step]() {
    scan_transform(1, 3) -= step;
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
        Eigen::AngleAxisf(angle_step, Eigen::Vector3f::UnitX()).toRotationMatrix();
    scan_transform = scan_transform * roll;
  });

  pangolin::RegisterKeyPressCallback('f', [&scan_transform, angle_step]() {
    Eigen::Matrix4f roll = Eigen::Matrix4f::Identity();
    roll.block<3, 3>(0, 0) =
        Eigen::AngleAxisf(-angle_step, Eigen::Vector3f::UnitX()).toRotationMatrix();
    scan_transform = scan_transform * roll;
  });

  pangolin::RegisterKeyPressCallback('p', [&scan_transform, angle_step]() {
    Eigen::Matrix4f pitch = Eigen::Matrix4f::Identity();
    pitch.block<3, 3>(0, 0) =
        Eigen::AngleAxisf(angle_step, Eigen::Vector3f::UnitY()).toRotationMatrix();
    scan_transform = scan_transform * pitch;
  });

  pangolin::RegisterKeyPressCallback('l', [&scan_transform, angle_step]() {
    Eigen::Matrix4f pitch = Eigen::Matrix4f::Identity();
    pitch.block<3, 3>(0, 0) =
        Eigen::AngleAxisf(-angle_step, Eigen::Vector3f::UnitY()).toRotationMatrix();
    scan_transform = scan_transform * pitch;
  });
  pangolin::RegisterKeyPressCallback('y', [&scan_transform, angle_step]() {
    Eigen::Matrix4f yaw = Eigen::Matrix4f::Identity();
    yaw.block<3, 3>(0, 0) =
        Eigen::AngleAxisf(angle_step, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    scan_transform = scan_transform * yaw;
  });
  pangolin::RegisterKeyPressCallback('h', [&scan_transform, angle_step]() {
    Eigen::Matrix4f yaw = Eigen::Matrix4f::Identity();
    yaw.block<3, 3>(0, 0) =
        Eigen::AngleAxisf(-angle_step, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    scan_transform = scan_transform * yaw;
  });

  pangolin::RegisterKeyPressCallback(' ', [this, &scan_transform]() {
    PCLPointCloud::Ptr transformed_scan(new PCLPointCloud);
    pcl::transformPointCloud(*this->scan_, *transformed_scan, scan_transform);
    Eigen::Matrix4f icp_transform = this->MyMatcher(transformed_scan, this->map_);
    scan_transform = icp_transform * scan_transform;
  });

  Eigen::Vector3f map_center = ComputePointCloudCenter(map_);
  Eigen::Vector3f scan_center = ComputePointCloudCenter(scan_);

  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);

    // 绘制地图点云（灰白色）
    glColor3f(map_r, map_g, map_b);
    glPointSize(2);
    glBegin(GL_POINTS);
    for (const auto& point : map_->points) {
      glVertex3f(point.x, point.y, point.z);
    }
    glEnd();

    // 绘制扫描点云（青色），应用变换
    glPushMatrix();
    glMultMatrixf(scan_transform.data());

    glColor3f(scan_r, scan_g, scan_b);
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
