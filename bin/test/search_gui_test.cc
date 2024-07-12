#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <random>
#include <thread>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "pangolin/display/display.h"
#include "pangolin/display/view.h"
#include "pangolin/gl/gldraw.h"
#include "pangolin/handler/handler.h"
#include "pangolin/pangolin.h"
#include "pangolin/var/var.h"
#include "pcl/filters/approximate_voxel_grid.h"
#include "pcl/io/pcd_io.h"
#include "pcl/point_types.h"
#include "pcl/registration/icp.h"

#include "common/kdtree.h"
#include "common/voxel_map.h"

typedef pcl::PointCloud<pcl::PointXYZ> PCLPointCloud;

DEFINE_string(scan_pcd_path, "../bin/data/lidar/scan.pcd", "scan点云路径");
DEFINE_string(map_pcd_path, "../bin/data/lidar/map.pcd", "地图点云路径");
DEFINE_double(ann_alpha, 1.0, "AAN的比例因子");

class SearchGuiTest : public testing::Test {
 public:
  Common::PointCloudPtr scan_ = nullptr;
  Common::PointCloudPtr map_ = nullptr;
  PCLPointCloud::Ptr pcl_scan_ = nullptr;
  PCLPointCloud::Ptr pcl_map_ = nullptr;
  std::shared_ptr<Common::KdTree> kdtree_ = nullptr;
  std::shared_ptr<Common::VoxelMap> voxel_map_ = nullptr;
  std::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZ>> pcl_kdtree_ = nullptr;

  // 随机生成10个点吧，然后空格键单步执行搜索任务
  std::vector<Eigen::Vector3d> random_points_;

  Utils::Timer timer_;
  bool enable_test_ = true;

 public:
  void SetUp() override {
    scan_ = std::make_shared<Common::PointCloud>();
    map_ = std::make_shared<Common::PointCloud>();
    scan_->LoadPCDFile<pcl::PointXYZ>(FLAGS_scan_pcd_path);
    map_->LoadPCDFile<pcl::PointXYZ>(FLAGS_map_pcd_path);
    pcl_scan_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pcl_map_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::io::loadPCDFile<pcl::PointXYZ>(FLAGS_scan_pcd_path, *pcl_scan_);
    pcl::io::loadPCDFile<pcl::PointXYZ>(FLAGS_map_pcd_path, *pcl_map_);
    kdtree_ = std::make_shared<Common::KdTree>();
    voxel_map_ = std::make_shared<Common::VoxelMap>(1.0, 100.0, 100);
    pcl_kdtree_ = std::make_shared<pcl::KdTreeFLANN<pcl::PointXYZ>>();

    // int32_t scan_size = scan_->size();
    timer_.StartTimer("Voxel Map Load map");
    voxel_map_->AddPoints(*scan_);
    timer_.StopTimer();
    timer_.PrintElapsedTime();

    timer_.StartTimer("Build KdTree");
    kdtree_->BuildTree(scan_);
    kdtree_->SetEnableANN(false);
    timer_.StopTimer();
    timer_.PrintElapsedTime();
    // LOG(INFO) << "Kd tree leaves: " << kdtree_->Size() << ", points: " << scan_size
    //           << ", time / size: " << timer_.GetElapsedTime(Utils::Timer::Microseconds) /
    //           scan_size;

    pcl_kdtree_->setInputCloud(pcl_scan_);

    // 从 scan_ 中随机选择10个点
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, scan_->size() - 1);
    for (int i = 0; i < 10; ++i) {
      int idx = dis(gen);
      random_points_.emplace_back(scan_->points().at(idx));
    }
  }

  enum class SearchMethod {
    KDTREE,
    PCLKDTREE,
    VOXELMAP,
  };

  void PerformSearch(const SearchMethod& method,
                     const Eigen::Vector3d& cur_point,
                     std::vector<Eigen::Vector3d>* const searched_points) {
    searched_points->clear();
    int32_t k_nums = 10;  // 最近邻的数量
    std::vector<int32_t> point_index(k_nums);
    std::vector<float> point_distance_square(k_nums);
    std::vector<std::pair<uint32_t, double>> cloest_index;
    std::vector<std::pair<Eigen::Vector3d, double>> voxel_res;
    switch (method) {
      case SearchMethod::KDTREE: {
        kdtree_->GetClosestPoint(cur_point, &cloest_index, k_nums);
        for (const auto& index_dist_pair : cloest_index) {
          uint32_t index = index_dist_pair.first;
          if (index < scan_->points().size()) {
            searched_points->emplace_back(scan_->points().at(index));
          }
        }
        break;
      }
      case SearchMethod::PCLKDTREE: {
        pcl::PointXYZ pt;
        pt.x = cur_point.x();
        pt.y = cur_point.y();
        pt.z = cur_point.z();
        pcl_kdtree_->nearestKSearch(pt, k_nums, point_index, point_distance_square);
        for (const auto& index : point_index) {
          Eigen::Vector3d pt(
              pcl_scan_->points[index].x, pcl_scan_->points[index].y, pcl_scan_->points[index].z);
          searched_points->emplace_back(pt);
        }
        break;
      }
      case SearchMethod::VOXELMAP: {
        voxel_map_->GetClosestNeighbor(cur_point, &voxel_res, k_nums);
        for (const auto& index : voxel_res) {
          searched_points->emplace_back(index.first);
        }
        break;
      }
    }
  }

  void DrawAxis(const Eigen::Vector3f& origin) {
    const float axis_length = 5.0f;
    const float axis_line_width = 5.0f;

    glLineWidth(axis_line_width);
    glBegin(GL_LINES);

    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(origin.x(), origin.y(), origin.z());
    glVertex3f(origin.x() + axis_length, origin.y(), origin.z());

    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(origin.x(), origin.y(), origin.z());
    glVertex3f(origin.x(), origin.y() + axis_length, origin.z());

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
};

TEST_F(SearchGuiTest, SearchTest) {
  float window_width = 1280.0f;
  float window_height = 720.0f;
  pangolin::CreateWindowAndBind("PointCloud Viewer", window_width, window_height);
  glEnable(GL_DEPTH_TEST);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(
          window_width, window_height, 420, 420, window_width / 2, window_height / 2, 0.2, 1000),
      pangolin::ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin::AxisZ));

  pangolin::View& d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, 0.0, 1.0, -window_width / window_height)
                              .SetHandler(new pangolin::Handler3D(s_cam));

  pangolin::Var<float> map_r("ui.Map R", 0.8f, 0.0f, 1.0f);
  pangolin::Var<float> map_g("ui.Map G", 0.8f, 0.0f, 1.0f);
  pangolin::Var<float> map_b("ui.Map B", 0.8f, 0.0f, 1.0f);

  pangolin::Var<float> random_r("ui.Random R", 0.0f, 0.0f, 1.0f);
  pangolin::Var<float> random_g("ui.Random G", 1.0f, 0.0f, 1.0f);
  pangolin::Var<float> random_b("ui.Random B", 0.0f, 0.0f, 1.0f);

  Eigen::Vector3f scan_center = ComputePointCloudCenter(pcl_scan_);

  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);

    // 绘制地图点云
    glPushMatrix();
    glColor3f(map_r, map_g, map_b);
    glPointSize(2);
    glBegin(GL_POINTS);
    for (const auto& point : scan_->points()) {
      glVertex3f(point.x(), point.y(), point.z());
    }
    glEnd();

    // 绘制随机点
    glPushMatrix();
    glPointSize(8);
    glBegin(GL_POINTS);
    for (const auto& point : random_points_) {
      glColor3f(0.0f, 1.0f, 0.0f);
      glVertex3f(point.x(), point.y(), point.z());
      // 在这里调用函数搜索最近点
      std::vector<Eigen::Vector3d> searched_points;
      PerformSearch(SearchMethod::KDTREE, point, &searched_points);
      // 绘制搜索到的近邻点
      glColor3f(1.0f, 0.0f, 0.0f);  // 设置颜色为红色
      for (const auto& search_point : searched_points) {
        glVertex3f(search_point.x(), search_point.y(), search_point.z());
      }
    }
    glEnd();

    // 绘制与 scan_ 绑定的坐标轴
    DrawAxis(scan_center);

    pangolin::FinishFrame();
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
