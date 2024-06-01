#include <thread>

#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "pcl/io/pcd_io.h"

#include "glog/logging.h"
#include "pangolin/display/display.h"
#include "pangolin/display/view.h"
#include "pangolin/gl/gldraw.h"
#include "pangolin/handler/handler.h"

DEFINE_string(scan_pcd_path, "../bin/data/lidar/scan.pcd", "scan点云路径");
DEFINE_string(map_pcd_path, "../bin/data/lidar/map.pcd", "地图点云路径");

class PangolinTest : public testing::Test {
 public:
  void SetUp() override {
    scan_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::io::loadPCDFile<pcl::PointXYZ>(FLAGS_scan_pcd_path, *scan_);
    map_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::io::loadPCDFile<pcl::PointXYZ>(FLAGS_map_pcd_path, *map_);
  }
  pcl::PointCloud<pcl::PointXYZ>::Ptr scan_ = nullptr;
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_ = nullptr;
  bool enable_test_ = true;
};

TEST_F(PangolinTest, GuiCreatTest) {
  if (enable_test_) {
    return;
  }
  // 创建一个窗口
  pangolin::CreateWindowAndBind("GuiCreatTest", 640, 480);
  // 是 OpenGL 中的一条命令，用于启用深度测试。
  // 深度测试的作用是确保在渲染3D场景时，能够正确处理物体之间的遮挡关系，使得靠近摄像机的物体能够遮挡住在其后方的物体。
  glEnable(GL_DEPTH_TEST);

  // 相机投影矩阵和相机视图矩阵
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
      pangolin::ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin::AxisY));

  // Create Interactive View in window
  pangolin::Handler3D handler(s_cam);
  pangolin::View& d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                              .SetHandler(&handler);

  while (!pangolin::ShouldQuit()) {
    // Clear screen and activate view to render into
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);

    // Render OpenGL Cube
    pangolin::glDrawColouredCube();

    // Swap frames and Process Events
    pangolin::FinishFrame();
  }
}

TEST_F(PangolinTest, MultiThreadGuiCreatTest) {
  if (enable_test_) {
    return;
  }

  // setup()
  {
    // create a window and bind its context to the main thread
    pangolin::CreateWindowAndBind("MultiThreadGuiCreatTest", 640, 480);
    // enable depth
    glEnable(GL_DEPTH_TEST);
    // unset the current context from the main thread
    pangolin::GetBoundWindow()->RemoveCurrent();
  }

  // use the context in a separate rendering thread
  std::thread render_loop;
  render_loop = std::thread([]() -> void {
    // fetch the context and bind it to this thread
    pangolin::BindToContext("MultiThreadGuiCreatTest");

    // we manually need to restore the properties of the context
    glEnable(GL_DEPTH_TEST);

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
        pangolin::ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin::AxisY));

    // Create Interactive View in window
    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                                .SetHandler(&handler);

    while (!pangolin::ShouldQuit()) {
      // Clear screen and activate view to render into
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      d_cam.Activate(s_cam);

      // Render OpenGL Cube
      pangolin::glDrawColouredCube();

      // Swap frames and Process Events
      pangolin::FinishFrame();
    }

    // unset the current context from the main thread
    pangolin::GetBoundWindow()->RemoveCurrent();
  });
  render_loop.join();
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
