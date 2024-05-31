#include <gtest/gtest.h>
#include "common/data/point_cloud.h"
#include <vector>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using namespace Common::Data;

// 测试默认构造函数
TEST(PointCloudTest, DefaultConstructor) {
    PointCloud pc;
    EXPECT_TRUE(pc.empty());
    EXPECT_EQ(pc.size(), 0);
}

// 测试插入点
TEST(PointCloudTest, EmplaceBack) {
    PointCloud pc;
    pc.emplace_back(1.0f, 2.0f, 3.0f);
    EXPECT_FALSE(pc.empty());
    EXPECT_EQ(pc.size(), 1);
    auto& point = pc.at(0);
    EXPECT_EQ(point.x(), 1.0f);
    EXPECT_EQ(point.y(), 2.0f);
    EXPECT_EQ(point.z(), 3.0f);
}

// 测试从Eigen::Vector3d向量构造
TEST(PointCloudTest, EigenConstructor) {
    std::vector<Eigen::Vector3d> eigen_points = {
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(4.0, 5.0, 6.0)
    };
    PointCloud pc(eigen_points);
    EXPECT_EQ(pc.size(), 2);
    EXPECT_EQ(pc.at(0).x(), 1.0f);
    EXPECT_EQ(pc.at(0).y(), 2.0f);
    EXPECT_EQ(pc.at(0).z(), 3.0f);
    EXPECT_EQ(pc.at(1).x(), 4.0f);
    EXPECT_EQ(pc.at(1).y(), 5.0f);
    EXPECT_EQ(pc.at(1).z(), 6.0f);
}

// 测试从PCL点云构造
TEST(PointCloudTest, PCLConstructor) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_points(new pcl::PointCloud<pcl::PointXYZ>());
    pcl_points->points.push_back(pcl::PointXYZ(1.0f, 2.0f, 3.0f));
    pcl_points->points.push_back(pcl::PointXYZ(4.0f, 5.0f, 6.0f));
    
    PointCloud pc;
    pc.GetPointsFromPCL<pcl::PointXYZ>(pcl_points);

    EXPECT_EQ(pc.size(), 2);
    EXPECT_EQ(pc.at(0).x(), 1.0f);
    EXPECT_EQ(pc.at(0).y(), 2.0f);
    EXPECT_EQ(pc.at(0).z(), 3.0f);
    EXPECT_EQ(pc.at(1).x(), 4.0f);
    EXPECT_EQ(pc.at(1).y(), 5.0f);
    EXPECT_EQ(pc.at(1).z(), 6.0f);
}

// 测试reserve函数
TEST(PointCloudTest, Reserve) {
    PointCloud pc;
    pc.reserve(10);
    EXPECT_TRUE(pc.empty());
    pc.emplace_back(1.0f, 2.0f, 3.0f);
    EXPECT_EQ(pc.size(), 1);
}

// 测试clear函数
TEST(PointCloudTest, Clear) {
    PointCloud pc;
    pc.emplace_back(1.0f, 2.0f, 3.0f);
    EXPECT_FALSE(pc.empty());
    pc.clear();
    EXPECT_TRUE(pc.empty());
    EXPECT_EQ(pc.size(), 0);
}

// 测试at函数的边界检查
TEST(PointCloudTest, AtFunction) {
    PointCloud pc;
    pc.emplace_back(1.0f, 2.0f, 3.0f);
    EXPECT_NO_THROW(pc.at(0));
    EXPECT_THROW(pc.at(1), std::out_of_range);
    const PointCloud& const_pc = pc;
    EXPECT_NO_THROW(const_pc.at(0));
    EXPECT_THROW(const_pc.at(1), std::out_of_range);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
