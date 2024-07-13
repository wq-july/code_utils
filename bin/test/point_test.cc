#include "common/point.h"

#include "gtest/gtest.h"

using namespace Common;

// Test fixture for PointXYZ struct
class PointXYZTest : public ::testing::Test {
 protected:
  PointXYZTest() {}
  virtual ~PointXYZTest() {}
  virtual void SetUp() {}
  virtual void TearDown() {}
};

// Test fixture for PointXYZI struct
class PointXYZITest : public ::testing::Test {
 protected:
  PointXYZITest() {}
  virtual ~PointXYZITest() {}
  virtual void SetUp() {}
  virtual void TearDown() {}
};

// Test constructors and member functions of PointXYZ
TEST_F(PointXYZTest, TestConstructorAndMemberFunctions) {
  PointXYZ point1(1.0, 2.0, 3.0);
  EXPECT_DOUBLE_EQ(point1.x, 1.0);
  EXPECT_DOUBLE_EQ(point1.y, 2.0);
  EXPECT_DOUBLE_EQ(point1.z, 3.0);

  Eigen::Vector3d vec(4.0, 5.0, 6.0);
  PointXYZ point2(vec);
  EXPECT_DOUBLE_EQ(point2.x, 4.0);
  EXPECT_DOUBLE_EQ(point2.y, 5.0);
  EXPECT_DOUBLE_EQ(point2.z, 6.0);

  PointXYZ point3 = point1;
  EXPECT_DOUBLE_EQ(point3.x, 1.0);
  EXPECT_DOUBLE_EQ(point3.y, 2.0);
  EXPECT_DOUBLE_EQ(point3.z, 3.0);

  Eigen::Vector3d vec2 = point2.ToVector3d();
  EXPECT_DOUBLE_EQ(vec2[0], 4.0);
  EXPECT_DOUBLE_EQ(vec2[1], 5.0);
  EXPECT_DOUBLE_EQ(vec2[2], 6.0);
}

// Test constructors and member functions of PointXYZI
TEST_F(PointXYZITest, TestConstructorAndMemberFunctions) {
  PointXYZI point1(1.0, 2.0, 3.0, 4.0);
  EXPECT_DOUBLE_EQ(point1.x, 1.0);
  EXPECT_DOUBLE_EQ(point1.y, 2.0);
  EXPECT_DOUBLE_EQ(point1.z, 3.0);
  EXPECT_FLOAT_EQ(point1.intensity, 4.0);

  Eigen::Vector3d vec(4.0, 5.0, 6.0);
  PointXYZI point2(vec, 7.0);
  EXPECT_DOUBLE_EQ(point2.x, 4.0);
  EXPECT_DOUBLE_EQ(point2.y, 5.0);
  EXPECT_DOUBLE_EQ(point2.z, 6.0);
  EXPECT_FLOAT_EQ(point2.intensity, 7.0);

  PointXYZI point3 = point1;
  EXPECT_DOUBLE_EQ(point3.x, 1.0);
  EXPECT_DOUBLE_EQ(point3.y, 2.0);
  EXPECT_DOUBLE_EQ(point3.z, 3.0);
  EXPECT_FLOAT_EQ(point3.intensity, 4.0);
}

// Test operator overloading of PointXYZ
TEST_F(PointXYZTest, TestOperatorOverloading) {
  PointXYZ point1(1.0, 2.0, 3.0);
  PointXYZ point2(4.0, 5.0, 6.0);
  PointXYZ point3 = point1 + point2;
  EXPECT_DOUBLE_EQ(point3.x, 5.0);
  EXPECT_DOUBLE_EQ(point3.y, 7.0);
  EXPECT_DOUBLE_EQ(point3.z, 9.0);

  point3 += point2;
  EXPECT_DOUBLE_EQ(point3.x, 9.0);
  EXPECT_DOUBLE_EQ(point3.y, 12.0);
  EXPECT_DOUBLE_EQ(point3.z, 15.0);

  point3 = point1 - point2;
  EXPECT_DOUBLE_EQ(point3.x, -3.0);
  EXPECT_DOUBLE_EQ(point3.y, -3.0);
  EXPECT_DOUBLE_EQ(point3.z, -3.0);

  point3 -= point2;
  EXPECT_DOUBLE_EQ(point3.x, -7.0);
  EXPECT_DOUBLE_EQ(point3.y, -8.0);
  EXPECT_DOUBLE_EQ(point3.z, -9.0);

  point3 = point1 * 2.0;
  EXPECT_DOUBLE_EQ(point3.x, 2.0);
  EXPECT_DOUBLE_EQ(point3.y, 4.0);
  EXPECT_DOUBLE_EQ(point3.z, 6.0);

  point3 = point1 / 2.0;
  EXPECT_DOUBLE_EQ(point3.x, 0.5);
  EXPECT_DOUBLE_EQ(point3.y, 1.0);
  EXPECT_DOUBLE_EQ(point3.z, 1.5);
}

// Test operator overloading of PointXYZI
TEST_F(PointXYZITest, TestOperatorOverloading) {
  PointXYZI point1(1.0, 2.0, 3.0, 4.0);
  PointXYZI point2(4.0, 5.0, 6.0, 7.0);
  PointXYZI point3 = point1 + point2;
  EXPECT_DOUBLE_EQ(point3.x, 5.0);
  EXPECT_DOUBLE_EQ(point3.y, 7.0);
  EXPECT_DOUBLE_EQ(point3.z, 9.0);
  EXPECT_FLOAT_EQ(point3.intensity, 0.0);

  point3 += point2;
  EXPECT_DOUBLE_EQ(point3.x, 9.0);
  EXPECT_DOUBLE_EQ(point3.y, 12.0);
  EXPECT_DOUBLE_EQ(point3.z, 15.0);
  EXPECT_FLOAT_EQ(point3.intensity, 0.0);

  point3 = point1 - point2;
  EXPECT_DOUBLE_EQ(point3.x, -3.0);
  EXPECT_DOUBLE_EQ(point3.y, -3.0);
  EXPECT_DOUBLE_EQ(point3.z, -3.0);
  EXPECT_FLOAT_EQ(point3.intensity, 0.0);

  point3 -= point2;
  EXPECT_DOUBLE_EQ(point3.x, -7.0);
  EXPECT_DOUBLE_EQ(point3.y, -8.0);
  EXPECT_DOUBLE_EQ(point3.z, -9.0);
  EXPECT_FLOAT_EQ(point3.intensity, 0.0);

  point3 = point1 * 2.0;
  EXPECT_DOUBLE_EQ(point3.x, 2.0);
  EXPECT_DOUBLE_EQ(point3.y, 4.0);
  EXPECT_DOUBLE_EQ(point3.z, 6.0);
  EXPECT_FLOAT_EQ(point3.intensity, 4.0);

  point3 = point1 / 2.0;
  EXPECT_DOUBLE_EQ(point3.x, 0.5);
  EXPECT_DOUBLE_EQ(point3.y, 1.0);
  EXPECT_DOUBLE_EQ(point3.z, 1.5);
  EXPECT_FLOAT_EQ(point3.intensity, 4.0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
