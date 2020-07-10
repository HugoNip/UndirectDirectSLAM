#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>

typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

// Camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline
double baseline = 0.573;

// data paths
std::string left_file = "../data/left.png";
std::string disparity_file = "../disparity.png";
boost::format fmt_others("../image_direct/%06d.png");    // other files

