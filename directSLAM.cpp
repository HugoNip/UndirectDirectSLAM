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

class JacobianAccumulator {
public:
    JacobianAccumulator(
            const cv::Mat &img1_,
            const cv::Mat &img2_,
            const VecVector2d &px_ref_,
            const std::vector<double> depth_ref_,
            Sophus::SE3d &T21_) :
            img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), T21(T21_) {
        projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));
    };

    // accumulate Jacobian in a range
    void accumulate_jacobian(const cv::Range &range);

    // get Hessian Matrix
    Matrix6d hessian() const { return H; }

    // get bias
    Vector6d bias() const { return b; }

    // get cost
    double cost_func() const { return cost; }

    // get projected points
    VecVector2d projected_points() const { return projection; }

    // reset H, b, cost to zero
    void reset() {
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0;
    }

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const VecVector2d &px_ref;
    const std::vector<double> depth_ref;
    Sophus::SE3d &T21;
    VecVector2d projection;

    std::mutex hessian_mutex;
    Matrix6d H = Matrix6d::Zero();
    Vector6d b = Vector6d::Zero();
    double cost = 0;
};


/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const std::vector<double> depth_ref,
        Sophus::SE3d &T21
        );


/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const std::vector<double> depth_ref,
        Sophus::SE3d &T21
);


// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}


int main(int argc, char** argv) {
    
    return 0;
}