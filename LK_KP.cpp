#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

// image files path
/**
 * Note: When using Clion to compile, comment these two lines and uncomment the next two lines.
 */
// std::string image_file1 = "./data/LK/LK1.png";
// std::string image_file2 = "./data/LK/LK2.png";
std::string image_file1 = "../data/LK/LK1.png";
std::string image_file2 = "../data/LK/LK2.png";

class OpticalFlowTracker {
public:
    OpticalFlowTracker(
            const cv::Mat &img1_,
            const cv::Mat &img2_,
            const std::vector<cv::KeyPoint> &kp1_,
            std::vector<cv::KeyPoint> &kp2_,
            std::vector<bool> &success_,
            bool inverse_ = true,
            bool has_initial_ = false) :
            img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_),
            has_initial(has_initial_) {}

    void calculateOpticalFlow(const cv::Range &range);

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const std::vector<cv::KeyPoint> &kp1;
    std::vector<cv::KeyPoint> &kp2;
    std::vector<bool> &success;
    bool inverse = true;
    bool has_initial = false;
};

