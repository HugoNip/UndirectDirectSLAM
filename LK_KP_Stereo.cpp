#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

// Read Images
/**
 * Note: When using Clion to compile, comment these two lines and uncomment the next two lines.
 */
// std::string image_file1 = "./data/image_2/000006_10.png";
// std::string image_file2 = "./data/image_3/000006_10.png";
std::string image_file1 = "../data/image_2/000002_10.png"; // Right
std::string image_file2 = "../data/image_3/000002_10.png"; // Left

struct myclass {
    bool operator() (int i,int j) { return (i<j);}
} myobject;

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

/**
 * Single Level Optical Flow
 * @param img1
 * @param img2
 * @param kp1
 * @param kp2
 * @param success
 * @param inverse
 * @param has_initial_guess
 */
void OpticalFlowSingleLevel(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const std::vector<cv::KeyPoint> &kp1,
        std::vector<cv::KeyPoint> &kp2,
        std::vector<bool> &success,
        bool inverse = false,
        bool has_initial_guess = false
);

/**
 * Multi Level Optical Flow, scale: 2(default)
 * @param img1
 * @param img2
 * @param kp1
 * @param kp2
 * @param success
 * @param inverse
 */
void OpticalFlowMultiLevel(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const std::vector<cv::KeyPoint> &kp1,
        std::vector<cv::KeyPoint> &kp2,
        std::vector<bool> &success,
        bool inverse = false
);

/**
 * Get a gray scale value from the reference image
 * Bi-Linear interpolated
 * @param img
 * @param x
 * @param y
 * @return interpolated value at position (x, y)
 */
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

/**
 * compute mean and standard deviation
 * @param data
 * @param mean
 * @param stD
 */
void calculateSD(std::vector<double> data, double &mean, double &stD);

int main(int argc, char** argv) {

    // Read images
    cv::Mat img1 = cv::imread(image_file1, 0);
    cv::Mat img2 = cv::imread(image_file2, 0);

    // Detect keypoints in image 1, using GFTT detector
    std::vector<cv::KeyPoint> kp1;
    cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(500, 0.01, 20);
    detector->detect(img1, kp1);
    // for (auto it = kp1.begin(); it != kp1.end(); it++)
    //     std::cout << (*it).pt << std::endl;

    // Method 2: Multi level LK
    std::vector<cv::KeyPoint> kp2_multi;
    std::vector<cv::KeyPoint> good_kp1, good_kp2;
    std::vector<bool> success_multi;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);
    std::cout << "KeyPoints detected by Multi-Level LK: " << kp2_multi.size() << std::endl;

    // Select good matches
    double thres = 0;
    std::vector<double> distXs_posi;
    for (int i = 0; i < kp2_multi.size(); i++) {
        // std::cout << "1: " << kp1[i].pt.x << ", 2: " << kp2_multi[i].pt.x << "." << std::endl;
        double distXs = kp1[i].pt.x - kp2_multi[i].pt.x;
        // std::cout << distXs << std::endl;
        if (distXs > thres)
            distXs_posi.push_back(distXs);
    }

    double distXs_mean, distXs_stD;
    calculateSD(distXs_posi, distXs_mean, distXs_stD);
    std::cout << "Mean: " << distXs_mean << std::endl;
    std::cout << "STD: " << distXs_stD << std::endl;

    // sort disparity from min to max
    std::sort (distXs_posi.begin(), distXs_posi.end(), myobject);
    std::cout << "myvector contains:";
    for (std::vector<double>::iterator it = distXs_posi.begin(); it != distXs_posi.end(); ++it)
        std::cout << ' ' << *it;
    std::cout << std::endl;

    // print minimum distance
    auto dist_minmax = std::minmax_element(distXs_posi.begin(), distXs_posi.end());
    std::cout << "Min Distance: " << *dist_minmax.first << std::endl;
    std::cout << "Max Distance: " << *dist_minmax.second << std::endl;

    for (int i = 0; i < kp2_multi.size(); i++) {
        double distX = kp1[i].pt.x - kp2_multi[i].pt.x;
        double distY = kp1[i].pt.y - kp2_multi[i].pt.y;

        if (abs(distY) < 1)
            if ((distX > 2 * (distXs_mean - distXs_stD)) && (kp2_multi[i].pt.x > 0) && (distX < (*dist_minmax.second) - 10))
            {
                kp1[i].pt.y = kp2_multi[i].pt.y;
                good_kp1.push_back(kp1[i]);
                good_kp2.push_back(kp2_multi[i]);
            }
    }
    std::cout << "Good Matches Number: " << good_kp1.size() << std::endl;
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Optical Flow by Gauss-Newton: " << time_used.count() << " seconds." << std::endl;

    // Plot results
    cv::Mat img1_multi;
    cv::cvtColor(img1, img1_multi, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < good_kp1.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img1_multi, good_kp1[i].pt, 1, cv::Scalar(0, 250, 0), 2); // green
            cv::circle(img1_multi, good_kp2[i].pt, 1, cv::Scalar(250, 0, 0), 2); // blue
            cv::line(img1_multi, good_kp1[i].pt, good_kp2[i].pt, cv::Scalar(0, 0, 250));
            /**
             * void line(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
             * Parameters:
             * img – Image.
             * pt1 – First point of the line segment.
             * pt2 – Second point of the line segment.
             * color – Line color.
             * thickness – Line thickness.
             * lineType -
             * Type of the line:
             *  8 (or omitted) - 8-connected line.
             *  4 - 4-connected line.
             *  CV_AA - antialiased line.
             * shift – Number of fractional bits in the point coordinates.
             */
        }
    }
    cv::imshow("Tracked multi level (Right, Green Circle)", img1_multi);
    cv::imwrite("../results/LK_Multi_Stereo_Right.png", img1_multi);

    cv::Mat img2_multi;
    cv::cvtColor(img2, img2_multi, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < good_kp2.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img2_multi, good_kp1[i].pt, 1, cv::Scalar(0, 250, 0), 2); // green
            cv::circle(img2_multi, good_kp2[i].pt, 1, cv::Scalar(250, 0, 0), 2); // blue
            cv::line(img2_multi, good_kp1[i].pt, good_kp2[i].pt, cv::Scalar(0, 0, 250));
        }
    }

    cv::imshow("Tracked multi level (Left, Blue Circle)", img2_multi);
    cv::imwrite("../results/LK_Multi_Stereo_Left.png", img2_multi);
    // cv::imwrite("./results/LK_Multi_Stereo.png", img2_multi);

    cv::waitKey(0);

    return 0;
}

void OpticalFlowSingleLevel(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const std::vector<cv::KeyPoint> &kp1,
        std::vector<cv::KeyPoint> &kp2,
        std::vector<bool> &success,
        bool inverse,
        bool has_initial_guess) {

    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial_guess);
    // multithreads
    cv::parallel_for_(cv::Range(0, kp1.size()),
                      std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, std::placeholders::_1));
}

void OpticalFlowMultiLevel(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const std::vector<cv::KeyPoint> &kp1,
        std::vector<cv::KeyPoint> &kp2,
        std::vector<bool> &success,
        bool inverse) {

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scale[] = {1.0, 0.5, 0.25, 0.125};

    // Create Image Pyramids
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    std::vector<cv::Mat> pyr1, pyr2;
    /**
     * in pyramid
     * i = 0, scale = 1.0
     * i = 1, scale = 0.5
     * i = 2, scale = 0.25
     * i = 3, scale = 0.125
     */
    for (int i = 0; i < pyramids; ++i) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Build pyramid time: " << time_used.count() << std::endl;

    std::vector<cv::KeyPoint> kp1_pyr, kp2_pyr;
    for (auto &kp : kp1) {
        auto kp_top = kp;
        kp_top.pt *= scale[pyramids - 1]; // x 0.125
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }

    // Coarse to fine LK tracking in Image Pyramids
    for (int level = pyramids - 1; level >= 0 ; level--) {
        // from coarse to fine
        success.clear();
        t1 = std::chrono::steady_clock::now();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);
        t2 = std::chrono::steady_clock::now();
        auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        std::cout << "Track pyramid " << level << " cost time: " << time_used.count() << std::endl;

        if (level > 0) {
            for (auto &kp : kp1_pyr)
                kp.pt /= pyramid_scale; // /0.5
            for (auto &kp : kp2_pyr)
                kp.pt /= pyramid_scale;
        }
    }

    for (auto &kp : kp2_pyr)
        kp2.push_back(kp);
}


void OpticalFlowTracker::calculateOpticalFlow(const cv::Range &range) {
    // parameters
    int half_patch_size = 4;
    int iterations = 10;
    for (size_t i = range.start; i < range.end; ++i) {
        auto kp = kp1[i];
        double dx = 0, dy = 0;
        if (has_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            // dy = kp2[i].pt.y - kp.pt.y;
            dy = 0;
        }

        double cost = 0, lastCost = 0;
        bool succ = true;

        // GN iterations
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        Eigen::Vector2d b = Eigen::Vector2d::Zero();
        Eigen::Vector2d J; // Jacobian Matrix

        for (int iter = 0; iter < iterations; ++iter) {
            if (inverse == false) {
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            } else {
                b = Eigen::Vector2d::Zero();
            }

            cost = 0;

            // Compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; ++x)
                for (int y = -half_patch_size; y < half_patch_size; ++y) {
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                   GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);

                    // Jacobian
                    if (inverse == false) {
                        J = -1.0 * Eigen::Vector2d(
                                0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                       GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                                0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                       GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
                        );
                    } else if (iter == 0) {
                        /**
                         * inverse == true, iter == 0
                         * Inverse Optical Flow
                         * Gradient of I1 is fixed for each iteration
                         * so save the results after the first iteration.
                         */
                        J = -1.0 * Eigen::Vector2d(
                                0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                       GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                                0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                       GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))
                        );
                    }
                    // Compute H, b and set cost
                    b += -error * J;
                    cost += error * error;
                    if (inverse == false || iter == 0) {
                        // also update H
                        H += J * J.transpose();
                    }
                }

            // Update H
            Eigen::Vector2d update = H.ldlt().solve(b);

            if (std::isnan(update[0])) {
                std::cout << "Update is nan" << std::endl;
                succ = false;
                break;
            }

            if (iter > 0 && cost > lastCost)
                break;

            // Update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            if (update.norm() < 1e-2)
                // Converge
                break;
        }

        success[i] = succ;

        // set kp2
        kp2[i].pt = kp.pt + cv::Point2f(dx, dy);
    }
}

void calculateSD(std::vector<double> data, double &mean, double &stD)
{
    double sum = 0.0, standardDeviation = 0.0;

    int i;

    for(i = 0; i < data.size(); ++i)
    {
        sum += data[i];
    }

    mean = sum/data.size();

    for(i = 0; i < data.size(); ++i)
        standardDeviation += pow(data[i] - mean, 2);

    stD = sqrt(standardDeviation / data.size());
}