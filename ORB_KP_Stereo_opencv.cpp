/*!
 * This is an ORB-based method to detect the Matching Points for stereo images.
 */

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

int main() {
    // Read Images
    /**
     * Note: uncomment it when using terminal to compile
     */
    // std::string image_file1 = "./data/image_2/000006_10.png";
    // std::string image_file2 = "./data/image_3/000006_10.png";

    // These two lines are used in Clion, when using terminal to compile, please comment them.
    std::string image_file1 = "../data/image_2/000006_10.png";
    std::string image_file2 = "../data/image_3/000006_10.png";

    cv::Mat img_1 = cv::imread(image_file1);
    cv::Mat img_2 = cv::imread(image_file2);

    // Initialization
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    // Oriented FAST Detection
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // BRIEF Descriptor
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "extract ORB cost = " << time_used.count() << " seconds." << std::endl;

    cv::Mat outimg1;
    cv::drawKeypoints(img_1, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("ORB features", outimg1);

    // Feature Matching based on Hamming Distance
    std::vector<cv::DMatch> matches;
    t1 = std::chrono::steady_clock::now();
    matcher->match(descriptors_1, descriptors_2, matches);
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Match ORB cost = " << time_used.count() << " seconds." << std::endl;

    // Select Good Matching
    // Calculate Minimum and Maximum Distance
    auto min_max = minmax_element(matches.begin(), matches.end(),
            [](const cv::DMatch &m1, const cv::DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Max Distance: %f \n", max_dist);
    printf("-- Min distance: %f \n", min_dist);

    // < 2 x minimized distance
    // same y coordinate
    // minimized distance > 60 (30 is the experience value)
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; ++i) {
        if (keypoints_1[matches[i].queryIdx].pt.y == keypoints_2[matches[i].trainIdx].pt.y)
            if (matches[i].distance <= cv::max(2 * min_dist, 50.0))
                good_matches.push_back(matches[i]);
    }

    // Print the Matched KeyPoints
    printf("\nThere are %d KeyPoints.\n", good_matches.size());
    std::cout << "Print the coordiates of matched KeyPoints:" << std::endl;

    for (int i = 0; i < (int) good_matches.size(); i++){

        std::cout << "KeyPoint in image 1: " << keypoints_1[good_matches[i].queryIdx].pt << ", "
                  << "KeyPoint in image 2: " << keypoints_2[good_matches[i].trainIdx].pt << std::endl;
    }

    // Draw Matching Results
    cv::Mat img_match;
    cv::Mat img_good_match;
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_good_match);
    cv::imshow("All Matches", img_match);
    cv::imshow("Good Matches", img_good_match);
    cv::waitKey(0);

    return 0;
}
