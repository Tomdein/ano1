#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>

#include "floodfill.hpp"
#include "color-generator.hpp"
#include "moments.hpp"

#define TEST_IMG_PATH "../../img/train.png"

int main(int argc, char **argv)
{
    cv::Mat image_in = cv::imread(TEST_IMG_PATH, cv::IMREAD_GRAYSCALE);

    if (!image_in.data)
    {
        printf("No image data \n");
        return -1;
    }

    std::cout << "Size: " << image_in.size[0] << "," << image_in.size[1] << std::endl;

    cv::namedWindow("Input", cv::WINDOW_AUTOSIZE);
    cv::imshow("Input", image_in);

    /* ============== THRESHOLDING ============== */
    unsigned char threshold = 50;

    cv::Mat image_threshold(image_in.size(), CV_8UC1);
    for (int y = 0; y < image_threshold.size[0]; y++)
    {
        for (int x = 0; x < image_threshold.size[1]; x++)
        {
            image_threshold.at<unsigned char>(y, x) = (image_in.at<unsigned char>(y, x) > threshold) ? 255 : 0;
        }
    }
    cv::namedWindow("Thresholding", cv::WINDOW_AUTOSIZE);
    cv::imshow("Thresholding", image_threshold);
    /* ============== THRESHOLDING ============== */

    /* ============== Indexing ============== */
    cv::Vec3b color_white(255, 255, 255);

    auto color = ano::GenerateRandomColorBGR(color_white);
    cv::Mat image_indexing = ano::FloodFill(image_threshold, 0, 0, color);

    unsigned char object_index = -2; // char max - 1 (as 255 is reserved for foreground)
    for (int y = 0; y < image_threshold.size[0]; y++)
    {
        for (int x = 0; x < image_threshold.size[1]; x++)
        {
            if (image_threshold.at<unsigned char>(y, x) == 255)
            {
                color = ano::GenerateRandomColorBGR(color_white);
                ano::FloodFillInPlace(image_threshold, image_indexing, x, y, color, object_index);
                object_index--;
            }
        }
    }
    cv::namedWindow("Indexing", cv::WINDOW_NORMAL || cv::WINDOW_KEEPRATIO);
    cv::imshow("Indexing", image_indexing);

    cv::namedWindow("Thresholding", cv::WINDOW_AUTOSIZE);
    cv::imshow("Thresholding", image_threshold);
    /* ============== Indexing ============== */

    /* ============== Moments ============== */
    // Iterate through all indexed objects
    for (unsigned char obj_index = object_index + 1; obj_index < 255; obj_index++)
    {
        auto center_of_mass = ano::CenterOfMass(image_threshold, obj_index);
        float area = ano::Area(image_threshold, obj_index);
        auto F1 = std::pow(ano::Circumference(image_threshold, obj_index), 2) / (100 * area);

        auto u20 = ano::CenteredMoment(image_threshold, 2, 0, obj_index);
        auto u02 = ano::CenteredMoment(image_threshold, 0, 2, obj_index);
        auto u11 = ano::CenteredMoment(image_threshold, 1, 1, obj_index);
        float umax = 0.5f * (u20 + u02) + 0.5f * std::sqrt(4 * std::pow(u11, 2) + std::pow(u20 - u02, 2));
        float umin = 0.5f * (u20 + u02) - 0.5f * std::sqrt(4 * std::pow(u11, 2) + std::pow(u20 - u02, 2));
        auto F2 = umin / umax;

        std::cout << "Object index: " << std::to_string(obj_index) << "\n"
                  << "\tCenter of mass: " << center_of_mass << "\n"
                  << "\tArea: " << area << "\n"
                  << "\tF1: " << F1 << "\n"
                  << "\tF2: " << F2 << "\n\n"
                  << std::endl;
    }
    /* ============== Moments ============== */

    cv::waitKey(0);

    return 0;
}
