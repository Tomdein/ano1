#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>
// #include <opencv2/imgcodecs.hpp>

#include "include/floodfill.hpp"
#include "include/color-generator.hpp"

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

    for (int y = 0; y < image_threshold.size[0]; y++)
    {
        for (int x = 0; x < image_threshold.size[1]; x++)
        {
            if (image_threshold.at<unsigned char>(y, x) == 255)
            {
                color = ano::GenerateRandomColorBGR(color_white);
                ano::FloodFillInPlace(image_threshold, image_indexing, x, y, color);
            }
        }
    }
    cv::namedWindow("Indexing", cv::WINDOW_NORMAL || cv::WINDOW_KEEPRATIO);
    cv::imshow("Indexing", image_indexing);

    cv::namedWindow("Thresholding", cv::WINDOW_AUTOSIZE);
    cv::imshow("Thresholding", image_threshold);
    /* ============== Indexing ============== */

    cv::waitKey(0);

    return 0;
}
