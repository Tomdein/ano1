#include <iostream>

#include <opencv2/opencv.hpp>
// #include <opencv2/imgcodecs.hpp>

#include "include/floodfill.hpp"

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

    cv::namedWindow("Input", cv::WINDOW_NORMAL || cv::WINDOW_KEEPRATIO);
    cv::imshow("Input", image_in);

    std::cout << image_in.type() << std::endl;

    /* ============== THRESHOLDING ============== */
    unsigned char threshold = 1;

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
    cv::waitKey(0);
    /* ============== THRESHOLDING ============== */

    /* ============== Indexing ============== */
    cv::Mat image_indexing = image_threshold.clone();

    unsigned char color_index = -2; // unsinged char max - 1 (unsigned char max is for foreground)
    for (int y = 0; y < image_indexing.size[0]; y++)

    {
        for (int x = 0; x < image_indexing.size[1]; x++)
        {
            if (image_indexing.at<unsigned char>(y, x) == 255)
            {
                ano::FloodFillInPlace(image_indexing, x, y, color_index);
                color_index -= 10;
                cv::namedWindow("Indexing", cv::WINDOW_NORMAL || cv::WINDOW_KEEPRATIO);
                cv::imshow("Indexing", image_indexing);
                cv::waitKey(0);
            }
        }
    }

    // cv::namedWindow("Indexing", cv::WINDOW_AUTOSIZE);
    // cv::imshow("Indexing", image_indexing);
    /* ============== Indexing ============== */

    cv::waitKey(0);

    return 0;
}
