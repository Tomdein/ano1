#pragma once

#include <opencv2/opencv.hpp>

namespace ano
{

    void FloodFillInPlace(cv::Mat &img, cv::Point starting_pixel, unsigned char index);
    void FloodFillInPlace(cv::Mat &img, int starting_x, int starting_y, unsigned char index);
    void FloodFillInPlaceScanline(cv::Mat &img, int starting_x, int starting_y, unsigned char index);
    void FloodFillInPlaceWiki(cv::Mat &img, int starting_x, int starting_y, unsigned char index);
    inline cv::Mat FloodFill(const cv::Mat &img, cv::Point starting_pixel, unsigned char index);

}