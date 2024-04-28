#pragma once

#include <opencv2/opencv.hpp>

namespace ano
{

    template <typename T>
    void FloodFillInPlace(cv::Mat &img, cv::Mat &img_out, const cv::Point &starting_pixel, const T &color);
    template <typename T>
    void FloodFillInPlace(cv::Mat &img, cv::Mat &img_out, int starting_x, int starting_y, const T &color);
    template <typename T>
    inline cv::Mat FloodFill(cv::Mat &img, const cv::Point &starting_pixel, const T &color);
    template <typename T>
    inline cv::Mat FloodFill(cv::Mat &img, int starting_x, int starting_y, const T &color);

}