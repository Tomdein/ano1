#pragma once

// Histogram of Oriented Gradients (HOG)
//
//    http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
//    http://mrl.cs.vsb.cz/people/gaura/ano/hog.pdf

#include <opencv2/opencv.hpp>

namespace ano
{
    cv::Mat HoG(const cv::Mat &img, int block_size, int cell_size, int nbins = 9);
    cv::Mat HoGVisualizeByAlpha(const cv::Mat &hog, const int cell_size, const int nbins, const float color_multiply = 2.0f);
    cv::Mat HoGVisualizeByLenght(const cv::Mat &hog, const int cell_size, const int nbins, const float length_multiply = 16.0f);
}