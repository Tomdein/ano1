#pragma once

// Histogram of Oriented Gradients (HOG)
//
//    http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
//    http://mrl.cs.vsb.cz/people/gaura/ano/hog.pdf

#include <opencv2/opencv.hpp>

namespace ano
{
    cv::Mat HoG(const cv::Mat &img, int block_size, int cell_size, int nbins = 9);
}