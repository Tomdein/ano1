#pragma once

#include <opencv2/opencv.hpp>

namespace ano
{
#define SLIC_CENTROID_COLOR (cv::Vec3b(0, 0, 255))
#define SLIC_CENTROID_STARTING_COLOR (cv::Vec3b(255, 0, 0))

    cv::Mat SLIC(const cv::Mat &img, int k_segments, float m_balance = 1.0f, int max_iterations = 1000, bool debug_view = false);
}
