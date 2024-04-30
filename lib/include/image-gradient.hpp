#pragma once

#include <opencv2/opencv.hpp>

namespace ano
{
    // Calculate gradients of the whole image. Both magnitude and orintation (angle -> 1 variable)
    inline cv::Mat ComputeGradients(const cv::Mat &img);
    // Calculate gradients of part of the image. Both magnitude and orintation (angle -> 1 variable)
    cv::Mat ComputeGradients(const cv::Mat &img, int x, int y, int w, int h);

    // Calculates color difference between 2 colors
    inline float DiffColor(const cv::Vec3b &color1, const cv::Vec3b &color2);
    // Calculates color difference in +x dir
    inline float XDiffNeighbourColor(const cv::Mat &img, int x, int y);
    // Calculates color difference in +y dir
    inline float YDiffNeighbourColor(const cv::Mat &img, int x, int y);
    // Calculates euclidian distance between 2 points
    inline float EuclidianDistance(int x1, int x2, int y1, int y2);
}