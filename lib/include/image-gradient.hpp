#pragma once

#include <opencv2/opencv.hpp>

namespace ano
{
    // Calculate gradients of part of the image. Both magnitude and orintation (angle -> 1 variable)
    cv::Mat ComputeGradients(const cv::Mat &img, int x, int y, int w, int h);

    // Calculate gradients of the whole image. Both magnitude and orintation (angle -> 1 variable)
    inline cv::Mat ComputeGradients(const cv::Mat &img)
    {
        return ComputeGradients(img, 0, 0, img.size[1], img.size[0]);
    }

    // Calculates color difference between 2 colors just like euclidian distance
    inline float DiffColor(const cv::Vec3b &color1, const cv::Vec3b &color2)
    {
        int color1_b = color1[0], color1_g = color1[1], color1_r = color1[2];
        int color2_b = color2[0], color2_g = color2[1], color2_r = color2[2];

        // diff = B_c1 - B_c2, G_c1 - G_c2, R_c1 - R_c2
        auto diff_b = (color1_b - color2_b);
        auto diff_g = (color1_g - color2_g);
        auto diff_r = (color1_r - color2_r);

        // sqrt((B_c1 - B_c2)^2, (G_c1 - G_c2)^2, (R_c1 - R_c2)^2)
        return std::sqrt(diff_b * diff_b + diff_g * diff_g + diff_r * diff_r);
    }

    // Calculates color difference in +x dir
    inline float XDiffNeighbourColor(const cv::Mat &img, int x, int y)
    {
        return DiffColor(img.at<cv::Vec3b>(y, x + 1), img.at<cv::Vec3b>(y, x));
    }

    // Calculates color difference in +y dir
    inline float YDiffNeighbourColor(const cv::Mat &img, int x, int y)
    {
        return DiffColor(img.at<cv::Vec3b>(y + 1, x), img.at<cv::Vec3b>(y, x));
    }

    // Calculates euclidian distance between 2 points
    inline float EuclidianDistance(int x1, int x2, int y1, int y2)
    {
        auto dx = (x1 - x2);
        auto dy = (y1 - y2);
        return sqrt(dx * dx + dy * dy);
    }
}