#include "image-gradient.hpp"

#include <cmath>

namespace ano
{
    inline cv::Mat ComputeGradients(const cv::Mat &img)
    {
        return ComputeGradients(img, 0, 0, img.size[1], img.size[0]);
    }

    cv::Mat ComputeGradients(const cv::Mat &img, int start_x, int start_y, int w, int h)
    {
        cv::Mat gradients(h, w, CV_32FC2);

        // Calculate all color differences between neighbours (excluding last row and col)
        for (int y = start_y; y < h - 2 + start_y; y++)
        {
            for (int x = start_x; x < w - 2 + start_x; x++)
            {
                auto dX = XDiffNeighbourColor(img, x, y);
                auto dY = YDiffNeighbourColor(img, x, y);

                // Calculate gradient magnitude and orientation
                auto orientation = std::atan2(dY, dX);
                auto magnitude = std::sqrt(dX * dX + dY * dY);

                gradients.at<cv::Vec2f>(y, x) = cv::Vec2f(orientation, magnitude);
            }
        }

        // If the part we are calculating is containing right edge
        if (start_x + w == img.size[1])
        {
            // Duplicate values on the right edge
            for (int y = 0; y < h; y++)
            {
                gradients.at<cv::Vec2f>(y, w - 1) = gradients.at<cv::Vec2f>(y, w - 2);
            }
        }

        // If the part we are calculating is containing right edge
        if (start_y + h == img.size[0])
        {
            // Duplicate values on the bottom edge
            for (int x = 0; x < w; x++)
            {
                gradients.at<cv::Vec2f>(h - 1, x) = gradients.at<cv::Vec2f>(h - 2, x);
            }
        }

        return gradients;
    }
}
