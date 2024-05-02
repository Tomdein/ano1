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
        assert(start_x >= 0 && start_y >= 0);
        assert(start_x + w <= img.size[1] && start_y + h <= img.size[0]);

        cv::Mat gradients(h, w, CV_32FC2);

        // Calculate all color differences between neighbours (excluding last row and col)
        for (int y = 0; y < h - 1; y++)
        {
            for (int x = 0; x < w - 1; x++)
            {
                auto dX = XDiffNeighbourColor(img, start_x + x, start_y + y);
                auto dY = YDiffNeighbourColor(img, start_x + x, start_y + y);

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
