#include "image-gradient.hpp"

#include <cmath>

namespace ano
{

    cv::Mat ComputeGradients(const cv::Mat &img, int start_x, int start_y, int w, int h)
    {
        cv::Mat img_greyscale;
        cv::cvtColor(img, img_greyscale, cv::COLOR_BGR2GRAY);

        assert(start_x >= 0 && start_y >= 0);
        assert(start_x + w <= img.size[1] && start_y + h <= img.size[0]);

        cv::Mat gradients(h, w, CV_32FC2);

        // Calculate all color differences between neighbours (excluding last row and col)
        for (int y = 0; y < h - 1; y++)
        {
            for (int x = 0; x < w - 1; x++)
            {
                auto dX = img_greyscale.at<unsigned char>(start_y + y, start_x + x + 1) - img_greyscale.at<unsigned char>(start_y + y, start_x + x);
                auto dY = img_greyscale.at<unsigned char>(start_y + y + 1, start_x + x) - img_greyscale.at<unsigned char>(start_y + y, start_x + x);

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
