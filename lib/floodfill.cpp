#include <iostream>
#include <tuple>
#include <queue>

#include "include/floodfill.hpp"

namespace ano
{

    // combined-scan-and-fill span filler
    template <typename T>
    void FloodFillInPlace(cv::Mat &img, cv::Mat &img_out, const cv::Point &starting_pixel, const T &color, const unsigned char index)
    {
        FloodFillInPlace(img, img_out, starting_pixel.x, starting_pixel.y, color, index);
    }

    // combined-scan-and-fill span filler
    template <typename T>
    void FloodFillInPlace(cv::Mat &img, cv::Mat &img_out, int starting_x, int starting_y, const T &color, const unsigned char index)
    {
        auto xmax = img.size[1];
        decltype(xmax) xmin = 0;
        auto ymax = img.size[0];
        decltype(ymax) ymin = 0;

        // Similar to: https://en.wikipedia.org/wiki/Flood_fill#:~:text=The%20final%2C%20combined%2Dscan%2Dand%2Dfill%20span%20filler%20was%20then%20published%20in%201990.%20In%20pseudo%2Dcode%20form
        if (img.at<unsigned char>(starting_y, starting_x) != 255)
        {
            std::cout << "FloodFill at [x,y]: [" << starting_x << ", " << starting_y << "] failed: Starting point on the background." << std::endl;
            return;
        }

        if (starting_x < 0 || starting_x > xmax || starting_y < 0 || starting_y > ymax)
        {
            std::cout << "FloodFill at [x,y]: [" << starting_x << ", " << starting_y << "] failed: Starting point outside of image." << std::endl;
            return;
        }

        // tuple of (xl,xr,y,dy) - left end of line, right end of line, y, direction: 1 = down, -1 = up (opencv has reversed y axis)
        using ff_tuple = std::tuple<int, int, int, int>;

        std::queue<ff_tuple> queue;
        queue.push(std::move(ff_tuple(starting_x, starting_x, starting_y, 1)));
        queue.push(std::move(ff_tuple(starting_x, starting_x, starting_y - 1, -1)));

        while (!queue.empty())
        {
            auto [xl, xr, y, dy] = queue.front();
            queue.pop();

            auto x = xl;

            // Find the first pixel in the given line segment from queue
            while (x < xr && !(img.at<unsigned char>(y, x) == 255))
            {
                x++;
            }

            // Find the left edge
            while (img.at<unsigned char>(y, x - 1) == 255)
            {
                img.at<unsigned char>(y, x - 1) = index;
                img_out.at<T>(y, x - 1) = color;
                x--;
            }

            // If expanded to the left -> add segment from left edge (x) to left side of checked segment (xl) in -dy
            if (x < xl)
            {
                queue.emplace(x, xl - 1, y - dy, -dy);
            }

            // x is now the the new left edge
            // Search between given line segment from queue
            while (xl <= xr)
            {
                // Search the right edge for new segment
                while (img.at<unsigned char>(y, xl) == 255)
                {
                    img.at<unsigned char>(y, xl) = index;
                    img_out.at<T>(y, xl) = color;
                    xl++;
                }

                // Found at least one pixel on the starting left edge -> add line segment to queue
                if (xl > x)
                {
                    // x is left edge, xl is one pixel to the right
                    queue.emplace(x, xl - 1, y + dy, dy);
                }

                // Expanded to the right -> add segment from right edge (xr) to right side of checked segment
                if (xl - 1 > xr)
                {
                    queue.emplace(xr + 1, xl - 1, y - dy, -dy);
                }

                // The first while ended on background -> can add +1 as "speedup"
                xl++;

                // Find next segment above given line segment from queue
                while (xl < xr && !(img.at<unsigned char>(y, xl) == 255))
                {
                    xl++;
                }

                // Move the left edge to the right
                x = xl;
            }
        }
    }

    template <typename T>
    inline cv::Mat FloodFill(cv::Mat &img, const cv::Point &starting_pixel, const T &color, const unsigned char index)
    {
        return FloodFill(img, starting_pixel.x, starting_pixel.y, color, index);
    }

    template <typename T>
    inline cv::Mat FloodFill(cv::Mat &img, int starting_x, int starting_y, const T &color, const unsigned char index)
    {
        cv::Mat img_out(img.size(), CV_8UC3);
        FloodFillInPlace(img, img_out, starting_x, starting_y, color, index);
        return img_out;
    }

    template void FloodFillInPlace<unsigned char>(cv::Mat &, cv::Mat &, int, int, const unsigned char &, const unsigned char);
    template void FloodFillInPlace<cv::Vec3b>(cv::Mat &, cv::Mat &, int, int, const cv::Vec3b &, const unsigned char);
    template void FloodFillInPlace<unsigned char>(cv::Mat &, cv::Mat &, const cv::Point &, const unsigned char &, const unsigned char);
    template void FloodFillInPlace<cv::Vec3b>(cv::Mat &, cv::Mat &, const cv::Point &, const cv::Vec3b &, const unsigned char);
    template cv::Mat FloodFill<unsigned char>(cv::Mat &, int, int, const unsigned char &, const unsigned char);
    template cv::Mat FloodFill<cv::Vec3b>(cv::Mat &, int, int, const cv::Vec3b &, const unsigned char);
    template cv::Mat FloodFill<unsigned char>(cv::Mat &, const cv::Point &, const unsigned char &, const unsigned char);
    template cv::Mat FloodFill<cv::Vec3b>(cv::Mat &, const cv::Point &, const cv::Vec3b &, const unsigned char);

}