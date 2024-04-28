#include <iostream>
#include <tuple>
#include <queue>

#include "include/floodfill.hpp"

namespace ano
{

    // combined-scan-and-fill span filler
    void FloodFillInPlace(cv::Mat &img, cv::Point starting_pixel, unsigned char index)
    {
        FloodFillInPlaceScanline(img, starting_pixel.x, starting_pixel.y, index);
    }

    // combined-scan-and-fill span filler
    void FloodFillInPlaceScanline(cv::Mat &img, int starting_x, int starting_y, unsigned char index)
    {
        // 0 - background color
        // 1 - item color
        // 255 - checked pixels
        assert(index != 0 && index != 255);

        auto xmax = img.size[1];
        decltype(xmax) xmin = 0;
        auto ymax = img.size[0];
        decltype(ymax) ymin = 0;

        // https://lodev.org/cgtutor/floodfill.html
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

        // tuple of (x1,x2,y,dy)
        using ff_tuple = std::tuple<int, int>;

        int x1;
        bool spanAbove, spanBelow;

        std::queue<ff_tuple> queue;
        queue.push(std::move(ff_tuple(starting_x, starting_y)));

        while (!queue.empty())
        {
            auto [x, y] = queue.front();
            queue.pop();

            x1 = x;
            while (x1 >= 0 && img.at<unsigned char>(y, x1) == 255)
                x1--;
            x1++;
            spanAbove = spanBelow = 0;
            while (x1 < xmax && img.at<unsigned char>(y, x1) == 255)
            {
                img.at<unsigned char>(y, x1) = index;
                if (!spanAbove && y > 0 && img.at<unsigned char>(y - 1, x1) == 255)
                {
                    queue.emplace(x1, y - 1);
                    spanAbove = 1;
                }
                else if (spanAbove && y > 0 && img.at<unsigned char>(y - 1, x1) != 255)
                {
                    spanAbove = 0;
                }
                if (!spanBelow && y < ymax - 1 && img.at<unsigned char>(y + 1, x1) == 255)
                {
                    queue.emplace(x1, y + 1);
                    spanBelow = 1;
                }
                else if (spanBelow && y < ymax - 1 && img.at<unsigned char>(y + 1, x1) != 255)
                {
                    spanBelow = 0;
                }
                x1++;
            }
        }
    }

    // combined-scan-and-fill span filler
    void FloodFillInPlaceWiki(cv::Mat &img, int starting_x, int starting_y, unsigned char index)
    {
        // 0 - background color
        // 1 - item color
        // 255 - checked pixels
        assert(index != 0 && index != 255);

        auto xmax = img.size[1];
        decltype(xmax) xmin = 0;
        auto ymax = img.size[0];
        decltype(ymax) ymin = 0;

        // https://en.wikipedia.org/wiki/Flood_fill#:~:text=The%20final%2C%20combined%2Dscan%2Dand%2Dfill%20span%20filler%20was%20then%20published%20in%201990.%20In%20pseudo%2Dcode%20form
        // Graphics Gems I
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

        // tuple of (x1,x2,y,dy)
        using ff_tuple = std::tuple<int, int, int, int>;

        std::queue<ff_tuple> queue;
        queue.push(std::move(ff_tuple(starting_x, starting_x, starting_y, 1)));
        queue.push(std::move(ff_tuple(starting_x, starting_x, starting_y + 1, -1)));

        while (!queue.empty())
        {
            auto [x1, x2, y, dy] = queue.front();
            queue.pop();

            auto x = x1;

            if (img.at<unsigned char>(y, x))
            {
                while (img.at<unsigned char>(y, x - 1))
                {
                    img.at<unsigned char>(y, x - 1) = index;
                    x -= 1;
                }

                if (x < x1)
                {
                    queue.push(std::move(ff_tuple(x, x1 - 1, y - dy, -dy)));
                }
            }

            while (x1 <= x2)
            {
                while (img.at<unsigned char>(y, x1))
                {
                    img.at<unsigned char>(y, x1) = index;
                    x1 += 1;
                }

                if (x1 > x)
                {
                    queue.push(std::move(ff_tuple(x, x1 - 1, y + dy, dy)));
                }

                if (x1 - 1 > x)
                {
                    queue.push(std::move(ff_tuple(x2 + 1, x1 - 1, y - dy, -dy)));
                }

                x1 = x1 + 1;

                while (x1 < x2 && !(img.at<unsigned char>(y, x1)))
                {
                    x1 += 1;
                }

                x = x1;
            }
        }
    }

    // combined-scan-and-fill span filler
    void FloodFillInPlace(cv::Mat &img, int starting_x, int starting_y, unsigned char index)
    {
        // 0 - background color
        // 1 - item color
        // 255 - checked pixels
        assert(index != 0 && index != 255);

        auto xmax = img.size[1];
        decltype(xmax) xmin = 0;
        auto ymax = img.size[0];
        decltype(ymax) ymin = 0;

        // https://en.wikipedia.org/wiki/Flood_fill#:~:text=The%20final%2C%20combined%2Dscan%2Dand%2Dfill%20span%20filler%20was%20then%20published%20in%201990.%20In%20pseudo%2Dcode%20form
        // Graphics Gems I
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

        // tuple of (x1,x2,y,dy)
        using ff_tuple = std::tuple<int, int, int, int>;

        std::queue<ff_tuple> queue;
        queue.push(std::move(ff_tuple(starting_x, starting_x, starting_y, 1)));
        queue.push(std::move(ff_tuple(starting_x, starting_x, starting_y - 1, -1)));

        while (!queue.empty())
        {
            auto [x1, x2, y, dy] = queue.front();
            queue.pop();

            auto x = x1;
            auto start = x;

            // Check all pixels to the left
            while (x >= xmin && img.at<unsigned char>(y, x) == 255)
            {
                img.at<unsigned char>(y, x) = index;
                x -= 1;
            }

            // Add pixel to queue if found pixel to the left
            if (x >= x1)
                goto skip;

            start = x + 1;

            if (start < x1)
            {
                queue.push(std::move(ff_tuple(y, start, x1 - 1, -dy)));
            }

            x = x1 + 1;

            do
            {
                // Check right pixels
                while (x <= xmax && img.at<unsigned char>(y, x) == 255)
                {
                    img.at<unsigned char>(y, x) = index;
                    x += 1;
                }

                queue.push(std::move(ff_tuple(start, x - 1, y, dy)));

                if (x > x2 + 1)
                {
                    queue.push(std::move(ff_tuple(x2 + 1, x - 1, y, -dy)));
                }

            skip:
                x += 1;

                while (x <= x2 && !(img.at<unsigned char>(y, x) != 255))
                {
                    x += 1;
                }

                start = x;
            } while (x <= x2);
        }
    }

    cv::Mat FloodFill(const cv::Mat &img, cv::Point starting_pixel, unsigned char index)
    {
        cv::Mat img_indexed(img.size(), img.type());
        FloodFillInPlace(img_indexed, starting_pixel, index);
        return std::move(img_indexed);
    }

}