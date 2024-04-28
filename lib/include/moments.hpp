#pragma once

#include <cassert>
#include <type_traits>

#include <opencv2/opencv.hpp>

namespace ano
{
    template <typename T>
    int Moment(const cv::Mat &img, unsigned char p, unsigned char q, const T &color)
    {
        static_assert(std::is_same_v<decltype(color), const unsigned char &>, "color must be unsigned char");

        auto ymax = img.size[0];
        auto xmax = img.size[1];

        int sum = 0;
        for (int y = 0; y < ymax; ++y)
        {
            for (int x = 0; x < xmax; ++x)
            {
                auto val = img.at<T>(y, x);
                if (img.at<T>(y, x) == color)
                {
                    sum += std::pow(x, p) * std::pow(y, q);
                }
            }
        }

        return sum;
    }

    template <typename T>
    int Area(const cv::Mat &img, const T &color)
    {
        return Moment<T>(img, 0, 0, color);
    }

    template <typename T>
    cv::Vec2i CenterOfMass(const cv::Mat &img, const T &color)
    {
        static_assert(std::is_same_v<decltype(color), const unsigned char &>, "color must be unsigned char");

        auto m00 = Moment<T>(img, 0, 0, color);
        auto m10 = Moment<T>(img, 1, 0, color);
        auto m01 = Moment<T>(img, 0, 1, color);

        auto xt = m10 / m00;
        auto yt = m01 / m00;

        return {xt, yt};
    }

    template <typename T>
    int Circumference(const cv::Mat &img, const T &color)
    {
        static_assert(std::is_same_v<decltype(color), const unsigned char &>, "color must be unsigned char");

        auto ymax = img.size[0];
        auto xmax = img.size[1];

        int sum = 0;
        for (int y = 0; y < ymax; ++y)
        {
            for (int x = 0; x < xmax; ++x)
            {
                if (img.at<T>(y, x) == color)
                {
                    // If none of the surrounding pixels are the same color, add 1 to circumference
                    if (!(img.at<T>(y - 1, x) == color && img.at<T>(y, x - 1) == color && img.at<T>(y, x + 1) == color && img.at<T>(y + 1, x) == color))
                    {
                        sum++;
                    }
                }
            }
        }

        return sum;
    }

    template <typename T>
    int CenteredMoment(const cv::Mat &img, unsigned char p, unsigned char q, const T &color, const cv::Vec2i &center)
    {
        static_assert(std::is_same_v<decltype(color), const unsigned char &>, "color must be unsigned char");

        auto ymax = img.size[0];
        auto xmax = img.size[1];

        int sum = 0;

        for (int y = 0; y < ymax; ++y)
        {
            for (int x = 0; x < xmax; ++x)
            {
                if (img.at<T>(y, x) == color)
                {
                    sum += std::pow(x - center[0], p) * std::pow(y - center[1], q);
                }
            }
        }

        return sum;
    }

    template <typename T>
    int CenteredMoment(const cv::Mat &img, unsigned char p, unsigned char q, const T &color)
    {
        static_assert(std::is_same_v<decltype(color), const unsigned char &>, "color must be unsigned char");

        auto center_of_mass = CenterOfMass(img, color);
        return CenteredMoment(img, p, q, color, center_of_mass);
    }
}