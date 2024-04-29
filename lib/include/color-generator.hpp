#pragma once

#include <opencv2/opencv.hpp>

namespace ano
{

    // Generates a random color and mixes it with the given color.
    // Great is to use white color to generate pastel colors. They go together well.
    // color_base - base color to which the random color will be mixed
    // mix_ratio_base - the ratio of the base color to and random color
    // lightness - the lightness of the random color. Divides the mixed color by this value
    cv::Vec3b GenerateRandomColorBGR(const cv::Vec3b &color_base = cv::Vec3b(255, 255, 255), float mix_ratio_base = 0.5f, float lightness = 1.0f);

}