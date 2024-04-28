#pragma once

#include <opencv2/opencv.hpp>

namespace ano
{

    // Generates a random color and mixes it with the given color.
    // Great is to use white color to generate pastel colors. They go together well.
    cv::Vec3b GenerateRandomColorBGR(const cv::Vec3b &color_mix);

}