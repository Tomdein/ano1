#include "color-generator.hpp"

#include <random>

const static int range_from = 0;
const static int range_to = 255;
static std::random_device rand_dev;
static std::mt19937 generator(rand_dev());
static std::uniform_int_distribution<int> distr(range_from, range_to);

namespace ano
{

    cv::Vec3b GenerateRandomColorBGR(const cv::Vec3b &color_base, float mix_ratio_base, float lightness)
    {
        int red = distr(generator);
        int green = distr(generator);
        int blue = distr(generator);

        auto mix_ratio_supplement = 1.0f - mix_ratio_base;

        red = (red * mix_ratio_supplement + color_base[0] * mix_ratio_base) * lightness;
        green = (green * mix_ratio_supplement + color_base[1] * mix_ratio_base) * lightness;
        blue = (blue * mix_ratio_supplement + color_base[2] * mix_ratio_base) * lightness;

        return cv::Vec3b(blue, green, red);
    }

}