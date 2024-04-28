#include "color-generator.hpp"

#include <random>

const static int range_from = 0;
const static int range_to = 255;
static std::random_device rand_dev;
static std::mt19937 generator(rand_dev());
static std::uniform_int_distribution<int> distr(range_from, range_to);

namespace ano
{

    cv::Vec3b GenerateRandomColorBGR(const cv::Vec3b &color_mix)
    {
        int red = distr(generator);
        int green = distr(generator);
        int blue = distr(generator);

        red = (red + color_mix[0]) / 2;
        green = (green + color_mix[1]) / 2;
        blue = (blue + color_mix[2]) / 2;

        return cv::Vec3b(blue, green, red);
    }

}