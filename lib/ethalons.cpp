#include "ethalons.hpp"

#include <opencv2/opencv.hpp>

#include "text.hpp"

namespace ano
{
    float GetEthalonF1(const DetectedObjectsVector &detected_objects, unsigned char id_class)
    {
        float ethalon = 0.0f;
        int count = 0;
        std::for_each(detected_objects.begin(), detected_objects.end(), [&id_class, &ethalon, &count](const DetectedObject &detected_object) -> void
                      {
            if(detected_object.id_class == id_class)
            {
                ethalon += detected_object.features.F1;
                count++;
            } });

        return ethalon / count;
    }

    float GetEthalonF2(const DetectedObjectsVector &detected_objects, unsigned char id_class)
    {
        float ethalon = 0.0f;
        int count = 0;
        std::for_each(detected_objects.begin(), detected_objects.end(), [&id_class, &ethalon, &count](const DetectedObject &detected_object) -> void
                      {
            if(detected_object.id_class == id_class)
            {
                ethalon += detected_object.features.F2;
                count++;
            } });

        return ethalon / count;
    }

    void DrawEthalon(cv::Mat &img, float ethalon_f1, float ethalon_f2, float size, const cv::Vec3b &color, float ethalon_f1_scale, float ethalon_f2_scale)
    {
        auto y = img.size[0] * ethalon_f1 * ethalon_f1_scale;
        auto x = img.size[1] * ethalon_f2 * ethalon_f2_scale;
        cv::circle(img, cv::Point(x, y), size, color, cv::FILLED);
    }

    void DrawEthalonWithText(cv::Mat &img, float ethalon_f1, float ethalon_f2, float size, const cv::Vec3b &color, float ethalon_f1_scale, float ethalon_f2_scale, unsigned char id_class)
    {
        auto text_x_offset = size + 4;

        auto y = img.size[0] * ethalon_f1 * ethalon_f1_scale;
        auto x = img.size[1] * ethalon_f2 * ethalon_f2_scale;
        cv::circle(img, cv::Point(x, y), size, color, cv::FILLED);
        cv::putText(img, std::to_string(ethalon_f1), cv::Point(x + text_x_offset, y + TEXT_LINE_HEIGHT), TEXT_FONT, TEXT_SIZE, color);
        cv::putText(img, std::to_string(ethalon_f2), cv::Point(x + text_x_offset, y + 2 * TEXT_LINE_HEIGHT), TEXT_FONT, TEXT_SIZE, color);

        if (id_class != 0)
        {
            cv::putText(img, "Class: " + std::to_string(id_class), cv::Point(x + text_x_offset, y + 3 * TEXT_LINE_HEIGHT), TEXT_FONT, TEXT_SIZE, color);
        }
    }
}
