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

    void Ethalons::AddEthalons(unsigned char id_class, const std::vector<float> &ethalons, const cv::Vec3b &color)
    {
        Ethalons::AddEthalons(id_class, std::vector<float>(ethalons), color);
    }

    void Ethalons::AddEthalons(unsigned char id_class, std::vector<float> &&ethalons, const cv::Vec3b &color)
    {
        this->ethalons.emplace_back(id_class, std::move(ethalons), color);
    }

    std::vector<float> Ethalons::GetEthalonsByClass(unsigned char id_class)
    {

        auto it = std::find_if(ethalons.begin(), ethalons.end(), [id_class](const auto &ethalon)
                               { return std::get<0>(ethalon) == id_class; });

        if (it == ethalons.end())
        {
            return {};
        }

        return std::get<1>(*it);
    }

    cv::Vec3b Ethalons::GetColorByClass(unsigned char id_class)
    {

        auto it = std::find_if(ethalons.begin(), ethalons.end(), [id_class](const auto &ethalon)
                               { return std::get<0>(ethalon) == id_class; });

        if (it == ethalons.end())
        {
            return {};
        }

        return std::get<2>(*it);
    }

    unsigned char Ethalons::FindClosestClass(const std::vector<float> &ethalons)
    {
        unsigned char closest_class = 0;
        float closest_distance = std::numeric_limits<float>::max();

        for (const auto &ethalon : this->ethalons)
        {
            float distance = 0;

            // Sum squares
            int i = 0;
            for (; i < std::get<1>(ethalon).size(); i++)
            {
                distance += std::pow(ethalons.at(i) - std::get<1>(ethalon).at(i), 2);
            }

            // Calculate euclidian distance
            distance = std::sqrt(distance);

            // Compare with closest distance
            if (distance < closest_distance)
            {
                closest_distance = distance;
                closest_class = std::get<0>(ethalon);
            }
        }

        return closest_class;
    }
}
