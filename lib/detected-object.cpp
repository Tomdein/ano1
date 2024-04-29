#include "detected-object.hpp"

#include <text.hpp>

namespace ano
{
    DetectedObjectsVector::iterator DetectedObjectsVectorGetByPixelID(DetectedObjectsVector &detected_objects, unsigned char id_pixel)
    {
        return std::find_if(detected_objects.begin(), detected_objects.end(), [&id_pixel](auto &object) -> bool
                            { return object.id_pixel == id_pixel; });
    }

    void DetectedObjectsVectorSetClass(DetectedObjectsVector &detected_objects, unsigned char id_pixel, unsigned char id_class)
    {
        auto object_it = DetectedObjectsVectorGetByPixelID(detected_objects, id_pixel);

        if (object_it != detected_objects.end())
        {
            object_it->id_class = id_class;
        }
    }

    void DetectedObject::DrawXY(cv::Mat &image, int x_offset, int y_offset, const cv::Vec3b &color) const
    {
        this->DrawText(image, "[" + std::to_string(this->x) + ", " + std::to_string(this->y) + "]", x_offset, y_offset, color);
    }

    void DetectedObject::DrawWH(cv::Mat &image, int x_offset, int y_offset, const cv::Vec3b &color) const
    {
        this->DrawText(image, "[" + std::to_string(this->width) + ", " + std::to_string(this->height) + "]", x_offset, y_offset, color);
    }

    void DetectedObject::DrawId(cv::Mat &image, int x_offset, int y_offset, const cv::Vec3b &color) const
    {
        this->DrawText(image, "ID: " + std::to_string(this->id_pixel), x_offset, y_offset, color);
    }

    void DetectedObject::DrawClass(cv::Mat &image, int x_offset, int y_offset, const cv::Vec3b &color) const
    {
        this->DrawText(image, "Class: " + std::to_string(this->id_class), x_offset, y_offset, color);
    }

    void DetectedObject::DrawText(cv::Mat &image, const std::string &text, int x_offset, int y_offset, const cv::Vec3b &color) const
    {
        cv::putText(image, text, cv::Point(this->x + x_offset, this->y + y_offset), TEXT_FONT, TEXT_SIZE, color);
    }
}
