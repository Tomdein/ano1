#pragma once

#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>

namespace ano
{
    struct DetectedObjectFeatures
    {
    public:
        cv::Vec2i center_of_mass = cv::Vec2i(0, 0);
        float area = 0.0f;
        float F1 = 0.0f;
        float F2 = 0.0f;

        DetectedObjectFeatures() = default;
        DetectedObjectFeatures(cv::Vec2i center_of_mass, float area, float F1, float F2) : center_of_mass(center_of_mass), area(area), F1(F1), F2(F2) {}
    };

    class DetectedObject
    {
    public:
        int x = 0;
        int y = 0;
        int width = 0;
        int height = 0;
        unsigned char id_pixel = 0;
        unsigned char id_class = 0;
        DetectedObjectFeatures features;

        DetectedObject(unsigned char id_pixel, unsigned char id_class, int x, int y, int width = 0, int height = 0, const DetectedObjectFeatures &features = DetectedObjectFeatures(cv::Vec2i(0, 0), 0.0, 0.0, 0.0))
            : id_pixel(id_pixel), id_class(id_class), x(x), y(y), width(width), height(height), features(features)
        {
        }

        bool operator==(const DetectedObject &other) const
        {
            return id_pixel == other.id_pixel;
        }

        void DrawXY(cv::Mat &image, int x_offset = 0, int y_offset = 0, const cv::Vec3b &color = {255, 255, 255}) const;
        void DrawWH(cv::Mat &image, int x_offset = 0, int y_offset = 0, const cv::Vec3b &color = {255, 255, 255}) const;
        void DrawId(cv::Mat &image, int x_offset = 0, int y_offset = 0, const cv::Vec3b &color = {255, 255, 255}) const;
        void DrawClass(cv::Mat &image, int x_offset = 0, int y_offset = 0, const cv::Vec3b &color = {255, 255, 255}) const;

    private:
        void DrawText(cv::Mat &image, const std::string &text, int x_offset = 0, int y_offset = 0, const cv::Vec3b &color = {255, 255, 255}) const;
    };

    using DetectedObjectsVector = std::vector<DetectedObject>;

    DetectedObjectsVector::iterator DetectedObjectsVectorGetByPixelID(DetectedObjectsVector &detected_objects, unsigned char id_pixel);

    void DetectedObjectsVectorSetClass(DetectedObjectsVector &detected_objects, unsigned char id_pixel, unsigned char id_class);
}