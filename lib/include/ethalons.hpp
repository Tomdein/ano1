#pragma once

#include <algorithm>
#include <vector>
#include <tuple>

#include <opencv2/opencv.hpp>

#include "detected-object.hpp"

namespace ano
{
    // Goes through all detected objects and returns the average F1 score for the given class.
    float GetEthalonF1(const DetectedObjectsVector &detected_objects, unsigned char id_class);
    // Goes through all detected objects and returns the average F2 score for the given class.
    float GetEthalonF2(const DetectedObjectsVector &detected_objects, unsigned char id_class);

    // Plots the given ethalons.
    void DrawEthalon(cv::Mat &img, float ethalon_f1, float ethalon_f2, float size, const cv::Vec3b &color, float ethalon_f1_scale = 1.0f, float ethalon_f2_scale = 0.5);
    // Plots the given ethalons with values and optionaly a class id.
    void DrawEthalonWithText(cv::Mat &img, float ethalon_f1, float ethalon_f2, float size, const cv::Vec3b &color, float ethalon_f1_scale = 1.0f, float ethalon_f2_scale = 0.5, unsigned char id_class = 0);

    class Ethalons
    {
    private:
        std::vector<std::tuple<unsigned char, std::vector<float>, cv::Vec3b>> ethalons;

    public:
        // Add more ethalons to storage
        void AddEthalons(unsigned char id_class, std::vector<float> &&ethalons, const cv::Vec3b &color);
        void AddEthalons(unsigned char id_class, const std::vector<float> &ethalons, const cv::Vec3b &color);

        // Get forward iterator to the first ethalon
        std::vector<std::tuple<unsigned char, std::vector<float>, cv::Vec3b>>::iterator begin();
        // Get forward iterator to one after the end ethalon
        std::vector<std::tuple<unsigned char, std::vector<float>, cv::Vec3b>>::iterator end();

        // Get ethalon features by class id
        std::vector<float> GetEthalonsByClass(unsigned char id_class);

        // Get ethalon color by class id
        cv::Vec3b GetColorByClass(unsigned char id_class);

        // Find closest class id of ethalon from given values
        unsigned char FindClosestClass(const std::vector<float> &ethalons);
    };
}