#pragma once

#include <algorithm>

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
}