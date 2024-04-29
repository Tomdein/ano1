#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>

#include "text.hpp"
#include "floodfill.hpp"
#include "color-generator.hpp"
#include "moments.hpp"
#include "detected-object.hpp"

#define TEST_IMG_PATH "../../img/train.png"

unsigned char id_map[][2] = {
    {243, 1}, {244, 1}, {245, 1}, {246, 1}, //
    {247, 2},
    {248, 2},
    {249, 2},
    {250, 2}, //
    {251, 3},
    {252, 3},
    {253, 3},
    {254, 3}, //
};

int main(int argc, char **argv)
{
    cv::Mat image_in = cv::imread(TEST_IMG_PATH, cv::IMREAD_GRAYSCALE);

    if (!image_in.data)
    {
        printf("No image data \n");
        return -1;
    }

    std::cout << "Size: " << image_in.size[0] << "," << image_in.size[1] << std::endl;

    cv::namedWindow("Input", cv::WINDOW_AUTOSIZE);
    cv::imshow("Input", image_in);

    /* ============== THRESHOLDING ============== */
    unsigned char threshold = 50;

    cv::Mat image_threshold(image_in.size(), CV_8UC1);
    for (int y = 0; y < image_threshold.size[0]; y++)
    {
        for (int x = 0; x < image_threshold.size[1]; x++)
        {
            image_threshold.at<unsigned char>(y, x) = (image_in.at<unsigned char>(y, x) > threshold) ? 255 : 0;
        }
    }
    cv::namedWindow("Thresholding", cv::WINDOW_AUTOSIZE);
    cv::imshow("Thresholding", image_threshold);
    /* ============== THRESHOLDING ============== */

    /* ============== Indexing ============== */
    cv::Vec3b color_white(255, 255, 255);

    auto color = ano::GenerateRandomColorBGR(color_white);
    cv::Mat image_indexing = ano::FloodFill(image_threshold, 0, 0, color);

    ano::DetectedObjectsVector detected_objects;

    unsigned char object_index = -2; // char max - 1 (as 255 is reserved for foreground)
    for (int y = 0; y < image_threshold.size[0]; y++)
    {
        for (int x = 0; x < image_threshold.size[1]; x++)
        {
            if (image_threshold.at<unsigned char>(y, x) == 255)
            {
                // Fill with random color
                color = ano::GenerateRandomColorBGR(color_white);
                ano::FloodFillInPlace(image_threshold, image_indexing, x, y, color, object_index);

                // Get the class label
                auto obj_id = std::find_if(std::begin(id_map), std::end(id_map), [&object_index](const auto &map_pair)
                                           { return map_pair[0] == object_index; });

                // Save the detected object
                if (obj_id != std::end(id_map))
                {
                    detected_objects.emplace_back(object_index, (*obj_id)[1], x, y);
                }
                else
                {
                    detected_objects.emplace_back(object_index, 0, x, y);
                }

                // Output info to the image
                cv::putText(image_indexing, "ID: " + std::to_string(object_index), cv::Point(x, y), TEXT_FONT, TEXT_SIZE, TEXT_COLOR, 1);
                cv::putText(image_indexing, "Class: " + std::to_string(object_index), cv::Point(x, y + TEXT_LINE_HEIGHT), TEXT_FONT, TEXT_SIZE, TEXT_COLOR, 1);

                // Decrement id
                object_index--;
            }
        }
    }
    cv::namedWindow("Indexing", cv::WINDOW_NORMAL || cv::WINDOW_KEEPRATIO);
    cv::imshow("Indexing", image_indexing);

    cv::namedWindow("Thresholding", cv::WINDOW_AUTOSIZE);
    cv::imshow("Thresholding", image_threshold);
    /* ============== Indexing ============== */

    /* ============== Moments ============== */
    // Iterate through all indexed objects
    for (unsigned char obj_index = object_index + 1; obj_index < 255; obj_index++)
    {

        auto center_of_mass = ano::CenterOfMass(image_threshold, obj_index);
        float area = ano::Area(image_threshold, obj_index);
        auto F1 = std::pow(ano::Circumference(image_threshold, obj_index), 2) / (100 * area);

        auto u20 = ano::CenteredMoment(image_threshold, 2, 0, obj_index);
        auto u02 = ano::CenteredMoment(image_threshold, 0, 2, obj_index);
        auto u11 = ano::CenteredMoment(image_threshold, 1, 1, obj_index);
        float umax = 0.5f * (u20 + u02) + 0.5f * std::sqrt(4 * std::pow(u11, 2) + std::pow(u20 - u02, 2));
        float umin = 0.5f * (u20 + u02) - 0.5f * std::sqrt(4 * std::pow(u11, 2) + std::pow(u20 - u02, 2));
        auto F2 = umin / umax;

        std::cout << "Object index: " << std::to_string(obj_index) << "\n"
                  << "\tCenter of mass: " << center_of_mass << "\n"
                  << "\tArea: " << area << "\n"
                  << "\tF1: " << F1 << "\n"
                  << "\tF2: " << F2 << "\n\n"
                  << std::endl;

        auto obj_it = ano::DetectedObjectsVectorGetByPixelID(detected_objects, obj_index);
        if (obj_it == detected_objects.end())
        {
            std::cout << "Object not found" << std::endl;
        }
        else
        {
            auto &obj_features = obj_it->features;
            obj_features.center_of_mass = center_of_mass;
            obj_features.area = area;
            obj_features.F1 = F1;
            obj_features.F2 = F2;
        }
    }
    /* ============== Moments ============== */

    cv::waitKey(0);

    return 0;
}
