#include <iostream>
#include <cmath>
#include <optional>

#include <opencv2/opencv.hpp>

#include "text.hpp"
#include "floodfill.hpp"
#include "color-generator.hpp"
#include "moments.hpp"
#include "detected-object.hpp"
#include "ethalons.hpp"
#include "k-means-clustering.hpp"
#include "backprop.hpp"

#define TRAIN_IMG_PATH "../../img/train.png"
#define TRAIN_IMG_NAME "Training_image"
#define TEST_IMG_PATH "../../img/test02.png"
#define TEST_IMG_NAME "Test_image"

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

void Threshold(const cv::Mat &image_in, cv::Mat &image_threshold, unsigned char threshold);
// Flood fill threshold image and save colored result in image_indexing. Also save info about detected objects (id, x, y) in detected_objects.
// object_index - starting id for indexing. Decrements for each new object.
// assign_class_from_map - used to assign class from id_map for training ethalons from training images.
void Indexing(cv::Mat &image_threshold, cv::Mat &image_indexing, ano::DetectedObjectsVector &detected_objects, unsigned char &object_index, const cv::Vec3b &color_for_mixing, bool assign_class_from_map = false);
// Calculate moments of all detected objects starting from id == object_index up to id == 254 (including 254)
void Moments(const cv::Mat &image_threshold, ano::DetectedObjectsVector &detected_objects, unsigned char object_index);
// Load image from file.
std::optional<cv::Mat> LoadImage(const cv::String &filename, const cv::String &window_name = "", bool show_img = true, int flags = 1);
// Calculate and draw class ethalons
void ClassEthalons(cv::Mat &image_ethalons, const ano::DetectedObjectsVector &detected_objects, ano::Ethalons &ethalons, float f1_scale, float f2_scale);

int main(int argc, char **argv)
{
    // Load training image.
    auto img_opt = LoadImage(TRAIN_IMG_PATH, TRAIN_IMG_NAME, true, cv::IMREAD_GRAYSCALE);
    if (!img_opt.has_value())
    {
        printf("No image\n");
        return -1;
    }
    cv::Mat image_in = img_opt.value();

    // Load test image.
    img_opt = LoadImage(TEST_IMG_PATH, TEST_IMG_NAME, true, cv::IMREAD_GRAYSCALE);
    if (!img_opt.has_value())
    {
        printf("No image\n");
        return -1;
    }
    cv::Mat image_test = img_opt.value();

    /* ============== THRESHOLDING ============== */
    unsigned char threshold = 50;

    // Threshold training image.
    cv::Mat image_threshold_train(image_in.size(), CV_8UC1);
    Threshold(image_in, image_threshold_train, threshold);

    // Threshold test image.
    cv::Mat image_threshold_test(image_test.size(), CV_8UC1);
    Threshold(image_test, image_threshold_test, threshold);
    /* ============== THRESHOLDING ============== */

    /* ============== Indexing ============== */
    cv::Mat image_indexing_train = cv::Mat(image_threshold_train.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat image_indexing_test = cv::Mat(image_threshold_test.size(), CV_8UC3, cv::Scalar(0, 0, 0));

    ano::DetectedObjectsVector detected_objects_train;
    ano::DetectedObjectsVector detected_objects_test;

    // Train
    // Starting id
    unsigned char object_index_train = -2; // char max - 1 (as 255 is reserved for foreground)
    Indexing(image_threshold_train, image_indexing_train, detected_objects_train, object_index_train, {255, 255, 255}, true);

    cv::namedWindow("Indexing train", cv::WINDOW_NORMAL || cv::WINDOW_KEEPRATIO);
    cv::imshow("Indexing train", image_indexing_train);

    cv::namedWindow("Thresholding train", cv::WINDOW_AUTOSIZE);
    cv::imshow("Thresholding train", image_threshold_train);

    // Test
    // Starting id
    unsigned char object_index_test = -2; // char max - 1 (as 255 is reserved for foreground)
    // Do not assign class from id_map
    Indexing(image_threshold_test, image_indexing_test, detected_objects_test, object_index_test, {255, 255, 255}, false);

    cv::namedWindow("Thresholding test", cv::WINDOW_AUTOSIZE);
    cv::imshow("Thresholding test", image_threshold_test);
    /* ============== Indexing ============== */

    /* ============== Moments ============== */
    // Iterate through all indexed objects
    Moments(image_threshold_train, detected_objects_train, object_index_train + 1);
    Moments(image_threshold_test, detected_objects_test, object_index_test + 1);
    /* ============== Moments ============== */

    /* ============== Neural Network ============== */

    // Create NN:
    // 2 input features: F1, F2. 5 hidden neurons. 3 output neurons: 3 classes.
    ano::bpnn::NN *nn = ano::bpnn::createNN(2, 5, 3);

    // Train NN:
    constexpr int n_in = 2;  // == number of features
    constexpr int n_out = 3; // == number of output neurons (classes)
    constexpr int num_training_objects = 12;

    double ***trainingSet = new double **[num_training_objects]; // N sets of outputs and inputs
    for (int i = 0; i < num_training_objects; i++)
    {
        trainingSet[i] = new double *[2];      // 2 arrays: input and output values
        trainingSet[i][0] = new double[n_in];  // Input values
        trainingSet[i][1] = new double[n_out]; // Output values

        // Get single training set:
        auto train_data = id_map[i];
        auto train_id = train_data[0];

        // Fill inputs (manually as they are stored in a struct)
        auto it = DetectedObjectsVectorGetByPixelID(detected_objects_train, train_id);
        if (it == detected_objects_train.end())
        {
            throw "Invalid id for nn";
        }
        trainingSet[i][0][0] = it->features.F1;
        trainingSet[i][0][1] = it->features.F2;

        // Fill outputs
        auto train_class = train_data[1];
        for (int j = 0; j < n_out; j++)
        {
            // Set single neuron (corresponding to class) to 1 and others to 0
            // neurons are numbered from 0, classes are numbered from 1 -> class - 1
            trainingSet[i][1][j] = (train_class - 1 == j) ? 1 : 0;
        }
    }

    constexpr int n = 1000; // Number of training iterations
    double error = 1.0;
    int i = 0;
    while (error > 0.001)
    {
        ano::bpnn::setInput(nn, trainingSet[i % num_training_objects][0]);
        ano::bpnn::feedforward(nn);
        error = ano::bpnn::backpropagation(nn, trainingSet[i % num_training_objects][1]);
        i++;
        printf("\rerr=%0.3f", error);
    }
    printf(" (%d iterations)\n", i);

    for (int i = 0; i < num_training_objects; i++)
    {
        delete[] trainingSet[i][0];
        delete[] trainingSet[i][1];
        delete[] trainingSet[i];
    }
    delete[] trainingSet;

    // Apply NN:
    double *in = new double[nn->n[0]];
    for (int i = 0; i < detected_objects_test.size(); i++)
    {
        for (auto &obj_test_it : detected_objects_test)
        {
            // Set input values
            in[0] = obj_test_it.features.F1;
            in[1] = obj_test_it.features.F2;

            // Set NN input
            ano::bpnn::setInput(nn, in, true);

            // Feedforward
            ano::bpnn::feedforward(nn);
            int output = ano::bpnn::getOutput(nn, true);

            // Set class of object
            obj_test_it.id_class = output + 1;
            obj_test_it.DrawClass(image_indexing_test, 0, TEXT_LINE_HEIGHT);
        }
    }
    delete[] in;

    // Free NN:
    ano::bpnn::releaseNN(nn);

    cv::namedWindow("Indexing test", cv::WINDOW_NORMAL || cv::WINDOW_KEEPRATIO);
    cv::imshow("Indexing test", image_indexing_test);
    /* ============== Neural Network ============== */

    cv::waitKey(0);

    return 0;
}

std::optional<cv::Mat> LoadImage(const cv::String &filename, const cv::String &window_name, bool show_img, int flags)
{
    cv::Mat image_in = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    if (!image_in.data)
    {
        printf("No image data \n");
        return {};
    }

    std::cout << "Image '" << filename << "', "
              << "Size: " << image_in.size[0] << "," << image_in.size[1] << "\n"
              << std::endl;

    if (show_img)
    {
        auto name = (window_name.empty()) ? filename : window_name;
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
        cv::imshow(window_name, image_in);
    }

    return image_in;
}

void Threshold(const cv::Mat &image_in, cv::Mat &image_threshold, unsigned char threshold)
{
    for (int y = 0; y < image_threshold.size[0]; y++)
    {
        for (int x = 0; x < image_threshold.size[1]; x++)
        {
            image_threshold.at<unsigned char>(y, x) = (image_in.at<unsigned char>(y, x) > threshold) ? 255 : 0;
        }
    }
}

void Indexing(cv::Mat &image_threshold, cv::Mat &image_indexing, ano::DetectedObjectsVector &detected_objects, unsigned char &object_index, const cv::Vec3b &color_for_mixing, bool assign_class_from_map)
{
    for (int y = 0; y < image_threshold.size[0]; y++)
    {
        for (int x = 0; x < image_threshold.size[1]; x++)
        {
            if (image_threshold.at<unsigned char>(y, x) == 255)
            {
                // Fill with random color
                auto color = ano::GenerateRandomColorBGR(color_for_mixing);
                ano::FloodFillInPlace(image_threshold, image_indexing, x, y, color, object_index);

                // Get the class label
                auto obj_id_it = std::find_if(std::begin(id_map), std::end(id_map), [&object_index](const auto &map_pair)
                                              { return map_pair[0] == object_index; });

                // Skip the class assignment if it is not required
                if (!assign_class_from_map)
                {
                    obj_id_it = std::end(id_map);
                }

                unsigned char obj_id = 0;
                if (obj_id_it != std::end(id_map))
                {
                    obj_id = (*obj_id_it)[1];
                }

                // Save the detected object
                detected_objects.emplace_back(object_index, obj_id, x, y);

                // Output info to the image
                cv::putText(image_indexing, "ID: " + std::to_string(object_index), cv::Point(x, y), TEXT_FONT, TEXT_SIZE, TEXT_COLOR, 1);
                if (obj_id != 0)
                {
                    cv::putText(image_indexing, "Class: " + std::to_string(obj_id), cv::Point(x, y + TEXT_LINE_HEIGHT), TEXT_FONT, TEXT_SIZE, TEXT_COLOR, 1);
                }

                // Decrement id
                object_index--;
            }
        }
    }
}

void Moments(const cv::Mat &image_threshold, ano::DetectedObjectsVector &detected_objects, unsigned char object_index)
{
    for (unsigned char obj_index = object_index; obj_index < 255; obj_index++)
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
}

void ClassEthalons(cv::Mat &image_ethalons, const ano::DetectedObjectsVector &detected_objects, ano::Ethalons &ethalons, float f1_scale, float f2_scale)
{
    // Calculate ethalons
    auto class1_ethalon_f1 = ano::GetEthalonF1(detected_objects, 1);
    auto class1_ethalon_f2 = ano::GetEthalonF2(detected_objects, 1);
    auto color1 = ano::GenerateRandomColorBGR();
    ethalons.AddEthalons(1, {class1_ethalon_f1, class1_ethalon_f2}, color1);

    auto class2_ethalon_f1 = ano::GetEthalonF1(detected_objects, 2);
    auto class2_ethalon_f2 = ano::GetEthalonF2(detected_objects, 2);
    auto color2 = ano::GenerateRandomColorBGR();
    ethalons.AddEthalons(2, {class2_ethalon_f1, class2_ethalon_f2}, color2);

    auto class3_ethalon_f1 = ano::GetEthalonF1(detected_objects, 3);
    auto class3_ethalon_f2 = ano::GetEthalonF2(detected_objects, 3);
    auto color3 = ano::GenerateRandomColorBGR();
    ethalons.AddEthalons(3, {class3_ethalon_f1, class3_ethalon_f2}, color3);

    // Draw ethalons
    ano::DrawEthalonWithText(image_ethalons, class1_ethalon_f1, class1_ethalon_f2, 3, color1 * 0.5f, f1_scale, f2_scale, 1);
    ano::DrawEthalonWithText(image_ethalons, class2_ethalon_f1, class2_ethalon_f2, 3, color2 * 0.5f, f1_scale, f2_scale, 2);
    ano::DrawEthalonWithText(image_ethalons, class3_ethalon_f1, class3_ethalon_f2, 3, color3 * 0.5f, f1_scale, f2_scale, 3);
}
