#include <iostream>
#include <cmath>
#include <optional>

#include <opencv2/opencv.hpp>

#include "text.hpp"
#include "slic.hpp"

#define TEST_IMG_PATH "../../img/slic_bears.jpg"
#define TEST_IMG_NAME "Test_image"

// Load image from file.
std::optional<cv::Mat> LoadImage(const cv::String &filename, const cv::String &window_name = "", bool show_img = true, int flags = 1);

int main(int argc, char **argv)
{
    // Load test image.
    auto img_opt = LoadImage(TEST_IMG_PATH, TEST_IMG_NAME, true);
    if (!img_opt.has_value())
    {
        printf("No image\n");
        return -1;
    }
    cv::Mat image_test = img_opt.value();

    auto image_slic = cv::Mat();
    /* ============== SLIC ============== */
    // auto num_centroids = 180, m = 100000 / num_centroids;
    // auto num_centroids = 18, m = 10; // From paper: https://www.researchgate.net/publication/44234783_SLIC_superpixels
    auto num_centroids = 18, m = 25;
    image_slic = ano::SLIC(image_test, num_centroids, m, 200, true);
    cv::namedWindow("SLIC", cv::WINDOW_AUTOSIZE);
    cv::imshow("SLIC", image_slic);
    /* ============== SLIC ============== */

    cv::waitKey(0);

    return 0;
}

std::optional<cv::Mat> LoadImage(const cv::String &filename, const cv::String &window_name, bool show_img, int flags)
{
    cv::Mat image_in = cv::imread(filename, flags);

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
