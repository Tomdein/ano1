#include "slic.hpp"

#include <cassert>
#include <cmath>
#include <algorithm>

#include "image-gradient.hpp"

namespace ano
{

#define SLIC_CENTEROIDS_MOVED_THRESHOLD (1e-2)

    // SLIC: Simple Linear Iterative Clustering
    cv::Mat SLIC(const cv::Mat &img, int k_segments, float m_balance, int max_iterations)
    {
        auto w = img.size[1];
        auto h = img.size[0];
        auto pixel_count = w * h;

        auto S = std::sqrt(pixel_count / k_segments);
        auto x_count = std::floor(w / S);
        auto y_count = std::floor(h / S);

        // SLIC: S is not an integer multiple of the image size
        // assert(S == x_count * y_count);

        // Copy of input image
        cv::Mat img_slic(img);
        // Distance + index
        cv::Mat img_slic_indices(img_slic.size(), CV_32FC2);

        const auto color_space_distance_balance = m_balance / S;

        // Step 0: Initialize centroids as a pair of color and (x, y) coordinates
        std::vector<std::pair<cv::Vec2f, cv::Vec3b>> centroids;
        auto S_offset = S / 2;
        for (int y = 0; y < y_count; ++y)
        {
            for (int x = 0; x < x_count; ++x)
            {
                auto x_center = static_cast<int>(S * x + S_offset);
                auto y_center = static_cast<int>(S * y + S_offset);

                assert(x_center - 1 >= 0 && y_center - 1 >= 0);

                using gradient_type = float[2];

                auto gradient = ano::ComputeGradients(img_slic, x_center - 1, y_center - 1, 3, 3);

                // Find the gradient with the smallest magnitude
                auto mat_begin_ptr = reinterpret_cast<gradient_type *>(gradient.ptr(0));
                auto mat_end_ptr = reinterpret_cast<gradient_type *>(gradient.ptr(10));
                auto min_ptr = std::min_element(mat_begin_ptr, mat_end_ptr, [](const gradient_type &a, const gradient_type &b)
                                                {
                                                    return a[1] < b[1]; // Compare gradient magnitudes
                                                });

                // Find the index of the gradient with the smallest magnitude
                int min_index = static_cast<int>(std::distance(mat_begin_ptr, min_ptr));
                auto min_index_x = min_index % 3;
                auto min_index_y = min_index / 3;

                auto centroid_x = x_center - 1 + min_index_x;
                auto centroid_y = y_center - 1 + min_index_y;

                // Add the centroid to the list
                centroids.push_back({cv::Vec2f(centroid_x, centroid_y), img.at<cv::Vec3b>(centroid_y, centroid_y)});
            }
        }

        int iterations = 0;
        bool centroids_moved = true;
        while (centroids_moved && max_iterations < 1000)
        {
            centroids_moved = false;

            // Used to set the distances to float max (reset distances)
            bool first_centroid_iteration = true;

            // Step 1: Assign each pixel to its closest centroid
            // For every center search in -S to +S (== 2S x 2S) square and assign the pixel to the closest centroid
            for (int i = 0; const auto &centroid : centroids)
            {
                const auto &centroid_x = centroid.first[0];
                const auto &centroid_y = centroid.first[1];
                const auto &centroid_color = centroid.second;

                auto y_max = std::min(static_cast<int>(centroid_y + S), h);
                for (int y = std::max(static_cast<int>(centroid_y - S), 0); y < y_max; ++y)
                {
                    auto x_max = std::min(static_cast<int>(centroid_x + S), w);
                    for (int x = std::max(static_cast<int>(centroid_x - S), 0); x < x_max; ++x)
                    {
                        auto &distance = img_slic_indices.at<cv::Vec2f>(y, x)[0];
                        auto &centroid_idx = img_slic_indices.at<cv::Vec2f>(y, x)[1];
                        const auto &pixel_color = img_slic.at<cv::Vec3b>(y, x);

                        // Zero the distance for first centroid
                        if (first_centroid_iteration)
                        {
                            distance = std::numeric_limits<float>::max();
                        }

                        auto new_xy_distance = ano::EuclidianDistance(centroid_x, x, centroid_y, y);
                        auto new_color_distance = ano::DiffColor(centroid_color, pixel_color);
                        auto new_distance = new_color_distance + color_space_distance_balance * new_xy_distance;

                        if (new_distance < distance)
                        {
                            distance = new_distance;
                            centroid_idx = i;
                        }
                    }
                }

                // Increment centroid index
                i++;
            }

            // Step 2: Update centroids

            // Go through all centroids and calculate their new position as centers of their assigned pixels
            auto size = centroids.size();
            if (size > std::numeric_limits<double>::max())
            {
                throw std::overflow_error("size is larger than DOUBLE_MAX");
            }
            auto centroids_count = static_cast<double>(size);

            std::vector<int> counts = std::vector<int>(centroids.size(), 0);
            std::vector<cv::Vec2d> sums_x_y = std::vector<cv::Vec2d>({centroids_count, cv::Vec2d(0.0, 0.0)});
            for (int y = 0; y < img_slic_indices.size[0]; ++y)
            {
                for (int x = 0; x < img_slic_indices.size[1]; ++x)
                {
                    auto centroid_idx = img_slic_indices.at<cv::Vec2f>(y, x)[1];

                    // Add the pixel to the sum of its coresponding centroid
                    sums_x_y[centroid_idx][0] += x;
                    sums_x_y[centroid_idx][1] += y;
                    counts[centroid_idx]++;
                }
            }

            // Divide all sums by their pixel counts
            for (int i = 0; i < centroids.size(); ++i)
            {
                auto &centroid_x = centroids[i].first[0];
                auto &centroid_y = centroids[i].first[1];

                auto new_centroid_x = static_cast<float>(sums_x_y[i][0]) / counts[i];
                auto new_centroid_y = static_cast<float>(sums_x_y[i][1]) / counts[i];

                // Check if the centroid has moved more than a threshold
                if (std::abs(new_centroid_x - centroid_x) > SLIC_CENTEROIDS_MOVED_THRESHOLD ||
                    std::abs(new_centroid_y - centroid_y) > SLIC_CENTEROIDS_MOVED_THRESHOLD)
                {
                    centroids_moved = true;
                }

                // Update the centroid centers
                centroid_x = new_centroid_x;
                centroid_y = new_centroid_y;
            }

            // Increment iteration counter
            iterations++;
        }

        // Step 3: Draw centroids:
        for (int i = 0; i < centroids.size(); ++i)
        {
            const auto &centroid_x = centroids[i].first[0];
            const auto &centroid_y = centroids[i].first[1];

            cv::circle(img_slic, cv::Point(centroid_x, centroid_y), 2, SLIC_CENTROID_COLOR, cv::FILLED);
        }

        return img_slic;
    }
}
