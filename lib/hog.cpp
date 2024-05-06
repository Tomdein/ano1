#include "hog.hpp"
// Histogram of Oriented Gradients (HOG) descriptor

#include <vector>
#include <cmath>

namespace ano
{
    // Compute HOG
    // If cells_in_block_count * cell_size != image size -> the histogram is averaged from less pixels
    cv::Mat HoG(const cv::Mat &img_gradients, const int cells_in_block_count, const int cell_size, const int nbins)
    {
        int y_cell_count = static_cast<int>(std::ceil(img_gradients.size[0] / static_cast<float>(cell_size))); // How many cells in y dir
        int x_cell_count = static_cast<int>(std::ceil(img_gradients.size[1] / static_cast<float>(cell_size))); // How many cells in x dir

        int y_block_count = static_cast<int>(std::ceil(img_gradients.size[0] / static_cast<float>(cells_in_block_count * cell_size))); // How many blocks in x dir
        int x_block_count = static_cast<int>(std::ceil(img_gradients.size[1] / static_cast<float>(cells_in_block_count * cell_size))); // How many blocks in x dir

        int block_size_pixels = cells_in_block_count * cell_size; // Size of block in pixels

        // 1. Add gradient magnitudes to cell's histogram and Sum all gradients in each block

        // Gradient histogram for every cell
        float *histogram = new float[y_cell_count * x_cell_count * nbins];            // n ints for every orientation + 1 for pixel count ()
        float *histogram_end = histogram + (y_cell_count * x_cell_count * nbins + 1); // Iterator end
        float *histogram_it = histogram;                                              // Iterator

        // Zero the histogram counts
        memset(histogram, 0, y_cell_count * x_cell_count * nbins);

        // Sum of all bins in block
        float *histogram_block_sum = new float[y_block_count * x_block_count]; // n floats for every orientation + 1 for pixel count ()

        // Zero the histogram counts
        memset(histogram_block_sum, 0, y_block_count * x_block_count);

        // Angle difference between bins

        const float bin_delta = 2.0f * M_PIf / nbins;

        // For every pixel in y direction
        for (int y = 0; y < img_gradients.size[0]; y++)
        {
            // Current cell in y dir
            int y_cell = y / cell_size;
            int y_block = y / block_size_pixels;

            // For every pixel in x direction
            for (int x = 0; x < img_gradients.size[1]; x++)
            {
                // Current cell in x dir
                int x_cell = x / cell_size;
                int x_block = x / block_size_pixels;

                // Move pointer to current cell's histogram counts
                histogram_it = histogram + ((y_cell * x_cell_count + x_cell) * nbins);

                // Calculate bin
                auto pixel = img_gradients.at<cv::Vec2f>(y, x);
                auto pixel_angle = std::min(std::max(pixel[0], -M_PIf + 1e-4f), M_PIf - 1e-4f); // Exclude -PI and +PI -> due to finding the bins

                int bin = static_cast<int>(std::floor((pixel_angle + M_PIf) / bin_delta)); // Move from [-PI, PI] to [0, 2*PI]

                assert(bin >= 0 && bin < nbins);

                // Add gradient magnitude to bin histogram
                *(histogram_it + bin) += pixel[1];

                // Sum gradient magnitudes in block
                histogram_block_sum[y_block * x_block_count + x_block] += pixel[1];
            }
        }

        // 2. Normalize the cell's histogram by block's gradient sum

        // For every cell in y direction
        for (int y_cell = 0; y_cell < y_cell_count; y_cell++)
        {
            // Current cell in y dir
            int y_block = y_cell / cells_in_block_count;

            // For every cell in x direction
            for (int x_cell = 0; x_cell < x_cell_count; x_cell++)
            {
                // Current cell in x direction
                int x_block = x_cell / cells_in_block_count;

                // Move pointer to current cell's histogram counts
                histogram_it = histogram + ((y_cell * x_cell_count + x_cell) * nbins);

                // Normalize cell's histogram by block's gradient sum
                for (int i = 0; i < nbins; i++)
                {
                    histogram_it[i] /= histogram_block_sum[y_block * x_block_count + x_block];
                }
            }
        }

        // Ctor with *data does not copy the data from pointer and DOES NOT DEALOCATE the data -> do a copy a be done with it
        cv::Mat img_histogram = cv::Mat(y_cell_count, x_cell_count, CV_32FC(nbins), histogram, cv::Mat::AUTO_STEP).clone(); // The final histogram

        delete[] (histogram_block_sum);
        delete[] (histogram);

        return img_histogram;
    }
#include "text.hpp"

    cv::Mat HoGVisualizeByAlpha(const cv::Mat &hog, const int cell_size, const int nbins, const float color_multiply)
    {
        auto dims = std::vector<int>{hog.size[0], hog.size[1] * nbins};
        auto hog_flat = hog.reshape(1, dims);

        cv::Mat visualization(cell_size * hog.size[0], cell_size * hog.size[1], CV_8UC1);

        auto cell_center = std::ceil(static_cast<float>(cell_size) / 2);
        auto line_lenght = (cell_size / 2) - 1;

        const float bin_delta = M_PIf / nbins;

        // For every y cell
        for (int y = 0; y < hog.size[0]; y++)
        {
            // For every x cell
            for (int x = 0; x < hog.size[1]; x++)
            {

                auto center_x = x * cell_size + cell_center;
                auto center_y = y * cell_size + cell_center;

                // For every bin in the cell
                for (int i = 0; i < nbins; i++)
                {
                    auto bin_angle = bin_delta * i - M_PIf; // [-PI, PI]

                    auto dx = std::cos(bin_angle) * line_lenght;
                    auto dy = -std::sin(bin_angle) * line_lenght; // y+ is downwards and x+ is right -> -sin(a)

                    auto alpha = hog_flat.at<float>(y, x * nbins + i);

                    // Draw the gradient orientation
                    cv::line(visualization,
                             cv::Point(center_x, center_y),
                             cv::Point(center_x + dx, center_y + dy),
                             cv::Vec3b(255, 255, 255) * alpha * color_multiply);
                }
            }
        }

        return visualization;
    }

    cv::Mat HoGVisualizeByLenght(const cv::Mat &hog, const int cell_size, const int nbins, const float length_multiply)
    {
        auto dims = std::vector<int>{hog.size[0], hog.size[1] * nbins};
        auto hog_flat = hog.reshape(1, dims);

        cv::Mat visualization(cell_size * hog.size[0], cell_size * hog.size[1], CV_8UC1);

        auto cell_center = std::ceil(static_cast<float>(cell_size) / 2);
        auto line_lenght = (cell_size / 2) - 1;

        const float bin_delta = M_PIf / nbins;

        // For every y cell
        for (int y = 0; y < hog.size[0]; y++)
        {
            // For every x cell
            for (int x = 0; x < hog.size[1]; x++)
            {

                auto center_x = x * cell_size + cell_center;
                auto center_y = y * cell_size + cell_center;

                // For every bin in the cell
                for (int i = 0; i < nbins; i++)
                {
                    auto bin_angle = bin_delta * i - M_PIf; // [-PI, PI]

                    auto alpha = hog_flat.at<float>(y, x * nbins + i);

                    auto dx = std::cos(bin_angle) * line_lenght * alpha * length_multiply;
                    auto dy = -std::sin(bin_angle) * line_lenght * alpha * length_multiply; // y+ is downwards and x+ is right -> -sin(a)

                    // Draw the gradient orientation
                    cv::line(visualization,
                             cv::Point(center_x, center_y),
                             cv::Point(center_x + dx, center_y + dy),
                             cv::Vec3b(255, 255, 255));
                }
            }
        }

        return visualization;
    }
}