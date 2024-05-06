#include "hog.hpp"
// Histogram of Oriented Gradients (HOG) descriptor

#include <vector>

namespace ano
{
    // Compute HOG
    // If cells_in_block_count * cell_size != image size -> the histogram is averaged from less pixels
    cv::Mat HoG(const cv::Mat &img_gradients, const int cells_in_block_count, const int cell_size, const int nbins)
    {
        int y_cell_count = static_cast<int>(std::ceil(img_gradients.size[0] / cell_size)); // How many cells in y dir
        int x_cell_count = static_cast<int>(std::ceil(img_gradients.size[1] / cell_size)); // How many cells in x dir

        int y_block_count = static_cast<int>(std::ceil(img_gradients.size[0] / (cells_in_block_count * cell_size))); // How many blocks in x dir
        int x_block_count = static_cast<int>(std::ceil(img_gradients.size[1] / (cells_in_block_count * cell_size))); // How many blocks in x dir

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
        const float bin_delta = 360.0f / nbins;

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
                histogram_it = histogram + (y_cell * x_cell_count + x_cell) * nbins;

                // Calculate bin
                auto pixel = img_gradients.at<cv::Vec2f>(y, x);
                int bin = static_cast<int>(std::floor(pixel[0] / bin_delta));

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
                histogram_it = histogram + (y_cell * x_cell_count + x_cell) * nbins;

                // Normalize cell's histogram by block's gradient sum
                for (int i = 0; i < nbins; i++)
                {
                    histogram_it[i] /= histogram_block_sum[y_block * x_block_count + x_block];
                }
            }
        }

        // Ctor with *data does not copy the data from pointer and DOES NOT DEALOCATE the data -> do a copy a be done with it
        cv::Mat img_histogram = cv::Mat(y_cell_count, x_cell_count, CV_32FC(nbins), histogram, cv::Mat::CONTINUOUS_FLAG).clone(); // The final histogram

        delete[] (histogram_block_sum);
        delete[] (histogram);

        return img_histogram;
    }
}