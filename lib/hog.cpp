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

        cv::Mat img_histogram(y_cell_count, x_cell_count, CV_32FC(nbins)); // The final histogram

        // 1. Add gradient to cell's histogram

        // Counts histogram occurances
        int *histogram_counts = new int[y_cell_count * x_cell_count * nbins];                     // n ints for every orientation + 1 for pixel count ()
        int *histogram_counts_end = histogram_counts + (y_cell_count * x_cell_count * nbins + 1); // Iterator end
        int *histogram_counts_it = histogram_counts;                                              // Iterator

        // Zero the histogram counts
        memset(histogram_counts, 0, y_cell_count * x_cell_count * nbins);

        // Angle difference between bins
        const float bin_delta = 360.0f / nbins;

        // For every pixel in y direction
        for (int y = 0; y < img_gradients.size[0]; y++)
        {
            // Current cell in y dir
            int y_cell = y / cell_size;

            // For every pixel in x direction
            for (int x = 0; x < img_gradients.size[1]; x++)
            {
                // Current cell in x dir
                int x_cell = x / cell_size;

                // Move pointer to current cell's histogram counts
                histogram_counts_it = histogram_counts + (y_cell * x_cell_count + x_cell) * nbins;

                // Calculate bin
                auto pixel = img_gradients.at<cv::Vec2f>(y, x);
                int bin = static_cast<int>(std::floor(pixel[0] / bin_delta));

                assert(bin >= 0 && bin < nbins);

                // Add +1 to bin histogram
                *(histogram_counts_it + bin) += 1;
            }
        }

        // 2. Sum all gradients in each block

        // Counts number of pixels in cell
        int *histogram_cell_pixels_count = new int[y_cell_count * x_cell_count];                                // n ints for every orientation + 1 for pixel count ()
        int *histogram_cell_pixels_count_end = histogram_cell_pixels_count + (y_cell_count * x_cell_count + 1); // Iterator end
        int *histogram_cell_pixels_count_counts_it = histogram_cell_pixels_count;                               // Iterator

        // Zero the pixel counts for cell
        memset(histogram_cell_pixels_count, 0, y_cell_count * x_cell_count);

        // Counts histogram occurances
        float *histogram_block_pixels_count = new float[y_block_count * x_block_count];                               // n floats for every orientation + 1 for pixel count ()
        float *histogram_block_pixels_count_end = histogram_block_pixels_count + (y_block_count * x_block_count + 1); // Iterator end
        float *histogram_block_pixels_count_it = histogram_block_pixels_count;                                        // Iterator

        // Zero the histogram counts
        memset(histogram_block_pixels_count, 0, y_block_count * x_block_count);

        for (int y = 0; y < y_cell_count; y++)
        {
            // Current cell in y dir
            int y_block = y / cells_in_block_count;

            for (int x = 0; x < x_cell_count; x++)
            {
                int x_block = x / cells_in_block_count;
            }
        }

        // 3. Normalize the cell's histogram by block's gradient sum

        for (int y = 0; y < y_cell_count; y++)
        {
            // Current cell in y dir
            int y_block = y / cells_in_block_count;

            for (int x = 0; x < x_cell_count; x++)
            {
                int x_block = x / cells_in_block_count;
            }
        }

        delete[] (histogram_block_pixels_count);
        delete[] (histogram_counts);
        delete[] (histogram_cell_pixels_count);

        return;
    }
}