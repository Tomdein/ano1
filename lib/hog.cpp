#include "hog.hpp"
// Histogram of Oriented Gradients (HOG) descriptor

#include <vector>

namespace ano
{
    // Compute HOG
    // If cells_in_block_count * cell_size != image size -> the histogram is averaged from less pixels
    cv::Mat HoG(const cv::Mat &img, const int cells_in_block_count, const int cell_size, const int nbins)
    {
        int y_cell_count = static_cast<int>(std::ceil(img.size[0] / cell_size)); // How many cells in y dir
        int x_cell_count = static_cast<int>(std::ceil(img.size[1] / cell_size)); // How many cells in x dir

        int x_block_count = static_cast<int>(std::ceil(img.size[1] / (cells_in_block_count * cell_size))); // How many blocks in x dir

        cv::Mat histogram(y_cell_count, x_cell_count, CV_32FC(nbins)); // The final histogram

        int *histogram_block_count = new int[x_block_count];                          // n ints for every orientation + 1 for pixel count ()
        int *histogram_block_count_end = histogram_block_count + (x_block_count + 1); // Iterator end
        int *histogram_block_count_counts_it = histogram_block_count;                 // Iterator

        // TODO: int *histogram_counts = new int[y_cell_count * x_cell_count * nbins];
        // TODO: Single 2D array
        int *histogram_counts = new int[x_cell_count * nbins];                     // n ints for every orientation + 1 for pixel count ()
        int *histogram_counts_end = histogram_counts + (x_cell_count * nbins + 1); // Iterator end
        int *histogram_counts_it = histogram_counts;                               // Iterator

        for (int y = 0; y < y_cell_count; ++y)
        {
            // Outside of image in y dir -> calculate histograms and skip
            if (y >= img.size[0])
            {
                continue;
            }

            // Just got outside previous cell -> calculate histograms and zero the histogram counts
            if ((y + 1) % cell_size == 0)
            {
                // Normalize cells over single block
                // For every block
                for (int x = 0; x < x_block_count; ++x)
                {
                    histogram_counts_it = histogram_counts + x * nbins;

                    // Normalize cells over single block
                    for (int i = 0; i < nbins; ++i)
                    {
                        // Divide by the number of pixels in the block
                        histogram_counts_it[i] /= static_cast<float>(histogram_block_count[x]);

                        // Increment pointer to the beginning of the next histogram data
                        histogram_counts_it += nbins;
                    }
                }

                // Save the histograms

                // Zero the histogram counts
                histogram_counts_it = histogram_counts;
                for (; histogram_counts_it != histogram_counts_end; histogram_counts_it++)
                {
                    *histogram_counts_it = 0;
                }
            }

            // Calculate histograms for every pixel in cell

            for (int x = 0; x < x_cell_count; ++x)
            {
                // Outside of image in x dir -> skip
                if (x >= img.size[1])
                {
                    continue;
                }

                auto x_pixels_in_cell = cells_in_block_count * cell_size; // Number of pixels in one cell in x direction
                for (int i = 0; i < x_pixels_in_cell; ++i)
                {
                }

                // Sum histogram counts for every pixel
            }
        }

        delete[] (histogram_counts);

        return;
    }
}