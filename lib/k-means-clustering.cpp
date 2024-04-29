#include "k-means-clustering.hpp"

#include <random>
#include <map>

#include "color-generator.hpp"

const static float range_from = 0.0f;
const static float range_to = 1.0f;
static std::random_device rand_dev;
static std::mt19937 generator(rand_dev());
static std::uniform_real_distribution<float> distr(range_from, range_to);

namespace ano
{
    ano::Ethalons EthalonsKMeansClustering(ano::DetectedObjectsVector &objects, int k, int num_features, int max_iterations)
    {
        Ethalons ethalons;

        // Stores centroids for each cluster
        std::vector<float> centroids[k];
        int closest_object_count[k];

        // Init ethalons randomly (starting from class 1)
        for (int i = 0; i < k; ++i)
        {
            auto features = std::vector<float>(num_features);
            for (int j = 0; j < num_features; ++j)
            {
                features[j] = distr(generator);
            }

            centroids[i] = features;
            // Start from class id == 1
            ethalons.AddEthalons(i + 1, std::move(features), ano::GenerateRandomColorBGR());
        }

        // Clustering
        for (int i = 0; i < max_iterations; ++i)
        {
            // Clear centroids
            for (auto &centroid : centroids)
            {
                centroid = std::vector<float>(num_features);
            }

            // Clear closest object counts
            for (auto &count : closest_object_count)
            {
                count = 0;
            }

            // For every object
            for (auto &object : objects)
            {
                // Find closest ethalon
                auto closest_ethalon_class = ethalons.FindClosestClass({object.features.F1, object.features.F2});

                // Assign object to closest ethalon
                object.id_class = closest_ethalon_class;

                // Add value to centroid
                centroids[object.id_class - 1][0] += object.features.F1;
                centroids[object.id_class - 1][1] += object.features.F2;
                closest_object_count[object.id_class - 1]++;
            }

            // std::for_each(objects.begin(), objects.end(), [&](ano::DetectedObject & object) -> void);

            bool centroid_moved = false;

            // Recalculate centroids
            for (auto &ethalon : ethalons)
            {
                auto ethalon_class = std::get<0>(ethalon);
                auto centroid = centroids[ethalon_class - 1];
                auto count = closest_object_count[ethalon_class - 1];

                // No objects assigned to this ethalon
                if (count == 0)
                {
                    std::cout << "No objects assigned to ethalon class: " << ethalon_class << std::endl;
                    return ethalons;
                }

                // Average the values
                centroid[0] /= count;
                centroid[1] /= count;

                std::vector<float> &features = std::get<1>(ethalon);

                // Check if centroid moved and update ethalons
                for (int j = 0; j < num_features; ++j)
                {
                    // Check if centroid moved
                    if (std::abs(centroid[j] - features[j]) > CLUSTERING_MIN_DISTANCE_MOVED)
                    {
                        centroid_moved = true;
                    }

                    // Update ethalon
                    features[j] = centroid[j];
                }
            }

            // No centroids moved -> stop clustering
            if (!centroid_moved)
            {
                break;
            }
        }

        return ethalons;
    }
}
