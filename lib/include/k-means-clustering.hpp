#pragma once

#include "ethalons.hpp"
#include "detected-object.hpp"

namespace ano
{
#define CLUSTERING_MIN_DISTANCE_MOVED ((double)(1.0E-5))

    // K-Means Clustering.
    // k number of centroids (classes) to be found.
    // Returns k ethalons and assigns classes to all objects.
    ano::Ethalons EthalonsKMeansClustering(ano::DetectedObjectsVector &objects, int k, int num_features, int max_iterations = 1000);
}