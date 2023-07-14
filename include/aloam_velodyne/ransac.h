#pragma once

#include <cmath>
#include <open3d/Open3D.h>

namespace open3d {
namespace pipelines {
namespace registration {

CorrespondenceSet LocalFeatureMatching(
    const geometry::KDTreeFlann &kdtree,
    const geometry::PointCloud &target,
    const Feature &source_features,
    const Feature &target_features,
    double max_correspondence_distance,
    const int num_similar_features,
    const double similarity_threshold,
    const bool reverse
);

RegistrationResult RANSACWithInitialPruning(
    const geometry::PointCloud &source,
    const geometry::PointCloud &target,
    const Feature &source_feature,
    const Feature &target_feature,
    const double max_pruning_distance, double max_correspondence_distance,
    int ransac_n, const int num_similar_features, const double similarity_threshold,
    const std::vector<std::reference_wrapper<const CorrespondenceChecker>> &checkers,
    const RANSACConvergenceCriteria &criteria
);

RegistrationResult RegistrationFromLocalToGlobal(
    const geometry::PointCloud &source,
    const geometry::PointCloud &target,
    const Feature &source_feature,
    const Feature &target_feature,
    double max_pruning_distance, double max_correspondence_distance,
    int num_similar_features, double similarity_threshold, double decay,
    const ICPConvergenceCriteria &criteria
);

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
