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


//updated Zhaoml 2023-07-15 11:57
int UniformRandInt(const int min, const int max);
static open3d::pipelines::registration::RegistrationResult GetRegistrationResultAndCorrespondences(
        const open3d::geometry::PointCloud &source,
        const open3d::geometry::PointCloud &target,
        const open3d::geometry::KDTreeFlann &target_kdtree,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transformation);
open3d::pipelines::registration::RegistrationResult ransac_based_on_given_model(
    const open3d::geometry::PointCloud &pc1,
    const open3d::geometry::PointCloud &pc2,
    const open3d::pipelines::registration::Feature &f1,
    const open3d::pipelines::registration::Feature &f2,
    const Eigen::Matrix4d &transformation_matrix, //given model of odometry
    double max_correspondence_distance,
    const open3d::pipelines::registration::TransformationEstimation &estimation,
    int ransac_epoches,
    const open3d::pipelines::registration::RANSACConvergenceCriteria &criteria,
    const std::vector<std::reference_wrapper<const open3d::pipelines::registration::CorrespondenceChecker>> &checkers);
