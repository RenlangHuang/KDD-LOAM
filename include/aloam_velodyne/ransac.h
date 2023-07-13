#pragma once

#include <cmath>
#include <open3d/Open3D.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

using namespace open3d;


void convert_o3d_to_pcl_XYZ(
    const geometry::PointCloud &ply,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr pcd
);


pipelines::registration::RegistrationResult RANSACWithInitialPruning(
    const geometry::PointCloud &source,
    const geometry::PointCloud &target,
    const pipelines::registration::Feature &source_feature,
    const pipelines::registration::Feature &target_feature,
    double max_correspondence_distance,
    int ransac_n,
    const std::vector<std::reference_wrapper<const pipelines::registration::CorrespondenceChecker>> &checkers,
    const pipelines::registration::RANSACConvergenceCriteria &criteria
);


pipelines::registration::RegistrationResult RANSACFromLocalToGlobal(
    const geometry::PointCloud &source,
    const geometry::PointCloud &target,
    const pipelines::registration::Feature &source_feature,
    const pipelines::registration::Feature &target_feature,
    double max_correspondence_distance,
    const std::vector<std::reference_wrapper<const pipelines::registration::CorrespondenceChecker>> &checkers,
    const pipelines::registration::RANSACConvergenceCriteria &criteria
);


pipelines::registration::RegistrationResult RANSAC(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const pipelines::registration::Feature &source_feature,
        const pipelines::registration::Feature &target_feature,
        double max_correspondence_distance,
        const pipelines::registration::TransformationEstimation &estimation,
        int ransac_n,
        const std::vector<std::reference_wrapper<const pipelines::registration::CorrespondenceChecker>> &checkers,
        const pipelines::registration::RANSACConvergenceCriteria &criteria
);
