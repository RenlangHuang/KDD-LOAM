#pragma once

#include <tuple>
#include <vector>
#include <Eigen/Core>

#include "kddloam_velodyne/AdaptiveThreshold.hpp"
#include "kddloam_velodyne/preprocessing.hpp"
#include "kddloam_velodyne/registration.hpp"
#include "kddloam_velodyne/VoxelHashMap.hpp"


struct ICPConfig {
    // map params
    double voxel_size = 1.0;
    double max_range = 100.0;
    double min_range = 5.0;
    int max_points_per_voxel = 20;

    // th parms
    double min_motion_th = 0.1;
    double initial_threshold = 2.0;

    // Motion compensation
    bool deskew = false;

    // Motion prediction via constant velocity model
    bool const_vel_pred = true;
};

class HybridICP {
public:
    using Vector3dVector = std::vector<Eigen::Vector3d>;
    using Vector3dVectorTuple = std::tuple<Vector3dVector, Vector3dVector>;

public:
    explicit HybridICP(const ICPConfig &config)
        : config_(config),
          local_map_(config.voxel_size, config.max_range, config.max_points_per_voxel),
          adaptive_threshold_(config.initial_threshold, config.min_motion_th, config.max_range) {}

    HybridICP() : HybridICP(ICPConfig{}) {}

public:
    double GetAdaptiveThreshold();
    Sophus::SE3d GetPredictionModel() const;
    bool HasMoved();

public:
    std::vector<Eigen::Vector3d> LocalMap() const { return local_map_.Pointcloud(); };
    std::vector<Sophus::SE3d> poses() const { return poses_; };

public:
    std::vector<Sophus::SE3d> poses_;
    ICPConfig config_;
    VoxelHashMap local_map_;
    AdaptiveThreshold adaptive_threshold_;
};


double HybridICP::GetAdaptiveThreshold() {
    if (!HasMoved()) return config_.initial_threshold;
    return adaptive_threshold_.ComputeThreshold();
}

Sophus::SE3d HybridICP::GetPredictionModel() const {
    Sophus::SE3d pred = Sophus::SE3d();
    const size_t N = poses_.size();
    if (N < 2) return pred;
    return poses_[N - 2].inverse() * poses_[N - 1];
}

bool HybridICP::HasMoved() {
    if (poses_.empty()) return false;
    const double motion = (poses_.front().inverse() * poses_.back()).translation().norm();
    return motion > 5.0 * config_.min_motion_th;
}

std::vector<Eigen::Vector3d> CorrectKITTIScan(const std::vector<Eigen::Vector3d> &frame) {
    constexpr double VERTICAL_ANGLE_OFFSET = (0.205 * M_PI) / 180.0;
    std::vector<Eigen::Vector3d> corrected_frame(frame.size());
    tbb::parallel_for(size_t(0), frame.size(), [&](size_t i) {
        const auto &pt = frame[i];
        const Eigen::Vector3d rotationVector = pt.cross(Eigen::Vector3d(0., 0., 1.));
        corrected_frame[i] = Eigen::AngleAxisd(VERTICAL_ANGLE_OFFSET, rotationVector.normalized()) * pt;
    });
    return corrected_frame;
}