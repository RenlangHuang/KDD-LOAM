#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include <Eigen/Core>

#include <tbb/parallel_for.h>
#include <tsl/robin_map.h>
#include <sophus/se3.hpp>


using Voxel = Eigen::Vector3i;
struct VoxelHash {
    size_t operator()(const Voxel &voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
    }
};


std::vector<Eigen::Vector3d> VoxelDownsample(const std::vector<Eigen::Vector3d> &frame, double voxel_size) {
    tsl::robin_map<Voxel, Eigen::Vector3d, VoxelHash> grid;
    grid.reserve(frame.size());
    for (const auto &point : frame) {
        const auto voxel = Voxel((point / voxel_size).cast<int>());
        if (grid.contains(voxel)) continue;
        grid.insert({voxel, point});
    }
    std::vector<Eigen::Vector3d> frame_downsampled;
    frame_downsampled.reserve(grid.size());
    for (auto it = grid.begin(); it != grid.end(); ++it) {
        frame_downsampled.emplace_back(it->second);
    }
    return frame_downsampled;
}

std::vector<Eigen::Vector3d> Preprocess(const std::vector<Eigen::Vector3d> &frame,
                                        double max_range,
                                        double min_range) {
    std::vector<Eigen::Vector3d> inliers;
    std::copy_if(frame.cbegin(), frame.cend(), std::back_inserter(inliers), [&](const auto &pt) {
        const double norm = pt.norm();
        return norm < max_range && norm > min_range;
    });
    return inliers;
}