#pragma once

#include <tsl/robin_map.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>


struct VoxelHashMap {
    using Vector3dVector = std::vector<Eigen::Vector3d>;
    using Vector3dVectorTuple = std::tuple<Vector3dVector, Vector3dVector>;
    using Vector3dVectorNormalTuple = std::tuple<Vector3dVector, Vector3dVector, Vector3dVector>;
    using Voxel = Eigen::Vector3i;
    struct VoxelBlock {
        int num_points_;
        std::vector<Eigen::Vector3d> points;
        std::vector<float> saliency;
        Eigen::Vector3d normal_{0, 0, 0};
        bool fitted = false;
        inline void AddPoint(const Eigen::Vector3d &point, const float score) {
            if (points.size() < static_cast<size_t>(num_points_)) {
                points.push_back(point);
                saliency.push_back(score);
            }
        }
        inline void VoxelFilter(double voxel_size) {
            tsl::robin_map<Voxel, size_t, VoxelHash> grid;
            grid.reserve(points.size());
            std::vector<size_t> remove_indices;
            remove_indices.reserve(points.size());
            for (size_t i = 0; i < points.size(); ++i) {
                const auto voxel = Voxel((points[i] / voxel_size).cast<int>());
                if (grid.contains(voxel)) remove_indices.emplace_back(i);
                grid.insert({voxel, i});
            }
            for (size_t i = 0; i < remove_indices.size(); ++i) {
                points.erase(points.begin() + remove_indices[i] - i);
                saliency.erase(saliency.begin() + remove_indices[i] - i);
            }
        }
        inline bool FitPlane(const double error_limit) {
            Eigen::Map<Eigen::Matrix3Xd> mat3xd(points.data()->data(), 3, points.size());
            Eigen::MatrixXd mat = mat3xd.transpose();
            Eigen::MatrixXd vec = Eigen::MatrixXd::Ones(points.size(), 1);
            Eigen::Vector3d norm = mat.colPivHouseholderQr().solve(-vec);
			normal_ = norm.normalized();
            Eigen::Matrix3Xd normal(normal_);
            Eigen::MatrixXd err(mat * normal);
            err = err + vec / norm.norm();
            for (size_t i = 0; i < points.size(); i++) {
                if (fabs(err(i, 0)) > error_limit) return false;
            }
            return true;
        }
    };
    struct VoxelHash {
        size_t operator()(const Voxel &voxel) const {
            const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
            return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
        }
    };

    explicit VoxelHashMap(double voxel_size, double max_distance, int max_points_per_voxel)
        : voxel_size_(voxel_size),
          max_distance_(max_distance),
          max_points_per_voxel_(max_points_per_voxel) {}

    VoxelHashMap::Vector3dVectorNormalTuple GetCorrespondences(const Vector3dVector &points, double max_correspondence_distance) const;
    inline void Clear() { map_.clear(); }
    inline bool Empty() const { return map_.empty(); }
    void Update(const Vector3dVector &points, const std::vector<float> &score, const Eigen::Vector3d &origin);
    void Update(const Vector3dVector &points, const std::vector<float> &score, const Sophus::SE3d &pose);
    void AddPoints(const std::vector<Eigen::Vector3d> &points, const std::vector<float> &score);
    void RemovePointsFarFromLocation(const Eigen::Vector3d &origin);
    std::vector<Eigen::Vector3d> Pointcloud() const;

    double voxel_size_;
    double max_distance_;
    int max_points_per_voxel_;
    tsl::robin_map<Voxel, VoxelBlock, VoxelHash> map_;
};


struct PointNormalTuple {
    PointNormalTuple(std::size_t n) {
        source.reserve(n);
        target.reserve(n);
        normal.reserve(n);
    }
    std::vector<Eigen::Vector3d> source;
    std::vector<Eigen::Vector3d> target;
    std::vector<Eigen::Vector3d> normal;
};