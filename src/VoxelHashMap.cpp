#include "kddloam_velodyne/VoxelHashMap.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

struct ResultTuple {
    ResultTuple(std::size_t n) {
        source.reserve(n);
        target.reserve(n);
    }
    std::vector<Eigen::Vector3d> source;
    std::vector<Eigen::Vector3d> target;
};


VoxelHashMap::Vector3dVectorNormalTuple VoxelHashMap::GetCorrespondences(
    const Vector3dVector &points, double max_correspondence_distance) const {
    // Lambda Function to obtain the KNN of one point, maybe refactor
    auto GetClosestNeighbor = [&](const Eigen::Vector3d &point) {
        auto kx = static_cast<int>(point[0] / voxel_size_);
        auto ky = static_cast<int>(point[1] / voxel_size_);
        auto kz = static_cast<int>(point[2] / voxel_size_);
        std::vector<Voxel> voxels;
        voxels.reserve(27);
        for (int i = kx - 1; i < kx + 1 + 1; ++i) {
            for (int j = ky - 1; j < ky + 1 + 1; ++j) {
                for (int k = kz - 1; k < kz + 1 + 1; ++k) {
                    voxels.emplace_back(i, j, k);
                }
            }
        }

        using Vector3dVector = std::vector<Eigen::Vector3d>;
        std::vector<bool> is_surfel;
        Vector3dVector neighbors, normals;
        normals.reserve(27 * max_points_per_voxel_);
        is_surfel.reserve(27 * max_points_per_voxel_);
        neighbors.reserve(27 * max_points_per_voxel_);
        std::for_each(voxels.cbegin(), voxels.cend(), [&](const auto &voxel) {
            auto search = map_.find(voxel);
            if (search != map_.end()) {
                const auto &points = search->second.points;
                if (!points.empty()) {
                    for (const auto &point : points) {
                        neighbors.emplace_back(point);
                        is_surfel.push_back((points.size() == 1));
                        normals.emplace_back(search->second.normal_);
                    }
                }
            }
        });

        Eigen::Vector3d closest_neighbor, normal{0, 0, 0};
        Eigen::Vector3d closest_point{0, 0, 0}, closest_surfel{0, 0, 0};
        double closest_point_distance = std::numeric_limits<double>::max();
        double closest_surfel_distance = std::numeric_limits<double>::max();
        for (size_t i = 0; i < neighbors.size(); ++i) {
            double distance = (neighbors[i] - point).squaredNorm();
            if (!is_surfel[i]) {
                if (distance < closest_point_distance) {
                    closest_point_distance = distance;
                    closest_point = neighbors[i];
                }
            }
            else {
                if (distance < closest_surfel_distance) {
                    closest_surfel_distance = distance;
                    closest_surfel = neighbors[i];
                    normal = normals[i];
                }
            }
        }
        double closest_distance = fabs(normal.dot(closest_surfel - point));
        closest_point_distance = sqrt(closest_point_distance);
        if (closest_distance < closest_point_distance && \
            closest_surfel_distance / 1.5 < closest_point_distance) {
            closest_neighbor = closest_surfel;
        } else {
            closest_distance = closest_point_distance;
            closest_neighbor = closest_point;
            normal << 0, 0, 0;
        }
        return std::make_tuple(closest_neighbor, normal, closest_distance);
    };
    
    using points_iterator = std::vector<Eigen::Vector3d>::const_iterator;
    const auto [source, target, normal] = tbb::parallel_reduce(
        tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()},
        PointNormalTuple(points.size()),
        // 1st lambda: Parallel computation
        [max_correspondence_distance, &GetClosestNeighbor](
            const tbb::blocked_range<points_iterator> &r, PointNormalTuple res) -> PointNormalTuple {
            auto &src = res.source, &tgt = res.target, &norm = res.normal;
            src.reserve(r.size());
            tgt.reserve(r.size());
            norm.reserve(r.size());
            for (const auto &point : r) {
                auto closest_neighbor = GetClosestNeighbor(point);
                if ((std::get<2>(closest_neighbor) < max_correspondence_distance)) {
                    src.emplace_back(point);
                    tgt.emplace_back(std::get<0>(closest_neighbor));
                    norm.emplace_back(std::get<1>(closest_neighbor));
                }
            }
            return res;
        },
        // 2nd lambda: Parallel reduction
        [](PointNormalTuple a, const PointNormalTuple &b) -> PointNormalTuple {
            auto &[src, tgt, norm] = a;
            const auto &[srcp, tgtp, normp] = b;
            src.insert(src.end(), std::make_move_iterator(srcp.begin()), std::make_move_iterator(srcp.end()));
            tgt.insert(tgt.end(), std::make_move_iterator(tgtp.begin()), std::make_move_iterator(tgtp.end()));
            norm.insert(norm.end(), std::make_move_iterator(normp.begin()), std::make_move_iterator(normp.end()));
            return a;
        });
        return std::make_tuple(source, target, normal);
}

std::vector<Eigen::Vector3d> VoxelHashMap::Pointcloud() const {
    std::vector<Eigen::Vector3d> points;
    points.reserve(max_points_per_voxel_ * map_.size());
    for (const auto &[voxel, voxel_block] : map_) {
        (void)voxel;
        for (const auto &point : voxel_block.points) {
            points.push_back(point);
        }
    }
    return points;
}

void VoxelHashMap::Update(const Vector3dVector &points, const std::vector<float> &score, const Eigen::Vector3d &origin) {
    AddPoints(points, score);
    RemovePointsFarFromLocation(origin);
}

void VoxelHashMap::Update(const Vector3dVector &points, const std::vector<float> &score, const Sophus::SE3d &pose) {
    Vector3dVector points_transformed(points.size());
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                   [&](const auto &point) { return pose * point; });
    const Eigen::Vector3d &origin = pose.translation();
    Update(points_transformed, score, origin);
}

void VoxelHashMap::AddPoints(const std::vector<Eigen::Vector3d> &points, const std::vector<float> &score) {
    for (size_t i = 0; i < points.size(); ++i) {
        if (score[i] > 1.08f) continue;
        auto &point = points[i];
        auto voxel = Voxel((point / voxel_size_).template cast<int>());
        auto search = map_.find(voxel);
        if (search != map_.end()) {
            auto &voxel_block = search.value();
            voxel_block.AddPoint(point, score[i]);
            if (!voxel_block.fitted && voxel_block.num_points_ <= (int)voxel_block.points.size())
                voxel_block.VoxelFilter(voxel_size_ / (double)voxel_block.num_points_ * 2.0);
            if (!voxel_block.fitted && voxel_block.num_points_ <= (int)voxel_block.points.size()) {
                voxel_block.fitted = voxel_block.FitPlane(voxel_size_ / (double)voxel_block.num_points_);
                if (voxel_block.fitted) {
                    Eigen::Vector3d centroid(voxel_block.points[0]);
                    for (size_t k = 1; k < voxel_block.points.size(); ++k)
                        centroid = centroid + voxel_block.points[k];
                    centroid = centroid / (double)voxel_block.points.size();
                    double min_dist2 = 1e8; size_t min_idx = 0;
                    for (size_t k = 0; k < voxel_block.points.size(); ++k) {
                        double dist2 = (centroid - voxel_block.points[k]).squaredNorm();
                        if (min_dist2 > dist2) {min_dist2 = dist2; min_idx = k;}
                    }
                    double saliency = voxel_block.saliency[min_idx];
                    Eigen::Vector3d pt(voxel_block.points[min_idx]);
                    voxel_block.points.clear();
                    voxel_block.points.emplace_back(pt);
                    voxel_block.saliency.clear();
                    voxel_block.saliency.push_back(saliency);
                    voxel_block.num_points_ = 1;
                }
            }
        } else if (score[i] < 1.02f) {
            map_.insert({voxel, VoxelBlock{max_points_per_voxel_, {point}, {score[i]}}});
        }
    }
    
    /*double resolution = voxel_size_ / (double)max_points_per_voxel_ * 2.0;

    for (size_t i = 0; i < points.size(); ++i) {
        auto &point = points[i];
        auto voxel = Voxel((point / voxel_size_).template cast<int>());
        auto search = map_.find(voxel);
        int max_num_pts = max_points_per_voxel_;
        if (score[i] > 1.13f) max_num_pts -= 5;
        if (score[i] < 0.90f) max_num_pts += 5;
        if (score[i] < 0.86f) max_num_pts += 5;
        if (search != map_.end()) {
            auto &voxel_block = search.value();
            voxel_block.AddPoint(point, score[i]);           
            if (voxel_block.num_points_ == (int)voxel_block.points.size())
                voxel_block.VoxelFilter(resolution);
            if (voxel_block.num_points_ < max_num_pts)
                voxel_block.num_points_ = max_num_pts;
        } else {
            map_.insert({voxel, VoxelBlock{max_num_pts, {point}, {score[i]}}});
        }
    }*/
}

void VoxelHashMap::RemovePointsFarFromLocation(const Eigen::Vector3d &origin) {
    for (const auto &[voxel, voxel_block] : map_) {
        const auto &pt = voxel_block.points.front();
        const auto max_distance2 = max_distance_ * max_distance_;
        if ((pt - origin).squaredNorm() > (max_distance2)) {
            map_.erase(voxel);
        }
    }
}