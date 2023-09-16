#pragma once

#include <tuple>
#include <vector>
#include <Eigen/Core>

#include "kddloam_velodyne/AdaptiveThreshold.hpp"
#include "kddloam_velodyne/preprocessing.hpp"
#include "kddloam_velodyne/registration.hpp"
#include "kddloam_velodyne/VoxelHashMap.hpp"


struct KddLOAMConfig {
    // map parameters
    double voxel_size = 1.0;
    double cube_size = 50.0;
    double max_range = 100.0;
    double min_range = 5.0;
    int max_points_per_voxel = 20;

    int cube_width = 21;
    int cube_height = 21;
    int cube_depth = 11;

    // threshold parameters
    double min_motion_th = 0.1;
    double initial_threshold = 2.0;

    // Motion compensation
    bool deskew = false;

    // Motion prediction via constant velocity model
    bool const_vel_pred = false;
};

class KddLOAM {
public:
    using Vector3dVector = std::vector<Eigen::Vector3d>;
    using Vector3dVectorTuple = std::tuple<Vector3dVector, Vector3dVector>;
    using VoxelHashMapPtr = std::shared_ptr<VoxelHashMap>;

public:
    explicit KddLOAM(const KddLOAMConfig &config)
        : config_(config), num_cubes_(config.cube_width * config.cube_height * config.cube_depth),
          adaptive_threshold_(config.initial_threshold, config.min_motion_th, config.max_range) {
            cube_map_.reserve(num_cubes_);
            for (int i = 0; i < num_cubes_; ++i) {
                VoxelHashMapPtr cube(new VoxelHashMap(config.voxel_size, config.max_range, config.max_points_per_voxel));
                cube_map_.push_back(cube);
            }
            laserCloudCenWidth = config_.cube_width / 2;
            laserCloudCenHeight = config_.cube_height / 2;
            laserCloudCenDepth = config_.cube_depth / 2;
          }

    KddLOAM() : KddLOAM(KddLOAMConfig{}) {}

public:
    Eigen::Vector3i FindCurrentCube(Eigen::Vector3d t_w_curr);
    Eigen::Vector3i TranslateTheMap(Eigen::Vector3d t_w_curr);
    VoxelHashMap::Vector3dVectorNormalTuple GetCorrespondences(
        const Vector3dVector &points, double max_correspondence_distance);
    Sophus::SE3d Registration(const Vector3dVector &frame, const Sophus::SE3d &initial_guess);
    void MapUpdate(Vector3dVector &points, const std::vector<float> &score, const Sophus::SE3d &pose);
    double GetAdaptiveThreshold();
    Sophus::SE3d GetPredictionModel() const;
    bool HasMoved();

public:
    std::vector<Sophus::SE3d> poses() const { return poses_; };
    Eigen::Vector3i GetCurrentCube() const {
        return Eigen::Vector3i(laserCloudCenWidth, laserCloudCenHeight, laserCloudCenDepth);
    };
    inline int _hash_cube(int i, int j, int k) {
        return i + config_.cube_width * j + config_.cube_width * config_.cube_height * k;
    };
    inline int _hash_cube(Eigen::Vector3i idx) {
        return idx[0] + config_.cube_width * idx[1] + config_.cube_width * config_.cube_height * idx[2];
    };

public:
    int num_cubes_;
    KddLOAMConfig config_;
    std::vector<Sophus::SE3d> poses_;
    std::vector<VoxelHashMapPtr> cube_map_;
    AdaptiveThreshold adaptive_threshold_;

private:
    int laserCloudCenWidth;
    int laserCloudCenHeight;
    int laserCloudCenDepth;
};

double KddLOAM::GetAdaptiveThreshold() {
    if (!HasMoved()) return config_.initial_threshold;
    return adaptive_threshold_.ComputeThreshold();
}

Sophus::SE3d KddLOAM::GetPredictionModel() const {
    Sophus::SE3d pred = Sophus::SE3d();
    const size_t N = poses_.size();
    if (N < 2) return pred;
    return poses_[N - 2].inverse() * poses_[N - 1];
}

bool KddLOAM::HasMoved() {
    if (poses_.empty()) return false;
    const double motion = (poses_.front().inverse() * poses_.back()).translation().norm();
    return motion > 5.0 * config_.min_motion_th;
}

Eigen::Vector3i KddLOAM::FindCurrentCube(Eigen::Vector3d t_w_curr) {
    double half_size = config_.cube_size / 2.0;
	int centerCubeI = int((t_w_curr.x() + half_size) / config_.cube_size) + laserCloudCenWidth;
	int centerCubeJ = int((t_w_curr.y() + half_size) / config_.cube_size) + laserCloudCenHeight;
	int centerCubeK = int((t_w_curr.z() + half_size) / config_.cube_size) + laserCloudCenDepth;

	if (t_w_curr.x() + half_size < 0) centerCubeI--;
	if (t_w_curr.y() + half_size < 0) centerCubeJ--;
	if (t_w_curr.z() + half_size < 0) centerCubeK--;

    return Eigen::Vector3i(centerCubeI, centerCubeJ, centerCubeK);
}


VoxelHashMap::Vector3dVectorNormalTuple KddLOAM::GetCorrespondences(
    const Vector3dVector &points, double max_correspondence_distance) {
    // Lambda Function to obtain the KNN of one point, maybe refactor
    auto GetClosestNeighbor = [&](const Eigen::Vector3d &point) {
        Eigen::Vector3i idx = FindCurrentCube(point);
        auto &map_ = cube_map_[_hash_cube(idx)]->map_;

        auto kx = static_cast<int>(point[0] / config_.voxel_size);
        auto ky = static_cast<int>(point[1] / config_.voxel_size);
        auto kz = static_cast<int>(point[2] / config_.voxel_size);
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
        normals.reserve(27 * config_.max_points_per_voxel);
        is_surfel.reserve(27 * config_.max_points_per_voxel);
        neighbors.reserve(27 * config_.max_points_per_voxel);
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


Sophus::SE3d KddLOAM::Registration(const KddLOAM::Vector3dVector &frame, const Sophus::SE3d &initial_guess) {
    const int MAX_NUM_ITERATIONS_ = 500;
    const double ESTIMATION_THRESHOLD_ = 0.0001;
    Eigen::Vector3i idx = TranslateTheMap(Eigen::Vector3d(initial_guess.translation()));
    printf("Successfully translate the map (%d, %d, %d)\n", idx[0], idx[1], idx[2]);
    bool not_empty = false;
    for (int i = idx[0] - 2; i <= idx[0] + 2; i++) {
		for (int j = idx[1] - 2; j <= idx[1] + 2; j++) {
			for (int k = idx[2] - 1; k <= idx[2] + 1; k++) {
				if (i >= 0 && i < config_.cube_width && \
                    j >= 0 && j < config_.cube_height && \
                    k >= 0 && k < config_.cube_depth)
				{
                    not_empty = (not_empty || cube_map_[_hash_cube(i,j,k)]->Empty());
				}
			}
		}
	}
    if (!not_empty) return initial_guess;
    
    double sigma = GetAdaptiveThreshold();
    double max_distance_threshold = 3.0 * sigma, kernel = sigma / 3.0;

    std::vector<Eigen::Vector3d> source = frame;
    TransformPoints(initial_guess, source);

    Sophus::SE3d T_icp = Sophus::SE3d();
    for (int j = 0; j < MAX_NUM_ITERATIONS_; ++j) {
        const auto &pair = GetCorrespondences(source, max_distance_threshold);
        const auto &jacobians = HybridMetric(std::get<0>(pair), std::get<1>(pair), std::get<2>(pair), kernel);
        const Eigen::Vector6d x = jacobians.JTJ.ldlt().solve(-jacobians.JTr);
        Sophus::SE3d estimation =  Sophus::SE3d::exp(x);

        TransformPoints(estimation, source);
        T_icp = estimation * T_icp;
        if (estimation.log().norm() < ESTIMATION_THRESHOLD_) break;
    }
    return T_icp * initial_guess;
}


void KddLOAM::MapUpdate(Vector3dVector &points, const std::vector<float> &score, const Sophus::SE3d &pose) {
    auto t_w_curr = pose.translation();
    auto q_w_curr = pose.unit_quaternion();
    for (size_t i = 0; i < points.size(); ++i) {
        points[i] = q_w_curr * points[i] + t_w_curr;
        if (score[i] > 1.08f) continue;
        int idx = _hash_cube(FindCurrentCube(points[i]));
        if (idx < 0 || idx >= num_cubes_) continue;
        auto &map_ = cube_map_[idx]->map_;
        
        auto voxel = Voxel((points[i] / config_.voxel_size).template cast<int>());
        auto search = map_.find(voxel);
        if (search != map_.end()) {
            auto &voxel_block = search.value();
            voxel_block.AddPoint(points[i], score[i]);
            if (!voxel_block.fitted && voxel_block.num_points_ <= (int)voxel_block.points.size())
                voxel_block.VoxelFilter(config_.voxel_size / (double)voxel_block.num_points_ * 2.0);
            if (!voxel_block.fitted && voxel_block.num_points_ <= (int)voxel_block.points.size()) {
                voxel_block.fitted = voxel_block.FitPlane(config_.voxel_size / (double)voxel_block.num_points_);
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
            map_.insert({voxel, VoxelHashMap::VoxelBlock{config_.max_points_per_voxel, {points[i]}, {score[i]}}});
        }
    }
}


Eigen::Vector3i KddLOAM::TranslateTheMap(Eigen::Vector3d t_w_curr) {
    Eigen::Vector3i idx = FindCurrentCube(t_w_curr);
    while (idx[0] < 3) {
		for (int j = 0; j < config_.cube_height; j++) {
			for (int k = 0; k < config_.cube_depth; k++) {
				int i = config_.cube_width - 1;
				VoxelHashMapPtr ptr = cube_map_[_hash_cube(i, j, k)];
				for (; i >= 1; i--) {
                    cube_map_[_hash_cube(i, j, k)] = cube_map_[_hash_cube(i-1, j, k)];
                }
				cube_map_[_hash_cube(i, j, k)] = ptr;
				ptr->Clear();
			}
		}
		idx[0]++; laserCloudCenWidth++;
	}

    while (idx[0] >= config_.cube_width - 3) {
		for (int j = 0; j < config_.cube_height; j++) {
			for (int k = 0; k < config_.cube_depth; k++) {
				int i = 0;
				VoxelHashMapPtr ptr = cube_map_[_hash_cube(i, j, k)];
				for (; i < config_.cube_width - 1; i++) {
					cube_map_[_hash_cube(i, j, k)] = cube_map_[_hash_cube(i+1, j, k)];
				}
				cube_map_[_hash_cube(i, j, k)] = ptr;
				ptr->Clear();
			}
		}
		idx[0]--; laserCloudCenWidth--;
	}

    while (idx[1] < 3) {
		for (int i = 0; i < config_.cube_width; i++) {
			for (int k = 0; k < config_.cube_depth; k++) {
				int j = config_.cube_height - 1;
				VoxelHashMapPtr ptr = cube_map_[_hash_cube(i, j, k)];
				for (; j >= 1; j--) {
					cube_map_[_hash_cube(i, j, k)] = cube_map_[_hash_cube(i, j-1, k)];
				}
				cube_map_[_hash_cube(i, j, k)] = ptr;
				ptr->Clear();
			}
		}
		idx[1]++; laserCloudCenHeight++;
	}

    while (idx[1] >= config_.cube_height - 3) {
		for (int i = 0; i < config_.cube_width; i++) {
			for (int k = 0; k < config_.cube_depth; k++) {
				int j = 0;
				VoxelHashMapPtr ptr = cube_map_[_hash_cube(i, j, k)];
				for (; j < config_.cube_height - 1; j++) {
					cube_map_[_hash_cube(i, j, k)] = cube_map_[_hash_cube(i, j+1, k)];
				}
				cube_map_[_hash_cube(i, j, k)] = ptr;
				ptr->Clear();
			}
		}
		idx[1]--; laserCloudCenHeight--;
	}

    while (idx[2] < 3) {
		for (int i = 0; i < config_.cube_width; i++) {
			for (int j = 0; j < config_.cube_height; j++) {
				int k = config_.cube_depth - 1;
				VoxelHashMapPtr ptr = cube_map_[_hash_cube(i, j, k)];
				for (; k >= 1; k--) {
					cube_map_[_hash_cube(i, j, k)] = cube_map_[_hash_cube(i, j, k-1)];
				}
				cube_map_[_hash_cube(i, j, k)] = ptr;
				ptr->Clear();
			}
		}
		idx[2]++; laserCloudCenDepth++;
	}

    while (idx[2] >= config_.cube_depth - 3) {
		for (int i = 0; i < config_.cube_width; i++) {
			for (int j = 0; j < config_.cube_height; j++) {
				int k = 0;
				VoxelHashMapPtr ptr = cube_map_[_hash_cube(i, j, k)];
				for (; k < config_.cube_depth - 1; k++) {
					cube_map_[_hash_cube(i, j, k)] = cube_map_[_hash_cube(i, j, k+1)];
				}
				cube_map_[_hash_cube(i, j, k)] = ptr;
				ptr->Clear();
			}
		}
		idx[2]--; laserCloudCenDepth--;
	}
    return idx;
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