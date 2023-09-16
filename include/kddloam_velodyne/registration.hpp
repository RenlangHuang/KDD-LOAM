#pragma once

#include <tuple>
#include <cmath>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include "VoxelHashMap.hpp"


namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
}

inline double square(double x) { return x * x; }

struct ResultTuple {
    ResultTuple() {
        JTJ.setZero();
        JTr.setZero();
    }

    ResultTuple operator+(const ResultTuple &other) {
        this->JTJ += other.JTJ;
        this->JTr += other.JTr;
        return *this;
    }

    Eigen::Matrix6d JTJ;
    Eigen::Vector6d JTr;
};

void TransformPoints(const Sophus::SE3d &T, std::vector<Eigen::Vector3d> &points) {
    std::transform(points.cbegin(), points.cend(), points.begin(),
                   [&](const auto &point) { return T * point; });
}


// point-to-point metric
ResultTuple PointToPointMetric(const std::vector<Eigen::Vector3d> &source,
                               const std::vector<Eigen::Vector3d> &target,
                               double th) {
    auto compute_jacobian_and_residual = [&](auto i) {
        const Eigen::Vector3d residual = source[i] - target[i];
        Eigen::Matrix3_6d J_r;
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(source[i]);
        return std::make_tuple(J_r, residual);
    };

    return tbb::parallel_reduce(
        tbb::blocked_range<size_t>{0, source.size()}, ResultTuple(),
        [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple {
            //auto Weight = [&](double residual2) { return square(th) / square(th + residual2); };
            auto Weight = [&](double residual2) { return 1.0 / square(th + residual2); };
            auto &JTJ_private = J.JTJ; auto &JTr_private = J.JTr;
            for (auto i = r.begin(); i < r.end(); ++i) {
                const auto &Jr = compute_jacobian_and_residual(i);
                const double w = Weight(std::get<1>(Jr).squaredNorm());
                JTJ_private.noalias() += std::get<0>(Jr).transpose() * w * std::get<0>(Jr);
                JTr_private.noalias() += std::get<0>(Jr).transpose() * w * std::get<1>(Jr);
            }
            return J;
        },
        [&](ResultTuple a, const ResultTuple &b) -> ResultTuple { return a + b; });
}

// point-to-line metric
ResultTuple PointToLineMetric(const std::vector<Eigen::Vector3d> &source,
                              const std::vector<Eigen::Vector3d> &target,
                              const std::vector<Eigen::Vector3d> &direction) {
    auto compute_jacobian_and_residual = [&](auto i) {
        Eigen::Matrix3_6d J_r;
        const Eigen::Vector3d residual = source[i] - target[i];
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(source[i]);
        return std::make_tuple(Sophus::SO3d::hat(direction[i]) * J_r, residual);
    };

    return tbb::parallel_reduce(
        tbb::blocked_range<size_t>{0, source.size()}, ResultTuple(),
        [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple {
            auto &JTJ_private = J.JTJ; auto &JTr_private = J.JTr;
            for (auto i = r.begin(); i < r.end(); ++i) {
                const auto &Jr = compute_jacobian_and_residual(i);
                JTJ_private.noalias() += std::get<0>(Jr).transpose() * std::get<0>(Jr);
                JTr_private.noalias() += std::get<0>(Jr).transpose() * std::get<1>(Jr);
            }
            return J;
        },
        [&](ResultTuple a, const ResultTuple &b) -> ResultTuple { return a + b; });
}

// point-to-plane metric
ResultTuple PointToPlaneMetric(const std::vector<Eigen::Vector3d> &source,
                               const std::vector<Eigen::Vector3d> &target,
                               const std::vector<Eigen::Vector3d> &normal) {
    auto compute_jacobian_and_residual = [&](auto i) {
        Eigen::Matrix3_6d J_r;
        const Eigen::Vector3d residual = source[i] - target[i];
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(source[i]);
        return std::make_tuple(normal[i] * normal[i].transpose() * J_r, residual);
    };

    return tbb::parallel_reduce(
        tbb::blocked_range<size_t>{0, source.size()}, ResultTuple(),
        [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple {
            auto &JTJ_private = J.JTJ; auto &JTr_private = J.JTr;
            for (auto i = r.begin(); i < r.end(); ++i) {
                const auto &Jr = compute_jacobian_and_residual(i);
                JTJ_private.noalias() += std::get<0>(Jr).transpose() * std::get<0>(Jr);
                JTr_private.noalias() += std::get<0>(Jr).transpose() * std::get<1>(Jr);
            }
            return J;
        },
        [&](ResultTuple a, const ResultTuple &b) -> ResultTuple { return a + b; });
}


ResultTuple HybridMetric(const std::vector<Eigen::Vector3d> &source,
                         const std::vector<Eigen::Vector3d> &target,
                         const std::vector<Eigen::Vector3d> &normal,
                         double th) {
    auto compute_jacobian_and_residual = [&](auto i) {
        Eigen::Matrix3_6d J_r;
        Eigen::Matrix3Xd normal_(normal[i]);
        Eigen::Vector3d residual = source[i] - target[i];
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(source[i]);
        if (normal[i].squaredNorm() > 0.5) {
            J_r = normal_ * normal_.transpose() * J_r;
            residual = normal_ * normal_.transpose() * residual;
        }
        return std::make_tuple(J_r, residual);
    };

    return tbb::parallel_reduce(
        tbb::blocked_range<size_t>{0, source.size()}, ResultTuple(),
        [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple {
            auto Weight = [&](double residual2) { return 1.0 / square(th + residual2); };
            auto &JTJ_private = J.JTJ; auto &JTr_private = J.JTr;
            for (auto i = r.begin(); i < r.end(); ++i) {
                const auto &Jr = compute_jacobian_and_residual(i);
                const double w = Weight(std::get<1>(Jr).squaredNorm());
                JTJ_private.noalias() += std::get<0>(Jr).transpose() * w * std::get<0>(Jr);
                JTr_private.noalias() += std::get<0>(Jr).transpose() * w * std::get<1>(Jr);
            }
            return J;
        },
        [&](ResultTuple a, const ResultTuple &b) -> ResultTuple { return a + b; });
}


Sophus::SE3d AlignClouds(const std::vector<Eigen::Vector3d> &source,
                         const std::vector<Eigen::Vector3d> &target,
                         double th) {
    const auto &jacobians = PointToPointMetric(source, target, th);
    const Eigen::Vector6d x = jacobians.JTJ.ldlt().solve(-jacobians.JTr);
    return Sophus::SE3d::exp(x);
}



constexpr int MAX_NUM_ITERATIONS_ = 500;
constexpr double ESTIMATION_THRESHOLD_ = 0.0001;


Sophus::SE3d KissICPRegistration(const std::vector<Eigen::Vector3d> &frame,
                                 const VoxelHashMap &voxel_map,
                                 const Sophus::SE3d &initial_guess,
                                 double max_correspondence_distance,
                                 double kernel) {
    if (voxel_map.Empty()) return initial_guess;

    std::vector<Eigen::Vector3d> source = frame;
    TransformPoints(initial_guess, source);

    Sophus::SE3d T_icp = Sophus::SE3d();
    for (int j = 0; j < MAX_NUM_ITERATIONS_; ++j) {        
        const auto &pair = voxel_map.GetCorrespondences(source, max_correspondence_distance);
        //Sophus::SE3d estimation = AlignClouds(std::get<0>(pair), std::get<1>(pair), kernel);
        
        const auto &jacobians = HybridMetric(std::get<0>(pair), std::get<1>(pair), std::get<2>(pair), kernel);
        const Eigen::Vector6d x = jacobians.JTJ.ldlt().solve(-jacobians.JTr);
        Sophus::SE3d estimation =  Sophus::SE3d::exp(x);

        TransformPoints(estimation, source);
        T_icp = estimation * T_icp;
        if (estimation.log().norm() < ESTIMATION_THRESHOLD_) break;
    }
    return T_icp * initial_guess;
}
