#pragma once

#include <sophus/se3.hpp>

struct AdaptiveThreshold {
    explicit AdaptiveThreshold(double initial_threshold, double min_motion_th, double max_range)
        : initial_threshold_(initial_threshold),
          min_motion_th_(min_motion_th),
          max_range_(max_range) {}

    /// Update the current belief of the deviation from the prediction model
    inline void UpdateModelDeviation(const Sophus::SE3d &current_deviation) {
        model_deviation_ = current_deviation;
    }

    /// Returns the KISS-ICP adaptive threshold used in registration
    double ComputeThreshold();

    // configurable parameters
    double initial_threshold_;
    double min_motion_th_;
    double max_range_;

    // Local cache for ccomputation
    double model_error_sse2_ = 0;
    int num_samples_ = 0;
    Sophus::SE3d model_deviation_ = Sophus::SE3d();
};


double AdaptiveThreshold::ComputeThreshold() {
    const double theta = Eigen::AngleAxisd(model_deviation_.rotationMatrix()).angle();
    const double delta_rot = 2.0 * max_range_ * std::sin(theta / 2.0);
    const double delta_trans = model_deviation_.translation().norm();
    double model_error = delta_trans + delta_rot;

    if (model_error > min_motion_th_) {
        model_error_sse2_ += model_error * model_error;
        num_samples_++;
    }

    if (num_samples_ < 1) {
        return initial_threshold_;
    }
    return std::sqrt(model_error_sse2_ / num_samples_);
}