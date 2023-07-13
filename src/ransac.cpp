#include "aloam_velodyne/ransac.h"


namespace open3d {
namespace pipelines {
namespace registration {

static RegistrationResult GetRegistrationResultAndCorrespondences(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const geometry::KDTreeFlann &target_kdtree,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transformation) {
    RegistrationResult result(transformation);
    if (max_correspondence_distance <= 0.0) {
        return result;
    }

    double error2 = 0.0;

#pragma omp parallel
    {
        double error2_private = 0.0;
        CorrespondenceSet correspondence_set_private;
#pragma omp for nowait
        for (int i = 0; i < (int)source.points_.size(); i++) {
            std::vector<int> indices(1);
            std::vector<double> dists(1);
            const auto &point = source.points_[i];
            if (target_kdtree.SearchHybrid(point, max_correspondence_distance,
                                           1, indices, dists) > 0) {
                error2_private += dists[0];
                correspondence_set_private.push_back(
                        Eigen::Vector2i(i, indices[0]));
            }
        }
#pragma omp critical(GetRegistrationResultAndCorrespondences)
        {
            for (int i = 0; i < (int)correspondence_set_private.size(); i++) {
                result.correspondence_set_.push_back(
                        correspondence_set_private[i]);
            }
            error2 += error2_private;
        }
    }

    if (result.correspondence_set_.empty()) {
        result.fitness_ = 0.0;
        result.inlier_rmse_ = 0.0;
    } else {
        size_t corres_number = result.correspondence_set_.size();
        result.fitness_ = (double)corres_number / (double)source.points_.size();
        result.inlier_rmse_ = std::sqrt(error2 / (double)corres_number);
    }
    return result;
}

static double EvaluateInlierCorrespondenceRatio(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transformation) {
    RegistrationResult result(transformation);

    int inlier_corres = 0;
    double max_dis2 = max_correspondence_distance * max_correspondence_distance;
    for (const auto &c : corres) {
        double dis2 =
                (source.points_[c[0]] - target.points_[c[1]]).squaredNorm();
        if (dis2 < max_dis2) {
            inlier_corres++;
        }
    }

    return double(inlier_corres) / double(corres.size());
}

RegistrationResult PointToPointICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &init,
        const TransformationEstimation &estimation,
        const ICPConvergenceCriteria &criteria) {
    if (max_correspondence_distance <= 0.0) {
        utility::LogError("Invalid max_correspondence_distance.");
    }

    Eigen::Matrix4d transformation = init;
    geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(target);
    geometry::PointCloud pcd = source;
    if (!init.isIdentity()) {
        pcd.Transform(init);
    }
    RegistrationResult result;
    result = GetRegistrationResultAndCorrespondences(
        pcd, target, kdtree, max_correspondence_distance, transformation
    );
    for (int i = 0; i < criteria.max_iteration_; i++) {
        utility::LogDebug("ICP Iteration #{:d}: Fitness {:.4f}, RMSE {:.4f}", i,
                          result.fitness_, result.inlier_rmse_);
        Eigen::Matrix4d update = estimation.ComputeTransformation(
                pcd, target, result.correspondence_set_);
        transformation = update * transformation;
        pcd.Transform(update);
        RegistrationResult backup = result;
        result = GetRegistrationResultAndCorrespondences(
                pcd, target, kdtree, max_correspondence_distance,
                transformation);
        if (std::abs(backup.fitness_ - result.fitness_) <
                    criteria.relative_fitness_ &&
            std::abs(backup.inlier_rmse_ - result.inlier_rmse_) <
                    criteria.relative_rmse_) {
            break;
        }
    }
    return result;
}

RegistrationResult RANSACBasedOnCorrespondence(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        double max_correspondence_distance,
        const TransformationEstimation &estimation
        /* = TransformationEstimationPointToPoint(false)*/,
        int ransac_n /* = 3*/,
        const std::vector<std::reference_wrapper<const CorrespondenceChecker>>
                &checkers /* = {}*/,
        const RANSACConvergenceCriteria &criteria
        /* = RANSACConvergenceCriteria()*/) {
    if (ransac_n < 3 || (int)corres.size() < ransac_n ||
        max_correspondence_distance <= 0.0) {
        return RegistrationResult();
    }

    RegistrationResult best_result;
    geometry::KDTreeFlann kdtree(target);
    int est_k_global = criteria.max_iteration_;
    int total_validation = 0;

#pragma omp parallel
    {
        CorrespondenceSet ransac_corres(ransac_n);
        RegistrationResult best_result_local;
        int est_k_local = criteria.max_iteration_;
        utility::random::UniformIntGenerator<int> rand_gen(0, corres.size() - 1);

#pragma omp for nowait
        for (int itr = 0; itr < criteria.max_iteration_; itr++) {
            if (itr < est_k_global) {
                for (int j = 0; j < ransac_n; j++) {
                    ransac_corres[j] = corres[rand_gen()];
                }

                Eigen::Matrix4d transformation =
                        estimation.ComputeTransformation(source, target, ransac_corres);

                // Check transformation: inexpensive
                bool check = true;
                for (const auto &checker : checkers) {
                    if (!checker.get().Check(source, target, ransac_corres, transformation)) {
                        check = false; break;
                    }
                }
                if (!check) continue;

                // Expensive validation
                geometry::PointCloud pcd = source;
                pcd.Transform(transformation);
                auto result = GetRegistrationResultAndCorrespondences(
                        pcd, target, kdtree, max_correspondence_distance,
                        transformation);

                if (result.IsBetterRANSACThan(best_result_local)) {
                    best_result_local = result;

                    double corres_inlier_ratio =
                            EvaluateInlierCorrespondenceRatio(
                                    pcd, target, corres,
                                    max_correspondence_distance,
                                    transformation);

                    // Update exit condition if necessary.
                    // If confidence is 1.0, then it is safely inf, we always
                    // consume all the iterations.
                    double est_k_local_d =
                            std::log(1.0 - criteria.confidence_) /
                            std::log(1.0 - std::pow(corres_inlier_ratio, ransac_n));
                    est_k_local =
                            est_k_local_d < est_k_global
                                    ? static_cast<int>(std::ceil(est_k_local_d))
                                    : est_k_local;
                    utility::LogDebug(
                            "Thread {:06d}: registration fitness={:.3f}, "
                            "corres inlier ratio={:.3f}, "
                            "Est. max k = {}",
                            itr, result.fitness_, corres_inlier_ratio,
                            est_k_local_d);
                }
#pragma omp critical
                {
                    total_validation += 1;
                    if (est_k_local < est_k_global) {
                        est_k_global = est_k_local;
                    }
                }
            }  // if
        }      // for loop

#pragma omp critical(RegistrationRANSACBasedOnCorrespondence)
        {
            if (best_result_local.IsBetterRANSACThan(best_result)) {
                best_result = best_result_local;
            }
        }
    }
    utility::LogDebug(
            "RANSAC exits after {:d} validations. Best inlier ratio {:e}, "
            "RMSE {:e}",
            total_validation, best_result.fitness_, best_result.inlier_rmse_);
    return best_result;
}

RegistrationResult RANSACBasedOnFeatureMatching(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const Feature &source_features,
        const Feature &target_features,
        bool mutual_filter,
        double max_correspondence_distance,
        const TransformationEstimation &estimation,
        int ransac_n,
        const std::vector<std::reference_wrapper<const CorrespondenceChecker>> &checkers,
        const RANSACConvergenceCriteria &criteria)
{
    if (ransac_n < 3 || max_correspondence_distance <= 0.0) {
        return RegistrationResult();
    }

    CorrespondenceSet corres = CorrespondencesFromFeatures(
        source_features, target_features, mutual_filter
    );

    return RegistrationRANSACBasedOnCorrespondence(
        source, target, corres, max_correspondence_distance,
        estimation, ransac_n, checkers, criteria);
}


CorrespondenceSet LocalFeatureMatching(
    geometry::KDTreeFlann kdtree,
    geometry::PointCloud target,
    const Feature &source_features,
    const Feature &target_features,
    double max_correspondence_distance,
    const int num_similar_features,
    const double similarity_threshold)
{
    const int kMaxThreads = std::max(utility::EstimateMaxThreads(), 1);
    
    std::vector<int> indices;
    std::vector<double> distance2;
    CorrespondenceSet corres;

#pragma omp parallel for num_threads(kMaxThreads)
    for (size_t i = 0; i < target.points_.size(); i++) {
        kdtree.SearchRadius(target.points_[i], max_correspondence_distance, indices, distance2);
        std::vector<double> similarity;
        std::vector<size_t> proposals;
            
        for (size_t j = 0; j < indices.size(); j++) {
            auto s = target_features.data_.col(i) - source_features.data_.col(indices[j]);
            double dist = s.norm(); // Euclidean distance of descriptors (L2-norm)
            if (dist <= similarity_threshold) {
                similarity.push_back(dist);
                proposals.push_back(j);
            }
        }
        for (size_t k = 0; k < num_similar_features; k++) {
            if (proposals.size() < 1) break;
            double best = similarity[0]; size_t idx = 0;
            for (size_t j = 1; j < proposals.size(); j++) {
                if (similarity[j] < best) {
                    best = similarity[j];
                    idx = j;
                }
            }

#pragma omp critical // only one thread can enter at a time
            { corres.push_back(Eigen::Vector2i((int)i, (int)proposals[idx])); }

            similarity[idx] = similarity.back();
            proposals[idx] = proposals.back();
            similarity.pop_back();
            proposals.pop_back();
        }
    }
    return corres;
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d


pipelines::registration::RegistrationResult RANSACWithInitialPruning(
    const geometry::PointCloud &source,
    const geometry::PointCloud &target,
    const pipelines::registration::Feature &source_feature,
    const pipelines::registration::Feature &target_feature,
    double max_correspondence_distance,
    int ransac_n,
    const std::vector<std::reference_wrapper<const pipelines::registration::CorrespondenceChecker>> &checkers,
    const pipelines::registration::RANSACConvergenceCriteria &criteria)
{
    pipelines::registration::RegistrationResult result;
    auto estimation = pipelines::registration::TransformationEstimationPointToPoint(false);

    return result;
}


pipelines::registration::RegistrationResult RANSACFromLocalToGlobal(
    const geometry::PointCloud &source,
    const geometry::PointCloud &target,
    const pipelines::registration::Feature &source_feature,
    const pipelines::registration::Feature &target_feature,
    double max_correspondence_distance,
    const std::vector<std::reference_wrapper<const pipelines::registration::CorrespondenceChecker>> &checkers,
    const pipelines::registration::RANSACConvergenceCriteria &criteria)
{
    pipelines::registration::RegistrationResult result;
    auto estimation = pipelines::registration::TransformationEstimationPointToPoint(false);
    if (max_correspondence_distance <= 0.0) {
        return pipelines::registration::RegistrationResult();
    }

    int total_validation = 0;
    bool finished_validation = false;
    const int num_similar_features = 1;
    const double similarity_threshold = 0.5;
    std::vector<std::vector<int>> similar_features(source.points_.size());
    pipelines::registration::CorrespondenceSet ransac_corres;

#ifdef _OPENMP
#pragma omp parallel // concurrent execution threads
    {
#endif
        geometry::KDTreeFlann kdtree(source);
        std::vector<int> indices;
        std::vector<double> distance2;

        for (size_t i = 0; i < target.points_.size(); i++) {
            kdtree.SearchRadius(target.points_[i], max_correspondence_distance, indices, distance2);
            std::vector<double> similarity;
            std::vector<size_t> proposals;
            
            for (size_t j = 0; j < indices.size(); j++) {
                auto s = target_feature.data_.col(i) - source_feature.data_.col(indices[j]);
                double dist = s.norm(); // Euclidean distance of descriptors (L2-norm)
                if (dist <= similarity_threshold) {
                    similarity.push_back(dist);
                    proposals.push_back(j);
                }
            }
            for (size_t k = 0; k < num_similar_features; k++) {
                if (proposals.size() < 1) continue;
                double best = similarity[0]; size_t idx = 0;
                for (size_t j = 1; j < proposals.size(); j++) {
                    if (similarity[j] < best) {
                        best = similarity[j];
                        idx = j;
                    }
                }
                Eigen::Vector2i pair;
                pair << idx, i;
#ifdef _OPENMP
#pragma omp critical // only one thread can enter at a time
#endif
                { ransac_corres.push_back(pair); }
            }
        }

#ifdef _OPENMP
    }
#endif

    printf("RANSAC: fitness {%.4f}, rmse %.4f, correspondences %ld\n",
        result.fitness_, result.inlier_rmse_, result.correspondence_set_.size()
    );
    return result;
}


pipelines::registration::RegistrationResult RANSAC(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const pipelines::registration::Feature &source_feature,
        const pipelines::registration::Feature &target_feature,
        double max_correspondence_distance,
        const pipelines::registration::TransformationEstimation &estimation,
        int ransac_n,
        const std::vector<std::reference_wrapper<const pipelines::registration::CorrespondenceChecker>> &checkers,
        const pipelines::registration::RANSACConvergenceCriteria &criteria)
{
    if (ransac_n < 3 || max_correspondence_distance <= 0.0) {
        return pipelines::registration::RegistrationResult();
    }

    pipelines::registration::RegistrationResult result;
    int total_validation = 0;
    bool finished_validation = false;
    int num_similar_features = 1;
    std::vector<std::vector<int>> similar_features(source.points_.size());

#ifdef _OPENMP
#pragma omp parallel // concurrent execution threads
    {
#endif
        pipelines::registration::CorrespondenceSet ransac_corres(ransac_n);
        geometry::KDTreeFlann kdtree(target);
        geometry::KDTreeFlann kdtree_feature(target_feature);
        pipelines::registration::RegistrationResult result_private;

#ifdef _OPENMP
#pragma omp for nowait // ignore barrier
#endif
        utility::random::UniformIntGenerator<int> sample(0, static_cast<int>(source.points_.size()));
        utility::random::UniformIntGenerator<int> gen(0, num_similar_features);

        for (int itr = 0; itr < criteria.max_iteration_; itr++) {
            if (!finished_validation) {
                std::vector<double> dists(num_similar_features);
                Eigen::Matrix4d transformation;
                for (int j = 0; j < ransac_n; j++) {
                    int source_sample_id = sample();
                    if (similar_features[source_sample_id].empty()) {
                        std::vector<int> indices(num_similar_features);
                        kdtree_feature.SearchKNN(
                            Eigen::VectorXd(source_feature.data_.col(source_sample_id)),
                            num_similar_features, indices, dists
                        );
#ifdef _OPENMP
#pragma omp critical // only one thread can enter at a time
#endif
                        { similar_features[source_sample_id] = indices; }
                    }
                    ransac_corres[j](0) = source_sample_id;
                    if (num_similar_features == 1)
                         ransac_corres[j](1) = similar_features[source_sample_id][0];
                    else ransac_corres[j](1) = similar_features[source_sample_id][gen()];
                }
                bool check = true;
                for (const auto &checker : checkers) {
                    if (!checker.get().require_pointcloud_alignment_ &&
                        !checker.get().Check(source, target, ransac_corres, transformation)) {
                        check = false; break;
                    }
                }
                if (!check) continue;
                transformation = estimation.ComputeTransformation(source, target, ransac_corres); //ICP
                check = true;
                for (const auto &checker : checkers) {
                    if (checker.get().require_pointcloud_alignment_ &&
                        !checker.get().Check(source, target, ransac_corres, transformation)) {
                        check = false; break;
                    }
                }
                if (!check) continue;
                geometry::PointCloud pcd = source;
                pcd.Transform(transformation);
                auto this_result = pipelines::registration::EvaluateRegistration(
                    pcd, target, max_correspondence_distance, transformation);
                if (this_result.fitness_ > result_private.fitness_ ||
                    (this_result.fitness_ == result_private.fitness_ &&
                     this_result.inlier_rmse_ < result_private.inlier_rmse_)) {
                    result_private = this_result;
                }
#ifdef _OPENMP
#pragma omp critical
#endif
                {
                    total_validation = total_validation + 1;
                    if (total_validation >= criteria.max_iteration_)
                        finished_validation = true;
                }
            }  // end of if statement
        }      // end of for-loop
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            if (result_private.fitness_ > result.fitness_ ||
                (result_private.fitness_ == result.fitness_ &&
                 result_private.inlier_rmse_ < result.inlier_rmse_)) {
                result = result_private;
            }
        }
#ifdef _OPENMP
    }
#endif
    printf("RANSAC: fitness {%.4f}, rmse %.4f, correspondences %ld\n",
        result.fitness_, result.inlier_rmse_, result.correspondence_set_.size()
    );
    return result;
}
