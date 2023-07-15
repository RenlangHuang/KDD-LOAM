#include "aloam_velodyne/ransac.h"


namespace open3d {
namespace pipelines {
namespace registration {

/* copy from Open3D C++ source code registration.cpp */
static RegistrationResult _GetRegistrationResultAndCorrespondences(
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


CorrespondenceSet LocalFeatureMatching(
    const geometry::KDTreeFlann &kdtree,
    const geometry::PointCloud &target,
    const Feature &source_features,
    const Feature &target_features,
    double max_correspondence_distance,
    const int num_similar_features,
    const double similarity_threshold,
    const bool reverse)
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
        for (int k = 0; k < num_similar_features; k++) {
            if (proposals.size() < 1) break;
            double best = similarity[0]; size_t idx = 0;
            for (size_t j = 1; j < proposals.size(); j++) {
                if (similarity[j] < best) {
                    best = similarity[j];
                    idx = j;
                }
            }

#pragma omp critical // only one thread can enter at a time
            {
                if (reverse) corres.push_back(Eigen::Vector2i((int)proposals[idx], (int)i));
                else corres.push_back(Eigen::Vector2i((int)i, (int)proposals[idx]));
            }

            similarity[idx] = similarity.back();
            proposals[idx] = proposals.back();
            similarity.pop_back();
            proposals.pop_back();
        }
    }
    return corres;
}


RegistrationResult RANSACWithInitialPruning(
    const geometry::PointCloud &source,
    const geometry::PointCloud &target,
    const Feature &source_feature,
    const Feature &target_feature,
    const double max_pruning_distance, double max_correspondence_distance,
    int ransac_n, const int num_similar_features, const double similarity_threshold,
    const std::vector<std::reference_wrapper<const CorrespondenceChecker>> &checkers,
    const RANSACConvergenceCriteria &criteria)
{
    geometry::KDTreeFlann kdtree(source);
    auto estimation = TransformationEstimationPointToPoint(false);

    CorrespondenceSet corres = LocalFeatureMatching(
        kdtree, target, source_feature, target_feature,
        max_pruning_distance, num_similar_features, similarity_threshold, true
    );

    return RegistrationRANSACBasedOnCorrespondence(
        source, target, corres, max_correspondence_distance,
        estimation, ransac_n, checkers, criteria
    );
}


RegistrationResult RegistrationFromLocalToGlobal(
    const geometry::PointCloud &source,
    const geometry::PointCloud &target,
    const Feature &source_feature,
    const Feature &target_feature,
    double max_pruning_distance, double max_correspondence_distance,
    int num_similar_features, double similarity_threshold, double decay,
    const ICPConvergenceCriteria &criteria)
{
    geometry::PointCloud pcd = target;
    geometry::KDTreeFlann kdtree(source);
    auto estimation = TransformationEstimationPointToPoint(false);
    printf("in RegistrationFromLocalToGlobal\n");

    CorrespondenceSet corres = LocalFeatureMatching(
        kdtree, target, source_feature, target_feature,
        max_pruning_distance, num_similar_features, similarity_threshold, true
    );
    printf("successful LocalFeatureMatching with %ld corres\n", corres.size());
    Eigen::Matrix4d transformation = estimation.ComputeTransformation(pcd, source, corres);
    pcd.Transform(transformation);
    RegistrationResult result = _GetRegistrationResultAndCorrespondences(
        pcd, source, kdtree, max_correspondence_distance, transformation
    );
    printf("[1] successful GetRegistrationResultAndCorrespondences\n");

    for (int i = 1; i < criteria.max_iteration_; i++) {
        max_pruning_distance = max_pruning_distance * decay;
        if (max_pruning_distance < max_correspondence_distance)
            max_pruning_distance = max_correspondence_distance;
        corres = LocalFeatureMatching(
            kdtree, pcd, source_feature, target_feature,
            max_pruning_distance, num_similar_features, similarity_threshold, true
        );
        printf("[%d] successful GetRegistrationResultAndCorrespondences\n", i+1);
        Eigen::Matrix4d update = estimation.ComputeTransformation(pcd, source, corres);
        transformation = update * transformation;
        pcd.Transform(update);
        RegistrationResult backup = result;
        result = _GetRegistrationResultAndCorrespondences(
            pcd, source, kdtree, max_correspondence_distance, transformation
        );
        if (std::abs(backup.fitness_ - result.fitness_) <
                    criteria.relative_fitness_ &&
            std::abs(backup.inlier_rmse_ - result.inlier_rmse_) <
                    criteria.relative_rmse_) {
            break;
        }
    }
    printf("registration done.\n");
    return result;
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d


//updated Zhaoml 2023-07-15 11:57
int UniformRandInt(const int min, const int max) {
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(generator);
}

static open3d::pipelines::registration::RegistrationResult GetRegistrationResultAndCorrespondences(
        const open3d::geometry::PointCloud &source,
        const open3d::geometry::PointCloud &target,
        const open3d::geometry::KDTreeFlann &target_kdtree,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transformation) {
    open3d::pipelines::registration::RegistrationResult result(transformation);
    if (max_correspondence_distance <= 0.0) {
        return result;
    }

    double error2 = 0.0;

#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        double error2_private = 0.0;
        open3d::pipelines::registration::CorrespondenceSet correspondence_set_private;
#ifdef _OPENMP
#pragma omp for nowait
#endif
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
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            for (int i = 0; i < (int)correspondence_set_private.size(); i++) {
                result.correspondence_set_.push_back(
                        correspondence_set_private[i]);
            }
            error2 += error2_private;
        }
#ifdef _OPENMP
    }
#endif

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
    const std::vector<std::reference_wrapper<const open3d::pipelines::registration::CorrespondenceChecker>> &checkers
){
    open3d::pipelines::registration::RegistrationResult result(transformation_matrix);
    if(ransac_epoches<3||max_correspondence_distance<=0){
        return result;
    }
    int total=0;
    bool finished=false;
    int num_similar_feature=1;
    std::vector<std::vector<int>> similar_features(pc1.points_.size());
    open3d::utility::Logger a();

#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        open3d::pipelines::registration::CorrespondenceSet ransac_correspondence(ransac_epoches);
        open3d::geometry::KDTreeFlann kdtree(pc2);
        open3d::geometry::KDTreeFlann kdtree_feature(f2);
        open3d::pipelines::registration::RegistrationResult result_tmp;
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for(int i=0;i<criteria.max_iteration_;i++){
            if(!finished){
                std::vector<double> dists(num_similar_feature);
                Eigen::Matrix4d transformation_tmp;
                for(int j=0;j<ransac_epoches;j++){
                    int pc1_sampled_index=UniformRandInt(0,static_cast<int>(pc1.points_.size())-1);
                    if(similar_features[pc1_sampled_index].empty()){
                        std::vector<int> indices(num_similar_feature);
                        kdtree_feature.SearchKNN(Eigen::VectorXd(f1.data_.col(pc1_sampled_index)),num_similar_feature,indices,dists);    
#ifndef _OPENMP
#pragma omp critical
#endif
                        {similar_features[pc1_sampled_index]=indices;}
                    }
                    ransac_correspondence[j](0)=pc1_sampled_index;
                    if(num_similar_feature==1){
                        ransac_correspondence[j](1)=similar_features[pc1_sampled_index][0];
                    }
                    else{
                        ransac_correspondence[j](1)=similar_features[pc1_sampled_index][UniformRandInt(0,num_similar_feature-1)];
                    }
                }
                bool check=true;
                for(const auto &checker:checkers){
                    if(!checker.get().require_pointcloud_alignment_&&!checker.get().Check(pc1,pc2,ransac_correspondence,transformation_tmp)){
                        check=false;
                        break;
                    }
                }
                if(!check) continue;
                open3d::geometry::PointCloud pcd=pc1;
                pcd.Transform(transformation_tmp);
                auto result_tmp=GetRegistrationResultAndCorrespondences(pcd,pc2,kdtree,max_correspondence_distance,transformation_tmp);
                if(result_tmp.fitness_>result.fitness_||(result_tmp.fitness_==result.fitness_&&result_tmp.inlier_rmse_<result.inlier_rmse_)){
                    result=result_tmp;
                }
#ifdef _OPENMP
#pragma omp critical
#endif
                {
                    total=total+1;
                    if(total>=criteria.max_iteration_){
                        finished=true;
                    }
                }
            }
        }
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            if(result_tmp.fitness_>result.fitness_||(result_tmp.fitness_==result.fitness_&&result_tmp.inlier_rmse_<result.inlier_rmse_)){
                result=result_tmp;
            }
        }

#ifdef _OPENMP
    }
#endif
    open3d::utility::Logger::LogDebug("total: {:d}",total);
    open3d::utility::Logger::LogDebug("ransac: fitness {:e}, rmse: {:e}",result.fitness_,result.inlier_rmse_);
    return result;
}
