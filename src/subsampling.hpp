#include <unordered_map>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include "aloam_velodyne/common.h"

struct Leaf {
    Leaf(): idx(-1) {}
    int idx;
};

typedef pcl::PointXYZID PointT;
class UniformSampling {
    private:
        Eigen::Vector3f leaf_size_;
        Eigen::Vector3f inverse_leaf_size_;
        std::unordered_map<std::size_t, Leaf> leaves_;
    
    public:
        UniformSampling() {}
        UniformSampling(double radius) {
            setRadiusSearch(radius);
        }
        UniformSampling(double x_size, double y_size, double z_size) {
            setLeafSize(x_size, y_size, z_size);
        }
        UniformSampling(Eigen::Vector3f leaf_size) {
            leaf_size_ = leaf_size;
            inverse_leaf_size_ << 1.0 / leaf_size[0], 1.0 / leaf_size[1], 1.0 / leaf_size[2];
        }

        void setRadiusSearch(double radius) {
            double curvature = 1.0 / radius;
            leaf_size_ << radius, radius, radius;
            inverse_leaf_size_ << curvature, curvature, curvature;
        }
        void setLeafSize(double x_size, double y_size, double z_size) {
            leaf_size_ << x_size, y_size, z_size;
            inverse_leaf_size_ << 1.0 / x_size, 1.0 / y_size, 1.0 / z_size;
        }

        void getMinMax3D(
            pcl::PointCloud<PointT>::Ptr pcd,
            Eigen::Vector3f &min_pt,
            Eigen::Vector3f &max_pt)
        {
            min_pt << (*pcd)[0].x, (*pcd)[0].y, (*pcd)[0].z;
            max_pt << (*pcd)[0].x, (*pcd)[0].y, (*pcd)[0].z;
            for (size_t i = 1; i < pcd->size(); i++) {
                if (min_pt[0] > (*pcd)[i].x)
                    min_pt[0] = (*pcd)[i].x;
                if (min_pt[1] > (*pcd)[i].y)
                    min_pt[1] = (*pcd)[i].y;
                if (min_pt[2] > (*pcd)[i].z)
                    min_pt[2] = (*pcd)[i].z;
                if (max_pt[0] < (*pcd)[i].x)
                    max_pt[0] = (*pcd)[i].x;
                if (max_pt[1] < (*pcd)[i].y)
                    max_pt[1] = (*pcd)[i].y;
                if (max_pt[2] < (*pcd)[i].z)
                    max_pt[2] = (*pcd)[i].z;
            }
        }

        void filter(
            pcl::PointCloud<PointT>::Ptr const pcd,
            pcl::PointCloud<PointT>::Ptr result)
        {
            Eigen::Vector3f min_pt, max_pt;
            Eigen::Vector3i min_b_, max_b_;
            this->getMinMax3D(pcd, min_pt, max_pt);
            leaves_.clear();
            result->clear();
            
            // Compute the minimum and maximum bounding box values
            min_b_[0] = static_cast<int>(std::floor(min_pt[0] * inverse_leaf_size_[0]));
            max_b_[0] = static_cast<int>(std::floor(max_pt[0] * inverse_leaf_size_[0]));
            min_b_[1] = static_cast<int>(std::floor(min_pt[1] * inverse_leaf_size_[1]));
            max_b_[1] = static_cast<int>(std::floor(max_pt[1] * inverse_leaf_size_[1]));
            min_b_[2] = static_cast<int>(std::floor(min_pt[2] * inverse_leaf_size_[2]));
            max_b_[2] = static_cast<int>(std::floor(max_pt[2] * inverse_leaf_size_[2]));

            // Compute the number of divisions needed along each axis
            Eigen::Vector3i div_b_ = max_b_ - min_b_ + Eigen::Vector3i::Ones();

            // Set up the division multiplier
            Eigen::Vector3i divb_mul_(1, div_b_[0], div_b_[0] * div_b_[1]);
            
            // First pass: build a set of leaves with the point index closest to the leaf center
            for (size_t i = 0; i < pcd->size(); ++i) {
                Eigen::Vector3i ijk = Eigen::Vector3i::Zero();
                ijk[0] = static_cast<int>(std::floor((*pcd)[i].x * inverse_leaf_size_[0]));
                ijk[1] = static_cast<int>(std::floor((*pcd)[i].y * inverse_leaf_size_[1]));
                ijk[2] = static_cast<int>(std::floor((*pcd)[i].z * inverse_leaf_size_[2]));
                
                Eigen::Vector3f center;
                center[0] = (static_cast<float>(ijk[0]) + 0.5f) * leaf_size_[0];
                center[1] = (static_cast<float>(ijk[1]) + 0.5f) * leaf_size_[1];
                center[2] = (static_cast<float>(ijk[2]) + 0.5f) * leaf_size_[2];

                // Compute the leaf index
                int idx = (ijk - min_b_).dot(divb_mul_);
                Leaf& leaf = leaves_[idx];
                if (leaf.idx == -1) {
                    leaf.idx = (int)i;
                    continue;
                }
                
                Eigen::Vector3f point_prev, point_curr;
                point_curr << (*pcd)[i].x, (*pcd)[i].y, (*pcd)[i].z;
                point_prev << (*pcd)[leaf.idx].x, (*pcd)[leaf.idx].y, (*pcd)[leaf.idx].z;

                // Check to see if this point is closer to the leaf center than the existing one
                float diff_curr = (point_curr - center).squaredNorm();
                float diff_prev = (point_prev - center).squaredNorm();

                // If current point is closer, copy its index instead
                if (diff_curr < diff_prev) leaf.idx = (int)i;
            }

            // Second pass: go over all leaves and copy data
            for (const auto& leaf : leaves_)
                result->push_back((*pcd)[leaf.second.idx]);
        }
};
