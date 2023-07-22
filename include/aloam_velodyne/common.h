#pragma once
#define PCL_NO_PRECOMPILE
#include <cmath>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>


typedef float array[32];
typedef pcl::PointXYZI PointType;
typedef Eigen::Matrix<float, 1, 32> Vector32;

namespace pcl {
    struct PointXYZID {
        PCL_ADD_POINT4D;
        float intensity;
        float saliency;
        array descriptor;
        PCL_MAKE_ALIGNED_OPERATOR_NEW;
    } EIGEN_ALIGN16;
}
POINT_CLOUD_REGISTER_POINT_STRUCT (
    pcl::PointXYZID,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, saliency, saliency)
    (array, descriptor, descriptor)
)


void pcl_convert_XYZID_to_XYZI(
    const pcl::PointCloud<pcl::PointXYZID>::Ptr source,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr target)
{
    target->clear();
    for (size_t i = 0; i < source->size(); i++) {
        pcl::PointXYZI point;
        point.x = source->points[i].x;
        point.y = source->points[i].y;
        point.z = source->points[i].z;
        point.intensity = source->points[i].intensity;
        target->push_back(point);
    }
}


float descriptor_distance(pcl::PointXYZID &a, pcl::PointXYZID &b) {
    Vector32 a_desc = Eigen::Map<Vector32>(a.descriptor);
    Vector32 b_desc = Eigen::Map<Vector32>(b.descriptor);
    return (a_desc - b_desc).norm();
}


void assignNearestDescriptor(
    const pcl::PointCloud<pcl::PointXYZID>::Ptr source,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr target,
    const pcl::PointCloud<pcl::PointXYZID>::Ptr result)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZI>);
    pcl_convert_XYZID_to_XYZI(source, tmp);
    
    pcl::KdTreeFLANN<PointType> kdtree;
    kdtree.setInputCloud(tmp);
    std::vector<int> Ind(1);
    std::vector<float> SqDists(1);
    result->clear();

    for (size_t i = 0; i < target->points.size(); i++) {
        pcl::PointXYZID point;
        kdtree.nearestKSearch(target->points[i], 1, Ind, SqDists);
        point.x = target->points[i].x;
        point.y = target->points[i].y;
        point.z = target->points[i].z;
        point.intensity = target->points[i].intensity;
        point.saliency = source->points[Ind[0]].saliency;
        for (int k = 0; k < 32; k++) {
            point.descriptor[k] = source->points[Ind[0]].descriptor[k];
        }
        result->push_back(point);
    }
}
