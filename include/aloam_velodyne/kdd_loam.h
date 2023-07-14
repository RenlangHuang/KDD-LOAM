#pragma once
#define PCL_NO_PRECOMPILE
#include <cmath>
#include <open3d/Open3D.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

typedef float array[32];
typedef pcl::PointXYZI PointType;
namespace pcl {
    typedef FPFHSignature33 Descriptor; 
}

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

void cloudConverter(
    const pcl::PointCloud<pcl::PointXYZID>::Ptr pcd,
    const pcl::PointCloud<pcl::Descriptor>::Ptr desc,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr xyz)
{
    for (size_t i = 0; i < pcd->size(); i++) {
        pcl::PointXYZ point;
        pcl::Descriptor feature;
        point.x = pcd->points[i].x;
        point.y = pcd->points[i].y;
        point.z = pcd->points[i].z;
        for (size_t k = 0; k < 32; k++) {
            feature.histogram[k] = pcd->points[i].descriptor[k];
        }
        desc->push_back(feature);
        xyz->push_back(point);
    }
}


void PCLO3DConverter(
    const pcl::PointCloud<pcl::PointXYZID>::Ptr pcd,
    const std::shared_ptr<open3d::geometry::PointCloud> xyz,
    const std::shared_ptr<open3d::pipelines::registration::Feature> desc)
{
    xyz->Clear();
    desc->Resize(32, pcd->size());
    for (size_t i = 0; i < pcd->size(); i++) {
        Eigen::Vector3d point(
            pcd->points[i].x, pcd->points[i].y, pcd->points[i].z
        );
        xyz->points_.push_back(point);
        for (int k = 0; k < 32; k++) {
            desc->data_(k, i) = pcd->points[i].descriptor[k];
        }
    }
}
