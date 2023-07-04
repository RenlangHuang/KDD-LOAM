#pragma once
#define PCL_NO_PRECOMPILE
#include <cmath>
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

inline double rad2deg(double radians)
{
    return radians * 180.0 / M_PI;
}

inline double deg2rad(double degrees)
{
    return degrees * M_PI / 180.0;
}

void cloudConverter(
    const pcl::PointCloud<pcl::PointXYZID>::Ptr pcd,
    const pcl::PointCloud<pcl::Descriptor>::Ptr desc,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr xyz)
{
    std::cout << "totally " << pcd->size() << " points\n";
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
    std::cout << desc->at(0) << std::endl;
    std::cout << desc->at(1) << std::endl;
    std::cout << xyz->at(0) << std::endl;
    std::cout << xyz->at(1) << std::endl;
}
