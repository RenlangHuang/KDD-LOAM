#pragma once
#define PCL_NO_PRECOMPILE
#include <cmath>
#include <vector>
#include <regex>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>


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


std::string FixFrameId(const std::string &frame_id) {
    return std::regex_replace(frame_id, std::regex("^/"), "");
}

void FillPointCloud2XYZ(const std::vector<Eigen::Vector3d> &points, sensor_msgs::PointCloud2 &msg) {
    sensor_msgs::PointCloud2Iterator<float> msg_x(msg, "x");
    sensor_msgs::PointCloud2Iterator<float> msg_y(msg, "y");
    sensor_msgs::PointCloud2Iterator<float> msg_z(msg, "z");
    for (size_t i = 0; i < points.size(); i++, ++msg_x, ++msg_y, ++msg_z) {
        const Eigen::Vector3d &point = points[i];
        *msg_x = point.x();
        *msg_y = point.y();
        *msg_z = point.z();
    }
}

auto CreatePointCloud2Msg(const size_t n_points, const std_msgs::Header &header, bool timestamp = false) {
    sensor_msgs::PointCloud2 cloud_msg;
    sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
    cloud_msg.header = header;
    cloud_msg.header.frame_id = FixFrameId(cloud_msg.header.frame_id);
    cloud_msg.fields.clear();
    int offset = 0;
    offset = addPointField(cloud_msg, "x", 1, sensor_msgs::PointField::FLOAT32, offset);
    offset = addPointField(cloud_msg, "y", 1, sensor_msgs::PointField::FLOAT32, offset);
    offset = addPointField(cloud_msg, "z", 1, sensor_msgs::PointField::FLOAT32, offset);
    offset += sizeOfPointField(sensor_msgs::PointField::FLOAT32);
    if (timestamp) {
        // asuming timestamp on a velodyne fashion for now (between 0.0 and 1.0)
        offset = addPointField(cloud_msg, "time", 1, sensor_msgs::PointField::FLOAT64, offset);
        offset += sizeOfPointField(sensor_msgs::PointField::FLOAT64);
    }

    // Resize the point cloud accordingly
    cloud_msg.point_step = offset;
    cloud_msg.row_step = cloud_msg.width * cloud_msg.point_step;
    cloud_msg.data.resize(cloud_msg.height * cloud_msg.row_step);
    modifier.resize(n_points);
    return cloud_msg;
}

std::vector<Eigen::Vector3d> PointCloud2ToEigen(const sensor_msgs::PointCloud2 &msg) {
    std::vector<Eigen::Vector3d> points;
    points.reserve(msg.height * msg.width);
    sensor_msgs::PointCloud2ConstIterator<float> msg_x(msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> msg_y(msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> msg_z(msg, "z");
    for (size_t i = 0; i < msg.height * msg.width; ++i, ++msg_x, ++msg_y, ++msg_z) {
        points.emplace_back(*msg_x, *msg_y, *msg_z);
    }
    return points;
}

sensor_msgs::PointCloud2 EigenToPointCloud2(const std::vector<Eigen::Vector3d> &points,
                                            const std_msgs::Header &header) {
    sensor_msgs::PointCloud2 msg = CreatePointCloud2Msg(points.size(), header);
    FillPointCloud2XYZ(points, msg);
    return msg;
}


float descriptor_distance(pcl::PointXYZID &a, pcl::PointXYZID &b) {
    Vector32 a_desc = Eigen::Map<Vector32>(a.descriptor);
    Vector32 b_desc = Eigen::Map<Vector32>(b.descriptor);
    return (a_desc - b_desc).norm();
}


float transformSaliencyToColor(float saliency) {
    const Eigen::Vector3f red(250.0, 20.0, 20.0);
    const Eigen::Vector3f yellow(250.0, 200.0, 80.0);
    const Eigen::Vector3f green(160.0, 240.0, 80.0);
    const Eigen::Vector3f cyan(25.0, 240.0, 240.0);
    const Eigen::Vector3f blue(50.0, 50.0, 240.0);
    const Eigen::Vector3f purple(230.0, 40.0, 210.0);
    Eigen::Vector3f color;

    if (saliency < 0.76f) color = red;
    else if (saliency < 0.86f) color = red + (saliency - 0.76f) / 0.1f * (yellow - red);
    else if (saliency < 0.90f) color = yellow + (saliency - 0.86f) / 0.04f * (green - yellow);
    else if (saliency < 0.97f) color = green + (saliency - 0.9f) / 0.07f * (cyan - green);
    else if (saliency < 1.05f) color = cyan + (saliency - 0.97f) / 0.08f * (blue - cyan);
    else if (saliency < 1.15f) color = blue + (saliency - 1.05f) / 0.1f * (purple - blue);
    else color = purple;

    uint32_t rgb = ((uint32_t)color[0] << 16 | (uint32_t)color[1] << 8 | (uint32_t)color[2]);
    return *reinterpret_cast<float*>(&rgb);
}
