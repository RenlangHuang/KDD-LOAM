#include <cmath>
#include <vector>
#include <string>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include "kddloam_velodyne/common.h"
#include "kddloam_velodyne/tic_toc.h"


constexpr double VERTICAL_ANGLE_OFFSET = (0.205 * M_PI) / 180.0;

pcl::VoxelGrid<PointType> downSizeFilter;
pcl::UniformSampling<PointType> uniformFilter;

pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
ros::Publisher pubLaserCloud;


void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    TicToc t_whole;

    pcl::fromROSMsg(*laserCloudMsg, *laserCloud);
    for (size_t i = 0; i < laserCloud->points.size(); ++i) {
        Eigen::Vector3d pt(laserCloud->points[i].x, laserCloud->points[i].y, laserCloud->points[i].z);
        const Eigen::Vector3d rotationVector = pt.cross(Eigen::Vector3d(0., 0., 1.));
        pt = Eigen::AngleAxisd(VERTICAL_ANGLE_OFFSET, rotationVector.normalized()) * pt;
        laserCloud->points[i].x = pt[0];
        laserCloud->points[i].y = pt[1];
        laserCloud->points[i].z = pt[2];
    }

    // voxel grid down sampling
    pcl::PointCloud<pcl::PointXYZI> GridSubsampledCloud;
    //downSizeFilter.setInputCloud(laserCloud);
    //downSizeFilter.filter(GridSubsampledCloud);
    uniformFilter.setInputCloud(laserCloud);
    uniformFilter.filter(GridSubsampledCloud);

    sensor_msgs::PointCloud2 laserCloudOutMsg;
    pcl::toROSMsg(GridSubsampledCloud, laserCloudOutMsg);
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    laserCloudOutMsg.header.frame_id = "camera_init";
    pubLaserCloud.publish(laserCloudOutMsg);

    printf("totally %ld points, ", GridSubsampledCloud.size());
    printf("time %.2f ms\n", t_whole.toc());
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "voxelGridSampling");
    ros::NodeHandle nh;

    uniformFilter.setRadiusSearch(0.3);
    downSizeFilter.setLeafSize(0.3, 0.3, 0.3);

    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);
    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud", 100);

    ros::spin();
    return 0;
}
