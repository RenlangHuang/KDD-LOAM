#include <cmath>
#include <vector>
#include <string>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

using std::atan2;

int N_SCANS = 64;
double threshold = 0.1;

pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
pcl::UniformSampling<pcl::PointXYZI> uniformFilter;
ros::Publisher pubLaserCloud, pubSurfPointsFlat;


void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    TicToc t_whole;
    PointType point;

    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);

    int count = laserCloudIn.points.size();
    int cloudSize = laserCloudIn.points.size();
    
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
    for (int i = 0; i < cloudSize; i++) {
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;

        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        int scanID = 0;

        if (N_SCANS == 16) {
            scanID = int((angle + 15) / 2 + 0.5);
            if (scanID > (N_SCANS - 1) || scanID < 0) {
                count--; continue;
            }
        }
        else if (N_SCANS == 32) {
            scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
            if (scanID > (N_SCANS - 1) || scanID < 0) {
                count--; continue;
            }
        }
        else if (N_SCANS == 64) {
            if (angle >= -8.83)
                scanID = int((2 - angle) * 3.0 + 0.5);
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // use [0 50]  > 50 remove outlies
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0) {
                count--; continue;
            }
        }
        else {
            printf("wrong scan number\n");
            ROS_BREAK();
        }
        laserCloudScans[scanID].push_back(point);
    }
    cloudSize = count;

    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr SurfaceCloud(new pcl::PointCloud<PointType>());

    for (int i = 0; i < N_SCANS; i++) {
        *laserCloud += laserCloudScans[i];
    }

    for (int i = 5; i < cloudSize - 5; i++) {
        float diffX = -10 * laserCloud->points[i].x;
        float diffY = -10 * laserCloud->points[i].y;
        float diffZ = -10 * laserCloud->points[i].z;
        for (int k = 1;  k <= 5; k++) {
            diffX = diffX + laserCloud->points[i - k].x + laserCloud->points[i + k].x;
            diffY = diffY + laserCloud->points[i - k].y + laserCloud->points[i + k].y;
            diffZ = diffZ + laserCloud->points[i - k].z + laserCloud->points[i + k].z;
        }
        laserCloud->points[i].intensity = diffX * diffX + diffY * diffY + diffZ * diffZ;
        /*
        if (laserCloud->points[i].intensity < threshold) {
            SurfaceCloud->points.push_back(laserCloud->points[i]);
        }
        */
    }
    for (int i = 0; i < 5; i++) {
        laserCloud->points[i].intensity = laserCloud->points[5].intensity;
        laserCloud->points[cloudSize - i - 1].intensity = laserCloud->points[cloudSize - 6].intensity;
    }

    pcl::PointCloud<pcl::PointXYZI> GridSubsampledCloud;
    downSizeFilter.setLeafSize(0.3, 0.3, 0.3);
    downSizeFilter.setInputCloud(laserCloud);
    downSizeFilter.filter(GridSubsampledCloud);
    
    cloudSize = GridSubsampledCloud.size();
    std::cout << cloudSize << " points, ";
    
    for (int i = 0; i < cloudSize; i++) {
        if (GridSubsampledCloud.points[i].intensity < threshold) {
            SurfaceCloud->points.push_back(GridSubsampledCloud.points[i]);
        }
    }
    
    //uniformFilter.setRadiusSearch(0.3);
    //uniformFilter.setInputCloud(SurfaceCloud);
    //uniformFilter.filter(*SurfaceCloud);
    std::cout << SurfaceCloud->size() << " surface points, ";

    sensor_msgs::PointCloud2 laserCloudOutMsg, surfPointsFlat;
    pcl::toROSMsg(GridSubsampledCloud, laserCloudOutMsg);
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    laserCloudOutMsg.header.frame_id = "camera_init";

    pcl::toROSMsg(*SurfaceCloud, surfPointsFlat);
    surfPointsFlat.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat.header.frame_id = "camera_init";

    pubLaserCloud.publish(laserCloudOutMsg);
    pubSurfPointsFlat.publish(surfPointsFlat);

    printf("curvature estimation time %f ms\n", t_whole.toc());
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "curvatureEstimate");
    ros::NodeHandle nh;

    nh.param<int>("scan_line", N_SCANS, 64);
    nh.param<double>("curvature_threshold", threshold, 0.1);
    printf("scan line number %d \n", N_SCANS);

    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }

    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);

    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud", 100);
    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);

    ros::spin();
    return 0;
}
