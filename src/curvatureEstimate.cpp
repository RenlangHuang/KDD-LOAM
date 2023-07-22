#include <cmath>
#include <vector>
#include <string>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "subsampling.hpp"

using std::atan2;

int N_SCANS = 64;
double threshold = 0.1;
int cloudSortInd[400000];
int cloudLabel[400000];

pcl::VoxelGrid<PointType> downSizeFilter;
pcl::UniformSampling<PointType> uniformFilter;
pcl::UniformSampling<PointType> downSizeFilterCorner;
pcl::UniformSampling<PointType> downSizeFilterSurf;

pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
ros::Publisher pubLaserCloud, pubCornerPointsLessSharp, pubSurfPointsLessFlat;

bool comp(int i,int j) { return (laserCloud->points[i].intensity < laserCloud->points[j].intensity); }


void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    TicToc t_whole;
    PointType point;
    std::vector<int> scanStartInd(N_SCANS, 0);
    std::vector<int> scanEndInd(N_SCANS, 0);

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

    laserCloud->clear();
    for (int i = 0; i < N_SCANS; i++) {
        scanStartInd[i] = laserCloud->size() + 5; //discard the first 5 points
        *laserCloud += laserCloudScans[i];
        scanEndInd[i] = laserCloud->size() - 6; //discard the last 6 points
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
        cloudSortInd[i] = i;
        cloudLabel[i] = 0;
    }
    for (int i = 0; i < 5; i++) {
        laserCloud->points[i].intensity = laserCloud->points[5].intensity;
        laserCloud->points[cloudSize - i - 1].intensity = laserCloud->points[cloudSize - 6].intensity;
    }

    // voxel grid down sampling
    pcl::PointCloud<pcl::PointXYZI> GridSubsampledCloud;
    downSizeFilter.setInputCloud(laserCloud);
    downSizeFilter.filter(GridSubsampledCloud);

    sensor_msgs::PointCloud2 laserCloudOutMsg;
    pcl::toROSMsg(GridSubsampledCloud, laserCloudOutMsg);
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    laserCloudOutMsg.header.frame_id = "camera_init";
    pubLaserCloud.publish(laserCloudOutMsg);

    // remove close points again
    GridSubsampledCloud.clear();
    for (size_t i = 0; i < laserCloud->size(); i++) {
        double dist = laserCloud->points[i].x * laserCloud->points[i].x;
        dist = dist + laserCloud->points[i].y * laserCloud->points[i].y;
        dist = dist + laserCloud->points[i].z * laserCloud->points[i].z;
        if (dist < 25.0) GridSubsampledCloud.points.push_back(laserCloud->points[i]);
    }
    *laserCloud = GridSubsampledCloud;


    pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>());
    
    for (int i = 0; i < N_SCANS; i++) {
        if (scanEndInd[i] - scanStartInd[i] < 6) continue;
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
        for (int j = 0; j < 6; j++) {
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

            std::sort(cloudSortInd + sp, cloudSortInd + ep + 1, comp);

            int largestPickedNum = 0;
            for (int k = ep; k >= sp; k--) {
                int ind = cloudSortInd[k]; 
                if (cloudLabel[ind] == 0 && laserCloud->points[ind].intensity > 0.1) {
                    largestPickedNum++;
                    if (largestPickedNum <= 20) {
                        cornerPointsLessSharp->push_back(laserCloud->points[ind]);
                    }
                    else break;

                    cloudLabel[ind] = 1;
                    for (int l = 1; l <= 5; l++) {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) break; // not neighbors
                        cloudLabel[ind + l] = 1; // avoid concentrated distribution
                    }
                    for (int l = -1; l >= -5; l--) {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) break;
                        cloudLabel[ind + l] = 1;
                    }
                }
            }
            for (int k = sp; k <= ep; k++) {
                if (cloudLabel[k] <= 0 && laserCloud->points[k].intensity < threshold) {
                    //surfPointsLessFlat->push_back(laserCloud->points[k]);
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
        }
        pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
        // voxel down sampling
        
        pcl::VoxelGrid<PointType> downSizeFilter;
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilter.filter(surfPointsLessFlatScanDS);
        
        // uniform down sampling
        /*
        pcl::UniformSampling<PointType> uniformFilter;
        uniformFilter.setInputCloud(surfPointsLessFlatScan);
        uniformFilter.setRadiusSearch(0.2);
        uniformFilter.filter(surfPointsLessFlatScanDS);
        */
        *surfPointsLessFlat += surfPointsLessFlatScanDS;
    }

    pcl::PointCloud<pcl::PointXYZI> cornerPointsSharp;
    pcl::PointCloud<pcl::PointXYZI> surfPointsFlat;

    // uniform down sampling
    downSizeFilterCorner.setInputCloud(cornerPointsLessSharp);
    downSizeFilterCorner.filter(cornerPointsSharp);
    downSizeFilterSurf.setInputCloud(surfPointsLessFlat);
    downSizeFilterSurf.filter(surfPointsFlat);
    
    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "camera_init";
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

    sensor_msgs::PointCloud2 surfPointsLessFlatMsg;
    pcl::toROSMsg(surfPointsFlat, surfPointsLessFlatMsg);
    surfPointsLessFlatMsg.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlatMsg.header.frame_id = "camera_init";
    pubSurfPointsLessFlat.publish(surfPointsLessFlatMsg);
    
    printf("totally %ld points, ", GridSubsampledCloud.size());
    printf("%ld edge points, ", cornerPointsSharp.size());
    printf("%ld surface points, ", surfPointsFlat.size());
    printf("time %.2f ms\n", t_whole.toc());
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "ascanRegistration");
    ros::NodeHandle nh;
    float lineRes = 0;
	float planeRes = 0;

    nh.param<int>("scan_line", N_SCANS, 64);
    nh.param<double>("curvature_threshold", threshold, 0.1);
    nh.param<float>("mapping_line_resolution", lineRes, 0.4);
	nh.param<float>("mapping_plane_resolution", planeRes, 0.8);
    printf("scan line number %d \n", N_SCANS);

    downSizeFilterCorner.setRadiusSearch(lineRes);
	downSizeFilterSurf.setRadiusSearch(planeRes);
    downSizeFilter.setLeafSize(0.3, 0.3, 0.3);

    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }

    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);

    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud", 100);
    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);
    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);

    ros::spin();
    return 0;
}
