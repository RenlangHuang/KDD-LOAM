#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"


// Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

// q_curr_last(x, y, z, w), t_curr_last
double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};

bool systemInited = false;
double pcd_time, kpt_time;

Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

pcl::PointCloud<PointType>::Ptr allpoints(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr keypoints(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr allpointsLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr keypointsLast(new pcl::PointCloud<PointType>());
std::queue<sensor_msgs::PointCloud2ConstPtr> pcdBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> kptBuf;
std::mutex mBuf;

void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloud)
{
    mBuf.lock();
    pcdBuf.push(laserCloud);
    mBuf.unlock();
}

void keypointsHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloud)
{
    mBuf.lock();
    kptBuf.push(laserCloud);
    mBuf.unlock();
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "kddOdometry");
    ros::NodeHandle nh;
    double publish_delay = 10.0;
    nh.getParam("publish_delay", publish_delay);

    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/py_velodyne_points", 100, laserCloudHandler);
    ros::Subscriber subKeypoints = nh.subscribe<sensor_msgs::PointCloud2>("/py_keypoints", 100, keypointsHandler);
    
    ros::Publisher pubLaserCloudLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_last", 100);
    ros::Publisher pubKeypointsLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_keypoints_last", 100);
    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);
    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);
    /*
    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);
    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);
    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);
    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);
    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);
    */
    nav_msgs::Path laserPath;

    int frameCount = 0;
    ros::Rate rate(100);

    while (ros::ok())
    {
        ros::spinOnce();
        if (!pcdBuf.empty() && !kptBuf.empty()) {
            pcd_time = pcdBuf.front()->header.stamp.toSec();
            kpt_time = kptBuf.front()->header.stamp.toSec();
            if (pcd_time != kpt_time) {
                printf("unsync message!");
                ROS_BREAK();
            }

            mBuf.lock();

            allpoints->clear();
            pcl::fromROSMsg(*pcdBuf.front(), *allpoints);
            pcdBuf.pop();

            keypoints->clear();
            pcl::fromROSMsg(*kptBuf.front(), *keypoints);
            kptBuf.pop();
            
            mBuf.unlock();

            TicToc t_whole;
            if (!systemInited) {
                systemInited = true;
                std::cout << "Initialization finished \n";
                continue;
            }
            std::cout << pcd_time << " ";
            std::cout << allpoints->points.size() << " ";
            std::cout << keypoints->points.size() << std::endl;
            //std::cout << keypoints->points[0] << std::endl;

            /*TODO： ransac based on feature matching, publish*/

            pcl::PointCloud<PointType>::Ptr laserCloudTemp;
            laserCloudTemp = allpoints;
            allpoints = allpointsLast;
            allpoints = laserCloudTemp;

            laserCloudTemp = keypoints;
            keypoints = keypointsLast;
            keypointsLast = laserCloudTemp;
            printf("%f ms \n", t_whole.toc());
        }
        rate.sleep();
    }
    return 0;
}