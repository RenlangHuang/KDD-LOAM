#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>

//#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "aloam_velodyne/kdd_loam.h"


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
/*
pcl::PointCloud<PointType>::Ptr allpoints(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr keypoints(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr allpointsLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr keypointsLast(new pcl::PointCloud<PointType>());
*/
pcl::PointCloud<pcl::PointXYZID>::Ptr allpoints(new pcl::PointCloud<pcl::PointXYZID>());
pcl::PointCloud<pcl::PointXYZID>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZID>());
pcl::PointCloud<pcl::PointXYZID>::Ptr allpointsLast(new pcl::PointCloud<pcl::PointXYZID>());
pcl::PointCloud<pcl::PointXYZID>::Ptr keypointsLast(new pcl::PointCloud<pcl::PointXYZID>());

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
    
    ros::Publisher pubLaserCloudLast = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);
    ros::Publisher pubKeypointsLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_keypoints_last", 100);
    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);
    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

    pcl::PointCloud<pcl::PointXYZ>::Ptr s_points(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr t_points(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr r_points(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Descriptor>::Ptr s_desc(new pcl::PointCloud<pcl::Descriptor>);
    pcl::PointCloud<pcl::Descriptor>::Ptr t_desc(new pcl::PointCloud<pcl::Descriptor>);
    pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::Descriptor> ransac;
    
    ransac.setMaximumIterations(50000);
    ransac.setNumberOfSamples(4);
    ransac.setSimilarityThreshold(0.8);
    ransac.setMaxCorrespondenceDistance(0.3);
    ransac.setCorrespondenceRandomness(1); //

    nav_msgs::Path laserPath;
    nav_msgs::Odometry laserOdometry;
    laserOdometry.header.frame_id = "camera_init";
    laserOdometry.child_frame_id = "/laser_odom";

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
            TicToc t_whole;
            mBuf.lock();

            allpoints->clear();
            pcl::fromROSMsg(*pcdBuf.front(), *allpoints);
            pcdBuf.pop();

            keypoints->clear();
            pcl::fromROSMsg(*kptBuf.front(), *keypoints);
            kptBuf.pop();
            
            mBuf.unlock();

            std::cout << pcd_time << " ";
            std::cout << allpoints->points.size() << " ";
            std::cout << keypoints->points.size() << std::endl;
            cloudConverter(keypoints, t_desc, t_points);

            if (!systemInited) {
                systemInited = true;
                std::cout << "Initialization finished \n";
            }
            else {
                ransac.setInputSource(s_points);
                ransac.setInputTarget(t_points);
                ransac.setSourceFeatures(s_desc);
                ransac.setTargetFeatures(t_desc);
                printf("ransac based on feature matching ...\n");
                ransac.align(*r_points);
                printf("done\n");
                if (ransac.hasConverged()) {
                    std::cout << "RANSAC registration has converged with fitness "; 
                    std::cout << ransac.getFitnessScore() << " (m), transformation matrix:\n";
                    std::cout << ransac.getFinalTransformation() << std::endl;
                }
                else std::cout << "FAILED\n";
                /*TODO: convert matrix to quaternion, publish*/
                t_w_curr = t_w_curr + q_w_curr * t_last_curr;
                q_w_curr = q_w_curr * q_last_curr;
            }
            pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_points;
            pcl::PointCloud<pcl::Descriptor>::Ptr tmp_desc;
            tmp_points = t_points; t_points = s_points; s_points = tmp_points;
            tmp_desc = t_desc; t_desc = s_desc; s_desc = tmp_desc;
            printf("%f ms \n", t_whole.toc());
        }
        rate.sleep();
    }
    return 0;
}
