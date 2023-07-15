#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl_conversions/pcl_conversions.h>
#include <open3d/Open3D.h>
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

bool systemInited = false;
double pcd_time, kpt_time;

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


// void PCLO3DConverter(
//     const pcl::PointCloud<pcl::PointXYZID>::Ptr pcd,
//     const std::shared_ptr<open3d::geometry::PointCloud> xyz,
//     const std::shared_ptr<open3d::pipelines::registration::Feature> desc)
// {
//     xyz->Clear();
//     desc->Resize(32, pcd->size());
//     for (size_t i = 0; i < pcd->size(); i++) {
//         Eigen::Vector3d point(
//             pcd->points[i].x, pcd->points[i].y, pcd->points[i].z
//         );
//         xyz->points_.push_back(point);
//         for (int k = 0; k < 32; k++) {
//             desc->data_(k, i) = pcd->points[i].descriptor[k];
//         }
//     }
// }


int main(int argc, char **argv)
{
    ros::init(argc, argv, "odometry");
    ros::NodeHandle nh;
    double publish_delay = 10.0;
    nh.getParam("publish_delay", publish_delay);

    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_2", 100, laserCloudHandler);
    ros::Subscriber subKeypoints = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_keypoints", 100, keypointsHandler);
    
    ros::Publisher pubLaserCloudLast = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);
    ros::Publisher pubKeypointsLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_keypoints_last", 100);
    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);
    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

    std::shared_ptr<open3d::geometry::PointCloud> s_points(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> t_points(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::pipelines::registration::Feature> s_desc(new open3d::pipelines::registration::Feature);
    std::shared_ptr<open3d::pipelines::registration::Feature> t_desc(new open3d::pipelines::registration::Feature);

    open3d::pipelines::registration::RegistrationResult trans;
    std::vector<std::reference_wrapper<const open3d::pipelines::registration::CorrespondenceChecker>> checkers;
    auto length_checker = open3d::pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength(0.8);
    auto distance_checker = open3d::pipelines::registration::CorrespondenceCheckerBasedOnDistance(0.3);
    checkers.push_back(length_checker);
    checkers.push_back(distance_checker);
    
    nav_msgs::Path laserPath;
    nav_msgs::Odometry laserOdometry;
    laserOdometry.header.frame_id = "camera_init";
    laserOdometry.child_frame_id = "/laser_odom";

    ros::Rate rate(20);

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

            printf("odometry %f, ", pcd_time);
            printf("%ld \n", allpoints->points.size());
            printf("%ld \n", keypoints->points.size());
            PCLO3DConverter(keypoints, t_points, t_desc);

            if (!systemInited) {
                systemInited = true;
                printf("Initialization finished\n");
            }
            else {
                TicToc t_reg;
                trans = open3d::pipelines::registration::RegistrationRANSACBasedOnFeatureMatching(
                    *t_points, *s_points, *t_desc, *s_desc, true, 0.3, 
                    open3d::pipelines::registration::TransformationEstimationPointToPoint(false), 4,
                    checkers, open3d::pipelines::registration::RANSACConvergenceCriteria(50000, 0.999)
                );
                printf("RANSAC time: %f ms, ", t_reg.toc());
                printf("fitness: %f, ", trans.fitness_);
                printf("inlier rmse: %f, ", trans.inlier_rmse_);
                printf("correspondences: %ld\n", trans.correspondence_set_.size());
                
				Eigen::Matrix3d rotmat;
                Eigen::Vector3d t_last_curr;
				rotmat << trans.transformation_(0,0), trans.transformation_(0,1), trans.transformation_(0,2), 
						  trans.transformation_(1,0), trans.transformation_(1,1), trans.transformation_(1,2),
						  trans.transformation_(2,0), trans.transformation_(2,1), trans.transformation_(2,2);
                t_last_curr << trans.transformation_(0,3), trans.transformation_(1,3), trans.transformation_(2,3);

				Eigen::Quaterniond q_last_curr(rotmat);
				q_last_curr.normalize();
                
                /*TODO: convert matrix to quaternion, publish*/
                t_w_curr = t_w_curr + q_w_curr * t_last_curr;
                q_w_curr = q_w_curr * q_last_curr;
            }
            std::shared_ptr<open3d::geometry::PointCloud> tmp_points;
            std::shared_ptr<open3d::pipelines::registration::Feature> tmp_desc;
            tmp_points = t_points; t_points = s_points; s_points = tmp_points;
            tmp_desc = t_desc; t_desc = s_desc; s_desc = tmp_desc;

            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "camera_init";
            laserOdometry.child_frame_id = "/laser_odom";
            laserOdometry.header.stamp = ros::Time().fromSec(pcd_time);
            laserOdometry.pose.pose.orientation.x = q_w_curr.x();
            laserOdometry.pose.pose.orientation.y = q_w_curr.y();
            laserOdometry.pose.pose.orientation.z = q_w_curr.z();
            laserOdometry.pose.pose.orientation.w = q_w_curr.w();
            laserOdometry.pose.pose.position.x = t_w_curr.x();
            laserOdometry.pose.pose.position.y = t_w_curr.y();
            laserOdometry.pose.pose.position.z = t_w_curr.z();
            pubLaserOdometry.publish(laserOdometry);

            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header;
            laserPose.pose = laserOdometry.pose.pose;
            laserPath.header.stamp = laserOdometry.header.stamp;
            laserPath.poses.push_back(laserPose);
            laserPath.header.frame_id = "camera_init";
            pubLaserPath.publish(laserPath);
            
            printf("%f ms \n", t_whole.toc());
        }
        rate.sleep();
    }
    return 0;
}
