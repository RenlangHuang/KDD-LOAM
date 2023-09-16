#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>


std::string dataset_folder, save_dir, sequence_number;

void odometry_path_save(nav_msgs::Odometry odomAftMapped){
    Eigen::Quaterniond q;
    q.w() = odomAftMapped.pose.pose.orientation.w;
    q.x() = odomAftMapped.pose.pose.orientation.x;
    q.y() = odomAftMapped.pose.pose.orientation.y;
    q.z() = odomAftMapped.pose.pose.orientation.z;
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(q.normalized().toRotationMatrix());
    T.pretranslate(Eigen::Vector3d(
        odomAftMapped.pose.pose.position.x,
        odomAftMapped.pose.pose.position.y,
        odomAftMapped.pose.pose.position.z)
    );

    std::ofstream pose;
    pose.open(save_dir + "_odom.txt", std::ios::app);
    pose.setf(std::ios::scientific, std::ios::floatfield);
    pose.precision(6);
    pose << T(0,0) << " " << T(0,1) << " " << T(0,2) << " " << T(0,3) << " "
         << T(1,0) << " " << T(1,1) << " " << T(1,2) << " " << T(1,3) << " "
         << T(2,0) << " " << T(2,1) << " " << T(2,2) << " " << T(2,3) << " " << std::endl;
    pose.close();
    
    std::ofstream stamps;
    stamps.open(save_dir + "_stamp_odom.txt", std::ios::app);
    stamps.setf(std::ios::scientific, std::ios::floatfield);
    stamps.precision(6);
    stamps << odomAftMapped.header.stamp.toSec() << std::endl;
    stamps.close();
}


void mapping_path_save(nav_msgs::Odometry odomAftMapped){

    Eigen::Quaterniond q;
    q.w() = odomAftMapped.pose.pose.orientation.w;
    q.x() = odomAftMapped.pose.pose.orientation.x;
    q.y() = odomAftMapped.pose.pose.orientation.y;
    q.z() = odomAftMapped.pose.pose.orientation.z;
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(q.normalized().toRotationMatrix());
    T.pretranslate(Eigen::Vector3d(
        odomAftMapped.pose.pose.position.x,
        odomAftMapped.pose.pose.position.y,
        odomAftMapped.pose.pose.position.z)
    );

    std::ofstream pose;
    pose.open(save_dir + "_mapped.txt", std::ios::app);
    pose.setf(std::ios::scientific, std::ios::floatfield);
    pose.precision(6);
    pose << T(0,0) << " " << T(0,1) << " " << T(0,2) << " " << T(0,3) << " "
         << T(1,0) << " " << T(1,1) << " " << T(1,2) << " " << T(1,3) << " "
         << T(2,0) << " " << T(2,1) << " " << T(2,2) << " " << T(2,3) << " " << std::endl;
    pose.close();
    
    std::ofstream stamps;
    stamps.open(save_dir + "_stamp_mapped.txt", std::ios::app);
    stamps.setf(std::ios::scientific, std::ios::floatfield);
    stamps.precision(6);
    stamps << odomAftMapped.header.stamp.toSec() << std::endl;
    stamps.close();
}


int main(int argc, char **argv){
    ros::init(argc, argv, "savePath");
    ros::NodeHandle nh;
    nh.getParam("dataset_folder", dataset_folder);
    nh.getParam("save_directory", save_dir);
    nh.getParam("sequence_number", sequence_number);
    save_dir = save_dir + sequence_number;

    std::ofstream mapped_pose, odom_pose, odom_stamps, mapped_stamps;
    odom_pose.open(save_dir + "_odom.txt", std::ios::out);
    mapped_pose.open(save_dir + "_mapped.txt", std::ios::out);
    odom_stamps.open(save_dir + "_stamp_odom.txt", std::ios::out);
    mapped_stamps.open(save_dir + "_stamp_mapped.txt", std::ios::out);
    
    odom_pose.close();
    mapped_pose.close();
    odom_stamps.close();
    mapped_stamps.close();

    ros::Subscriber subOdom = nh.subscribe<nav_msgs::Odometry>("/kdd_odom_to_init", 100, odometry_path_save);
    //ros::Subscriber subOdom = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 100, odometry_path_save);
    ros::Subscriber subMapped = nh.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 100, mapping_path_save);
    ros::spin();
}
