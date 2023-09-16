#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <string>
#include <iostream>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <eigen3/Eigen/Dense>

#include "kddloam_velodyne/common.h"
#include "kddloam_velodyne/tic_toc.h"
#include "kddloam_velodyne/KddLOAM.hpp"


int frameCount = 0;

double alpha;
KddLOAMConfig config_;
KddLOAM kdd_loam(config_);

constexpr double VERTICAL_ANGLE_OFFSET = (0.205 * M_PI) / 180.0;

// input: from odom
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZID>::Ptr keypointsCloud(new pcl::PointCloud<pcl::PointXYZID>());

Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

pcl::UniformSampling<pcl::PointXYZI> uniformFilter;

std::queue<sensor_msgs::PointCloud2ConstPtr> keypointsBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::mutex mBuf;

double timeLaserOdometry, timeLaserCloudFullRes, timeKeyPointsLast;

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap;

nav_msgs::Path laserAfterMappedPath;


struct AdaptiveVoxelBlock {
	size_t num_points_ = 1;
    std::vector<Eigen::Vector3d> points;
    inline void AddPoint(const Eigen::Vector3d &point) {
        if (points.size() < num_points_) points.push_back(point);
    }
};


void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloud)
{
	mBuf.lock();
	fullResBuf.push(laserCloud);
	mBuf.unlock();
}

void laserCloudDeepHandler(const sensor_msgs::PointCloud2ConstPtr &laserKeypoints)
{
	mBuf.lock();
	keypointsBuf.push(laserKeypoints);
	mBuf.unlock();
}

void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
{
	mBuf.lock();
	odometryBuf.push(laserOdometry);
	mBuf.unlock();
}


void mapping()
{
	while(1)
	{
		while (!fullResBuf.empty() && !keypointsBuf.empty() && !odometryBuf.empty())
		{
			mBuf.lock();
			while (!keypointsBuf.empty() && keypointsBuf.front()->header.stamp.toSec() < odometryBuf.front()->header.stamp.toSec())
				keypointsBuf.pop();
			if (keypointsBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			while (!fullResBuf.empty() && fullResBuf.front()->header.stamp.toSec() < odometryBuf.front()->header.stamp.toSec())
				fullResBuf.pop();
			if (fullResBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();
			timeKeyPointsLast = keypointsBuf.front()->header.stamp.toSec();
            timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();

			if (timeLaserCloudFullRes != timeLaserOdometry || timeKeyPointsLast != timeLaserOdometry) {
				printf("unsync message!");
				mBuf.unlock(); break;
			}

			keypointsCloud->clear();
			pcl::fromROSMsg(*keypointsBuf.front(), *keypointsCloud);
			keypointsBuf.pop();

			laserCloud->clear();
			pcl::fromROSMsg(*fullResBuf.front(), *laserCloud);
            std_msgs::Header frame_header = fullResBuf.front()->header;
			fullResBuf.pop();

			q_w_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
			q_w_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
			q_w_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
			q_w_curr.w() = odometryBuf.front()->pose.pose.orientation.w;
			t_w_curr.x() = odometryBuf.front()->pose.pose.position.x;
			t_w_curr.y() = odometryBuf.front()->pose.pose.position.y;
			t_w_curr.z() = odometryBuf.front()->pose.pose.position.z;
			odometryBuf.pop();

			while(!odometryBuf.empty())
			    odometryBuf.pop();
			mBuf.unlock();

			TicToc t_whole;

			pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
			for (size_t i = 0; i < keypointsCloud->size(); i++) {
				pcl::PointXYZ point;
				point.x = keypointsCloud->points[i].x;
				point.y = keypointsCloud->points[i].y;
				point.z = keypointsCloud->points[i].z;
				tmp->push_back(point);
			}
			pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
			kdtree.setInputCloud(tmp);
			
            std::vector<Eigen::Vector3d> scan;

            for (size_t i = 0; i < laserCloud->points.size(); i++) {
				auto &pt = laserCloud->points[i];
				Eigen::Vector3d point(pt.x, pt.y, pt.z);
				double dist = point.norm();
				if (pt.z < -3.0 || dist < config_.min_range || dist > config_.max_range) continue;
				const Eigen::Vector3d rotationVector = point.cross(Eigen::Vector3d(0., 0., 1.));
        		point = Eigen::AngleAxisd(VERTICAL_ANGLE_OFFSET, rotationVector.normalized()) * point;
                scan.push_back(point);
            }

            auto frame_downsample = VoxelDownsample(scan, config_.voxel_size * alpha);

			std::vector<float> saliency(frame_downsample.size());
			tbb::parallel_for(size_t(0), frame_downsample.size(), [&](size_t i) {
				pcl::PointXYZ pt(frame_downsample[i][0], frame_downsample[i][1], frame_downsample[i][2]);
				std::vector<int> Ind(1);
				std::vector<float> SqDists(1);
				kdtree.nearestKSearch(pt, 1, Ind, SqDists);
				saliency[i] = keypointsCloud->points[Ind[0]].saliency;
			});
            
			const Sophus::SE3d new_pose(q_w_curr, t_w_curr);
			kdd_loam.MapUpdate(frame_downsample, saliency, new_pose);
            frameCount ++;

			frame_header.frame_id = "camera_init";
            frame_header.stamp = ros::Time().fromSec(timeLaserOdometry);
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr local_map(new pcl::PointCloud<pcl::PointXYZRGB>);

            std::set<int> localCubes;
            //Eigen::Vector3i idx = kdd_loam.FindCurrentCube(t_w_curr);
			Eigen::Vector3i idx = kdd_loam.TranslateTheMap(t_w_curr);
            for (int i = idx[0] - 3; i <= idx[0] + 3; i++)
                for (int j = idx[1] - 3; j <= idx[1] + 3; j++)
                    for (int k = idx[2] - 1; k <= idx[2] + 1; k++)
                        if (i >= 0 && i < config_.cube_width && \
							j >= 0 && j < config_.cube_height && \
							k >= 0 && k < config_.cube_depth) {
                            localCubes.insert(kdd_loam._hash_cube(i,j,k));
                        }
            
			TicToc t_pub;
            for (auto it = localCubes.begin(); it != localCubes.end(); ++it){
                auto &map_ = kdd_loam.cube_map_[*it]->map_;
				for (const auto &block : map_) {
					const auto &vox = std::get<1>(block);
					for (size_t i = 0; i < vox.saliency.size(); ++i) {
						pcl::PointXYZRGB pt;
						pt.rgb = transformSaliencyToColor(vox.saliency[i]);
						pt.x = vox.points[i][0];
						pt.y = vox.points[i][1];
						pt.z = vox.points[i][2];
						local_map->push_back(pt);
					}
				}
			}
			sensor_msgs::PointCloud2 laserCloudLocalMap;
			pcl::toROSMsg(*local_map, laserCloudLocalMap);
			laserCloudLocalMap.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			laserCloudLocalMap.header.frame_id = "camera_init";
			pubLaserCloudSurround.publish(laserCloudLocalMap);
			printf("publish local map %f ms\n", t_pub.toc());
			local_map->clear();

            printf("whole global mapping time %f ms +++++\n\n", t_whole.toc());
		}
		std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
	}
}


void publish_global_map() {
	while (true) {
		std::chrono::milliseconds dura(2000);
        std::this_thread::sleep_for(dura);

        TicToc t_pub;
		std::set<int> localCubes;
		pcl::PointCloud<pcl::PointXYZRGB> global_map;
        Eigen::Vector3i idx = kdd_loam.FindCurrentCube(t_w_curr);
        for (int i = idx[0] - 2; i <= idx[0] + 2; i++)
            for (int j = idx[1] - 2; j <= idx[1] + 2; j++)
                for (int k = idx[2] - 1; k <= idx[2] + 1; k++)
                    if (i >= 0 && i < config_.cube_width && \
						j >= 0 && j < config_.cube_height && \
						k >= 0 && k < config_.cube_depth) {
                        localCubes.insert(kdd_loam._hash_cube(i,j,k));
                    }
		
        for(int i = 0; i < kdd_loam.num_cubes_; ++i) {
            if (localCubes.find(i) != localCubes.end()) continue;
            auto &map_ = kdd_loam.cube_map_[i]->map_;
            for (const auto &block : map_) {
                const auto &vox = std::get<1>(block);
                for (size_t k = 0; k < vox.saliency.size(); ++k) {
                    pcl::PointXYZRGB pt;
                    pt.rgb = transformSaliencyToColor(vox.saliency[k]);
                    pt.x = vox.points[k][0];
                    pt.y = vox.points[k][1];
                    pt.z = vox.points[k][2];
                    global_map.push_back(pt);
                }
            }
        }
        sensor_msgs::PointCloud2 laserCloudMap;
        pcl::toROSMsg(global_map, laserCloudMap);
        laserCloudMap.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        laserCloudMap.header.frame_id = "camera_init";
        pubLaserCloudMap.publish(laserCloudMap);
        printf("publish global map %f ms\n", t_pub.toc());
        global_map.clear();
    }
}


int main(int argc, char **argv)
{
	ros::init(argc, argv, "global_mapping");
	ros::NodeHandle nh;

    nh.param<double>("voxel_size_map", config_.voxel_size, 1.0);
    nh.param<double>("factor_voxel_size_map_merge", alpha, 0.5);

	ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);
	ros::Subscriber subDeepLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/deep_laser_cloud", 100, laserCloudDeepHandler);
	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 100, laserOdometryHandler);

	pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100);
	pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_global_map", 100);

	std::thread mapping_process{mapping};
	std::thread global_mapping{publish_global_map};
	ros::spin();
	return 0;
}