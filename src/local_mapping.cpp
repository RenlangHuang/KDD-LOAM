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
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>

#include "kddloam_velodyne/common.h"
#include "kddloam_velodyne/tic_toc.h"
#include "kddloam_velodyne/HybridICP.hpp"


int frameCount = 0;
int memory_count = 0;
unsigned long map_memory = 0;

double alpha, beta;
ICPConfig config_;
HybridICP kdd_icp(config_);

constexpr double VERTICAL_ANGLE_OFFSET = (0.205 * M_PI) / 180.0;

// input: from odom
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZID>::Ptr keypointsCloud(new pcl::PointCloud<pcl::PointXYZID>());

Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

// transformation between odom's world and map's world frame
Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);

Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);

pcl::UniformSampling<pcl::PointXYZI> uniformFilter;

std::queue<sensor_msgs::PointCloud2ConstPtr> keypointsBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::mutex mBuf;

ros::Publisher pubLaserCloudMap, pubLaserCloudFullRes;
ros::Publisher pubOdomAftMapped, pubOdomAftMappedHighFrec, pubLaserAfterMappedPath;

nav_msgs::Path laserAfterMappedPath;


struct AdaptiveVoxelBlock {
	size_t num_points_ = 1;
    std::vector<Eigen::Vector3d> points;
    inline void AddPoint(const Eigen::Vector3d &point) {
        if (points.size() < num_points_) points.push_back(point);
    }
};

// set initial guess
void transformAssociateToMap()
{
	q_w_curr = q_wmap_wodom * q_wodom_curr;
	t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}

void transformUpdate()
{
	q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
	t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}

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

	Eigen::Quaterniond q_wodom_curr;
	Eigen::Vector3d t_wodom_curr;
	q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;
	q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
	q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
	q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
	t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
	t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
	t_wodom_curr.z() = laserOdometry->pose.pose.position.z;

	Eigen::Quaterniond q_w_curr = q_wmap_wodom * q_wodom_curr;
	Eigen::Vector3d t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom; 
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

			double timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();
			double timeKeyPointsLast = keypointsBuf.front()->header.stamp.toSec();
            double timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();

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

			q_wodom_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
			q_wodom_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
			q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
			q_wodom_curr.w() = odometryBuf.front()->pose.pose.orientation.w;
			t_wodom_curr.x() = odometryBuf.front()->pose.pose.position.x;
			t_wodom_curr.y() = odometryBuf.front()->pose.pose.position.y;
			t_wodom_curr.z() = odometryBuf.front()->pose.pose.position.z;
			odometryBuf.pop();

			while(!odometryBuf.empty())
			{
				odometryBuf.pop();
				printf("drop lidar frame in mapping for real time performance \n");
			}
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
			TicToc t_registration;

            transformAssociateToMap();
            Sophus::SE3d pred(q_w_curr, t_w_curr);

            auto frame_downsample = VoxelDownsample(scan, config_.voxel_size * alpha);
            
            tsl::robin_map<Voxel, AdaptiveVoxelBlock, VoxelHash> grid;
			grid.reserve(keypointsCloud->points.size());
			double voxel_size = config_.voxel_size * beta;

			std::vector<float> saliency(frame_downsample.size());
			tbb::parallel_for(size_t(0), frame_downsample.size(), [&](size_t i) {
				pcl::PointXYZ pt(frame_downsample[i][0], frame_downsample[i][1], frame_downsample[i][2]);
				std::vector<int> Ind(1);
				std::vector<float> SqDists(1);
				kdtree.nearestKSearch(pt, 1, Ind, SqDists);
				saliency[i] = keypointsCloud->points[Ind[0]].saliency;
			});
            
            //auto source = VoxelDownsample(frame_downsample, config_.voxel_size * beta);

			for (size_t i = 0; i < keypointsCloud->points.size(); ++i) {
				auto &kpt = keypointsCloud->points[i];
				Eigen::Vector3d point(kpt.x, kpt.y, kpt.z);
				if (point.squaredNorm() < 25.0) continue;
				size_t max_num_pts = 1;
				if (kpt.saliency > 1.13f) continue;
				if (kpt.saliency < 0.90f) max_num_pts ++;
				if (kpt.saliency < 0.86f) max_num_pts ++;

				const auto voxel = Voxel((point / voxel_size).cast<int>());
				auto search = grid.find(voxel);
				if (search != grid.end()) {
					auto &voxel_block = search.value();
					if (voxel_block.num_points_ < max_num_pts)
						voxel_block.num_points_ = max_num_pts;
				}
                else grid.insert({voxel, AdaptiveVoxelBlock{max_num_pts, {}}});
			}

			for (size_t i = 0; i < frame_downsample.size(); ++i) {
				const auto &point = frame_downsample[i];
				const auto voxel = Voxel((point / voxel_size).cast<int>());
				//size_t max_num_pts = 1;
				//if (saliency[i] > 1.13f) continue;
				//if (saliency[i] < 0.90f) max_num_pts ++;
				//if (saliency[i] < 0.86f) max_num_pts ++;
				auto search = grid.find(voxel);
				if (search != grid.end()) {
					auto &voxel_block = search.value();
					//if (voxel_block.num_points_ < max_num_pts)
					//	voxel_block.num_points_ = max_num_pts;
					voxel_block.AddPoint(point);
				}
                //else grid.insert({voxel, AdaptiveVoxelBlock{max_num_pts, {point}}});
			}
			
			std::vector<Eigen::Vector3d> source;
			source.reserve(frame_downsample.size());
			for (auto it = grid.begin(); it != grid.end(); ++it) {
				auto &voxel_block = it.value();
				for (auto pt : voxel_block.points) {
					source.emplace_back(pt);
				}
			}
            
            const double sigma = kdd_icp.GetAdaptiveThreshold();

            if (config_.const_vel_pred) {
                const auto prediction = kdd_icp.GetPredictionModel();
                const auto last_pose = !kdd_icp.poses_.empty() ? kdd_icp.poses_.back() : Sophus::SE3d();
                pred = last_pose * prediction;
            }

            const Sophus::SE3d new_pose = KissICPRegistration(
                source, kdd_icp.local_map_, pred, 3.0 * sigma, sigma / 3.0);
            const auto model_deviation = pred.inverse() * new_pose;
            kdd_icp.adaptive_threshold_.UpdateModelDeviation(model_deviation);
            kdd_icp.local_map_.Update(frame_downsample, saliency, new_pose);
            kdd_icp.poses_.push_back(new_pose);

            const auto pose = kdd_icp.poses().back();
            t_w_curr = pose.translation();
            q_w_curr = pose.unit_quaternion();
            transformUpdate();

			printf("scan-to-map registration time %f ms \n", t_registration.toc());

			frame_header.frame_id = "camera_init";
            frame_header.stamp = ros::Time().fromSec(timeLaserOdometry);
			
			unsigned long memory = 0; frameCount++;
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr local_map(new pcl::PointCloud<pcl::PointXYZRGB>);
			for (const auto &block : kdd_icp.local_map_.map_) {
				const auto &vox = std::get<1>(block);
				if (vox.fitted) memory = memory + sizeof(double) * 6;
				else memory = memory + sizeof(double) * 3 * vox.saliency.size();
				for (size_t i = 0; i < vox.saliency.size(); ++i) {
					pcl::PointXYZRGB pt;
					pt.rgb = transformSaliencyToColor(vox.saliency[i]);
					pt.x = vox.points[i][0];
					pt.y = vox.points[i][1];
					pt.z = vox.points[i][2];
					local_map->push_back(pt);
				}
			}
			printf("occupied memory of valid map data: %ld B\n", memory);
			if (frameCount % 50 == 0 && frameCount > 50){
				memory_count ++;
				map_memory += memory;
				printf("\n[map memory] %d:\t\t\t\t\t%.4f B\n", memory_count, (float)map_memory/(float)memory_count);
			}

			sensor_msgs::PointCloud2 laserCloudMap;
			pcl::toROSMsg(*local_map, laserCloudMap);
			laserCloudMap.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			laserCloudMap.header.frame_id = "camera_init";
			pubLaserCloudMap.publish(laserCloudMap);

			TransformPoints(pose, source);
			pubLaserCloudFullRes.publish(EigenToPointCloud2(source, frame_header));

			nav_msgs::Odometry odomAftMapped;
			odomAftMapped.header.frame_id = "camera_init";
			odomAftMapped.child_frame_id = "/aft_mapped";
			odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
			odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
			odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
			odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
			odomAftMapped.pose.pose.position.x = t_w_curr.x();
			odomAftMapped.pose.pose.position.y = t_w_curr.y();
			odomAftMapped.pose.pose.position.z = t_w_curr.z();
			pubOdomAftMapped.publish(odomAftMapped);

			geometry_msgs::PoseStamped laserAfterMappedPose;
			laserAfterMappedPose.header = odomAftMapped.header;
			laserAfterMappedPose.pose = odomAftMapped.pose.pose;
			laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
			laserAfterMappedPath.header.frame_id = "camera_init";
			laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
			pubLaserAfterMappedPath.publish(laserAfterMappedPath);

			tf::Quaternion q;
            tf::Transform transform;
            static tf::TransformBroadcaster br;
            transform.setOrigin(tf::Vector3(t_w_curr(0), t_w_curr(1), t_w_curr(2)));
			q.setW(q_w_curr.w());
			q.setX(q_w_curr.x());
			q.setY(q_w_curr.y());
			q.setZ(q_w_curr.z());
			transform.setRotation(q);
			br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "/aft_mapped"));
            printf("whole mapping time %f ms +++++\n\n", t_whole.toc());
		}
		std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
	}
}


int main(int argc, char **argv)
{
	ros::init(argc, argv, "local_mapping");
	ros::NodeHandle nh;

    nh.param<double>("voxel_size_map", config_.voxel_size, 1.0);
    nh.param<double>("factor_voxel_size_map_merge", alpha, 0.5);
    nh.param<double>("factor_voxel_size_registration", beta, 1.5);

	config_.const_vel_pred = false;
	config_.initial_threshold = 2.0;
	kdd_icp = HybridICP(config_);

	ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);
	ros::Subscriber subDeepLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/deep_laser_cloud", 100, laserCloudDeepHandler);
	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/kdd_odom_to_init", 100, laserOdometryHandler);

	pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 100);
	pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);
	pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100);
	pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100);

	std::thread mapping_process{mapping};
	ros::spin();
	return 0;
}
