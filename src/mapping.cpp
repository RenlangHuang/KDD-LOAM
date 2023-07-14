#include <math.h>
#include <vector>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/uniform_sampling.h>
#include <open3d/Open3D.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "aloam_velodyne/kdd_loam.h"
#include "aloam_velodyne/ransac.h"


/*
Here we have two schemes for mapping,
the first one is just using probabilistically sampled keypoints for mapping,
while the second one is mapping with salient regions and non-salient regions
separately (sorted by saliency uncertainty).
This code implements scheme one.
*/


int frameCount = 0;

double timeSurfPointsLast = 0;
double timeKeyPointsLast = 0;
double timeAllPointsLast = 0;
double timeLaserOdometry = 0;


int laserCloudCenWidth = 10;
int laserCloudCenHeight = 10;
int laserCloudCenDepth = 5;
const int laserCloudWidth = 21;
const int laserCloudHeight = 21;
const int laserCloudDepth = 11;
int num_salient_points = 6000;


const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth;


int laserCloudValidInd[125];
int laserCloudSurroundInd[125];

// input: from odom
pcl::PointCloud<pcl::PointXYZI>::Ptr surfpointsLast(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZID>::Ptr keypointsLast(new pcl::PointCloud<pcl::PointXYZID>());
pcl::PointCloud<pcl::PointXYZID>::Ptr salientRegionsLast(new pcl::PointCloud<pcl::PointXYZID>());
pcl::PointCloud<pcl::PointXYZID>::Ptr nonSalientRegionsLast(new pcl::PointCloud<pcl::PointXYZID>());

// output: all visible cube points
pcl::PointCloud<pcl::PointXYZID>::Ptr laserCloudSurround(new pcl::PointCloud<pcl::PointXYZID>());

// surround points in map to build tree
pcl::PointCloud<pcl::PointXYZID>::Ptr laserCloudKeypointsFromMap(new pcl::PointCloud<pcl::PointXYZID>());

//input & output: points in one frame. local --> global
pcl::PointCloud<pcl::PointXYZID>::Ptr laserCloudFull(new pcl::PointCloud<pcl::PointXYZID>());

// points in every cube
pcl::PointCloud<pcl::PointXYZID>::Ptr SalientPointsArray[laserCloudNum];
pcl::PointCloud<pcl::PointXYZID>::Ptr NonSalientPointsArray[laserCloudNum];

pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtreeKeypointsFromMap(new pcl::KdTreeFLANN<pcl::PointXYZ>());

std::vector<std::reference_wrapper<const open3d::pipelines::registration::CorrespondenceChecker>> checkers;
open3d::pipelines::registration::ICPConvergenceCriteria criteria(1e-6, 1e-6, 10);

double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);
//Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
//Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);

// wmap_T_odom * odom_T_curr = wmap_T_curr;
// transformation between odom's world and map's world frame
Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);

Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);


std::queue<sensor_msgs::PointCloud2ConstPtr> surfpointsBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> keypointsBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> allpointsBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::mutex mBuf;

pcl::UniformSampling<pcl::PointXYZ> downSizeFilter;

std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;

pcl::PointXYZID pointOri, pointSel;

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes;
ros::Publisher pubOdomAftMapped, pubOdomAftMappedHighFreq, pubLaserAfterMappedPath;

nav_msgs::Path laserAfterMappedPath;

pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

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

void pointAssociateToMap(pcl::PointXYZID const *const pi, pcl::PointXYZID *const po)
{
	Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
	po->x = point_w.x();
	po->y = point_w.y();
	po->z = point_w.z();
	po->intensity = pi->intensity;
	po->saliency = pi->saliency;
	for (int i = 0; i < 32; i++)
		po->descriptor[i] = pi->descriptor[i];
}

void pointAssociateTobeMapped(pcl::PointXYZID const *const pi, pcl::PointXYZID *const po)
{
	Eigen::Vector3d point_w(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_curr = q_w_curr.inverse() * (point_w - t_w_curr);
	po->x = point_curr.x();
	po->y = point_curr.y();
	po->z = point_curr.z();
	po->intensity = pi->intensity;
	po->saliency = pi->saliency;
	for (int i = 0; i < 32; i++)
		po->descriptor[i] = pi->descriptor[i];
}

void keypointsCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserKeypointsLast)
{
	mBuf.lock();
	keypointsBuf.push(laserKeypointsLast);
	mBuf.unlock();
}

void flatCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserFlatPointsLast)
{
    mBuf.lock();
	surfpointsBuf.push(laserFlatPointsLast);
	mBuf.unlock();
}

void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloud)
{
	mBuf.lock();
	allpointsBuf.push(laserCloud);
	mBuf.unlock();
}

void odometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
{
	mBuf.lock();
	odometryBuf.push(laserOdometry);
	mBuf.unlock();

	// high frequence publish
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

	nav_msgs::Odometry odomAftMapped;
	odomAftMapped.header.frame_id = "camera_init";
	odomAftMapped.child_frame_id = "/aft_mapped";
	odomAftMapped.header.stamp = laserOdometry->header.stamp;
	odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
	odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
	odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
	odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
	odomAftMapped.pose.pose.position.x = t_w_curr.x();
	odomAftMapped.pose.pose.position.y = t_w_curr.y();
	odomAftMapped.pose.pose.position.z = t_w_curr.z();
	pubOdomAftMappedHighFreq.publish(odomAftMapped);
}


void mapping()
{
    while (1)
    {
        while (!allpointsBuf.empty() && !keypointsBuf.empty() && !odometryBuf.empty() && !surfpointsBuf.empty())
        {
			TicToc t_whole;
            mBuf.lock();
            while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < keypointsBuf.front()->header.stamp.toSec())
				odometryBuf.pop();
			if (odometryBuf.empty())
			{
				mBuf.unlock();
				break;
			}
			while (!allpointsBuf.empty() && allpointsBuf.front()->header.stamp.toSec() < keypointsBuf.front()->header.stamp.toSec())
				allpointsBuf.pop();
			if (allpointsBuf.empty())
			{
				mBuf.unlock();
				break;
			}
            while (!surfpointsBuf.empty() && surfpointsBuf.front()->header.stamp.toSec() < keypointsBuf.front()->header.stamp.toSec())
				surfpointsBuf.pop();
			if (surfpointsBuf.empty())
			{
				mBuf.unlock();
				break;
			}

            timeSurfPointsLast = surfpointsBuf.front()->header.stamp.toSec();
            timeKeyPointsLast = keypointsBuf.front()->header.stamp.toSec();
			timeAllPointsLast = allpointsBuf.front()->header.stamp.toSec();
			timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();

			if (timeKeyPointsLast != timeAllPointsLast ||
				timeAllPointsLast != timeLaserOdometry ||
                timeLaserOdometry != timeSurfPointsLast)
			{
				printf("unsync message!");
				mBuf.unlock();
				break;
			}

            surfpointsLast->clear();
			pcl::fromROSMsg(*surfpointsBuf.front(), *surfpointsLast);
			surfpointsBuf.pop();

			keypointsLast->clear();
			pcl::fromROSMsg(*keypointsBuf.front(), *keypointsLast);
			keypointsBuf.pop();

			laserCloudFull->clear();
			pcl::fromROSMsg(*allpointsBuf.front(), *laserCloudFull);
			allpointsBuf.pop();

			q_wodom_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
			q_wodom_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
			q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
			q_wodom_curr.w() = odometryBuf.front()->pose.pose.orientation.w;
			t_wodom_curr.x() = odometryBuf.front()->pose.pose.position.x;
			t_wodom_curr.y() = odometryBuf.front()->pose.pose.position.y;
			t_wodom_curr.z() = odometryBuf.front()->pose.pose.position.z;
			odometryBuf.pop();

            printf("stamp %.2f, buffer length %ld\n", timeLaserOdometry, odometryBuf.size());

			while(!keypointsBuf.empty())
			{
				keypointsBuf.pop();
				printf("drop lidar frame in mapping for real time performance \n");
			}
			mBuf.unlock();

			transformAssociateToMap();

			TicToc t_prepare;

			// find the cube in which the current pose is located
			int centerCubeI = int((t_w_curr.x() + 25.0) / 50.0) + laserCloudCenWidth;
			int centerCubeJ = int((t_w_curr.y() + 25.0) / 50.0) + laserCloudCenHeight;
			int centerCubeK = int((t_w_curr.z() + 25.0) / 50.0) + laserCloudCenDepth;

			if (t_w_curr.x() + 25.0 < 0) centerCubeI--;
			if (t_w_curr.y() + 25.0 < 0) centerCubeJ--;
			if (t_w_curr.z() + 25.0 < 0) centerCubeK--;

			// translate the local cube map in the local FOV
			//  so that the cube in which the current pose is located
			//  can be closer to the center of the local FOV,
			// i.e., 3 < centerCubeI < 18, 3 < centerCubeJ < 18, 3 < centerCubeK < 8,
			// then it will be more convenient to extent the cube map later.
			while (centerCubeI < 3) {
				for (int j = 0; j < laserCloudHeight; j++) {
					for (int k = 0; k < laserCloudDepth; k++) { 
						int i = laserCloudWidth - 1;
						pcl::PointCloud<pcl::PointXYZID>::Ptr cubePointer =
							SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k]; 
						for (; i >= 1; i--) {
							SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								SalientPointsArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						}
						SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = cubePointer;
						cubePointer->clear();
					}
				}
				centerCubeI++;
				laserCloudCenWidth++;
			}

			while (centerCubeI >= laserCloudWidth - 3)
			{ 
				for (int j = 0; j < laserCloudHeight; j++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int i = 0;
						pcl::PointCloud<pcl::PointXYZID>::Ptr cubePointer =
							SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; i < laserCloudWidth - 1; i++)
						{
							SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								SalientPointsArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						}
						SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =	cubePointer;
						cubePointer->clear();
					}
				}
				centerCubeI--;
				laserCloudCenWidth--;
			}

			while (centerCubeJ < 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int j = laserCloudHeight - 1;
						pcl::PointCloud<pcl::PointXYZID>::Ptr cubePointer =
							SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; j >= 1; j--)
						{
							SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								SalientPointsArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
						}
						SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = cubePointer;
						cubePointer->clear();
					}
				}
				centerCubeJ++;
				laserCloudCenHeight++;
			}

			while (centerCubeJ >= laserCloudHeight - 3) {
				for (int i = 0; i < laserCloudWidth; i++) {
					for (int k = 0; k < laserCloudDepth; k++) {
						int j = 0;
						pcl::PointCloud<pcl::PointXYZID>::Ptr cubePointer =
							SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; j < laserCloudHeight - 1; j++)
						{
							SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								SalientPointsArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
						}
						SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = cubePointer;
						cubePointer->clear();
					}
				}
				centerCubeJ--;
				laserCloudCenHeight--;
			}

			while (centerCubeK < 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int j = 0; j < laserCloudHeight; j++)
					{
						int k = laserCloudDepth - 1;
						pcl::PointCloud<pcl::PointXYZID>::Ptr cubePointer =
							SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; k >= 1; k--)
						{
							SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
						}
						SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = cubePointer;
						cubePointer->clear();
					}
				}
				centerCubeK++;
				laserCloudCenDepth++;
			}

			while (centerCubeK >= laserCloudDepth - 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int j = 0; j < laserCloudHeight; j++)
					{
						int k = 0;
						pcl::PointCloud<pcl::PointXYZID>::Ptr cubePointer =
							SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; k < laserCloudDepth - 1; k++)
						{
							SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
						}
						SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = cubePointer;
						cubePointer->clear();
					}
				}
				centerCubeK--;
				laserCloudCenDepth--;
			}

			// extend 2 cubes in two directions along axis I&J(x&y)
			// extend 1 cube  in two directions along axis K(z), (250m*2)*(250m*2)
			// totally 125 cubes (75???) select those in the local FOV

			int laserCloudValidNum = 0;
			int laserCloudSurroundNum = 0;

			for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++) {
				for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++) {
					for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++) {
						if (i >= 0 && i < laserCloudWidth && j >= 0 && j < laserCloudHeight && k >= 0 && k < laserCloudDepth)
						{
							int ind = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
							laserCloudValidInd[laserCloudValidNum++] = ind;
							laserCloudSurroundInd[laserCloudSurroundNum++] = ind;
						}
					}
				}
			}

			laserCloudKeypointsFromMap->clear();
			for (int i = 0; i < laserCloudValidNum; i++) {
				*laserCloudKeypointsFromMap += *SalientPointsArray[laserCloudValidInd[i]];
			}
			int laserCloudKeypointsFromMapNum = laserCloudKeypointsFromMap->points.size();
			int laserCloudKeypointsStackNum = keypointsLast->points.size();

			std::shared_ptr<open3d::geometry::PointCloud> t_points(new open3d::geometry::PointCloud);
			std::shared_ptr<open3d::pipelines::registration::Feature> t_desc(new open3d::pipelines::registration::Feature);
			PCLO3DConverter(keypointsLast, t_points, t_desc);

			printf("map prepare time %f ms\n", t_prepare.toc());
			printf("map keypoints num %d\n", laserCloudKeypointsFromMapNum);

			/* scan-to-map registration */
			
			if (laserCloudKeypointsFromMapNum > 500) {
				TicToc t_reg;

				std::shared_ptr<open3d::geometry::PointCloud> s_points(new open3d::geometry::PointCloud);
				std::shared_ptr<open3d::pipelines::registration::Feature> s_desc(
					new open3d::pipelines::registration::Feature);
				PCLO3DConverter(laserCloudKeypointsFromMap, s_points, s_desc);

				/*

				Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
				T.rotate(q_w_curr); T.pretranslate(t_w_curr);
				Eigen::Matrix4d T_w_curr = T.matrix();
				t_points->Transform(T_w_curr);

				open3d::pipelines::registration::RegistrationResult trans;
				trans = open3d::pipelines::registration::RegistrationFromLocalToGlobal(
					*s_points, *t_points, *s_desc, *t_desc, 1.8, 0.3, 2, 0.7, 0.9, criteria
				);

				printf("Registration time: %f ms, ", t_reg.toc());
				printf("fitness: %f, ", trans.fitness_);
				printf("inlier rmse: %f, ", trans.inlier_rmse_);
				printf("correspondences: %ld\n", trans.correspondence_set_.size());
				t_points->Transform(trans.transformation_);
				T_w_curr = trans.transformation_ * T_w_curr;

				//q_w_curr = Eigen::Quaterniond(T_w_curr.block<3, 3>(0, 0)).normalized();
				//t_w_curr << T_w_curr(0,3), T_w_curr(1,3), T_w_curr(2,3);
				
				Eigen::Matrix3d rotmat;
				Eigen::Vector3d t_last_curr;
				rotmat << trans.transformation_(0,0), trans.transformation_(0,1), trans.transformation_(0,2), 
						  trans.transformation_(1,0), trans.transformation_(1,1), trans.transformation_(1,2),
						  trans.transformation_(2,0), trans.transformation_(2,1), trans.transformation_(2,2);
				t_last_curr << trans.transformation_(0,3), trans.transformation_(1,3), trans.transformation_(2,3);

				Eigen::Quaterniond q_last_curr(rotmat);
				q_last_curr.normalize();

				//t_w_curr = t_w_curr + q_w_curr * t_last_curr;
				//q_w_curr = q_w_curr * q_last_curr;
				
				t_w_curr = t_last_curr + q_last_curr * t_w_curr;
				q_w_curr = q_last_curr * q_w_curr;*/
			}
			/*TODO: update the map and publish*/
			transformUpdate();

			TicToc t_add;
			for (int i = 0; i < laserCloudKeypointsStackNum; i++) {
				keypointsLast->points[i].x = t_points->points_[i][0];
				keypointsLast->points[i].y = t_points->points_[i][1];
				keypointsLast->points[i].z = t_points->points_[i][2];

				int cubeI = int((keypointsLast->points[i].x + 25.0) / 50.0) + laserCloudCenWidth;
				int cubeJ = int((keypointsLast->points[i].y + 25.0) / 50.0) + laserCloudCenHeight;
				int cubeK = int((keypointsLast->points[i].z + 25.0) / 50.0) + laserCloudCenDepth;

				if (keypointsLast->points[i].x + 25.0 < 0) cubeI--;
				if (keypointsLast->points[i].y + 25.0 < 0) cubeJ--;
				if (keypointsLast->points[i].z + 25.0 < 0) cubeK--;

				if (cubeI >= 0 && cubeI < laserCloudWidth &&
					cubeJ >= 0 && cubeJ < laserCloudHeight &&
					cubeK >= 0 && cubeK < laserCloudDepth)
				{
					int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
					SalientPointsArray[cubeInd]->push_back(keypointsLast->points[i]);
				}
			}
			printf("add points time %f ms\n", t_add.toc());

			TicToc t_filter;
			for (int i = 0; i < laserCloudValidNum; i++) {
				int ind = laserCloudValidInd[i];
				if (SalientPointsArray[ind]->size()<1) continue;

				pcl::PointCloud<pcl::PointXYZID>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZID>());
				pcl::PointCloud<pcl::Descriptor>::Ptr desc(new pcl::PointCloud<pcl::Descriptor>());
    			pcl::PointCloud<pcl::PointXYZ>::Ptr xyz(new pcl::PointCloud<pcl::PointXYZ>());
				cloudConverter(SalientPointsArray[ind], desc, xyz);

				downSizeFilter.setInputCloud(xyz);
				downSizeFilter.filter(*xyz);
				pcl::copyPointCloud(*SalientPointsArray[ind], *downSizeFilter.getIndices(), *tmp);
				SalientPointsArray[ind] = tmp;
				printf("uniform sampling from %ld to %ld\n",
					desc->size(), SalientPointsArray[ind]->size() - downSizeFilter.getRemovedIndices()->size());
			}
			printf("filter time %f ms \n", t_filter.toc());
			
			TicToc t_pub;
			//publish local surrounding map for every 5 frames
			if (frameCount % 5 == 0) {
				laserCloudSurround->clear();
				for (int i = 0; i < laserCloudSurroundNum; i++) {
					int ind = laserCloudSurroundInd[i];
					*laserCloudSurround += *SalientPointsArray[ind];
				}

				sensor_msgs::PointCloud2 laserCloudSurround3;
				pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
				laserCloudSurround3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudSurround3.header.frame_id = "camera_init";
				pubLaserCloudSurround.publish(laserCloudSurround3);
			}
			//publish global map for every 20 frames
			if (frameCount % 20 == 0) {
				pcl::PointCloud<pcl::PointXYZID> laserCloudMap;
				for (int i = 0; i < 4851; i++) {
					laserCloudMap += *SalientPointsArray[i];
				}
				sensor_msgs::PointCloud2 laserCloudMsg;
				pcl::toROSMsg(laserCloudMap, laserCloudMsg);
				laserCloudMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudMsg.header.frame_id = "camera_init";
				pubLaserCloudMap.publish(laserCloudMsg);
			}

			int laserCloudFullResNum = laserCloudFull->points.size();
			for (int i = 0; i < laserCloudFullResNum; i++) {
				pointAssociateToMap(&laserCloudFull->points[i], &laserCloudFull->points[i]);
			}

			sensor_msgs::PointCloud2 laserCloudFullRes3;
			pcl::toROSMsg(*laserCloudFull, laserCloudFullRes3);
			laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			laserCloudFullRes3.header.frame_id = "camera_init";
			pubLaserCloudFullRes.publish(laserCloudFullRes3);

			printf("mapping pub time %f ms \n", t_pub.toc());

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

			static tf::TransformBroadcaster br;
			tf::Transform transform;
			tf::Quaternion q;
			transform.setOrigin(tf::Vector3(t_w_curr(0),t_w_curr(1),t_w_curr(2)));
			q.setW(q_w_curr.w());
			q.setX(q_w_curr.x());
			q.setY(q_w_curr.y());
			q.setZ(q_w_curr.z());
			transform.setRotation(q);
			br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "/aft_mapped"));

			frameCount++;

			printf("mapping process: %.2f ms\n\n", t_whole.toc());
        }
		std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}


int main(int argc, char **argv)
{
	ros::init(argc, argv, "mapping");
	ros::NodeHandle nh;

	nh.param<int>("num_salient_points", num_salient_points, 6000);

	//downSizeFilter.setRadiusSearch(0.6);
	downSizeFilter.setRadiusSearch(0.3);

	ros::Subscriber subLaserKeypointsLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_keypoints_last", 100, keypointsCloudHandler);
	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 100, odometryHandler);
	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_3", 100, laserCloudHandler);
    ros::Subscriber subLaserFlatLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, flatCloudHandler);

	pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100);
	pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 100);
	pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);
	pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100);
	pubOdomAftMappedHighFreq = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 100);
	pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100);

	for (int i = 0; i < laserCloudNum; i++) {
		SalientPointsArray[i].reset(new pcl::PointCloud<pcl::PointXYZID>());
        NonSalientPointsArray[i].reset(new pcl::PointCloud<pcl::PointXYZID>());
	}

    auto length_checker = open3d::pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength(0.8);
    auto distance_checker = open3d::pipelines::registration::CorrespondenceCheckerBasedOnDistance(0.3);
    checkers.push_back(length_checker);
    checkers.push_back(distance_checker);

	std::thread mapping_process{mapping};
	ros::spin();
	return 0;
}
