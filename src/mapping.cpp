#include <math.h>
#include <vector>
#include <aloam_velodyne/common.h>
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

//#include "aloam_velodyne/common.h"
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

//kd-tree
pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtreeKeypointsFromMap(new pcl::KdTreeFLANN<pcl::PointXYZ>());

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

			printf("map prepare time %f ms\n", t_prepare.toc());
			printf("map keypoints num %d\n", laserCloudKeypointsFromMapNum);

			/*TODO: scan-to-map registration*/

			/*TODO: update the map and publish*/

			printf("mapping process: %.2f ms\n", t_whole.toc());
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

	downSizeFilter.setRadiusSearch(0.6);

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

	std::thread mapping_process{mapping};
	ros::spin();
	return 0;
}
