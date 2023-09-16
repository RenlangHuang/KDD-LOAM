#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <string>
#include <iostream>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <eigen3/Eigen/Dense>

#include "kddloam_velodyne/common.h"
#include "kddloam_velodyne/tic_toc.h"

pcl::PointCloud<pcl::PointXYZID> keypointsCloud;
std::queue<sensor_msgs::PointCloud2ConstPtr> keypointsBuf;
ros::Publisher pubRendered;
std::mutex mBuf;

void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserKeypoints)
{
	mBuf.lock();
	keypointsBuf.push(laserKeypoints);
	mBuf.unlock();
}

void render()
{
	while(1)
	{
		while (!keypointsBuf.empty())
		{
			mBuf.lock();
			keypointsCloud.clear();
            double timeStamp = keypointsBuf.front()->header.stamp.toSec();
			pcl::fromROSMsg(*keypointsBuf.front(), keypointsCloud);
			keypointsBuf.pop();
			mBuf.unlock();

            pcl::PointCloud<pcl::PointXYZRGB> colored;
            for (size_t i = 0; i < keypointsCloud.points.size(); i++) {
                pcl::PointXYZRGB point;
                point.rgb = transformSaliencyToColor(keypointsCloud.points[i].saliency);
                point.x = keypointsCloud.points[i].x;
                point.y = keypointsCloud.points[i].y;
                point.z = keypointsCloud.points[i].z;
                colored.push_back(point);
            }
			sensor_msgs::PointCloud2 renderedCurrentScan;
			pcl::toROSMsg(colored, renderedCurrentScan);
			renderedCurrentScan.header.stamp = ros::Time().fromSec(timeStamp);
			renderedCurrentScan.header.frame_id = "camera_init";
			pubRendered.publish(renderedCurrentScan);
		}
		std::chrono::milliseconds dura(50);
        std::this_thread::sleep_for(dura);
	}
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserRender");
	ros::NodeHandle nh;

	ros::Subscriber subDeepLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/deep_laser_cloud", 100, laserCloudHandler);
	pubRendered = nh.advertise<sensor_msgs::PointCloud2>("/deep_current_scan", 100);

	std::thread render_process{render};
	ros::spin();
	return 0;
}
