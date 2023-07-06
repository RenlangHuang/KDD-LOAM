import time
import rospy
import argparse
import threading
import numpy as np
import open3d as o3d
from typing import List
from copy import deepcopy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree


paser = argparse.ArgumentParser()
paser.add_argument("--multi_threads_mode", type=bool, default=False)
args = paser.parse_args()


# transformation between odom's world and map's world frame
q_wmap_wodom = R.from_matrix(np.eye(3))
t_wmap_wodom = np.zeros([3, 1])

q_wodom_curr = R.from_matrix(np.eye(3))
t_wodom_curr = np.zeros([3, 1])


fields = ['x','y','z','intensity','saliency','descriptor']
allpointsBuf: List[List[PointCloud2]] = list()
keypointsBuf: List[List[PointCloud2]] = list()
odometryBuf: List[List[Odometry]] = list()
mBuf = threading.Lock()

params = {
    "ransac_inlier_threshold": 0.3,
    "ransac_max_iters": 50000,
    "edge_prune": 0.8,
    "dist_prune": 0.3,
    "ransac_n": 4,
}

pubmsg = PointCloud2()
pubmsg.height = 1
pubmsg.point_step = 37*4
pubmsg.is_bigendian = False
pubmsg.is_dense = False
pubmsg.fields = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name='saliency', offset=16, datatype=PointField.FLOAT32, count=1),
    PointField(name='descriptor',offset=20, datatype=PointField.FLOAT32, count=32)
]
pubKeypointsMsg = deepcopy(pubmsg)


def laserCloudHandler(data:PointCloud2):

    def laserCloudHandlerThread(data:PointCloud2):
        pcd = point_cloud2.read_points(data, field_names=fields)
        pcd = np.array(list(pcd), dtype=np.float32)
        print(rospy.Time.to_sec(data.header.stamp), pcd.shape)

        mBuf.acquire()
        allpointsBuf.append([data, pcd])
        mBuf.release()
    
    if args.multi_threads_mode:
        threading.Thread(target=laserCloudHandlerThread, args=[data]).start()
    else: laserCloudHandlerThread(data)


def keypointsCloudHandler(data:PointCloud2):

    def keypointsCloudHandlerThread(data:PointCloud2):
        pcd = point_cloud2.read_points(data, field_names=fields)
        pcd = np.array(list(pcd), dtype=np.float32)
        print(rospy.Time.to_sec(data.header.stamp), pcd.shape)

        mBuf.acquire()
        keypointsBuf.append([data, pcd])
        mBuf.release()
    
    if args.multi_threads_mode:
        threading.Thread(target=keypointsCloudHandlerThread, args=[data]).start()
    else: keypointsCloudHandlerThread(data)


def odometryHandler(data:Odometry):

    def odometryHandlerThread(data:Odometry):
        q_wodom_curr = np.zeros(4)
        t_wodom_curr = np.zeros([3, 1])
        q_wodom_curr[0] = data.pose.pose.orientation.x
        q_wodom_curr[1] = data.pose.pose.orientation.y
        q_wodom_curr[2] = data.pose.pose.orientation.z
        q_wodom_curr[3] = data.pose.pose.orientation.w
        t_wodom_curr[0][0] = data.pose.pose.position.x
        t_wodom_curr[1][0] = data.pose.pose.position.y
        t_wodom_curr[2][0] = data.pose.pose.position.z
        q_wodom_curr = R.from_quat(q_wodom_curr)

        mBuf.acquire()
        odometryBuf.append([data, q_wodom_curr, t_wodom_curr])
        mBuf.release()

        q_w_curr = (q_wmap_wodom * q_wodom_curr).as_quat()
        t_w_curr = q_wmap_wodom.as_matrix() @ t_wodom_curr + t_wmap_wodom 
        print(rospy.Time.to_sec(data.header.stamp), 'odom')

        odomAftMapped = Odometry()
        odomAftMapped.header.frame_id = "camera_init"
        odomAftMapped.child_frame_id = "/aft_mapped"
        odomAftMapped.header.stamp = data.header.stamp
        odomAftMapped.pose.pose.orientation.x = q_w_curr[0]
        odomAftMapped.pose.pose.orientation.y = q_w_curr[1]
        odomAftMapped.pose.pose.orientation.z = q_w_curr[2]
        odomAftMapped.pose.pose.orientation.w = q_w_curr[3]
        odomAftMapped.pose.pose.position.x = t_w_curr[0][0]
        odomAftMapped.pose.pose.position.y = t_w_curr[1][0]
        odomAftMapped.pose.pose.position.z = t_w_curr[2][0]
        pubOdomAftMappedHighFreq.publish(odomAftMapped)

    if args.multi_threads_mode:
        threading.Thread(target=odometryHandlerThread, args=[data])
    else: odometryHandlerThread(data)


def main():

    rospy.init_node("mapping", anonymous=False)
    rospy.Subscriber("/laser_odom_to_init", Odometry, odometryHandler, queue_size=100)
    rospy.Subscriber("/velodyne_cloud_3", PointCloud2, laserCloudHandler, queue_size=100)
    rospy.Subscriber("/laser_cloud_keypoints_last", PointCloud2, keypointsCloudHandler, queue_size=100)

    global pubOdomAftMappedHighFreq, q_wmap_wodom, t_wmap_wodom

    pubLaserCloudSurround = rospy.Publisher("/laser_cloud_surround", PointCloud2, queue_size=100)
    pubLaserCloudMap = rospy.Publisher("/laser_cloud_map", PointCloud2, queue_size=100)
    pubLaserCloudFull = rospy.Publisher("/velodyne_cloud_registered", PointCloud2, queue_size=100)
    pubOdomAftMapped = rospy.Publisher("/aft_mapped_to_init", Odometry, queue_size=100)
    pubOdomAftMappedHighFreq = rospy.Publisher("/aft_mapped_to_init_high_frec", Odometry, queue_size=100)
    pubLaserAfterMappedPath = rospy.Publisher("/aft_mapped_path", Path, queue_size=100)

    s_pcd = o3d.geometry.PointCloud()
    t_pcd = o3d.geometry.PointCloud()
    s_descriptors = o3d.pipelines.registration.Feature()
    t_descriptors = o3d.pipelines.registration.Feature()
    
    laserCloudCenWidth, laserCloudCenHeight, laserCloudCenDepth = 10, 10, 5
    laserCloudWidth, laserCloudHeight, laserCloudDepth = 21, 21, 11
    laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth; #4851

    laserCloudValidInd = [None for _ in range(125)]
    laserCloudSurroundInd = [None for _ in range(125)]

    # points in every cube
    laserKeypointsArray: List[List[List[list]]] = list()
    for i in range(laserCloudWidth):
        laserKeypointsArray.append(list())
        for j in range(laserCloudHeight):
            laserKeypointsArray[i].append(list())
            for k in range(laserCloudDepth):
                laserKeypointsArray[i][j].append(list())
    
    laserPath = Path()
    laserPath.header.frame_id = "camera_init"
    rate = rospy.Rate(50)
    
    while not rospy.is_shutdown():
        
        if len(allpointsBuf)>0 and len(keypointsBuf)>0 and len(odometryBuf)>0:
            timeAllPoints = rospy.Time.to_sec(allpointsBuf[0][0].header.stamp)
            timeKeyPoints = rospy.Time.to_sec(keypointsBuf[0][0].header.stamp)
            timeOdometry = rospy.Time.to_sec(odometryBuf[0][0].header.stamp)
            if  timeAllPoints != timeKeyPoints or \
                timeKeyPoints != timeOdometry or \
                timeOdometry != timeAllPoints:
                print("unsync message!")
                raise rospy.ROSInterruptException
            print('buffer length: ', len(odometryBuf)-1)
            
            start = time.time()
            mBuf.acquire()
            allpoints = allpointsBuf.pop(0)
            keypoints = keypointsBuf.pop(0)
            odom = odometryBuf.pop(0)
            mBuf.release()
            
            odom, q_wodom_curr, t_wodom_curr = odom
            q_w_curr = q_wmap_wodom * q_wodom_curr
            t_w_curr = q_wmap_wodom.as_matrix() @ t_wodom_curr + t_wmap_wodom
            
            # find the cube in which the current pose is located
            centerCubeI = int((t_w_curr[0][0] + 25.0) / 50.0) + laserCloudCenWidth
            centerCubeJ = int((t_w_curr[1][0] + 25.0) / 50.0) + laserCloudCenHeight
            centerCubeK = int((t_w_curr[2][0] + 25.0) / 50.0) + laserCloudCenDepth

            if t_w_curr[0][0] + 25.0 < 0: centerCubeI -= 1
            if t_w_curr[1][0] + 25.0 < 0: centerCubeJ -= 1
            if t_w_curr[2][0] + 25.0 < 0: centerCubeK -= 1

            # translate the local cube map in the local FOV
            # so that the cube in which the current pose is located
            # can be closer to the center of the local FOV,
            # i.e., 3 < centerCubeI < 18， 3 < centerCubeJ < 18, 3 < centerCubeK < 8,
            # then it will be more convenient to extent the cube map later.
            while centerCubeI < 3:
                for j in range(laserCloudHeight):
                    for k in range(laserCloudDepth):
                        laserKeypointsArray[-1][j][k].clear()
                newcube = laserKeypointsArray.pop()
                laserKeypointsArray.insert(0, newcube)
                centerCubeI = centerCubeI + 1
                laserCloudCenWidth = laserCloudCenWidth + 1
            
            while centerCubeI >= laserCloudWidth - 3:
                for j in range(laserCloudHeight):
                    for k in range(laserCloudDepth):
                        laserKeypointsArray[0][j][k].clear()
                newcube = laserKeypointsArray.pop(0)
                laserKeypointsArray.append(newcube)
                centerCubeI = centerCubeI - 1
                laserCloudCenWidth = laserCloudCenWidth - 1
            
            while centerCubeJ < 3:
                for i in range(laserCloudWidth):
                    for k in range(laserCloudDepth):
                        laserKeypointsArray[i][-1][k].clear()
                    newcube = laserKeypointsArray[i].pop()
                    laserKeypointsArray[i].insert(0, newcube)
                centerCubeJ = centerCubeJ + 1
                laserCloudCenHeight = laserCloudCenHeight + 1
            
            while centerCubeJ >= laserCloudHeight - 3:
                for i in range(laserCloudWidth):
                    for k in range(laserCloudDepth):
                        laserKeypointsArray[i][0][k].clear()
                    newcube = laserKeypointsArray[i].pop(0)
                    laserKeypointsArray[i].append(newcube)
                centerCubeJ = centerCubeJ - 1
                laserCloudCenHeight = laserCloudCenHeight - 1
            
            while centerCubeK < 3:
                for i in range(laserCloudWidth):
                    for j in range(laserCloudDepth):
                        laserKeypointsArray[i][j][-1].clear()
                        newcube = laserKeypointsArray[i][j].pop()
                        laserKeypointsArray[i][j].insert(0, newcube)
                centerCubeK = centerCubeK + 1
                laserCloudCenDepth = laserCloudCenDepth + 1
            
            while centerCubeK >= laserCloudDepth - 3:
                for i in range(laserCloudWidth):
                    for k in range(laserCloudDepth):
                        laserKeypointsArray[i][j][0].clear()
                        newcube = laserKeypointsArray[i][j].pop(0)
                        laserKeypointsArray[i][j].append(newcube)
                centerCubeK = centerCubeK - 1
                laserCloudCenDepth = laserCloudCenDepth - 1
            
            # extract local cubemap
            # centerCubeI-2 ~ centerCubeI+2
            # centerCubeJ-2 ~ centerCubeJ+2
            # centerCubeK-1 ~ centerCubeK+1
            local_cubemap = list()
            for i in range(centerCubeI-2, centerCubeI+3):
                for j in range(centerCubeJ-2, centerCubeJ+3):
                    for k in range(centerCubeK-1, centerCubeK+2):
                        local_cubemap += laserKeypointsArray[i][j][k]
            
            local_cubemap = np.array(local_cubemap)
            laserCloudKeypointsFromMapNum = local_cubemap.shape[0]
            # voxel grid downsample keypoints[1] (0.4m?0.8m??) TODO
            keypointsLast = keypoints[1]
            print(local_cubemap.shape, 'prepare local map:', time.time()-start)
            
            if local_cubemap.shape[0] > 50:
                kdtree = cKDTree(local_cubemap[:, 3])
                queries = q_w_curr.as_matrix() @ keypointsLast[:, :3].T + t_w_curr
                # TODO: we need to implement each of our scan-to-map registration methods
                dist, ind = kdtree.query(queries, k=5, distance_upper_bound=0.3) # ...
                # the scan-to-map registration results are stored in q_w_curr and t_w_curr
                q_wmap_wodom = q_w_curr * q_wodom_curr.inv()
                t_wmap_wodom = t_w_curr - q_wmap_wodom.as_matrix() * t_wodom_curr

                # TODO: update local cubemap and publish topics
            
            print(rospy.Time.to_sec(odom.header.stamp), end=' ')
            print(time.time()-start, '\n')
        
        rate.sleep()
    
    rospy.spin()

if __name__ == '__main__':
    main()
