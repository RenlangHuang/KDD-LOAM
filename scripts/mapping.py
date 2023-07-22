import time
import rospy
import threading
import numpy as np
import open3d as o3d
from copy import deepcopy
from typing import List, Dict
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from utils.VoxelFilter import uniform_sampling
from utils.registration import *


# transformation between odom's world and map's world frame
q_wmap_wodom = R.from_matrix(np.eye(3))
t_wmap_wodom = np.zeros([3, 1])

PointXYZISD = ['x','y','z','intensity','saliency','descriptor']
allpointsBuf: List[List[PointCloud2]] = list()
keypointsBuf: List[List[PointCloud2]] = list()
surfpointsBuf: List[List[PointCloud2]] = list()
odometryBuf: List[List[Odometry]] = list()
mBuf = threading.Lock()


surroundMsg = PointCloud2()
surroundMsg.height = 1
surroundMsg.point_step = 5*4
surroundMsg.is_bigendian = False
surroundMsg.is_dense = False
surroundMsg.fields = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name='saliency', offset=16, datatype=PointField.FLOAT32, count=1)
]
mapMsg = deepcopy(surroundMsg)
scanMsg = deepcopy(surroundMsg)


def laserCloudHandler(data:PointCloud2):

    def laserCloudHandlerThread(data:PointCloud2):
        pcd = point_cloud2.read_points(data, field_names=PointXYZISD)
        pcd = np.array(list(pcd), dtype=np.float32)
        #print(rospy.Time.to_sec(data.header.stamp), pcd.shape)

        mBuf.acquire()
        allpointsBuf.append([data, pcd])
        mBuf.release()
    
    threading.Thread(target=laserCloudHandlerThread, args=[data]).start()


def keypointsCloudHandler(data:PointCloud2):

    def keypointsCloudHandlerThread(data:PointCloud2):
        pcd = point_cloud2.read_points(data, field_names=PointXYZISD)
        pcd = np.array(list(pcd), dtype=np.float32)
        #print(rospy.Time.to_sec(data.header.stamp), pcd.shape)

        mBuf.acquire()
        keypointsBuf.append([data, pcd])
        mBuf.release()
    
    threading.Thread(target=keypointsCloudHandlerThread, args=[data]).start()


def flatCloudHandler(data:PointCloud2):

    def flatCloudHandlerThread(data:PointCloud2):
        pcd = point_cloud2.read_points(data, field_names=PointXYZISD)
        pcd = np.array(list(pcd), dtype=np.float32)
        #print(rospy.Time.to_sec(data.header.stamp), pcd.shape)

        mBuf.acquire()
        surfpointsBuf.append([data, pcd])
        mBuf.release()
    
    threading.Thread(target=flatCloudHandlerThread, args=[data]).start()


def odometryHandler(data:Odometry):

    def odometryHandlerThread(data:Odometry):
        _q_wodom_curr = np.zeros(4)
        _t_wodom_curr = np.zeros([3, 1])
        _q_wodom_curr[0] = data.pose.pose.orientation.x
        _q_wodom_curr[1] = data.pose.pose.orientation.y
        _q_wodom_curr[2] = data.pose.pose.orientation.z
        _q_wodom_curr[3] = data.pose.pose.orientation.w
        _t_wodom_curr[0][0] = data.pose.pose.position.x
        _t_wodom_curr[1][0] = data.pose.pose.position.y
        _t_wodom_curr[2][0] = data.pose.pose.position.z
        _q_wodom_curr = R.from_quat(_q_wodom_curr)

        mBuf.acquire()
        odometryBuf.append([data, _q_wodom_curr, _t_wodom_curr])
        mBuf.release()

        _q_w_curr = (q_wmap_wodom * _q_wodom_curr).as_quat()
        _t_w_curr = q_wmap_wodom.as_matrix() @ _t_wodom_curr + t_wmap_wodom 
        #print(rospy.Time.to_sec(data.header.stamp), 'odom')

        odomAftMapped = Odometry()
        odomAftMapped.header.frame_id = "camera_init"
        odomAftMapped.child_frame_id = "/aft_mapped"
        odomAftMapped.header.stamp = data.header.stamp
        odomAftMapped.pose.pose.orientation.x = _q_w_curr[0]
        odomAftMapped.pose.pose.orientation.y = _q_w_curr[1]
        odomAftMapped.pose.pose.orientation.z = _q_w_curr[2]
        odomAftMapped.pose.pose.orientation.w = _q_w_curr[3]
        odomAftMapped.pose.pose.position.x = _t_w_curr[0][0]
        odomAftMapped.pose.pose.position.y = _t_w_curr[1][0]
        odomAftMapped.pose.pose.position.z = _t_w_curr[2][0]
        pubOdomAftMappedHighFreq.publish(odomAftMapped)

    threading.Thread(target=odometryHandlerThread, args=[data]).start()


def main():

    rospy.init_node("mapping", anonymous=False)
    rospy.Subscriber("/kdd_odom_to_init", Odometry, odometryHandler, queue_size=100)
    rospy.Subscriber("/laser_cloud_3", PointCloud2, laserCloudHandler, queue_size=100)
    rospy.Subscriber("/laser_keypoints_last", PointCloud2, keypointsCloudHandler, queue_size=100)
    rospy.Subscriber("/flat_cloud", PointCloud2, flatCloudHandler, queue_size=100)
    
    global pubOdomAftMappedHighFreq, q_wmap_wodom, t_wmap_wodom

    pubLaserCloudSurround = rospy.Publisher("/laser_cloud_surround", PointCloud2, queue_size=100)
    pubLaserCloudMap = rospy.Publisher("/laser_cloud_map", PointCloud2, queue_size=100)
    pubLaserCloudFull = rospy.Publisher("/velodyne_cloud_registered", PointCloud2, queue_size=100)
    pubOdomAftMapped = rospy.Publisher("/aft_mapped_to_init", Odometry, queue_size=100)
    pubOdomAftMappedHighFreq = rospy.Publisher("/aft_mapped_to_init_high_frec", Odometry, queue_size=100)
    pubLaserAfterMappedPath = rospy.Publisher("/aft_mapped_path", Path, queue_size=100)

    rate = rospy.Rate(50)
    frameCount = 0

    laserPath = Path()
    laserPath.header.frame_id = "camera_init"

    laserCloudCenWidth = 10
    laserCloudCenHeight = 10
    laserCloudCenDepth = 5
    laserCloudWidth = 21
    laserCloudHeight = 21
    laserCloudDepth = 11

    laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth
    SalientPointsArray:List[List[np.ndarray]] = [list() for _ in range(laserCloudNum)]
    NonSalientPointsArray:List[List[np.ndarray]] = [list() for _ in range(laserCloudNum)]


    while not rospy.is_shutdown():
        
        while len(allpointsBuf)>0 and len(keypointsBuf)>0 and \
              len(odometryBuf)>0 and len(surfpointsBuf)>0:
            
            mBuf.acquire()
            
            while len(odometryBuf)>0 and odometryBuf[0][0].header.stamp.to_sec() < allpointsBuf[0][0].header.stamp.to_sec():
                odometryBuf.pop(0)
            if len(odometryBuf)==0:
                mBuf.release(); break
            
            while len(keypointsBuf)>0 and keypointsBuf[0][0].header.stamp.to_sec() < allpointsBuf[0][0].header.stamp.to_sec():
                keypointsBuf.pop(0)
            if len(keypointsBuf)==0:
                mBuf.release(); break
            
            while len(surfpointsBuf)>0 and surfpointsBuf[0][0].header.stamp.to_sec() < allpointsBuf[0][0].header.stamp.to_sec():
                surfpointsBuf.pop(0)
            if len(surfpointsBuf)==0:
                mBuf.release(); break
            
            t_total = time.time()
        
            timeSurfPointsLast = surfpointsBuf[0][0].header.stamp.to_sec()
            timeKeyPointsLast = keypointsBuf[0][0].header.stamp.to_sec()
            timeAllPointsLast = allpointsBuf[0][0].header.stamp.to_sec()
            timeLaserOdometry = odometryBuf[0][0].header.stamp.to_sec()
            if  timeAllPointsLast != timeKeyPointsLast or \
                timeKeyPointsLast != timeLaserOdometry or \
                timeLaserOdometry != timeSurfPointsLast:
                print("unsync message!")
                mBuf.release(); break

            odom = odometryBuf.pop(0)
            surfpointsLast = surfpointsBuf.pop(0)
            keypointsLast = keypointsBuf.pop(0)
            allpointsLast = allpointsBuf.pop(0)
            
            print("stamp %.2f, "%timeLaserOdometry, end='')
            print("buffer length %d"%len(odometryBuf))

            allpointsBuf.clear()
            mBuf.release()

            keypoints:np.ndarray = keypointsLast[1]
            odom, q_wodom_curr, t_wodom_curr = odom
            q_w_curr = q_wmap_wodom * q_wodom_curr
            t_w_curr = q_wmap_wodom.as_matrix() @ t_wodom_curr + t_wmap_wodom
            print("synchronization time %f s"%(time.time()-t_total))

            t_prepare = time.time()
            
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
            # i.e., 3 < centerCubeI < 18, 3 < centerCubeJ < 18, 3 < centerCubeK < 8,
            # then it will be more convenient to extent the cube map later.
            while centerCubeI < 3:
                for j in range(laserCloudHeight):
                    for k in range(laserCloudDepth): 
                        i = laserCloudWidth - 1
                        cubePointer = SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k]
                        for i in range(laserCloudWidth - 1, 0, -1):
                            SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = \
                                SalientPointsArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k]
                        SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = cubePointer
                        cubePointer.clear()
                centerCubeI = centerCubeI + 1
                laserCloudCenWidth = laserCloudCenWidth + 1
            
            while centerCubeI >= laserCloudWidth - 3:
                for j in range(laserCloudHeight):
                    for k in range(laserCloudDepth): 
                        i = 0
                        cubePointer = SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k]
                        for i in range(0, laserCloudWidth - 1, 1):
                            SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = \
                                SalientPointsArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k]
                        SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = cubePointer
                        cubePointer.clear()
                centerCubeI = centerCubeI - 1
                laserCloudCenWidth = laserCloudCenWidth - 1
            
            while centerCubeJ < 3:
                for i in range(laserCloudWidth):
                    for k in range(laserCloudDepth): 
                        j = laserCloudHeight - 1
                        cubePointer = SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k]
                        for i in range(laserCloudHeight - 1, 0, -1):
                            SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = \
                                SalientPointsArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k]
                        SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = cubePointer
                        cubePointer.clear()
                centerCubeJ = centerCubeJ + 1
                laserCloudCenHeight = laserCloudCenHeight + 1
            
            while centerCubeJ >= laserCloudHeight - 3:
                for i in range(laserCloudWidth):
                    for k in range(laserCloudDepth): 
                        j = 0
                        cubePointer = SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k]
                        for i in range(0, laserCloudHeight - 2, 1):
                            SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = \
                                SalientPointsArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k]
                        SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = cubePointer
                        cubePointer.clear()
                centerCubeJ = centerCubeJ - 1
                laserCloudCenHeight = laserCloudCenHeight - 1

            while centerCubeK < 3:
                for i in range(laserCloudWidth):
                    for j in range(laserCloudHeight): 
                        k = laserCloudDepth - 1
                        cubePointer = SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k]
                        for k in range(laserCloudDepth - 1, 0, -1):
                            SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = \
                                SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)]
                        SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = cubePointer
                        cubePointer.clear()
                centerCubeK = centerCubeK + 1
                laserCloudCenDepth = laserCloudCenDepth + 1
            
            while centerCubeK >= laserCloudDepth - 3:
                for i in range(laserCloudWidth):
                    for j in range(laserCloudHeight): 
                        k = 0
                        cubePointer = SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k]
                        for k in range(0, laserCloudDepth - 2, 1):
                            SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = \
                                SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)]
                        SalientPointsArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = cubePointer
                        cubePointer.clear()
                centerCubeK = centerCubeK - 1
                laserCloudCenDepth = laserCloudCenDepth - 1
            
            # extend 2 cubes in two directions along axis I&J(x&y)
            # extend 1 cube  in two directions along axis K(z), (250m*2)*(250m*2)
            # totally 125 cubes (75???) select those in the local FOV

            laserCloudSurroundNum = 0
            laserCloudSurroundInd:List[int] = list()

            for i in range(centerCubeI - 2, centerCubeI + 3):
                for j in range(centerCubeJ - 2, centerCubeJ + 3):
                    for k in range(centerCubeK - 1, centerCubeK + 2):
                        if  i >= 0 and i < laserCloudWidth and \
                            j >= 0 and j < laserCloudHeight and \
                            k >= 0 and k < laserCloudDepth:
                            ind = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k
                            laserCloudSurroundInd.append(ind)
                            laserCloudSurroundNum = laserCloudSurroundNum + 1
            
            laserCloudKeypointsFromMap:List[np.ndarray] = list()
            for i in range(laserCloudSurroundNum):
                laserCloudKeypointsFromMap += SalientPointsArray[laserCloudSurroundInd[i]]
            print("local map with %d points"%(len(laserCloudKeypointsFromMap)))
            print("map preparation time %f s"%(time.time()-t_prepare))

            if len(laserCloudKeypointsFromMap) > 50:
                laserCloudKeypointsFromMap = np.array(laserCloudKeypointsFromMap)
                """ TODO: scan-to-map registration """

                t_reg = time.time()
                T_w_curr = np.eye(4)
                T_w_curr[:3, 3:] = t_w_curr
                T_w_curr[:3, :3] = q_w_curr.as_matrix()

                dist = np.linalg.norm(laserCloudKeypointsFromMap[:, :3] - t_w_curr.T, axis=1)
                laserCloudKeypointsFromMap = laserCloudKeypointsFromMap[dist < 90.]

                src = o3d.geometry.PointCloud()
                tar = o3d.geometry.PointCloud()
                src.points = o3d.utility.Vector3dVector(laserCloudKeypointsFromMap[:, :3])
                tar.points = o3d.utility.Vector3dVector(keypoints[:, :3])
                """
                t_match = time.time()
                
                TwoStageLocalFeatureMatching(
                    deepcopy(src), tar, laserCloudKeypointsFromMap[:, -32:],
                    keypoints[:, -32:], np.linalg.inv(T_w_curr), 2.4, 3.0
                )
                print("matching time: %.4fs"%(time.time() - t_match))
                """
                """
                corres = LocalFeatureMatching(
                    keypoints[:, :3], laserCloudKeypointsFromMap[:, :3],
                    keypoints[:, -32:], laserCloudKeypointsFromMap[:, -32:],
                    1.2, T_w_curr, 20, 0.7, 8
                )
                print("%d correspondences, %.4fs"%(len(corres), time.time() - t_match))
                t = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                    tar, src, o3d.utility.Vector2iVector(corres), 0.3,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
                    [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.3)],
                    o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.999)
                )
                """
                """
                t = RegistrationBasedOnCorrespondences(
                    tar, src, 0.3, corres, T_w_curr,
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5),
                    #init_correspondence_distance=0.9, decay_rate=0.9
                )
                """
                
                t = o3d.pipelines.registration.registration_icp(
                    tar, src, 0.3, T_w_curr, #, T_w_curr
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
                )
                """
                t = o3d.pipelines.registration.registration_generalized_icp(
                    tar, src, 0.3, T_w_curr,
                    o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
                )
                """

                T_w_curr = np.array(t.transformation); print(t)
                q_w_curr = R.from_matrix(T_w_curr[:3, :3])
                t_w_curr = T_w_curr[:3, -1:]

                print("registration time: %f s"%(time.time() - t_reg))
            
            q_wmap_wodom = q_w_curr * q_wodom_curr.inv()
            t_wmap_wodom = t_w_curr - q_wmap_wodom.as_matrix() @ t_wodom_curr

            t_add = time.time()
            if frameCount == 0: keypoints = allpointsLast[1]
            keypoints[:, :3] = (q_w_curr.as_matrix() @ keypoints[:, :3].T + t_w_curr).T
            cube = np.floor((keypoints[:, :3] + 25.0) / 50.0).astype(np.int32)
            cube = cube + np.array([[laserCloudCenWidth, laserCloudCenHeight, laserCloudCenDepth]])
            idx = cube[:, 0] + laserCloudWidth * cube[:, 1] + laserCloudWidth * laserCloudHeight * cube[:, 2]
            for i in range(keypoints.shape[0]):
                SalientPointsArray[idx[i]].append(keypoints[i])
            ValidCubes = set(idx).intersection(set(laserCloudSurroundInd))
            print("adding points time %f s"%(time.time()-t_add))

            t_filter = time.time()
            for ind in ValidCubes:
                if len(SalientPointsArray[ind])<1: continue
                SalientPointsArray[ind] = uniform_sampling(0.3, SalientPointsArray[ind])

            print("filter time %f s"%(time.time()-t_filter))

            t_publish = time.time()

            if frameCount % 5 == 0:
                laserCloudSurround: List[np.ndarray] = list()
                for i in laserCloudSurroundInd:
                    laserCloudSurround += SalientPointsArray[i]

                """ TODO: render the point cloud according to saliency"""
                laserCloudSurround = np.array(laserCloudSurround)[:, :5]
                surroundMsg.header.stamp = odom.header.stamp
                surroundMsg.header.frame_id = "camera_init"
                surroundMsg.width = laserCloudSurround.shape[0]
                surroundMsg.row_step = surroundMsg.point_step * surroundMsg.width
                surroundMsg.data = laserCloudSurround.tobytes()
                pubLaserCloudSurround.publish(surroundMsg)
            
            if frameCount % 15 == 0:
                laserCloudMap: List[np.ndarray] = list()
                for i in range(4851):
                    laserCloudMap += SalientPointsArray[i]

                """ TODO: render the point cloud according to saliency"""
                laserCloudMap = np.array(laserCloudMap)[:, :5]
                mapMsg.header.stamp = odom.header.stamp
                mapMsg.header.frame_id = "camera_init"
                mapMsg.width = laserCloudMap.shape[0]
                mapMsg.row_step = mapMsg.point_step * mapMsg.width
                mapMsg.data = laserCloudMap.tobytes()
                pubLaserCloudMap.publish(mapMsg)

            """ TODO: render the point cloud according to saliency"""
            scan = allpointsLast[1][:, :5]
            scan[:, :3] = (q_w_curr.as_matrix() @ scan[:, :3].T + t_w_curr).T
            scanMsg.header.stamp = odom.header.stamp
            scanMsg.header.frame_id = "camera_init"
            scanMsg.width = scan.shape[0]
            scanMsg.row_step = scanMsg.point_step * scanMsg.width
            scanMsg.data = scan.tobytes()
            pubLaserCloudFull.publish(scanMsg)

            q = q_w_curr.as_quat()
            odomAftMapped = Odometry()
            odomAftMapped.header.frame_id = "camera_init"
            odomAftMapped.child_frame_id = "/aft_mapped"
            odomAftMapped.header.stamp = odom.header.stamp
            odomAftMapped.pose.pose.orientation.x = q[0]
            odomAftMapped.pose.pose.orientation.y = q[1]
            odomAftMapped.pose.pose.orientation.z = q[2]
            odomAftMapped.pose.pose.orientation.w = q[3]
            odomAftMapped.pose.pose.position.x = t_w_curr[0][0]
            odomAftMapped.pose.pose.position.y = t_w_curr[1][0]
            odomAftMapped.pose.pose.position.z = t_w_curr[2][0]
            pubOdomAftMapped.publish(odomAftMapped)

            laserPose = PoseStamped()
            laserPose.header = odomAftMapped.header
            laserPose.pose = odomAftMapped.pose.pose
            laserPath.header.stamp = odomAftMapped.header.stamp
            laserPath.poses.append(laserPose)
            pubLaserAfterMappedPath.publish(laserPath)
            print("publish time %f s"%(time.time() - t_publish))

            print(rospy.Time.to_sec(odom.header.stamp), end=' ')
            print(time.time()-t_total, '\n')
            frameCount = frameCount + 1
        
    rospy.spin()

if __name__=="__main__":
    main()
