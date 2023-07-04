import time
import rospy
import threading
import numpy as np
import open3d as o3d
from typing import List
from copy import deepcopy
from scipy.spatial import cKDTree
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R


laserCloudCenWidth = 10
laserCloudCenHeight = 10
laserCloudCenDepth = 5
laserCloudWidth = 21
laserCloudHeight = 21
laserCloudDepth = 11
laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth; #4851

laserCloudValidInd = [None for _ in range(125)]
laserCloudSurroundInd = [None for _ in range(125)]

# points in every cube
laserKeypointsArray: List[np.ndarray] = [np.array([]) for _ in laserCloudNum]

# KD-trees

# wmap_T_odom * odom_T_curr = wmap_T_curr
# transformation between odom's world and map's world frame
q_wmap_wodom = R.from_matrix(np.eye(3))
t_wmap_wodom = np.zeros([3, 1])

q_wodom_curr = R.from_matrix(np.eye(3))
t_wodom_curr = np.zeros([3, 1])

q_w_curr = R.from_matrix(np.eye(3))
t_w_curr = np.zeros([3, 1])

fields = ['x','y','z','intensity','saliency','descriptor']
keypointsBuf: List[List[PointCloud2]] = list()
allpointsBuf: List[PointCloud2] = list()
odometryBuf: List[Odometry] = list()
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
    print(rospy.Time.to_sec(data.header.stamp))
    mBuf.acquire()
    allpointsBuf.append(data)
    mBuf.release()


def keypointsCloudHandler(data:PointCloud2):
    pcd = point_cloud2.read_points(data, field_names=fields)
    pcd = np.array(list(pcd), dtype=np.float32)
    print(rospy.Time.to_sec(data.header.stamp), pcd.shape)

    mBuf.acquire()
    keypointsBuf.append([data, pcd])
    mBuf.release()


def odometryHandler(data:Odometry):
    mBuf.acquire()
    odometryBuf.append(data)
    mBuf.release()


def main():

    rospy.init_node("mapping", anonymous=False)
    rospy.Subscriber("/laser_odom_to_init", Odometry, odometryHandler, queue_size=100)
    rospy.Subscriber("/velodyne_cloud_3", PointCloud2, laserCloudHandler, queue_size=100)
    rospy.Subscriber("/laser_cloud_keypoints_last", PointCloud2, keypointsCloudHandler, queue_size=100)

    pubLaserCloudLast = rospy.Publisher("/velodyne_cloud_3", PointCloud2, queue_size=100)
    pubKeypointsLast = rospy.Publisher("/laser_cloud_keypoints_last", PointCloud2, queue_size=100)
    pubLaserOdometry = rospy.Publisher("/laser_odom_to_init", Odometry, queue_size=100)
    pubLaserPath = rospy.Publisher("/laser_odom_path", Path, queue_size=100)

    s_pcd = o3d.geometry.PointCloud()
    t_pcd = o3d.geometry.PointCloud()
    s_descriptors = o3d.pipelines.registration.Feature()
    t_descriptors = o3d.pipelines.registration.Feature()

    q_w_curr = R.from_matrix(np.eye(3))
    t_w_curr = np.zeros([3, 1])

    q_last_curr = R.from_matrix(np.eye(3))
    t_last_curr = np.zeros([3, 1])
    
    laserPath = Path()
    laserPath.header.frame_id = "camera_init"

    BUFFER_SIZE = 10
    systemInited = False

    while not rospy.is_shutdown():
        if len(allpointsBuf)>0 and len(keypointsBuf)>0:
            timeAllPoints = rospy.Time.to_sec(allpointsBuf[0].header.stamp)
            timeKeyPoints = rospy.Time.to_sec(keypointsBuf[0][0].header.stamp)
            if timeAllPoints != timeKeyPoints:
                print("unsync message!")
                raise rospy.ROSInterruptException
            
            start = time.time()
            mBuf.acquire()
            allpoints = allpointsBuf.pop(0)
            keypoints = keypointsBuf.pop(0)
            while len(allpointsBuf)>BUFFER_SIZE:
                allpointsBuf.pop(0)
                keypointsBuf.pop(0)
            print('buffer length: ', len(allpointsBuf))
            mBuf.release()
            
            t_points = keypoints[1][:, :3]
            t_desc = keypoints[1][:, -32:]
            
            if systemInited is not True:
                systemInited = True
                print("Initialization finished.")
            
            else:
                s_pcd.points = o3d.utility.Vector3dVector(s_points)
                t_pcd.points = o3d.utility.Vector3dVector(t_points)
                
                s_descriptors.data = s_desc.T
                t_descriptors.data = t_desc.T

                trans = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                    t_pcd, s_pcd, t_descriptors, s_descriptors, True, params["ransac_inlier_threshold"],
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False), params["ransac_n"],
                    [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(params["edge_prune"]),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(params["dist_prune"])],
                    o3d.pipelines.registration.RANSACConvergenceCriteria(params["ransac_max_iters"], 0.999))
                print(trans); trans = np.array(trans.transformation)
            
                t_last_curr  = trans[:3, -1:]
                q_last_curr = R.from_matrix(trans[:3, :3])
                t_w_curr = t_w_curr + q_w_curr.as_matrix() @ t_last_curr
                q_w_curr = q_w_curr * q_last_curr
                q_w_curr_ = q_w_curr.as_quat()
                
                laserOdometry = Odometry()
                laserOdometry.header = allpoints.header
                laserOdometry.header.frame_id = "camera_init"
                laserOdometry.child_frame_id = "/laser_odom"
                laserOdometry.pose.pose.orientation.x = q_w_curr_[0]
                laserOdometry.pose.pose.orientation.y = q_w_curr_[1]
                laserOdometry.pose.pose.orientation.z = q_w_curr_[2]
                laserOdometry.pose.pose.orientation.w = q_w_curr_[3]
                laserOdometry.pose.pose.position.x = t_w_curr[0][0]
                laserOdometry.pose.pose.position.y = t_w_curr[1][0]
                laserOdometry.pose.pose.position.z = t_w_curr[2][0]

                laserPose = PoseStamped()
                laserPose.header = laserOdometry.header
                laserPose.pose = laserOdometry.pose.pose
                laserPath.header.stamp = laserOdometry.header.stamp
                laserPath.poses.append(laserPose)
                
                pcd = point_cloud2.read_points(allpoints, field_names=fields)
                pcd = np.array(list(pcd), dtype=np.float32)
                pubmsg.header = allpoints.header
                pubmsg.width = pcd.shape[0]
                pubmsg.row_step = pubmsg.point_step * pubmsg.width
                pubmsg.data = pcd.tobytes()

                pubKeypointsMsg.header = keypoints[0].header
                pubKeypointsMsg.width = keypoints[1].shape[0]
                pubKeypointsMsg.row_step = pubKeypointsMsg.point_step * pubKeypointsMsg.width
                pubKeypointsMsg.data = keypoints[1].tobytes()

                pubLaserCloudLast.publish(pubmsg) 
                pubKeypointsLast.publish(pubKeypointsMsg)
                pubLaserOdometry.publish(laserOdometry)
                pubLaserPath.publish(laserPath)
            
            s_points, s_desc = t_points, t_desc
            print(rospy.Time.to_sec(allpoints.header.stamp), end=' ')
            print(time.time()-start, '\n')
    
    rospy.spin()

if __name__ == '__main__':
    main()