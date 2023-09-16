import time
import rospy
import argparse
import threading
import numpy as np
import open3d as o3d
from typing import List
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R


paser = argparse.ArgumentParser()
paser.add_argument("--num_keypoints", type=int, default=4500)
paser.add_argument("--ransac_iterations", type=int, default=20000)
paser.add_argument("--multi_threads_mode", type=bool, default=True)
args = paser.parse_args()


fields = ['x','y','z','intensity','saliency','descriptor']
allpointsBuf: List[List[PointCloud2]] = list()
mBuf = threading.Lock()

params = {
    "ransac_max_iters": args.ransac_iterations,
    "ransac_inlier_threshold": 0.3,
    "edge_prune": 0.8,
    "dist_prune": 0.3,
    "ransac_n": 4,
}


def laserCloudHandler(data:PointCloud2):

    def laserCloudHandlerThread(data:PointCloud2):
        pcd = point_cloud2.read_points(data, field_names=fields)
        pcd = np.array(list(pcd), dtype=np.float32)
        pcd = pcd[np.argsort(pcd[:, 4])] # sort by saliency
        print(rospy.Time.to_sec(data.header.stamp), pcd.shape)

        mBuf.acquire()
        allpointsBuf.append([data, pcd])
        mBuf.release()
    
    if args.multi_threads_mode:
        threading.Thread(target=laserCloudHandlerThread, args=[data]).start()
    else: laserCloudHandlerThread(data)


def main():

    rospy.init_node("odometry", anonymous=False)
    rospy.Subscriber("/deep_laser_cloud", PointCloud2, laserCloudHandler, queue_size=100)

    pubLaserOdometry = rospy.Publisher("/kdd_odom_to_init", Odometry, queue_size=100)
    pubLaserPath = rospy.Publisher("/kdd_odom_path", Path, queue_size=100)

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
    rate = rospy.Rate(50)

    while not rospy.is_shutdown():
        if len(allpointsBuf)>0:
            
            start = time.time()
            mBuf.acquire()
            allpoints = allpointsBuf.pop(0)
            while len(allpointsBuf)>BUFFER_SIZE:
                allpointsBuf.pop(0)
            print('buffer length: ', len(allpointsBuf))
            mBuf.release()
            
            t_points = allpoints[1][:args.num_keypoints, :3]
            t_desc = allpoints[1][:args.num_keypoints, -32:]
            
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
                print(trans) #trans = np.array(trans.transformation)
                trans = o3d.pipelines.registration.TransformationEstimationPointToPoint(False).compute_transformation(
                    t_pcd, s_pcd, o3d.utility.Vector2iVector(trans.correspondence_set))

                t_last_curr  = trans[:3, -1:]
                q_last_curr = R.from_matrix(trans[:3, :3])
                t_w_curr = t_w_curr + q_w_curr.as_matrix() @ t_last_curr
                q_w_curr = q_w_curr * q_last_curr
                q_w_curr_ = q_w_curr.as_quat()
                
                laserOdometry = Odometry()
                laserOdometry.header = allpoints[0].header
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
                
                pubLaserOdometry.publish(laserOdometry)
                pubLaserPath.publish(laserPath)
            
            s_points, s_desc = t_points, t_desc
            print(rospy.Time.to_sec(allpoints[0].header.stamp), end=' ')
            print(time.time()-start, '\n')
        
        rate.sleep()
    
    rospy.spin()

if __name__ == '__main__':
    main()
