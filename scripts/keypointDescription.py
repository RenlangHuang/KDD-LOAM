import time
import torch
import rospy
import threading
import numpy as np
from typing import List
from copy import deepcopy
from models.kpfcnn import KPFCNN
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from utils.precompute import precompute_data


msgBuf: List[list] = list()
mBuf = threading.Lock()
BUFFER_SIZE = 10

knn = [34, 32, 34, 34, 39]
feature = torch.ones([40000, 1], device='cuda:0')
model = KPFCNN(1, 32, 64, 15, 0.9, 0.6, 'group_norm', 32)
model.load_state_dict(torch.load('./checkpoints/kitti_HCL64_augm_23500.pth'))
model.eval().cuda()


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
    pcd = point_cloud2.read_points(data)
    pcd = np.array(list(pcd), dtype=np.float32)
    xyz = torch.from_numpy(pcd[:, :3])
    feat = feature[:xyz.shape[0]]
    mBuf.acquire()
    msgBuf.append([data, pcd, xyz, feat])
    mBuf.release()


def main():
    rospy.init_node("keypointDescription", anonymous=False)
    pubLaserCloudLast = rospy.Publisher("/deep_laser_cloud", PointCloud2, queue_size=100)
    rospy.Subscriber("/velodyne_cloud", PointCloud2, laserCloudHandler, queue_size=100)
    
    while not rospy.is_shutdown():
        if len(msgBuf) > 0:
            start = time.time()
            mBuf.acquire()
            msg = msgBuf.pop(0)
            while len(msgBuf)>BUFFER_SIZE: msgBuf.pop(0)
            print('buffer length: ', len(msgBuf))
            mBuf.release()

            data = precompute_data(msg[2], 5, 0.3, 0.9, knn)
            data['features'] = msg[3]
            with torch.no_grad():
                descriptor, saliency = model(data)
            pubmsg.header = msg[0].header
            pubmsg.width = msg[1].shape[0]
            
            pubmsg.row_step = pubmsg.point_step * msg[1].shape[0]
            data = [msg[1], saliency.cpu().numpy(), descriptor.cpu().numpy()]
            data = np.concatenate(data, axis=-1, dtype=np.float32)
            pubmsg.data = data.tobytes()
            pubLaserCloudLast.publish(pubmsg)
            
            print(rospy.Time.to_sec(msg[0].header.stamp), end=' ')
            print(time.time()-start, data.shape, '\n')
    
    rospy.spin()

if __name__ == '__main__':
    main()
