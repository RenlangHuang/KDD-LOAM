import os
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R


DATA_PATH = 'YOUR_DATA_PATH'
KITTI_path = DATA_PATH + 'KITTI_data_odometry/'
#path = os.path.dirname(os.path.abspath(__file__))


class KITTIDataset(object):
    def __init__(self, mode=None, augmented=0.):
        self.MINIMUM_RANGE = 2.5
        self.MAXIMUM_RANGE = 100.0
        self.augmented = augmented
        if isinstance(mode, str):
            if mode=='train':
                self.seqs = ['%02d'%i for i in range(6)]
            elif mode=='val':
                self.seqs = ['%02d'%i for i in range(6,8)]
            elif mode=='test':
                self.seqs = ['%02d'%i for i in range(8,11)]
            elif mode=='trainval':
                self.seqs = ['%02d'%i for i in range(8)]
            else: # all of the sequences
                self.seqs = ['%02d'%i for i in range(11)]
        else: # mode is a list of sequences
            self.seqs = ['%02d'%i for i in mode]
        self._pairs, self._gt = list(), list()

        if not os.path.exists(KITTI_path + 'velodyne/pairs'):
            os.makedirs(KITTI_path + 'velodyne/pairs')
            for i in range(11): # only 00~10 sequences have ground truth 
                if os.path.exists(KITTI_path + 'velodyne/sequences/%02d'%i):
                    self.generate_pairs('%02d'%i) # generate point cloud pairs
        
        for i in self.seqs:
            with np.load(KITTI_path+'velodyne/pairs/'+i+'.npz') as data:
                self._pairs.append([np.array([[int(i)]*data['pairs'].shape[0]]).T])
                self._pairs[-1].append(data['pairs']) # [N, 3], (seq, src, tar)
                self._pairs[-1] = np.concatenate(self._pairs[-1],axis=-1)
                self._gt.append(data['gt'])

        self.kdts = o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=16)
        self._pairs = np.concatenate(self._pairs, axis=0)
        self._gt = np.concatenate(self._gt, axis=0)
        self._size = self._pairs.shape[0]
    
    def __len__(self):
        return 100000000 #self._size
    
    def __getitem__(self, index):
        if self._size == 0: raise StopIteration
        pair = self._pairs[index % self._size]
        trans = self._gt[index % self._size]

        ply1 = o3d.geometry.PointCloud()
        ply2 = o3d.geometry.PointCloud()
        ply1.points = o3d.utility.Vector3dVector(
            self.read_point_cloud('%02d'%pair[0], '%06d.bin'%pair[1])
        )
        ply2.points = o3d.utility.Vector3dVector(
            self.read_point_cloud('%02d'%pair[0], '%06d.bin'%pair[2])
        )
        ply1 = ply1.voxel_down_sample(voxel_size=0.3)
        ply2 = ply2.voxel_down_sample(voxel_size=0.3)
        #ply1.estimate_normals(search_param=self.kdts)
        #ply2.estimate_normals(search_param=self.kdts)
        #ply1.orient_normals_towards_camera_location(np.array([0.,0.,2.]))
        #ply2.orient_normals_towards_camera_location(np.array([0.,0.,2.]))

        # ply1, _ = ply1.remove_statistical_outlier(nb_neighbors=8, std_ratio=4.2)
        # ply2, _ = ply2.remove_statistical_outlier(nb_neighbors=8, std_ratio=4.2)

        if np.random.random()<self.augmented: # random rotation
            r1 = self.sample_random_rotation(axis_scale=0.)
            r2 = self.sample_random_rotation(axis_scale=0.)
            ply1.transform(r1); ply2.transform(r2)
            trans = r1 @ trans @ np.linalg.inv(r2)
        
        xyz = [torch.from_numpy(np.array(ply1.points, dtype=np.float32)),
               torch.from_numpy(np.array(ply2.points, dtype=np.float32))]
        features = [torch.ones_like(xyz[0][:,:1]), torch.ones_like(xyz[1][:,:1])]
        return xyz[0], xyz[1], features[0], features[1], trans, None
    
    def read_point_cloud(self, dataset, filename):
        pcd = KITTI_path + 'velodyne/sequences/' + dataset + '/velodyne/' + filename
        pcd = np.fromfile(pcd, dtype=np.float32, count=-1)
        pcd = np.reshape(pcd, [-1, 4])[:, :-1] # [x, y, z, intensity], return [N, 3]
        rmv = np.linalg.norm(pcd[:, :2], ord=2, axis=-1)
        rmv = np.logical_and(rmv>self.MINIMUM_RANGE, rmv<self.MAXIMUM_RANGE)
        rmv = np.logical_and(rmv, pcd[:, 2]>-5.0)
        pcd = pcd[np.argwhere(rmv)[:, 0]]
        return pcd
    
    def read_groundtruth(self, dataset):
        gt = np.genfromtxt(KITTI_path + 'results/' + dataset + '.txt').reshape([-1, 3, 4])
        gt = np.concatenate([gt,np.repeat(np.array([[[0,0,0,1.]]]),gt.shape[0],axis=0)],axis=1)

        # these transformations are under the left camera's coordinate system
        calibf = open(KITTI_path + 'sequences/' + dataset + '/calib.txt')
        for t in calibf.readlines(): # [P0, P1, P2, P3, Tr]
            if t[0]=='T': t = t[4:]; break # Tr (camera0->LiDAR)
        calibf.close(); calib = np.eye(4, 4)
        calib[:-1, :] = np.array([float(c) for c in t.split(' ')]).reshape([3, 4])
        return np.linalg.inv(calib) @ gt @ calib
    
    def read_sequence(self, dataset):
        scans = os.listdir(KITTI_path + 'velodyne/sequences/' + dataset + '/velodyne/')
        return len(scans)
    
    def generate_pairs(self, dataset, min_dist=10.):
        kitti_pairs, refined_trans = list(), list()
        lenseq = self.read_sequence(dataset)
        trans = self.read_groundtruth(dataset); gt = trans[:, :3, 3]
        pdist = gt.reshape([1, -1, 3]) - gt.reshape([-1, 1, 3])
        pdist = np.sqrt(np.sum(pdist**2,axis=-1)) >= min_dist
        curr_time = 0 # generate pairs >= 10m
        print('Start generating KITTI pairs (Sequence %02d)...'%int(dataset))
        
        while curr_time < lenseq:
            next_time = np.where(pdist[curr_time][curr_time:curr_time + 100])[0]
            if len(next_time) == 0: curr_time = curr_time + 1
            else: # next_time <= lenseq-1
                next_time = next_time[0] + curr_time - 1
                if (int(dataset),curr_time,next_time) in [(8, 15, 58)]:
                    curr_time = next_time + 1 # problematic pair !
                else: # refine the transformation estimation
                    kitti_pairs.append((curr_time, next_time));print(kitti_pairs[-1])
                    ply1 = o3d.geometry.PointCloud()
                    ply2 = o3d.geometry.PointCloud()
                    xyz1 = self.read_point_cloud(dataset, '%06d.bin'%curr_time)
                    xyz2 = self.read_point_cloud(dataset, '%06d.bin'%next_time)
                    
                    ply1.points = o3d.utility.Vector3dVector(xyz1)
                    ply2.points = o3d.utility.Vector3dVector(xyz2)
                    ply1 = ply1.voxel_down_sample(voxel_size=0.05)
                    ply2 = ply2.voxel_down_sample(voxel_size=0.05)
                    
                    t = np.linalg.inv(trans[curr_time]) @ trans[next_time]
                    t = o3d.pipelines.registration.registration_icp(
                        ply2, ply1, 0.2, t, # refine the transformation matrix via ICP
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500))
                    refined_trans.append(t.transformation)
                    curr_time = next_time + 1

        print('Done with %d pairs.'%len(kitti_pairs))
        kitti_pairs = np.array(kitti_pairs,dtype=np.int32)
        refined_trans = np.array(refined_trans,dtype=np.float32)
        np.savez(KITTI_path+'velodyne/pairs/'+dataset+'.npz', pairs=kitti_pairs, gt=refined_trans)
    
    def sample_random_rotation(self, centroid=np.zeros(3), axis_scale=0.05, angle_scale=np.pi):
        r = np.random.normal(scale=axis_scale,size=3)
        r[-1] = 1.; r = r/np.linalg.norm(r)
        r = R.from_rotvec(r*np.random.uniform(-angle_scale,angle_scale))
        trans = np.eye(4); trans[:3, :3] = r.as_matrix()
        trans[:3, -1] = -trans[:3, :3] @ centroid
        return trans
