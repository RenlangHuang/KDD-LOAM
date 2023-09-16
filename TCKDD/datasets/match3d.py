import os
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R


#path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = '/media/jacko/SSD/' #'F:/' #'/remote-home/share/ums_huangrenlang/'
match3d_path = DATA_PATH + '3dmatch_predator/metadata/3DMatch/'
lomatch3d_path = DATA_PATH + '3dmatch_predator/metadata/3DLoMatch/'
match3d_train_path = DATA_PATH + '3dmatch_predator/data/train/'
match3d_test_path = DATA_PATH + '3dmatch_predator/data/test/'


class ThreeDMatchTrainDataset(Dataset):
    def __init__(self, scenes=None, augmented=0., overlap=0.3):
        self.augmented = augmented # 0. for no augmentation (as the probability)
        if isinstance(scenes, list): self.seqs = scenes
        else: self.seqs = os.listdir(match3d_train_path)
        self.seqs.sort() # keep them in order
        self._pairs, self._gt = list(), list()
        for (i, seq) in enumerate(self.seqs):
            pairs, trans, _ = self.read_ground_truth(seq, overlap)
            self._pairs.append([np.array([[i] * pairs.shape[0]]).T])
            self._pairs[-1].append(pairs) # [N, 3], (seq, src, tar)
            self._pairs[-1] = np.concatenate(self._pairs[-1],axis=-1)
            self._gt.append(trans)
        
        self.kdts = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16)
        self._pairs = np.concatenate(self._pairs, axis=0)
        self._gt = np.concatenate(self._gt, axis=0)
        self._size = self._pairs.shape[0]
    
    def __len__(self):
        return 100000000 #self._size
    
    def __getitem__(self, index):
        if self._size == 0: raise StopIteration
        pair = self._pairs[index % self._size]
        trans = self._gt[index % self._size]
        seq, pair = self.seqs[pair[0]], pair[1:]

        pcd1 = self.read_point_cloud(seq, pair[0]).voxel_down_sample(voxel_size=0.03)
        pcd2 = self.read_point_cloud(seq, pair[1]).voxel_down_sample(voxel_size=0.03)
        
        #pcd1.estimate_normals(search_param=self.kdts)
        #pcd2.estimate_normals(search_param=self.kdts)
        #pcd1.orient_normals_towards_camera_location(np.array([0.,0.,0.]))
        #pcd2.orient_normals_towards_camera_location(np.array([0.,0.,0.]))

        if np.random.random()<self.augmented: # random rotation
            r1 = self.sample_random_rotation()
            r2 = self.sample_random_rotation()
            pcd1.transform(r1); pcd2.transform(r2)
            trans = r1 @ trans @ np.linalg.inv(r2)
        
        xyz = [torch.from_numpy(np.array(pcd1.points, dtype=np.float32)),
               torch.from_numpy(np.array(pcd2.points, dtype=np.float32))]
        features = [torch.ones_like(xyz[0][:,:1]), torch.ones_like(xyz[1][:,:1])]
        return xyz[0], xyz[1], features[0], features[1], trans, None
    
    def read_point_cloud(self, seq, id):
        return o3d.io.read_point_cloud(match3d_train_path + seq + '/fragments/cloud_bin_%d.ply'%id)
    
    def read_extrinsic(self, seq, id):
        trans = list() # homogeneous transformation matrix
        f = open(match3d_train_path + seq + '/poses/cloud_bin_%d.txt'%id, 'r')
        for (i,t) in enumerate(f.readlines()): # [info, T0, T1, T2, T3]
            if i==0: info = t.split() # scene, sequence, frames (start, end)
            else: trans.append([float(c) for c in t.split()])
        trans = np.array(trans, dtype=np.float32)
        f.close() #; print(info); print(trans)
        return trans
    
    def read_ground_truth(self, seq, overlap=0.3):
        data = np.load(match3d_train_path + seq + '/pairs-%.1f.npz'%overlap)
        return data['pairs'], data['trans'], data['overlap']
    
    def read_sequence(self, seq):
        scans = os.listdir(match3d_train_path + seq + '/fragments/')
        scans = [int(s[:-4].split('_')[-1]) for s in scans]; scans.sort()
        return scans
    
    def generate_pairs(self, seq, voxel_size=0.03):
        extrinsics = list() # transformation matrices
        fragments = self.read_sequence(seq)
        length = len(os.listdir(match3d_train_path + seq + '/poses/'))
        for i in fragments:
            extrinsics.append(self.read_extrinsic(seq, i))
        extrinsics = np.array(extrinsics)
        pairs, trans, overlap = [], [], []
        
        for i in range(length):
            pcd1 = self.read_point_cloud(seq, fragments[i])
            pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size).points
            kdtree = cKDTree(np.array(pcd1) @ extrinsics[i,:3,:3].T + extrinsics[i,:3,-1:].T)
            itrans1 = np.linalg.inv(extrinsics[i])
            for k in range(i+1, length):
                pcd2 = self.read_point_cloud(seq, fragments[k])
                pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)
                pcd2.transform(extrinsics[k]); pcd2 = pcd2.points
                dist, _ = kdtree.query(np.array(pcd2), k=1, distance_upper_bound=voxel_size)
                ind2 = np.argwhere(dist != np.inf)
                overlap.append(len(ind2)/dist.shape[0])
                trans.append(itrans1 @ extrinsics[k])
                pairs.append((fragments[i], fragments[k]))
                print(pairs[-1], overlap[-1])
        
        pairs = np.array(pairs); trans = np.array(trans); overlap = np.array(overlap)
        np.savez(match3d_train_path + seq + '/pairs.npz', pairs=pairs, trans=trans, overlap=overlap)
        print('Totally %d pairs of point clouds:'%pairs.shape[0])
        for overlap_ratio in [0.1, 0.3, 0.5, 0.7]:
            ind = np.squeeze(np.argwhere(overlap >= overlap_ratio))
            pairs = pairs[ind]; trans = trans[ind]; overlap = overlap[ind]
            np.savez(match3d_train_path + seq + '/pairs-%.1f.npz'%overlap_ratio,
                     pairs=pairs, trans=trans, overlap=overlap)
            print('Generate %d pairs of point clouds with overlap>%.1f.'%(pairs.shape[0],overlap_ratio))

    def sample_random_rotation(self, pitch_scale=np.pi/3., roll_scale=np.pi/4.):
        roll = np.random.random() * roll_scale * 2. - roll_scale
        pitch = np.random.random() * pitch_scale * 2. - pitch_scale
        r = R.from_euler('xyz', [roll, pitch, 0.], degrees=False)
        trans = np.eye(4); trans[:3, :3] = r.as_matrix()
        return trans


class ThreeDMatchTestDataset(Dataset):
    def __init__(self, lomatch=False):
        self.path = lomatch3d_path if lomatch else match3d_path
        self.seqs = os.listdir(match3d_test_path)
        self.seqs.sort() # keep them in order
        self._pairs, self._gt, self._cov = [], [], []
        for (i, seq) in enumerate(self.seqs):
            _, trans = self.read_transformation_log(seq)
            pairs, cov = self.read_covariance_log(seq)
            self._pairs.append([np.array([[i] * pairs.shape[0]]).T])
            self._pairs[-1].append(pairs) # [N, 3], (seq, src, tar)
            self._pairs[-1] = np.concatenate(self._pairs[-1],axis=-1)
            self._gt.append(trans); self._cov.append(cov)
        
        self.kdts = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16)
        self._pairs = np.concatenate(self._pairs, axis=0)
        self._cov = np.concatenate(self._cov, axis=0)
        self._gt = np.concatenate(self._gt, axis=0)
        self._size = self._pairs.shape[0]
    
    def __len__(self):
        return self._size
    
    def __getitem__(self, index):
        if self._size == 0: raise StopIteration
        pair = self._pairs[index % self._size]
        trans = self._gt[index % self._size]
        cov = self._cov[index % self._size]
        seq, pair = self.seqs[pair[0]], pair[1:]
        
        pcd1 = self.read_point_cloud(seq, pair[0]).voxel_down_sample(voxel_size=0.03)
        pcd2 = self.read_point_cloud(seq, pair[1]).voxel_down_sample(voxel_size=0.03)
        
        #pcd1.estimate_normals(search_param=self.kdts)
        #pcd2.estimate_normals(search_param=self.kdts)
        #pcd1.orient_normals_towards_camera_location(np.array([0.,0.,0.]))
        #pcd2.orient_normals_towards_camera_location(np.array([0.,0.,0.]))
        
        xyz = [torch.from_numpy(np.array(pcd1.points, dtype=np.float32)),
               torch.from_numpy(np.array(pcd2.points, dtype=np.float32))]
        features = [torch.ones_like(xyz[0][:,:1]), torch.ones_like(xyz[1][:,:1])]
        return xyz[0], xyz[1], features[0], features[1], trans, cov
    
    def read_point_cloud(self, seq:str, id:int):
        return o3d.io.read_point_cloud(match3d_test_path + seq + '/fragments/cloud_bin_%d.ply'%id)
    
    def read_transformation_log(self, seq:str):
        with open(self.path + seq + '/gt.log') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        pairs, trans = list(), list()
        num_pairs = len(lines) // 5
        for i in range(num_pairs):
            line_id = i * 5
            pairs.append(lines[line_id].split())
            pairs[-1] = [int(pairs[-1][0]), int(pairs[-1][1])]
            transform = list()
            for j in range(1, 5):
                transform.append(lines[line_id + j].split())
            trans.append(np.array(transform, dtype=np.float32))
        return np.array(pairs), np.array(trans)
    
    def read_covariance_log(self, seq:str):
        with open(self.path + seq + '/gt.info') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        pairs, cov = list(), list()
        num_pairs = len(lines) // 7
        for i in range(num_pairs):
            line_id = i * 7
            pairs.append(lines[line_id].split())
            pairs[-1] = [int(pairs[-1][0]), int(pairs[-1][1])]
            covariance = list()
            for j in range(1, 7):
                covariance.append(lines[line_id + j].split())
            cov.append(np.array(covariance, dtype=np.float32))
        return np.array(pairs), np.array(cov)
