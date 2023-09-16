import numpy as np
import open3d as o3d
import torch
from math import acos,sqrt,pi
#from nibabel import quaternions as nq
from scipy.spatial.transform import Rotation as R

class Metrics(object):
    def __init__(self) -> None:
        self.tau1 = 0.1
        self.tau2 = 0.05
        self.rmse_bound = 0.2
        self.eps = 0.1
        self.positive_correspondence_bound = 0.2

        self.bidirectional = True
        self.feature_match_threshold = 1.3
        self.feature_saliency_ratio = None
        self.icp_inlier_threshold = 0.1
        self.ransac_inlier_threshold = 0.15
        self.ransac_max_iters = 50000
        self.edge_prune = 0.8
        self.dist_prune = 0.4
        self.ransac_n = 3

        self.rte_threshold = 2.0
        self.rre_threshold = 5.0
            
    def match_features(self, source, target):
        '''
        Input: source [N, F], target [N, F], torch.FloatTensor
        Algorithm: match the features (bidirectionally) based on Euclidean distances in the feature space
        Output: correspondence set, numpy.ndarray [*,2], np.int32
        '''
        dist = torch.unsqueeze(source, dim=1) - torch.unsqueeze(target, dim=0)
        dist = torch.norm(dist, p=2, dim=-1).cpu().numpy()
        corr = np.zeros_like(dist, dtype=np.int32)
        corr[dist<=self.feature_match_threshold] = 1

        # bidirectional nearest points/descriptors
        indices_r = np.expand_dims(np.argmin(dist, axis=-1),axis=-1) # argmin of the rows
        indices_c = np.expand_dims(np.argmin(dist, axis=-2),axis=-2) # argmin of the columns
        mask_r = np.zeros_like(dist, dtype=np.int32)
        mask_c = np.zeros_like(dist, dtype=np.int32)
        np.put_along_axis(mask_r, indices_r, 1, axis=-1)
        np.put_along_axis(mask_c, indices_c, 1, axis=-2)

        if self.bidirectional is True:
            corr = corr * mask_r * mask_c
        else: corr = corr * mask_r

        # secondary nearest neighbor
        if self.feature_saliency_ratio is not None:
            dist2 = dist + mask_r*1e8
            saliency_ratio = np.min(dist,axis=-1)/np.min(dist2,axis=-1)
            saliency_ratio = (saliency_ratio <= self.feature_saliency_ratio)
            corr = corr * saliency_ratio
        
        match_pairs = np.argwhere(corr)
        return match_pairs.astype(np.int32)

    def feature_matching_recall(self, cxyz1, cxyz2, trans):
        '''
        Input: cxyz1 [N, 3], cxyz2 [N, 3], numpy.ndarray (matched points)
        Output: boolean feature matching indicator for FMR, inlier ratio
        '''
        cxyz2 = cxyz2 @ trans[:3,:3].T + trans[:3,-1:].T
        fmr = np.linalg.norm(cxyz1 - cxyz2, axis=-1)
        fmr = fmr<=self.tau1
        if fmr.shape[0]==0: inlier_ratio = 0.
        else: inlier_ratio = np.sum(fmr)/fmr.shape[0]
        return inlier_ratio>=self.tau2, inlier_ratio
    

    def ransac_based_on_feature_matching(self, xyz1, xyz2, desc1, desc2):
        pcd = [o3d.geometry.PointCloud(), o3d.geometry.PointCloud()]
        pcd[0].points = o3d.utility.Vector3dVector(xyz1)
        pcd[1].points = o3d.utility.Vector3dVector(xyz2)

        s_desc = o3d.pipelines.registration.Feature()
        t_desc = o3d.pipelines.registration.Feature()
        s_desc.data = desc1.T
        t_desc.data = desc2.T

        trans = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd[1], pcd[0], t_desc, s_desc, True, self.ransac_inlier_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), self.ransac_n,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(self.edge_prune),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.dist_prune)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(self.ransac_max_iters, 0.999))
        return trans
    

    def ransac_based_on_correspondence(self, xyz1, xyz2, match_pairs):
        pcd = [o3d.geometry.PointCloud(), o3d.geometry.PointCloud()]
        pcd[0].points = o3d.utility.Vector3dVector(xyz1)
        pcd[1].points = o3d.utility.Vector3dVector(xyz2)
        match_pairs = np.stack([match_pairs[:,1], match_pairs[:,0]], axis=1)

        trans = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            pcd[1], pcd[0], o3d.utility.Vector2iVector(match_pairs), self.ransac_inlier_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), self.ransac_n,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(self.edge_prune),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.dist_prune)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(self.ransac_max_iters, 0.999))
        return trans
    
    def icp_registration(self, xyz1, xyz2, trans=np.eye(4)):
        point_cloud = [o3d.geometry.PointCloud(), o3d.geometry.PointCloud()]
        point_cloud[0].points = o3d.utility.Vector3dVector(xyz1)
        point_cloud[1].points = o3d.utility.Vector3dVector(xyz2)

        trans = o3d.pipelines.registration.registration_icp(
            point_cloud[1], point_cloud[0], self.icp_inlier_threshold, trans,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))
        
        return trans

    def compute_transform_error(self, transform, covariance, estimated_transform):
        relative_transform = np.matmul(np.linalg.inv(transform), estimated_transform)
        # q = nq.mat2quat(relative_transform[:3, :3])[1:]
        q = R.from_matrix(relative_transform[:3, :3]).as_quat()[:3]
        er = np.concatenate([relative_transform[:3, 3], q], axis=0)
        p = er.reshape(1, 6) @ covariance @ er.reshape(6, 1) / covariance[0, 0]
        return sqrt(abs(p.item()))
    
    def registration_recall(self, trans, covariance, ptrans):
        '''
        Input:  covariance [6, 6], numpy.ndarray, covariance matrix
                trans [4, 4], numpy.ndarray, groundtruth transformation matrix
                ptrans[4, 4], numpy.ndarray, estimated transformation matrix
        Output: boolean registration indicator for registration recall, rmse
        '''
        rmse = self.compute_transform_error(trans, covariance, ptrans)
        return rmse<=self.rmse_bound, rmse

    def approximate_registration_recall(self, xyz1, xyz2, trans, ptrans):
        '''
        Input:  xyz1 [N, 3], xyz2 [N, 3], torch.FloatTensor
                trans [4, 4], torch.FloatTensor, groundtruth transformation matrix
                ptrans[4, 4], torch.FloatTensor, estimated transformation matrix
        Output: boolean registration indicator for registration recall, rmse
        '''
        threshold = self.feature_match_threshold
        self.feature_match_threshold = self.positive_correspondence_bound**2
        cxyz2 = xyz2 @ trans[:3,:3].T + trans[:3,-1:].T
        corr = self.match_features(xyz1, cxyz2)
        xyz1, xyz2 = xyz1[corr[:,0]], xyz2[corr[:,1]]
        self.feature_match_threshold = threshold

        xyz2 = xyz2 @ ptrans[:3,:3].T + ptrans[:3,-1:].T
        dist = torch.norm(xyz1 - xyz2, p=2, dim=-1)**2
        rmse = sqrt(np.mean(dist.cpu().numpy()))
        return rmse<=self.rmse_bound, rmse
    
    def registration_error(self, trans, ptrans):
        rte = np.linalg.norm(trans[:3,-1]-ptrans[:3,-1])
        rre = np.trace(ptrans[:3,:3].T @ trans[:3,:3])
        rre = np.clip((rre - 1.)/2., -1., 1.)
        rre = acos(rre) / pi * 180.
        return rte, rre
