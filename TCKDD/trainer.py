import torch
import numpy as np
import torch.nn.functional as F
from scipy.spatial import cKDTree


def euclidean_distance(src, tar):
    # src and tar are tensors [N1, *], [N2, *], return Euclidean distance tensor [N1, N2]
    return torch.norm(torch.unsqueeze(src, dim=1) - torch.unsqueeze(tar, dim=0), p=2, dim=-1)


class MetricLoss:
    def __init__(self, pos_threshold, neg_threshold, squared=False,
                 pos_margin=0.1, neg_margin=2.0, num_pos=1000, num_hn=2000, num_rn=0):
        self.num_rn = num_rn # number of random negative pairs
        self.num_hn = num_hn # number of negative candidates
        self.num_pos = num_pos # number of positive pairs
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.squared = squared
    
    def get_matches_indices(self, source, target, trans=np.eye(4)):
        threshold = self.pos_threshold
        kdtree = cKDTree(np.array(target @ trans[:3, :3].T + trans[:3, -1:].T))
        dist, ind2 = kdtree.query(np.array(source),k=1,distance_upper_bound=threshold)
        ind1 = np.argwhere(dist!=np.inf)
        return np.concatenate([ind1, ind2[ind1]], axis=-1).astype(np.int32)
    
    def sample_candidate_points(self, points, nsample):
        if points[0].shape[0]<nsample: return points
        # points should be a list with the first element to be `xyz`
        ind = np.random.choice(points[0].shape[0], nsample, replace=False)
        return (e[ind] for e in points)
    
    def sample_positive_pairs(self, matches, points1, points2, nsample):
        # points(1/2) should be a list with the first element to be `xyz`
        if matches.shape[0] > nsample:
            sel = np.random.choice(matches.shape[0], nsample, replace=False)
            matches = matches[sel] # randomly sampling
        sel1, sel2 = matches[:, 0], matches[:, 1]
        points = [e[sel1] for e in points1] + [e[sel2] for e in points2]
        return points


class ContrastiveLoss(MetricLoss):
    
    def sample_random_negative(self, anc_xyz1, xyz2):
        ind = np.random.choice(xyz2.shape[0], anc_xyz1.shape[0])
        neg_xyz2 = xyz2[ind]
        with torch.no_grad():
            mask = torch.norm(anc_xyz1 - neg_xyz2, p=2, dim=-1)
            mask = mask > self.neg_threshold
            mask = mask.float()
        return ind, mask
    
    def loss(self, matches, xyz1, xyz2, desc1, desc2, sample_negative_from_matches=True):
        anc_xyz1, anc_desc1, anc_xyz2, anc_desc2 = \
            self.sample_positive_pairs(matches, [xyz1, desc1], [xyz2, desc2], self.num_rn)
        
        if sample_negative_from_matches:
            neg_xyz1, neg_xyz2 = xyz1[matches[:, 0]], xyz2[matches[:, 1]]
            neg_desc1, neg_desc2 = desc1[matches[:, 0]], desc2[matches[:, 1]]
            ind1, mask1 = self.sample_random_negative(anc_xyz1, neg_xyz2)
            ind2, mask2 = self.sample_random_negative(anc_xyz2, neg_xyz1)
            neg_desc2, neg_desc1 = neg_desc2[ind1], neg_desc1[ind2]
        
        else: # sample negative from all points with correspondence
            ind1, mask1 = self.sample_random_negative(anc_xyz1, xyz2)
            ind2, mask2 = self.sample_random_negative(anc_xyz2, xyz1)
            neg_desc2, neg_desc1 = desc2[ind1], desc1[ind2]
        
        pos_desc_loss = F.relu(torch.sum((anc_desc1 - anc_desc2)**2, dim=-1) - self.pos_margin)
        desc_loss1 = F.relu(self.neg_margin - torch.sum((anc_desc1 - neg_desc2)**2, dim=-1))
        desc_loss2 = F.relu(self.neg_margin - torch.sum((anc_desc2 - neg_desc1)**2, dim=-1))
        if self.squared:
            desc_loss1 = desc_loss1**2
            desc_loss2 = desc_loss2**2
            pos_desc_loss = pos_desc_loss**2
        desc_loss1 = torch.sum((desc_loss1 + pos_desc_loss) * mask1)/torch.sum(mask1)
        desc_loss2 = torch.sum((desc_loss2 + pos_desc_loss) * mask2)/torch.sum(mask2)
        return desc_loss1 + desc_loss2


class HardestContrastiveLoss(MetricLoss):
    def loss(self, matches, xyz1, xyz2, desc1, desc2, sal1, sal2, 
             negative_from_anchors=True, sample_negative_from_matches=True):
        
        anc_xyz1, anc_desc1, anc_sal1, anc_xyz2, anc_desc2, anc_sal2 = \
            self.sample_positive_pairs(matches, [xyz1, desc1, sal1], [xyz2, desc2, sal2], self.num_pos)
        
        if negative_from_anchors:
            neg_xyz1, neg_desc1 = anc_xyz1, anc_desc1
            neg_xyz2, neg_desc2 = anc_xyz2, anc_desc2
        else: # sample negative from matches or raw point clouds
            if sample_negative_from_matches:
                neg_xyz1, neg_desc1 = xyz1[matches[:, 0]], desc1[matches[:, 0]]
                neg_xyz2, neg_desc2 = xyz1[matches[:, 1]], desc1[matches[:, 1]]
            else: # sample negative from the whole point cloud
                neg_xyz1, neg_desc1 = xyz1, desc1
                neg_xyz2, neg_desc2 = xyz2, desc2
        
        if neg_xyz1.shape[0] > self.num_hn:
            neg_xyz1, neg_desc1 = self.sample_candidate_points([neg_xyz1, neg_desc1], self.num_hn)
            neg_xyz2, neg_desc2 = self.sample_candidate_points([neg_xyz2, neg_desc2], self.num_hn)
        
        with torch.no_grad():
            mask1 = euclidean_distance(anc_xyz1, neg_xyz2) > self.neg_threshold
            mask2 = euclidean_distance(anc_xyz2, neg_xyz1) > self.neg_threshold
            mask1, mask2 = mask1.float(), mask2.float()
        
        desc_loss1 = F.relu(self.neg_margin - euclidean_distance(anc_desc1, neg_desc2))
        desc_loss2 = F.relu(self.neg_margin - euclidean_distance(anc_desc2, neg_desc1))
        pos_desc_loss = F.relu(torch.norm((anc_desc1 - anc_desc2), p=2, dim=-1) - self.pos_margin)
        
        if self.squared:
            desc_loss1 = desc_loss1**2
            desc_loss2 = desc_loss2**2
            pos_desc_loss = pos_desc_loss**2
        neg_desc_loss1, _ = torch.max(desc_loss1 * mask1, dim=-1)
        neg_desc_loss2, _ = torch.max(desc_loss2 * mask2, dim=-1)

        anc_sal1 = torch.squeeze(anc_sal1, dim=-1)
        anc_sal2 = torch.squeeze(anc_sal2, dim=-1)
        det_loss = (neg_desc_loss1 + pos_desc_loss)/anc_sal1 + (neg_desc_loss2 + pos_desc_loss)/anc_sal2
        det_loss = torch.mean(det_loss + torch.log(anc_sal1) + torch.log(anc_sal2))

        return torch.mean(pos_desc_loss), torch.mean(neg_desc_loss1 + neg_desc_loss2), det_loss

