import torch
import numpy as np
import torch.nn.functional as F
from scipy.spatial import cKDTree


def euclidean_distance(src, tar) -> torch.Tensor:
    # src and tar are tensors [N1, *], [N2, *], return Euclidean distance tensor [N1, N2]
    return torch.norm(torch.unsqueeze(src, dim=1) - torch.unsqueeze(tar, dim=0), p=2, dim=-1)


class HardestContrastiveLoss(torch.nn.Module):
    def __init__(self, pos_threshold, neg_threshold, pos_margin=0.1, neg_margin=1.4, num_pos=64, num_neg=512):
        super(HardestContrastiveLoss, self).__init__()
        self.num_neg = num_neg # number of negative candidates
        self.num_pos = num_pos # number of positive pairs
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.weight_det_loss = 1.
    
    def get_correspondences(self, source: torch.Tensor, target: torch.Tensor):
        kdtree = cKDTree(target.cpu().numpy())
        dist, ind2 = kdtree.query(source.cpu().numpy(),k=1,distance_upper_bound=self.pos_threshold)
        ind1 = np.argwhere(dist!=np.inf)
        indices = np.concatenate([ind1, ind2[ind1]], axis=-1)
        return torch.from_numpy(indices).long().to(source.device)
    
    def random_sample(self, x: torch.Tensor, nsample: int):
        if x.shape[0] > nsample:
            return torch.randperm(x.shape[0], dtype=torch.int64, device=x.device)[:nsample]
        else:
            return torch.arange(x.shape[0], dtype=torch.int64, device=x.device)
        
    def forward(self, trans, xyz1, xyz2, desc1, desc2, sal1, sal2, 
                negative_from_anchors=True, sample_negative_from_matches=True):
        
        xyz2 = xyz2 @ trans[:3, :3].T + trans[:3, -1:].T
        matches = self.get_correspondences(xyz1, xyz2)
        
        matches = matches[self.random_sample(matches, self.num_pos)]
        anc_xyz1, anc_desc1, anc_sal1 = xyz1[matches[:, 0]], desc1[matches[:, 0]], sal1[matches[:, 0]]
        anc_xyz2, anc_desc2, anc_sal2 = xyz2[matches[:, 1]], desc2[matches[:, 1]], sal2[matches[:, 1]]
        
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
        
        if neg_xyz1.shape[0] > self.num_neg:
            indices = self.random_sample(neg_xyz1, self.num_neg)
            neg_xyz1, neg_desc1 = neg_xyz1[indices], neg_desc1[indices]
            indices = self.random_sample(neg_xyz2, self.num_neg)
            neg_xyz2, neg_desc2 = neg_xyz2[indices], neg_desc2[indices]
        
        with torch.no_grad():
            mask1 = euclidean_distance(anc_xyz1, neg_xyz2).gt(self.neg_threshold).float()
            mask2 = euclidean_distance(anc_xyz2, neg_xyz1).gt(self.neg_threshold).float()
        
        desc_loss1 = F.relu(self.neg_margin - euclidean_distance(anc_desc1, neg_desc2))
        desc_loss2 = F.relu(self.neg_margin - euclidean_distance(anc_desc2, neg_desc1))
        pos_desc_loss = F.relu(torch.norm((anc_desc1 - anc_desc2), p=2, dim=-1) - self.pos_margin)
        
        neg_desc_loss1, _ = torch.max(desc_loss1 * mask1, dim=-1)
        neg_desc_loss2, _ = torch.max(desc_loss2 * mask2, dim=-1)

        anc_sal1 = torch.squeeze(anc_sal1, dim=-1)
        anc_sal2 = torch.squeeze(anc_sal2, dim=-1)
        det_loss = (neg_desc_loss1 + pos_desc_loss)/anc_sal1 + (neg_desc_loss2 + pos_desc_loss)/anc_sal2
        det_loss = torch.mean(det_loss + torch.log(anc_sal1) + torch.log(anc_sal2))

        pos_desc_loss = torch.mean(pos_desc_loss)
        neg_desc_loss = torch.mean(neg_desc_loss1 + neg_desc_loss2)
        desc_loss = pos_desc_loss * 2. + neg_desc_loss

        return {
            'loss': desc_loss + det_loss * self.weight_det_loss,
            'desc_loss': desc_loss,
            'pos_loss': pos_desc_loss,
            'neg_loss': neg_desc_loss,
            'det_loss': det_loss,
        }
