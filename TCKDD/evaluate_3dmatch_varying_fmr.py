import time
import argparse
from metrics import Metrics
import torch
import numpy as np
from models.kpfcnn import KPFCNN
from datasets.dataloader import get_dataloader
from datasets.match3d import ThreeDMatchTestDataset
from utils.selection import *


paser = argparse.ArgumentParser()
paser.add_argument("--keypoint_detector", type=str, choices=["rand","prob"])
args = paser.parse_args()


neighbor_limits = [37, 31, 34, 37, 35]
model = KPFCNN(1, 32, 64, 15, 0.075, 0.06, 'group_norm', 32)
model.load_state_dict(torch.load('./checkpoints/3dmatch_kpfcnn_HCL64_40000.pth'))
model.eval().cuda()

indoor = ThreeDMatchTestDataset(False)
dloader, _ = get_dataloader(indoor, 0.03, 2.5, 5, 16, False, neighbor_limits)
iterator = dloader.__iter__() # dataset iterator
print(len(indoor))


metrics = Metrics()
metrics.feature_match_threshold = 1.3
metrics.ransac_inlier_threshold = 0.05
metrics.icp_inlier_threshold = 0.08
metrics.ransac_max_iters = 50000
metrics.dist_prune = 0.05


metrics.tau2 = 0.05
tau1 = np.arange(0., 0.2, 0.01) + 0.01
FMR_dist = [list() for i in range(tau1.shape[0])]
tau2 = np.arange(0., 0.21, 0.01) + 0.01
FMR_ratio = [list() for i in range(tau2.shape[0])]

for ids in range(len(indoor)):
    seq = indoor.seqs[indoor._pairs[ids][0]]
    start = time.time()
    inputs = iterator.next()
    preprocess_time = time.time() - start
    start = time.time()

    pcd0 = inputs['src_pcd']
    pcd1 = inputs['tar_pcd']
    trans = inputs['transform'].numpy()
    cov = inputs['covariance'].numpy()
    for k in pcd0.keys():
        if type(pcd0[k]) == list:
            pcd0[k] = [item.cuda() for item in pcd0[k]]
            pcd1[k] = [item.cuda() for item in pcd1[k]]
        else:
            pcd0[k] = pcd0[k].cuda()
            pcd1[k] = pcd1[k].cuda()
    
    with torch.no_grad():
        gpu_loading_time = time.time() - start
        start = time.time()
        desc0, saliency0 = model(pcd0)
        inference_time = time.time() - start
        desc1, saliency1 = model(pcd1)
    
    print(ids+1, indoor._pairs[ids][1:], seq)
    print('preprocessing time: %.4fs, loading time: %.4fs, inference time: %.4fs'\
        %(preprocess_time, gpu_loading_time, inference_time))
    
    xyz0, xyz1 = pcd0['points'][0], pcd1['points'][0]
    print(xyz0.shape, xyz1.shape, desc0.shape, desc1.shape, saliency0.shape, saliency1.shape)
    
    if args.keypoint_detector == "rand":
        sxyz0, sdesc0, ssal0 = random_sample_keypoints(xyz0, desc0, saliency0, 5000)
        sxyz1, sdesc1, ssal1 = random_sample_keypoints(xyz1, desc1, saliency1, 5000)
    
    else: # args.keypoint_detector == "prob"
        score0 = saliency0.cpu().numpy().squeeze()
        score1 = saliency1.cpu().numpy().squeeze()
        sxyz0, sdesc0 = random_sample_keypoints_with_scores(xyz0, desc0, score0, 5000)
        sxyz1, sdesc1 = random_sample_keypoints_with_scores(xyz1, desc1, score1, 5000)
    
    # match descriptors
    start = time.time()
    match_pairs = metrics.match_features(sdesc0, sdesc1)
    print(match_pairs.shape, time.time()-start)

    # feature matching recall and inlier ratio
    metrics.tau2 = 0.05
    cxyz1 = sxyz0[match_pairs[:,0]].cpu().numpy()
    cxyz2 = sxyz1[match_pairs[:,1]].cpu().numpy()
    
    for i in range(len(tau1)):
        metrics.tau1 = tau1[i]
        fmri, _ = metrics.feature_matching_recall(cxyz1, cxyz2, trans)
        FMR_dist[i].append(fmri)
    
    metrics.tau1 = 0.1
    _, inlier_ratio = metrics.feature_matching_recall(cxyz1, cxyz2, trans)
    for i in range(len(tau2)):
        FMR_ratio[i].append(inlier_ratio>=tau2[i])
        if i==5: ir = inlier_ratio
    
    print('feature matching recall: ', FMR_dist[10][-1], ', inlier ratio: %.4f\n'%ir)


print("FMR v.s. distance threshold:\n", np.mean(np.array(FMR_dist), axis=-1))
print("FMR v.s. inlier ratio threshold:\n", np.mean(np.array(FMR_ratio), axis=-1))