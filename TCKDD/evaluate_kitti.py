#import os
#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='0'
import time
import argparse
import numpy as np
from engine.metrics import Metrics
import torch
from engine.detector import *
from models.kpfcnn import KPFCNN
from datasets.dataloader import get_dataloader
from datasets.kitti import KITTIDataset


paser = argparse.ArgumentParser()
paser.add_argument("--keypoint_detector", type=str, default="sort", choices=["rand","prob","sort"])
paser.add_argument("--num_keypoints", type=int, default=5000)
args = paser.parse_args()


model = KPFCNN(1, 32, 64, 15, 0.9, 0.6, 'group_norm', 32)
model.load_state_dict(torch.load('./ckpt/kitti_HCL64_augm_23500.pth'))
model.eval().cuda()

kitti = KITTIDataset('test')
neighbor_limits = [34, 32, 34, 34, 39]
dloader, _ = get_dataloader(kitti, 0.3, 3.0, 5, 16, False, neighbor_limits)
iterator = dloader.__iter__() # dataset iterator
length = kitti._gt.shape[0]; print(length)


metrics = Metrics()
metrics.feature_match_threshold = 1.3
metrics.ransac_inlier_threshold = 0.3
metrics.ransac_max_iters = 50000
metrics.dist_prune = 0.3
metrics.ransac_n = 4

rrs, rtes, rres = [], [], []


for ids in range(length):
    start = time.time()
    inputs = iterator.next()
    preprocess_time = time.time() - start
    start = time.time()

    pcd0 = inputs['src_pcd']
    pcd1 = inputs['tar_pcd']
    trans = inputs['transform'].numpy()
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
    
    print(ids+1, kitti._pairs[ids])
    print('preprocessing time: %.4fs, loading time: %.4fs, inference time: %.4fs'\
        %(preprocess_time, gpu_loading_time, inference_time))
    
    xyz0, xyz1 = pcd0['points'][0], pcd1['points'][0]
    print(xyz0.shape, xyz1.shape, desc0.shape, desc1.shape, saliency0.shape, saliency1.shape)
    
    start = time.time()
    if args.keypoint_detector == "rand":
        sxyz0, sdesc0, ssal0 = random_sample_keypoints(xyz0, desc0, saliency0, args.num_keypoints)
        sxyz1, sdesc1, ssal1 = random_sample_keypoints(xyz1, desc1, saliency1, args.num_keypoints)
    
    elif args.keypoint_detector == "sort":
        score0 = (saliency0.cpu().numpy().squeeze()-1.0)/0.1
        score1 = (saliency1.cpu().numpy().squeeze()-1.0)/0.1
        sxyz0, sdesc0 = sample_keypoints_with_scores(xyz0, desc0, score0, args.num_keypoints)
        sxyz1, sdesc1 = sample_keypoints_with_scores(xyz1, desc1, score1, args.num_keypoints)
    
    else: # args.keypoint_detector == "prob"
        score0 = (saliency0.cpu().numpy().squeeze()-1.0)/0.1
        score1 = (saliency1.cpu().numpy().squeeze()-1.0)/0.1
        sxyz0, sdesc0 = random_sample_keypoints_with_scores(xyz0, desc0, score0, args.num_keypoints)
        sxyz1, sdesc1 = random_sample_keypoints_with_scores(xyz1, desc1, score1, args.num_keypoints)
    
    print(sxyz0.shape, sxyz1.shape, sdesc0.shape, sdesc1.shape)

    # registration recall
    print(trans) # ground-truth
    sxyz0, sxyz1 = sxyz0.cpu().numpy(), sxyz1.cpu().numpy()
    sdesc0, sdesc1 = sdesc0.cpu().numpy(), sdesc1.cpu().numpy()
    
    trans_ransac = metrics.ransac_based_on_feature_matching(sxyz0, sxyz1, sdesc0, sdesc1)
    print('ransac registration time: %.4fs'%(time.time() - start))
    rte, rre = metrics.registration_error(trans,trans_ransac.transformation)
    rr = rte < metrics.rte_threshold and rre < metrics.rre_threshold
    print('ransac matching (feature matching):\n',trans_ransac)
    print(trans_ransac.transformation)
    print('registration recall (RANSAC):',rr)
    rrs.append(rr)

    if rr==True:
        rtes.append(rte*100.); rres.append(rre)
        print('RTE: %.2fcm, RRE: %.2f°'%(rtes[-1], rre))

    print()


print('Success rate (registration recall): %.4f'%np.mean(np.array(rrs)))
print('Average RTE (cm): %.2f, STD: %.2f'%(np.mean(np.array(rtes)), np.std(np.array(rtes))))
print('Average RRE (°): %.2f, STD: %.2f'%(np.mean(np.array(rres)), np.std(np.array(rres))))
