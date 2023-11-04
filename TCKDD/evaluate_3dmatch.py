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
paser.add_argument("--keypoint_detector", type=str, default="rand", choices=["rand","prob","nms","nmsp"])
paser.add_argument("--num_keypoints", type=int, default=5000)
paser.add_argument("--model_path", type=str, default="./checkpoints/3dmatch_kpfcnn_HCL64_40000.pth")
args = paser.parse_args()


neighbor_limits = [37, 31, 34, 37, 35]
model = KPFCNN(1, 32, 64, 15, 0.075, 0.06, 'group_norm', 32)
model.load_state_dict(torch.load(args.model_path))
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

fmr, rrs, ir, rtes, rres = dict(), dict(), dict(), dict(), dict()
for seq in indoor.seqs: fmr[seq], rrs[seq], ir[seq], rtes[seq], rres[seq] = [], [], [], [], []
radius = {5000 : 0.03, 2500 : 0.03, 1000 : 0.05, 500 : 0.07, 250 : 0.1}


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
        sxyz0, sdesc0, ssal0 = random_sample_keypoints(xyz0, desc0, saliency0, args.num_keypoints)
        sxyz1, sdesc1, ssal1 = random_sample_keypoints(xyz1, desc1, saliency1, args.num_keypoints)
    
    elif args.keypoint_detector == "prob":
        score0 = saliency0.cpu().numpy().squeeze()
        score1 = saliency1.cpu().numpy().squeeze()
        if args.num_keypoints < 2500:
            score0 = (saliency0.cpu().numpy().squeeze() - 1.0)/0.2
            score1 = (saliency1.cpu().numpy().squeeze() - 1.0)/0.2
        sxyz0, sdesc0 = random_sample_keypoints_with_scores(xyz0, desc0, score0, args.num_keypoints)
        sxyz1, sdesc1 = random_sample_keypoints_with_scores(xyz1, desc1, score1, args.num_keypoints)
    
    elif args.keypoint_detector == "nms":
        score0 = (saliency0.cpu().numpy().squeeze() - 1.0)/0.2
        score1 = (saliency1.cpu().numpy().squeeze() - 1.0)/0.2
        sxyz0, sdesc0 = sample_keypoints_with_nms(
            xyz0.cpu().numpy(), desc0, score0, args.num_keypoints, radius[args.num_keypoints]
        )
        sxyz1, sdesc1 = sample_keypoints_with_nms(
            xyz1.cpu().numpy(), desc1, score1, args.num_keypoints, radius[args.num_keypoints]
        )
        sxyz0 = torch.FloatTensor(sxyz0).cuda()
        sxyz1 = torch.FloatTensor(sxyz1).cuda()
    
    else: # args.keypoint_detector == "nmsp"
        score0 = (saliency0.cpu().numpy().squeeze() - 1.0)/0.2
        score1 = (saliency1.cpu().numpy().squeeze() - 1.0)/0.2
        sxyz0, sdesc0 = random_sample_keypoints_with_nms(
            xyz0.cpu().numpy(), desc0, score0, args.num_keypoints, radius[args.num_keypoints]
        )
        sxyz1, sdesc1 = random_sample_keypoints_with_nms(
            xyz1.cpu().numpy(), desc1, score1, args.num_keypoints, radius[args.num_keypoints]
        )
        sxyz0 = torch.FloatTensor(sxyz0).cuda()
        sxyz1 = torch.FloatTensor(sxyz1).cuda()
    
    print(sxyz0.shape, sxyz1.shape, sdesc0.shape, sdesc1.shape)
    
    # match descriptors
    start = time.time()
    match_pairs = metrics.match_features(sdesc0, sdesc1)
    print(match_pairs.shape, time.time()-start)

    # feature matching recall and inlier ratio
    cxyz1 = sxyz0[match_pairs[:,0]].cpu().numpy()
    cxyz2 = sxyz1[match_pairs[:,1]].cpu().numpy()
    fmri, inlier_ratio = metrics.feature_matching_recall(cxyz1, cxyz2, trans)
    print('feature matching recall:', fmri)
    print('inlier ratio:',inlier_ratio)
    fmr[seq].append(fmri)
    ir[seq].append(inlier_ratio)

    # registration recall
    print(trans) # ground-truth
    sxyz0, sxyz1 = sxyz0.cpu().numpy(), sxyz1.cpu().numpy()
    sdesc0, sdesc1 = sdesc0.cpu().numpy(), sdesc1.cpu().numpy()
    trans_ransac = metrics.ransac_based_on_feature_matching(sxyz0, sxyz1, sdesc0, sdesc1)
    rr,rmse = metrics.registration_recall(trans, cov, trans_ransac.transformation)
    print('ransac matching (feature matching):\n',trans_ransac)
    print(trans_ransac.transformation)
    print('registration recall (RANSAC):',rr,rmse)
    rrs[seq].append(rr)

    if rr==True:
        rte, rre = metrics.registration_error(trans,trans_ransac.transformation)
        rtes[seq].append(rte); rres[seq].append(rre)
        print('RTE: %.2fcm, RRE: %.2f°'%(rte*100., rre))
    
    print()


FMRs, RRs, IRs, RTEs, RREs, STD = [], [], [], [], [], []
for seq in indoor.seqs:
    print(seq, len(fmr[seq]))
    print('feature matching recall:', np.mean(np.array(fmr[seq])))
    print('registration recall:', np.mean(np.array(rrs[seq])))
    print('inlier ratio:', np.mean(np.array(ir[seq])))
    print('average rte:', np.mean(np.array(rtes[seq])))
    print('average rre:', np.mean(np.array(rres[seq])))
    STD.append(np.mean(np.array(fmr[seq])))
    FMRs = FMRs + fmr[seq]
    RRs = RRs + rrs[seq]
    IRs = IRs + ir[seq]
    RTEs = RTEs + rtes[seq]
    RREs = RREs + rres[seq]
    print()

print('average feature matching recall:',np.mean(np.array(FMRs)))
print('average registration recall:',np.mean(np.array(RRs)))
print('average inlier ratio:',np.mean(np.array(IRs)))
print('average relative translation error:',np.mean(np.array(RTEs)))
print('average relative rotation error:',np.mean(np.array(RREs)))
print('average feature matching recall (across sequences):',np.mean(np.array(STD)))
print('standard derivation of feature matching recall',np.std(np.array(STD)))
