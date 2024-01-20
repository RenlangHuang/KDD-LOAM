import time
import argparse
import numpy as np
import open3d as o3d
from engine.metrics import Metrics
import matplotlib.pyplot as plt
import torch
from engine.detector import *
from models.kpfcnn import KPFCNN
from datasets.match3d import ThreeDMatchTrainDataset
from datasets.dataloader import collate_fn_precompute_data_stack_mode


parser = argparse.ArgumentParser()
parser.add_argument("--keypoint_detector", type=str, default="nmsp", choices=["rand","prob","nms","nmsp"])
parser.add_argument("--num_keypoints", type=int, default=250)
parser.add_argument("--sample_index", type=int, default=1000) #20
parser.add_argument("--model_path", type=str, default="./ckpt/3dmatch_kpfcnn_HCL64_40000.pth")
args = parser.parse_args()


neighbor_limits = [37, 31, 34, 37, 35]
model = KPFCNN(1, 32, 64, 15, 0.075, 0.06, 'group_norm', 32)
model.load_state_dict(torch.load(args.model_path))
model.eval().cuda()

indoor = ThreeDMatchTrainDataset()

metrics = Metrics()
metrics.feature_match_threshold = 1.3
metrics.ransac_inlier_threshold = 0.05
metrics.icp_inlier_threshold = 0.08
metrics.ransac_max_iters = 50000
metrics.dist_prune = 0.05

radius = {5000 : 0.03, 2500 : 0.03, 1000 : 0.05, 500 : 0.07, 250 : 0.1}


start = time.time()
pair = indoor._pairs[args.sample_index]
trans = indoor._gt[args.sample_index]
# cov = indoor._cov[args.sample_index]
seq, pair = indoor.seqs[pair[0]], pair[1:]
seq = indoor.seqs[indoor._pairs[args.sample_index][0]]
inputs = indoor.__getitem__(args.sample_index)
inputs = collate_fn_precompute_data_stack_mode([inputs], 5, 0.03, 0.03*2.5, neighbor_limits)
preprocess_time = time.time() - start

start = time.time()
pcd0 = inputs['src_pcd']
pcd1 = inputs['tar_pcd']
trans = inputs['transform'].numpy()
# cov = inputs['covariance'].numpy()
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
        
print(seq, pair)
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

# registration recall
print(trans) # ground-truth
sxyz0, sxyz1 = sxyz0.cpu().numpy(), sxyz1.cpu().numpy()
sdesc0, sdesc1 = sdesc0.cpu().numpy(), sdesc1.cpu().numpy()
trans_ransac = metrics.ransac_based_on_feature_matching(sxyz0, sxyz1, sdesc0, sdesc1)
#rr,rmse = metrics.registration_recall(trans, cov, trans_ransac.transformation)
print('ransac matching (feature matching):\n',trans_ransac)
print(trans_ransac.transformation)
#print('registration recall (RANSAC):',rr,rmse)

rte, rre = metrics.registration_error(trans,trans_ransac.transformation)
print('RTE: %.2fcm, RRE: %.2f°'%(rte*100., rre))


def saliency_hierarchical_render(points, saliency):
    p1 = np.percentile(saliency, 20)
    p2 = np.percentile(saliency, 40)
    p3 = np.percentile(saliency, 60)
    p4 = np.percentile(saliency, 80)
    #p1, p2, p3, p4 = 5, 6, 8, 10
    colors = np.zeros_like(np.array(points.points))+0.3
    colors[:,0] = np.where(saliency< p1, 0.9-(p1-saliency)*0.1, colors[:,0]) # [0.9, 0.3, 0.3]
    colors[:,0] = np.where(saliency>=p1, 0.9-(saliency-p1)*0.2, colors[:,0]) # [0.7, 0.8, 0.5]
    colors[:,1] = np.where(saliency>=p1, 0.3+(saliency-p1)*0.5, colors[:,1])
    colors[:,2] = np.where(saliency>=p1, 0.3+(saliency-p1)*0.2, colors[:,2])

    colors[:,0] = np.where(saliency>=p2, 0.7-(saliency-p2)*0.3, colors[:,0]) # [0.1, 1.0, 0.9]
    colors[:,1] = np.where(saliency>=p2, 0.8+(saliency-p2)*0.1, colors[:,1])
    colors[:,2] = np.where(saliency>=p2, 0.5+(saliency-p2)*0.2, colors[:,2])

    colors[:,0] = np.where(saliency>=p3, 0.1+(saliency-p3)*0.2, colors[:,0]) # [0.5, 0.4, 0.9]
    colors[:,1] = np.where(saliency>=p3, 1.0-(saliency-p3)*0.3, colors[:,1])
    colors[:,2] = np.where(saliency>=p3, 0.9, colors[:,2])

    colors[:,0] = np.where(saliency>=p4, 0.5+(saliency-p4)*0.2, colors[:,0]) # [0.9, 0.5, 0.9]
    colors[:,1] = np.where(saliency>=p4, 0.4, colors[:,1])
    colors[:,2] = np.where(saliency>=p4, 0.9, colors[:,2])
    points.colors = o3d.utility.Vector3dVector(colors)

def keypoints_to_spheres(keypoints, radius=0.03, color=[1.0, 0.25, 0.0]):
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(keypoint)
        spheres += sphere
    spheres.paint_uniform_color(color)
    return spheres


xyz0 = xyz0.cpu().numpy()
xyz1 = xyz1.cpu().numpy()
desc0 = desc0.cpu().numpy()
desc1 = desc1.cpu().numpy()
saliency0 = saliency0.cpu().numpy()
saliency1 = saliency1.cpu().numpy()


keypoints = o3d.geometry.PointCloud()
keypoints.points = o3d.utility.Vector3dVector(sxyz1)
keypoints = keypoints_to_spheres(keypoints,0.02)

print(np.min(saliency1), np.max(saliency1), np.std(saliency1))
plt.hist(saliency1,bins=30,density=False,facecolor='tab:blue',alpha=0.7);plt.show()
print(saliency1.shape)


colors = np.repeat(np.array([[0.5, 1, 1]]),xyz1.shape[0],0)

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(xyz1)
point_cloud.colors = o3d.utility.Vector3dVector(colors)
min_saliency = np.min(saliency1)
max_saliency = np.max(saliency1)
saliency1 = (saliency1-min_saliency)/(max_saliency-min_saliency)*(12-4) + 4
saliency_hierarchical_render(point_cloud, np.squeeze(saliency1))

vis = o3d.visualization.Visualizer()
vis.create_window()
render_option: o3d.visualization.RenderOption = vis.get_render_option()	
render_option.background_color = np.array([0, 0, 0])
render_option.background_color = np.array([1, 1, 1])
render_option.point_size = 3.0
vis.add_geometry(point_cloud)
vis.add_geometry(keypoints)
vis.run()
