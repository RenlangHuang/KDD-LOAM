import time
import argparse
import numpy as np
import open3d as o3d
from metrics import Metrics
import torch
from models.kpfcnn import KPFCNN
from datasets.kitti import KITTIDataset
from datasets.dataloader import collate_fn_precompute_data_stack_mode


parser = argparse.ArgumentParser()
parser.add_argument("--num_keypoints", type=int, default=5000)
parser.add_argument("--sample_index", type=int, default=59)
args = parser.parse_args()


neighbor_limits = [34, 32, 34, 34, 39]
model = KPFCNN(1, 32, 64, 15, 0.9, 0.6, 'group_norm', 32)
model.load_state_dict(torch.load('./checkpoints/kitti_HCL64_augm_23500.pth'))
model.eval().cuda()

kitti = KITTIDataset('train')

metrics = Metrics()
metrics.feature_match_threshold = 1.3
metrics.ransac_inlier_threshold = 0.3
metrics.ransac_max_iters = 50000
metrics.dist_prune = 0.3


def uniform_sampling(radius, source: np.ndarray):
    leaf_size_ = np.array([[radius, radius, radius]])
    inverse_leaf_size = 1. / leaf_size_

    pcd = np.array(source)
    min_pt = np.min(pcd[:, :3], keepdims=True)
    max_pt = np.max(pcd[:, :3], keepdims=True)
    
    min_b_ = np.floor(min_pt * inverse_leaf_size).astype(np.int64)
    max_b_ = np.floor(max_pt * inverse_leaf_size).astype(np.int64)
    
    div_b_ = np.squeeze(max_b_ - min_b_ + 1)
    divb_mul_ = np.array([[1, div_b_[0], div_b_[0]*div_b_[1]]])
    
    ijk = np.floor(pcd[:, :3] * inverse_leaf_size).astype(np.int64)
    idx = np.sum((ijk - min_b_) * divb_mul_, axis=-1)
    center = (ijk + 0.5) * leaf_size_
    priority = np.linalg.norm(pcd[:, :3] - center, axis=-1)
    max_priority = np.max(priority)
    priority = (max_priority - priority) / (max_priority - np.min(priority))
    indx = np.argsort(idx + priority / 2.0)
    leaves_ = dict(zip(idx[indx], pcd[indx]))
    return list(leaves_.values())


def saliency_hierarchical_render(points, saliency):
    red = np.array([[250.0, 20.0, 20.0]])
    yellow = np.array([[250.0, 200.0, 80.0]])
    green = np.array([[160.0, 240.0, 80.0]])
    cyan = np.array([[25.0, 240.0, 240.0]])
    blue = np.array([[50.0, 50.0, 240.0]])
    purple = np.array([[230.0, 40.0, 210.0]])

    node = [0.78, 0.86, 0.90, 0.97, 1.05, 1.15]

    score = np.stack([saliency]*3, axis=-1)
    color = np.ones([saliency.shape[0], 3]) * red
    color = np.where(score > node[0], red + (score - node[0]) / (node[1] - node[0]) * (yellow - red), color)
    color = np.where(score > node[1], yellow + (score - node[1]) / (node[2] - node[1]) * (green - yellow), color)
    color = np.where(score > node[2], green + (score - node[2]) / (node[3] - node[2]) * (cyan - green), color)
    color = np.where(score > node[3], cyan + (score - node[3]) / (node[4] - node[3]) * (blue - cyan), color)
    color = np.where(score > node[4], blue + (score - node[4]) / (node[5] - node[4]) * (purple - blue), color)
    color = np.where(score > node[5], purple, color)

    points.colors = o3d.utility.Vector3dVector(color / 255.)


args.sample_index = 59
start = time.time()
pair = kitti._pairs[args.sample_index]
trans = kitti._gt[args.sample_index]
seq, pair = pair[0], pair[1:]
inputs = kitti.__getitem__(args.sample_index)
inputs = collate_fn_precompute_data_stack_mode(
    [inputs], 5, 0.3, 0.3*3.0, neighbor_limits)
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

print(seq, pair)
print('preprocessing time: %.4fs, loading time: %.4fs, inference time: %.4fs'\
    %(preprocess_time, gpu_loading_time, inference_time))


xyz0, xyz1 = pcd0['points'][0], pcd1['points'][0]
print(xyz0.shape, xyz1.shape, desc0.shape, desc1.shape, saliency0.shape, saliency1.shape)


xyz0 = xyz0.cpu().numpy()
xyz1 = xyz1.cpu().numpy()
desc0 = desc0.cpu().numpy()
desc1 = desc1.cpu().numpy()
saliency0 = saliency0.cpu().numpy()
saliency1 = saliency1.cpu().numpy()

# sorted keypoints
idx0 = np.argsort(np.squeeze(saliency0))[:args.num_keypoints]
idx1 = np.argsort(np.squeeze(saliency1))[:args.num_keypoints]
sxyz0, sdesc0 = xyz0[idx0], desc0[idx0]
sxyz1, sdesc1 = xyz1[idx1], desc1[idx1]
print(sxyz0.shape, sxyz1.shape, sdesc0.shape, sdesc1.shape)

# match descriptors
start = time.time()
match_pairs = metrics.match_features(torch.FloatTensor(sdesc0), torch.FloatTensor(sdesc1))
print(match_pairs.shape, time.time()-start)

# registration recall
print(trans) # ground-truth
trans_ransac = metrics.ransac_based_on_feature_matching(sxyz0, sxyz1, sdesc0, sdesc1)
print('ransac matching (feature matching):\n',trans_ransac)
print(trans_ransac.transformation)

rte, rre = metrics.registration_error(trans,trans_ransac.transformation)
print('RTE: %.2fcm, RRE: %.2f°'%(rte*100., rre))

sxyz0 = sxyz0[match_pairs[:, 0]]
sxyz1 = sxyz1[match_pairs[:, 1]]
corres = np.linalg.norm(sxyz1 @ trans[:3, :3].T + trans[:3, -1:].T - sxyz0, axis=-1)
print(np.sum(corres < 0.45))

sxyz0 = uniform_sampling(1.5, np.concatenate([sxyz0, np.arange(sxyz0.shape[0]).reshape([-1,1])], axis=-1))
sxyz0 = np.array(sxyz0)
sxyz0, idx = sxyz0[:, :3], sxyz0[:, -1].astype(np.int64)
sxyz1, sdesc0, sdesc1 = sxyz1[idx], sdesc0[idx], sdesc1[idx]
lines = [np.argwhere(corres[idx] < 0.45)]
vertices = np.concatenate([sxyz0, sxyz1 + np.array([[0., 0., 30.]])], axis=0)
lines.append(lines[-1] + sxyz0.shape[0])
lines = np.concatenate(lines, axis=-1)
colors = np.repeat(np.array([[0., 1., .5]]),lines.shape[0],0)


framework = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15., origin=[0,0,0])

point_cloud0 = o3d.geometry.PointCloud()
point_cloud0.points = o3d.utility.Vector3dVector(xyz0)
point_cloud0.colors = o3d.utility.Vector3dVector(np.zeros_like(xyz0)+0.4)
saliency_hierarchical_render(point_cloud0, np.squeeze(saliency0))

point_cloud1 = o3d.geometry.PointCloud()
point_cloud1.points = o3d.utility.Vector3dVector(xyz1 + np.array([[0., 0., 30.]]))
point_cloud1.colors = o3d.utility.Vector3dVector(np.zeros_like(xyz1)+0.4)
saliency_hierarchical_render(point_cloud1, np.squeeze(saliency1))

line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(vertices),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)

vis = o3d.visualization.Visualizer()
vis.create_window()
render_option: o3d.visualization.RenderOption = vis.get_render_option()	
render_option.background_color = np.array([0, 0, 0])
render_option.background_color = np.array([1, 1, 1])
render_option.point_size = 2.0
render_option.line_width = 0.1
vis.add_geometry(point_cloud0)
vis.add_geometry(point_cloud1)
vis.add_geometry(framework)
vis.add_geometry(line_set)
vis.run()