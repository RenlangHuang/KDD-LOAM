import time
import torch
import numpy as np
from functools import partial
from torch.utils.data import DataLoader
import cpp_extensions.cpp_neighbors.radius_neighbors as radius_neighbors
import cpp_extensions.cpp_subsampling.grid_subsampling as grid_subsampling


def batch_grid_subsampling(points, batches_len, sampleDl=0.1):
    s_points, s_len = grid_subsampling.subsample_batch(points, batches_len, sampleDl=sampleDl, max_p=0, verbose=0)
    return torch.from_numpy(s_points), torch.from_numpy(s_len)


def batch_knn_query(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of batch query length
    :param s_batches: (B) the list of batch support length
    :param radius: float32
    :return: neighbors indices [n_points, n_neighbors], torch.int
    """
    neighbors = radius_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
    if max_neighbors > 0: return torch.from_numpy(neighbors[:, :max_neighbors])
    else: return torch.from_numpy(neighbors)


def precompute_data(points, num_blocks, voxel_size, radius, neighbor_limits):
    assert num_blocks == len(neighbor_limits)
    points_list, length_list = list(), list()
    neighbors_list, downsampling_list, upsampling_list = list(), list(), list()
    start = time.time()
    for i in range(num_blocks):
        if i>0: points, length = batch_grid_subsampling(points, length, voxel_size)
        else: length = torch.IntTensor([points.shape[0]])
        points_list.append(points)
        length_list.append(length)
        voxel_size = voxel_size * 2.
    #print('grid subsampling: %fs'%(time.time()-start))
    
    start = time.time()
    for i in range(num_blocks):
        neighbors_list.append(batch_knn_query(
            points_list[i], points_list[i],
            length_list[i], length_list[i],
            radius, neighbor_limits[i]
        ))
        if i < num_blocks - 1:
            downsampling_list.append(batch_knn_query(
                points_list[i+1], points_list[i],
                length_list[i+1], length_list[i],
                radius, neighbor_limits[i]
            ))
            upsampling_list.append(batch_knn_query(
                points_list[i], points_list[i+1],
                length_list[i], length_list[i+1],
                radius*2., neighbor_limits[i+1]
            ))
        radius = radius * 2.
    print('ball grouping: %fs'%(time.time()-start))

    return {
        'points': points_list,
        'stack_lengths': length_list,
        'neighbors': [v.long() for v in neighbors_list],
        'subsampling': [v.long() for v in downsampling_list],
        'upsampling': [v.long() for v in upsampling_list],
    }


def collate_fn_precompute_data_stack_mode(list_data, num_blocks, voxel_size, radius, neighbor_limits):
    assert num_blocks == len(neighbor_limits)
    xyz1, xyz2, feat1, feat2, trans, cov = list_data[0] # batch size = 1
    #start = time.time()
    pcd1 = precompute_data(xyz1, num_blocks, voxel_size, radius, neighbor_limits)
    pcd2 = precompute_data(xyz2, num_blocks, voxel_size, radius, neighbor_limits)
    #print('precompute_data_stack_mode: %fs'%(time.time()-start))
    pcd1['features'], pcd2['features'] = feat1, feat2
    return {
        'src_pcd': pcd1,
        'tar_pcd': pcd2,
        'transform': torch.from_numpy(trans),
        'covariance': torch.from_numpy(cov) if cov is not None else None
    }


def calibrate_neighbors(dataset, voxel_size, radius_factor=2.5, num_layers=5, keep_ratio=0.8, samples_threshold=2000):
    hist_n = int(np.ceil(4 / 3 * np.pi * (radius_factor + 1.) ** 3))
    neighb_hists = np.zeros((num_layers, hist_n), dtype=np.int32)

    for i in range(len(dataset)):
        batch = collate_fn_precompute_data_stack_mode(
            [dataset[i]], num_layers, voxel_size, voxel_size*radius_factor, [hist_n]*num_layers
        )['tar_pcd']
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in batch['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)
        print(f"Calibrate Neighbors {i:08d}")
        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold: break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n-1, :]), axis=0)
    neighbor_limits = percentiles; print()
    return neighbor_limits


def get_dataloader(dataset, voxel_size, radius_factor=2.5, num_layers=5, num_workers=4, shuffle=True, neighbor_limits=None):
    # neighbor_limits = [37, 31, 34, 37, 35] # [25, 23, 24, 25, 29]
    if neighbor_limits is None:
        neighbor_limits = calibrate_neighbors(dataset, voxel_size, radius_factor, num_layers)
    print("neighborhood:", neighbor_limits)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers,
        collate_fn=partial(collate_fn_precompute_data_stack_mode, num_blocks=num_layers, voxel_size=voxel_size,
                           radius=voxel_size*radius_factor, neighbor_limits=neighbor_limits))
    return dataloader, neighbor_limits
