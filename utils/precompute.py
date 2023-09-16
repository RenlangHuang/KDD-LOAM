import time
import torch
import threading
import numpy as np
from typing import List
from scipy.spatial import cKDTree
import cpp_extensions.cpp_subsampling.grid_subsampling as grid_subsampling


def batch_grid_subsampling(points, batches_len, sampleDl=0.1):
    s_points, s_len = grid_subsampling.subsample_batch(points, batches_len, sampleDl=sampleDl, max_p=0, verbose=0)
    return torch.from_numpy(s_points), torch.from_numpy(s_len)

def build_kdtree(data, idx, trees):
        trees[idx] = cKDTree(data)

def ball_query(kdtree:cKDTree, queries, tree_size, num, radius, idx, result):
        dist, ind = kdtree.query(queries, k=num, distance_upper_bound=radius)
        ind = np.where(dist < radius, ind, tree_size)
        result[idx] = torch.from_numpy(ind).cuda()


def precompute_data(points:torch.Tensor, num_blocks, voxel_size, radius, neighbor_limits):    
    assert num_blocks == len(neighbor_limits)
    points_list: List[torch.Tensor] = [None for _ in range(num_blocks)]
    length_list: List[torch.Tensor] = [None for _ in range(num_blocks)]
    kdtrees: List[cKDTree] = [None for _ in range(num_blocks)]
    threads: List[threading.Thread] = list()
    
    start = time.time()
    points_list[0] = points
    length_list[0] = torch.IntTensor([points.shape[0]])
    threads.append(threading.Thread(
        target=build_kdtree, args=(points_list[0], 0, kdtrees)
    ))
    threads[-1].start()

    for i in range(1, num_blocks):
        voxel_size = voxel_size * 2.
        points_list[i], length_list[i] = batch_grid_subsampling(
            points_list[i - 1], length_list[i - 1], voxel_size)
        threads.append(threading.Thread(
            target=build_kdtree, args=(points_list[i], i, kdtrees)
        ))
        threads[-1].start()
    
    for thread in threads: thread.join()
    threads.clear()
    
    neighbors_list: List[torch.Tensor] = [None for _ in range(num_blocks)]
    downsampling_list: List[torch.Tensor] = [None for _ in range(num_blocks-1)]
    upsampling_list: List[torch.Tensor] = [None for _ in range(num_blocks-1)]

    start = time.time()
    for i in range(num_blocks):
        threads.append(threading.Thread(target=ball_query, args=(
            kdtrees[i], points_list[i],
            length_list[i].item(), neighbor_limits[i],
            radius, i, neighbors_list
        )))
        threads[-1].start()
        if i < num_blocks - 1:
            threads.append(threading.Thread(target=ball_query, args=(
                kdtrees[i], points_list[i+1],
                length_list[i].item(), neighbor_limits[i],
                radius, i, downsampling_list
            )))
            threads[-1].start()
            threads.append(threading.Thread(target=ball_query, args=(
                kdtrees[i+1], points_list[i],
                length_list[i+1].item(), neighbor_limits[i+1],
                radius*2., i, upsampling_list
            )))
            threads[-1].start()
        radius = radius * 2.
    
    for thread in threads: thread.join()
    print('ball grouping: %fs'%(time.time()-start))

    return {
        'points': [v.cuda() for v in points_list],
        'stack_lengths': [v.cuda() for v in length_list],
        'neighbors': [v.long() for v in neighbors_list],
        'subsampling': [v.long() for v in downsampling_list],
        'upsampling': [v.long() for v in upsampling_list],
    }