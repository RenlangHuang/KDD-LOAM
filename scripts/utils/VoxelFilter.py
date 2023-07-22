import numpy as np
from typing import Dict, List


class Leaf:
    def __init__(self):
        self.idx:int = -1


def UniformSampling(radius, source: List[np.ndarray], target: List[np.ndarray]):
    leaf_size_ = np.array([[radius, radius, radius]])
    inverse_leaf_size = 1. / leaf_size_
    leaves_: Dict[int, Leaf] = dict()

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

    for (i, ind) in enumerate(idx):
        if ind not in leaves_.keys():
            leaf = Leaf()
            leaf.idx = i
            leaves_[ind] = leaf
            continue
        leaf = leaves_[ind]
        if priority[i] < priority[leaf.idx]:
            leaf.idx = i
    
    for leaf in leaves_.values():
        target.append(pcd[leaf.idx])


def uniform_sampling(radius, source: List[np.ndarray]):
    leaf_size_ = np.array([[radius, radius, radius]])
    inverse_leaf_size = 1. / leaf_size_
    leaves_: Dict[int, Leaf] = dict()

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


if __name__=="__main__":
    import time
    import threading
    import multiprocessing
    from typing import List
    
    threads: List[threading.Thread] = list()
    processes: List[multiprocessing.Process] = list()
    pcd = np.random.normal(0., 3., [30000, 4])
    #pcd = np.random.random([30000, 4]) * 3.

    itrs = 5

    start = time.time()

    for _ in range(itrs):
        threads.append(threading.Thread(
            target=uniform_sampling,
            args=(0.3, pcd)
        ))
        threads[-1].start()
    
    for thread in threads:
        thread.join()
    
    print(time.time() - start)

    threads.clear()


    start = time.time()

    for _ in range(itrs):
        uniform_sampling(0.3, pcd)
    
    print(time.time() - start)
    
    start = time.time()

    for _ in range(itrs):
        UniformSampling(0.3, pcd, list())
    
    print(time.time() - start)