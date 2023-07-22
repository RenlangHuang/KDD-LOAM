import time
import typing
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree


def LocalFeatureMatching(
    source: np.ndarray,
    target: np.ndarray,
    source_features: np.ndarray,
    target_features: np.ndarray,
    max_correspondence_distance: float,
    init: np.ndarray,
    num_neighbors: int,
    similarity_threshold: float,
    num_workers: int = 1
) -> typing.Tuple[np.ndarray, cKDTree]:

    kdtree = cKDTree(target)
    source = source @ init[:3, :3].T + init[:3, -1:].T
    dist, ind2 = kdtree.query(
        source, k=num_neighbors, p=2,
        distance_upper_bound=1e6,
        workers=num_workers
    )
    ind1 = np.expand_dims(np.arange(source.shape[0]),1)
    ind1 = np.repeat(ind1, num_neighbors, 1)
    ind = np.stack([ind1, ind2], axis=-1).reshape([-1, 2])
    
    desc_dist = target_features[ind[:, 1]] - source_features[ind[:, 0]]
    desc_dist = np.linalg.norm(desc_dist, axis=-1)
    desc_dist = np.reshape(desc_dist, [-1, num_neighbors])

    selection = np.argmin(desc_dist, axis=1)
    selection = np.expand_dims(selection, axis=1)
    corres = np.zeros_like(desc_dist) + 1e6
    np.put_along_axis(corres, selection, 1, axis=1)
    corres = (corres * desc_dist) < similarity_threshold
    corres = np.logical_and(corres, dist < max_correspondence_distance)
    return ind.reshape([-1, num_neighbors, 2])[corres]


def TwoStageLocalFeatureMatching(
    source:o3d.geometry.PointCloud,
    target:o3d.geometry.PointCloud,
    source_feature:np.ndarray, target_feature:np.ndarray,
    init: np.ndarray, super_grid_size:float,
    radius:float = 1., num_k:int = 5
):

    def gaussian(X, Y, sigma=1., euler=False):
        X = np.array(X)
        Y = np.array(Y)
        D2 = np.sum(X*X, axis=1, keepdims=True) + np.sum(Y*Y, axis=1, keepdims=True).T - 2 * np.dot(X, Y.T)
        if euler==True: return D2
        else: return np.exp(-D2 / (2. * sigma ** 2))
    
    # 共用当前帧超点
    source.transform(init)
    super_point = target.voxel_down_sample(voxel_size=super_grid_size)
    super_point_kdtree = o3d.geometry.KDTreeFlann(super_point)
    
    source_full_num = np.array(source.points).shape[0]
    target_full_num = np.array(target.points).shape[0]
    # 建立source和target对于超点的最近邻配对数组
    source_n2s = np.empty(shape=source_full_num)
    target_n2s = np.empty(shape=target_full_num)
    for source_index,point in enumerate(source.points):
        k, indice, _ = super_point_kdtree.search_hybrid_vector_3d(point, radius, 1)
        if k == 0: continue
        source_n2s[source_index] = indice.pop()
    for target_index,point in enumerate(target.points):
        k, indice, _ = super_point_kdtree.search_hybrid_vector_3d(point, radius, 1)
        if k == 0: continue
        target_n2s[target_index] = indice.pop()

    for super_index, point in enumerate(super_point.points):
        # 建立每个超点下存放普通点描述子的数组
        source_vector = np.empty(shape = (0, 32))
        target_vector = np.empty(shape = (0, 32))
        tmp_source = np.where(source_n2s == super_index)
        tmp_target = np.where(target_n2s == super_index)
        # 把对应的普通点描述子放进去
        source_vector = np.append(source_vector, source_feature[tmp_source],axis=0)
        target_vector = np.append(target_vector, target_feature[tmp_target],axis=0)
        # 如果有一个点云超点没有任何普通点就跳过这一对
        if source_vector.shape[0]==0 or target_vector.shape[0]==0: continue
        # 计算高斯相关矩阵
        matrix = torch.exp(torch.Tensor(gaussian(source_vector,target_vector)))
        # 归一化
        target_matching_scores = matrix / matrix.sum(dim=1, keepdim=True)
        source_matching_scores = matrix / matrix.sum(dim=0, keepdim=True)
        matching_scores = target_matching_scores * source_matching_scores
        # 确定k值
        num_correspondences = min(num_k, matching_scores.numel())
        # 展开为一维数组取top_k
        corr_scores, corr_indices = matching_scores.view(-1).topk(k=num_correspondences, largest=True)
        # 通过矩阵维度计算位置
        source_indices = torch.div(corr_indices,matching_scores.shape[1],rounding_mode='floor')
        target_indices = corr_indices % matching_scores.shape[1] # one dimensional



def RegistrationBasedOnCorrespondences(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_correspondence_distance: float,
    correspondences: np.ndarray,
    init: np.ndarray,
    criteria: o3d.pipelines.registration.ICPConvergenceCriteria,
    init_correspondence_distance: float = -0.1,
    decay_rate: float = 1.0,
) -> o3d.pipelines.registration.RegistrationResult:
    
    source_ = source
    source_.transform(init)
    target_points = np.array(target.points)
    transformation = init
    
    result = o3d.pipelines.registration.evaluate_registration(
        source_, target, max_correspondence_distance, np.eye(4)
    )
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)

    if init_correspondence_distance < 0:
        correspondence_distance = max_correspondence_distance
    else:
        correspondence_distance = init_correspondence_distance

    for _ in range(criteria.max_iteration):
        t_iter = time.time()
        src = np.array(source_.points)[correspondences[:, 0]]
        tar = target_points[correspondences[:, 1]]
        dist = np.linalg.norm(src - tar, axis=-1)
        dist = np.argwhere(dist < correspondence_distance)
        corres = correspondences[dist.squeeze()]
        print(corres.shape, end=' ')

        update = estimation.compute_transformation(
            source_, target, o3d.utility.Vector2iVector(corres)
        )
        transformation = update @ transformation
        source_.transform(update)
        backup = result

        result = o3d.pipelines.registration.evaluate_registration(
            source_, target, max_correspondence_distance, np.eye(4)
        )
        correspondence_distance = correspondence_distance * decay_rate
        if correspondence_distance < max_correspondence_distance:
            correspondence_distance = max_correspondence_distance
        print(time.time() - t_iter)

        if  abs(backup.fitness - result.fitness) < criteria.relative_fitness and \
            abs(backup.inlier_rmse - result.inlier_rmse) < criteria.relative_rmse:
            break
    
    result.transformation = transformation
    return result


if __name__=="__main__":
    pass