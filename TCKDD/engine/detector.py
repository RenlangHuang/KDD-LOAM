import numpy as np
from typing import Tuple


def random_sample_keypoints(
    points: np.ndarray,
    descriptors: np.ndarray,
    saliency: np.ndarray,
    num_keypoints: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    num_points = points.shape[0]
    if num_points > num_keypoints:
        indices = np.random.choice(num_points, num_keypoints, replace=False)
        points = points[indices]
        descriptors = descriptors[indices]
        saliency = saliency[indices]
    return points, descriptors, saliency


def sample_keypoints_with_scores(
    points: np.ndarray,
    descriptors: np.ndarray,
    saliency: np.ndarray,
    num_keypoints: int,
) -> Tuple[np.ndarray, np.ndarray]:
    
    num_points = points.shape[0]
    if num_points > num_keypoints:
        indices = np.argsort(saliency)[:num_keypoints]
        points = points[indices]
        descriptors = descriptors[indices]
    return points, descriptors


def random_sample_keypoints_with_scores(
    points: np.ndarray,
    descriptors: np.ndarray,
    saliency: np.ndarray,
    num_keypoints: int,
) -> Tuple[np.ndarray, np.ndarray]:
    
    num_points = points.shape[0]
    saliency = np.exp(-saliency)
    if num_points > num_keypoints:
        indices = np.arange(num_points)
        probs = saliency / np.sum(saliency)
        indices = np.random.choice(indices, num_keypoints, replace=False, p=probs)
        points = points[indices]
        descriptors = descriptors[indices]
    return points, descriptors


def sample_keypoints_with_nms(
    points: np.ndarray,
    descriptors: np.ndarray,
    saliency: np.ndarray,
    num_keypoints: int,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    
    num_points = points.shape[0]
    if num_points > num_keypoints:
        radius2 = radius ** 2
        masks = np.ones(num_points, dtype=np.bool)
        sorted_indices = np.argsort(saliency)
        sorted_points = points[sorted_indices]
        sorted_feats = descriptors[sorted_indices]
        indices = list()
        for i in range(num_points):
            if masks[i]:
                indices.append(i)
                if len(indices) == num_keypoints: break
                if i + 1 < num_points:
                    current_masks = np.sum((sorted_points[i+1:]-sorted_points[i])**2, axis=1)<radius2
                    masks[i+1:] = masks[i+1:] & ~current_masks
        points = sorted_points[indices]
        descriptors = sorted_feats[indices]
    return points, descriptors


def random_sample_keypoints_with_nms(
    points: np.ndarray,
    descriptors: np.ndarray,
    saliency: np.ndarray,
    num_keypoints: int,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    
    num_points = points.shape[0]
    if num_points > num_keypoints:
        radius2 = radius ** 2
        masks = np.ones(num_points, dtype=np.bool)
        sorted_indices = np.argsort(saliency)
        sorted_points = points[sorted_indices]
        sorted_feats = descriptors[sorted_indices]
        indices = list()
        for i in range(num_points):
            if masks[i]:
                indices.append(i)
                if i + 1 < num_points:
                    current_masks = np.sum((sorted_points[i+1:]-sorted_points[i])**2, axis=1)<radius2
                    masks[i+1:] = masks[i+1:] & ~current_masks
        indices = np.array(indices)
        if len(indices) > num_keypoints:
            sorted_scores = saliency[sorted_indices]
            saliency = sorted_scores[indices]
            saliency = np.exp(-saliency)
            probs = saliency / np.sum(saliency)
            indices = np.random.choice(indices, num_keypoints, replace=False, p=probs)
        points = sorted_points[indices]
        descriptors = sorted_feats[indices]
    return points, descriptors
