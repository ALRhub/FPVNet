import einops
import torch

from pytorch3d.ops import knn_points, knn_gather

def interpolate_point_cloud_features(features, feature_point_cloud, target_point_cloud):
    """
    Interpolate point cloud features based on the point cloud coordinates.
    Args:
        features: (B, N, D) tensor, features to interpolate
        feature_point_cloud: (B, N, 3) tensor, coordinates of the features
        target_point_cloud: (B, M, 3) tensor, coordinates of the target points
    Returns:
        interpolated_features: (B, M, D) tensor, interpolated features
    """

    dists = torch.cdist(target_point_cloud, feature_point_cloud)
    weights = 1 / (dists + 1e-10) ** 2
    interpolated_features = torch.einsum(
        "bij, bjk -> bik", weights, features
    ) / torch.sum(weights, dim=-1, keepdim=True)

    return interpolated_features

def average_pc_features_from_nearest_points(features, feature_point_cloud, target_point_cloud, k_nearest, return_min_distance=False):
    """
    Average point cloud features from the nearest points.
    Args:
        features: (B, num_cam, N, D) tensor, features to interpolate
        feature_point_cloud: (B, num_cam, N, 3) tensor, coordinates of the features
        target_point_cloud: (B, M, 3) tensor, coordinates of the target points
        k_nearest: int, number of nearest points to consider
    Returns:
        averaged_features: (B, M, D) tensor, averaged features
    """
    batch_size = features.shape[0]
    num_cam = features.shape[1]
    target_num_points = target_point_cloud.shape[1]
    feature_dim = features.shape[-1]

    min_dists = torch.zeros(batch_size, target_num_points, num_cam, k_nearest, device=features.device)
    min_features = torch.zeros(batch_size, target_num_points, num_cam, k_nearest, feature_dim, device=features.device)

    for i in range(feature_point_cloud.shape[1]):
        dists, idx, _ = knn_points(target_point_cloud, feature_point_cloud[:, i], K=k_nearest, return_sorted=False, return_nn=False)
        min_dists[:, :, i] = dists
        min_features[:, :, i] = knn_gather(features[:, i], idx)

    min_dists = einops.rearrange(min_dists, 'b n num_cam k -> b n (num_cam k)')
    min_features = einops.rearrange(min_features, 'b n num_cam k d -> b n (num_cam k) d')

    weights = 1 / (min_dists + 1e-10) ** 2
    averaged_features = torch.einsum("bij, bijk -> bik", weights, min_features) / torch.sum(weights, dim=-1, keepdim=True)

    if return_min_distance:
        return averaged_features, torch.min(min_dists, dim=-1, keepdim=True).values

    return averaged_features

    
