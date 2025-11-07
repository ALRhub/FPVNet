from pytorch3d.ops import sample_farthest_points, knn_points, knn_gather
from torch import nn

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size, normalize_neighborhood=False):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.normalize_neighborhood = normalize_neighborhood

    def forward(self, points):
        '''
            input: B N 3+x
            ---------------------------
            output: B G M 3+x
            center : B G 3
        '''

        xyz = points[..., :3]
 
        center, _ = sample_farthest_points(xyz, K=self.num_group)  # B G 3
        _, idx, test_neigh = knn_points(center, xyz, K=self.group_size, return_nn=True)

        neighborhood = knn_gather(points, idx)

        if self.normalize_neighborhood:
            neighborhood[..., :3] = neighborhood[..., :3] - center.unsqueeze(2)

        return neighborhood, center