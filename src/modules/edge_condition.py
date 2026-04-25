import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeFeatureInjector(nn.Module):
    """Project edge maps to multi-scale feature tensors and add them as conditions."""

    def __init__(self, channel_list, init_scale=0.1):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(1, c, kernel_size=1) for c in channel_list])
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, edge_map, feature_list):
        conditioned_features = []
        for feature, proj in zip(feature_list, self.projections):
            resized_edge = F.interpolate(edge_map, size=feature.shape[-2:], mode="bilinear", align_corners=False)
            conditioned_features.append(feature + self.scale * proj(resized_edge))
        return conditioned_features
