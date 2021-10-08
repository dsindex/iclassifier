# ---------------------------------------------------------------------------------------------------------- #
# source from https://github.com/dlmacedo/entropic-out-of-distribution-detection/blob/master/losses/isomax.py
# ---------------------------------------------------------------------------------------------------------- #

import torch
import torch.nn as nn
import torch.nn.functional as F

class IsoMax(nn.Module):
    """Replaces the model classifier last layer nn.Linear()"""
    def __init__(self, num_features, num_classes):
        super(IsoMaxLayer, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        nn.init.constant_(self.prototypes, 0.0)

    def forward(self, features):
        distances = F.pairwise_distance(features.unsqueeze(2), self.prototypes.t().unsqueeze(0), p=2.0)
        logits = -distances
        return logits
