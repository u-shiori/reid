import torch.nn as nn
import torch.nn.functional as F

class ClusterTripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin_a, margin_b=None):
        super(ClusterTripletLoss, self).__init__()
        self.margin_a = margin_a
        self.margin_b = margin_b

    def forward(self, anchor, positive, negative, distance_cluster=None, size_average=True):

        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        
        losses = F.relu(distance_positive - distance_negative + self.margin_a)
        if distance_cluster is not None:
            losses += F.relu(self.margin_b-distance_cluster)
        return losses.mean() if size_average else losses.sum()


