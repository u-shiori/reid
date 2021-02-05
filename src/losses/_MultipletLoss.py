import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin_a, margin_b=None):
        super(TripletLoss, self).__init__()
        self.margin_a = margin_a
        self.margin_b = margin_b

    def forward(self, anchor, positive, negative, size_average=True):

        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin_a)
        if self.margin_b is not None:
            losses += F.relu(distance_positive - self.margin_b)
        return losses.mean() if size_average else losses.sum()


class QuadrupletLoss(nn.Module):
    """
    Quadruplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin_a, margin_b):
        super(QuadrupletLoss, self).__init__()
        self.margin_a = margin_a
        self.margin_b = margin_b

    def forward(self, anchor, positive, negative1, negative2, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative1 = (anchor - negative1).pow(2).sum(1)  # .pow(.5)
        distance_negative2 = (negative1 - negative2).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative1 + self.margin_a) \
                + F.relu(distance_positive - distance_negative2 + self.margin_b)
        return losses.mean() if size_average else losses.sum()


