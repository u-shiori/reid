import numpy as np
import torch

def triplet_euclidean(anchor, positive, negative):
    distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
    distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)

    return distance_positive, distance_negative


def doublet_euclidean(xs,ys):
    return (xs - ys).pow(2).sum(1)


def triplet_cosin(anchor, positive, negative):

    distance_positive = []
    distance_negative = []
    anchor = anchor.data.cpu().numpy()
    positive = positive.data.cpu().numpy()
    negative = negative.data.cpu().numpy()
    
    for i in range(len(anchor)):
        p_dist = 1 - np.dot(anchor[i], positive[i]) / (np.linalg.norm(anchor[i]) * np.linalg.norm(positive[i]))
        n_dist = 1 - np.dot(anchor[i], negative[i]) / (np.linalg.norm(anchor[i]) * np.linalg.norm(negative[i]))

        distance_positive.append(p_dist)
        distance_negative.append(n_dist)

    return distance_positive, distance_negative

def doublet_cosin(xs,ys):
    dot = (xs*ys).sum(1,keepdim=True).squeeze()#内積
    return 1. -  dot / (torch.norm(xs, dim=1) * torch.norm(ys, dim=1))
