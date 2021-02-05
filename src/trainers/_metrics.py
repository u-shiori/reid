import numpy as np

def euclidean_metric(anchor, positive, negative=None):
    distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
    if negative is None:
        distance_negative = None
    else:
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)

    return distance_positive, distance_negative

    
def cosin_metric(anchor, positive, negative=None):

    anchor = anchor.data.cpu().numpy()
    positive = positive.data.cpu().numpy()
    distance_positive = []
    if negative is None:
        distance_negative = None
    else:
        distance_negative = []
        negative = negative.data.cpu().numpy()
    
    for i in range(len(anchor)):
        p_dist = 1 - np.dot(anchor[i], positive[i]) / (np.linalg.norm(anchor[i]) * np.linalg.norm(positive[i]))
        distance_positive.append(p_dist)
        if negative is not None:
            n_dist = 1 - np.dot(anchor[i], negative[i]) / (np.linalg.norm(anchor[i]) * np.linalg.norm(negative[i]))
            distance_negative.append(n_dist)
        

    return distance_positive, distance_negative