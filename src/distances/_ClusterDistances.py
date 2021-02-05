import torch



#最短距離法
class single_method:
    def __init__(self, object_distance):
        self.object_distance = object_distance
    
    def __call__(self, features1, features2):

        assert features1.shape[1] == features2.shape[1]
        
        features1_for_dist \
            = torch.cat([features1 for _ in range(len(features2))])
        features2_for_dist \
            = torch.stack([features2[i] for i in range(len(features2)) for _ in range(len(features1))])
        dists = self.object_distance(features1_for_dist, features2_for_dist)#クラス間の全サンプルごとに類似度関数を計算
        assert features1.shape[0] * features2.shape[0] == len(dists)
        single_dist = dists.min().tolist()
        return single_dist

#最長距離法
class complete_method:
    def __init__(self, object_distance):
        self.object_distance = object_distance

    def __call__(self, features1, features2):
        assert features1.shape[1] == features2.shape[1]
        features1_for_dist \
            = torch.cat([features1 for _ in range(len(features2))])
        features2_for_dist \
            = torch.stack([features2[i] for i in range(len(features2)) for _ in range(len(features1))])
        dists = self.object_distance(features1_for_dist, features2_for_dist)#クラス間の全サンプルごとに類似度関数を計算
        assert features1.shape[0] * features2.shape[0] == len(dists)
        complete_dist = dists.max().tolist()
        return complete_dist

#群平均法
class average_method:
    def __init__(self, object_distance):
        self.object_distance = object_distance

    def __call__(self, features1, features2):
        assert features1.shape[1] == features2.shape[1]
        features1_for_dist \
            = torch.cat([features1 for _ in range(len(features2))])
        features2_for_dist \
            = torch.stack([features2[i] for i in range(len(features2)) for _ in range(len(features1))])
        dists = self.object_distance(features1_for_dist, features2_for_dist)#クラス間の全サンプルごとに類似度関数を計算
        assert features1.shape[0] * features2.shape[0] == len(dists)
        average_dist = dists.mean().tolist()
        return average_dist

#重心法
class centroid_method:
    def __init__(self, object_distance):
        self.object_distance = object_distance
    
    def __call__(self, features1, features2):

        assert features1.shape[1] == features2.shape[1]

        #各クラスの重心を求める．
        center1 = features1.mean(axis=0).reshape((1,features1.shape[1]))
        center2 = features2.mean(axis=0).reshape((1,features2.shape[1]))
        
        #重心間距離を求める．
        dists = self.object_distance(center1, center2)
        assert len(dists) == 1
        centroid_dist = dists[0]
        return centroid_dist

#内部平方距離法
class ward_method:
    def __init__(self, object_distance):
        self.object_distance = object_distance
        
    def __call__(self, features1, features2):

        assert features1.shape[1] == features2.shape[1]

        #各クラスの重心を求める．
        center1 = features1.mean(axis=0).reshape((1,features1.shape[1]))
        center2 = features2.mean(axis=0).reshape((1,features2.shape[1]))

        #クラス内要素の重心までの距離の２乗和 L を求める．
        dists1 = self.object_distance(center1, features1)
        assert features1.shape[0] == len(dists1)
        L1 = torch.einsum('i, i -> i', dists1, dists1).sum()

        dists2 = self.object_distance(center2, features2)
        assert features2.shape[0] == len(dists2)
        L2 = torch.einsum('i, i -> i', dists2, dists2).sum()
        

        #２クラス共通の重心を求める
        features1or2 = torch.cat([features1, features2])
        assert len(features1) + len(features2) == len(features1or2)
        center1or2 = features1or2.mean(axis=0).reshape((1,features1or2.shape[1]))

        #２クラス共通の重心までの距離の２乗和𝐿(1∨2)を求める．
        dists1or2 = self.object_distance(center1or2, features1or2)
        assert features1or2.shape[0] == len(dists1or2)
        L1or2 = torch.einsum('i, i -> i', dists1or2, dists1or2).sum()

        #𝑳(1v2)−𝑳(1)−𝑳(2)を求める．
        ward_dist = (L1or2 - L1 - L2).tolist()

        return ward_dist


    

if __name__ == "__main__":
    from _ObjectDistances import doublet_euclidean

    features1 = torch.tensor([[0,0],[0,1],[1,0],[1,1]])
    features2 = torch.tensor([[-1,0], [0,-2]])

    single_dist = single_method(features1, features2, doublet_euclidean)
    assert single_dist == 1
    complete_dist = complete_method(features1, features2, doublet_euclidean)
    assert complete_dist == 10



    
    