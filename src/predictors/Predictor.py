import torch
import os
import numpy as np
import csv

class Predictor:
    def __init__(self, model, cluster_distance, output_dir=None):
        """ある特定場所において，通行人が以前通った誰かなのか，もしくは未知の人物かを予測する
        Parameters
        ----------
        model :
            モデル
        cluster_distance :
            クラスタ間距離
        output_dir : str, default None

        """
        self.model = model
        self.cluster_distance = cluster_distance
        self.output_dir = output_dir
        self.cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedded_features = None
        self.embedded_labels = None
        self.embedded_paths = None
        self.embedded_set_labels = None
        self.cluster_distances = None



    def setEmbeddingSpace(self, data_loader):
        """data_loaderを特徴空間へ埋め込む

        Parameters
        ----------
        data_loader

        """
        features, labels, paths = self._getFeatures(data_loader)

        self.embedded_features = features
        self.embedded_labels = labels
        self.embedded_paths = paths
        self.embedded_set_labels = set(labels)

        cluster_distances, min_cluster_distances = self._calcEmbeddedClusterDistances()
        self.cluster_distances = cluster_distances
        self.thresholds = min_cluster_distances
        assert len(self.embedded_set_labels) == len(min_cluster_distances)

    def calcDistribution(self):
        """クラス内分散とクラス間分散を求める．
        クラス内分散...各クラスの要素と平均との最大距離の平均
        クラス間分散...各クラスと他クラスとの最小クラスタ間距離の平均

        returens
        ----------
        inclass_distributions : list
            クラスごとのクラス内分散
        interclass_distributions : list
            クラスごとのクラス間分散
        inclass_distribution : float
            クラス内分散
        intercalss_distribution : float
            クラス間分散
        """
        
        inclass_distributions = list()
        interclass_distributions = list()
        for inlabel in self.embedded_set_labels:
            # クラス内分散を求める
            features_inlabel = self.embedded_features[np.where(np.array(self.embedded_labels)==inlabel)[0]]
            mean_feature_inlabel = features_inlabel.mean(0)
            object_distances_inlabel = self.cluster_distance.object_distance(mean_feature_inlabel,features_inlabel)
            inclass_distributions.append(max(object_distances_inlabel).tolist())

            # クラス間分散を求める
            cluster_distances_interlabel = list()
            for outlabel in self.embedded_set_labels-{inlabel}:
                features_outlabel = self.embedded_features[np.where(np.array(self.embedded_labels)==outlabel)[0]]
                cluster_distance_interlabel = self.cluster_distance(features_inlabel,features_outlabel)
                cluster_distances_interlabel.append(cluster_distance_interlabel)
            interclass_distributions.append(min(cluster_distances_interlabel))


        inclass_distribution = np.array(inclass_distributions).mean()
        interclass_distribution = np.array(interclass_distributions).mean()
        return inclass_distributions, interclass_distributions, inclass_distribution, interclass_distribution



    def predictID(self, data_loader, eval=False):
        """IDを予測する（既存人物の誰かなのか，もしくは未知人物なのか）

        Parameters
        ----------
        data_loader
        eval : bool
            評価用か否か

        returns
        ---
        preds : list
            予測結果 (len(set_labels),)
        set_labels : set
            data_loaderに存在するlabelの集合
        cluster_distances : list
            クラスタ間距離の集合
    
        """
        features, labels, paths = self._getFeatures(data_loader)
        set_labels = set(labels)
        cluster_distances, min_cluster_distances, argmin_cluster_distances = self._calcClusterDistances(features, labels, set_labels)

        preds = list()
        for min_cluster_distance, argmin_cluster_distance in zip(min_cluster_distances, argmin_cluster_distances):
            if self.thresholds[argmin_cluster_distance] < min_cluster_distance:
                preds.append(-1)#未知人物と予測
            else:
                preds.append(list(self.embedded_set_labels)[argmin_cluster_distance])
        if eval:
            return preds, set_labels, cluster_distances, features, labels
        else:
            return preds


    def _getFeatures(self, data_loader):
        """画像から特徴を得る．

        Parameters
        ----------
        data_loader

        Notes
        -----
        現在は，正解ラベルがある前提
        """

        self.model.eval()
        with torch.no_grad():
            # if self.cuda:
            #     features = torch.tensor([]).cuda()
            # else:
            features = torch.tensor([])
            labels = torch.tensor([], dtype=torch.int64)
            paths = list()

            for img_paths, data, targets in data_loader:
                if self.cuda:    data = data.cuda()
                outputs = self.model(data)
                if self.cuda:   outputs = outputs.cpu()
                features = torch.cat((features, outputs), 0)
                labels = torch.cat((labels, targets), 0)
                img_paths = [os.path.basename(path) for path in img_paths]
                paths.extend(img_paths)
        labels = labels.tolist()
        return features, labels, paths

    def _calcEmbeddedClusterDistances(self):
        """埋め込み済みの特徴で，各ラベルの他クラスとのクラスタ間距離を計算する

        return
        ----
        cluster_distance : list
            各ラベルの他クラスとのクラスタ間距離集合
        min_cluster_distances : list
            各ラベルの他クラスとの最小クラスタ間距離
        Notes
        -----
        cluster_distance はヒストグラムの図示のために出力．
        min_cluster_distancesは閾値に用いる．→現在はラベルごとに閾値が異なる．
        """
        cluster_distances = list()
        min_cluster_distances = list()

        for label in self.embedded_set_labels:
            inds_in_label = np.where(np.array(self.embedded_labels)==label)[0]
            features_in_label = self.embedded_features[inds_in_label]

            cluster_distances_from_label_to_other_labels = list()
            other_set_labels = self.embedded_set_labels - {label}
            for another_label in other_set_labels:
                inds_in_another_label = np.where(np.array(self.embedded_labels)==another_label)[0]
                features_in_another_label = self.embedded_features[inds_in_another_label]
                cluster_distance_from_label_to_another_label \
                    = self.cluster_distance(features_in_label, features_in_another_label)
                cluster_distances_from_label_to_other_labels.append(\
                    cluster_distance_from_label_to_another_label)

            cluster_distances.append(cluster_distances_from_label_to_other_labels)
            min_cluster_distances.append(min(cluster_distances_from_label_to_other_labels))
        return cluster_distances, min_cluster_distances

    def _calcClusterDistances(self, features, labels, set_labels):
        """set_labelsの各ラベルの他クラスとのクラスタ間距離を計算する
        Parameters
        ----
        features : list
        labels : list
        set_labels :set
            set_labels=set(labels)ごとに

        return
        ----
        cluster_distance : list
            各ラベルの他クラスとのクラスタ間距離集合
        min_cluster_distances : list
            各ラベルの他クラスとの最小クラスタ間距離
        argmin_cluster_distances : list
            各ラベルの他クラスとの最小クラスタ間距離をとるindex集合

        Notes
        -----
        cluster_distance はヒストグラムの図示のために出力．
        min_cluster_distancesは閾値に用いる．→現在はラベルごとに閾値が異なる．
        argmin_cluster_distancesは，既知人物と判別されたときにどのラベルと最も近かったのかをはかるために用いる．
        """

        cluster_distances = list()
        min_cluster_distances = list()
        argmin_cluster_distances = list()

        for label in set_labels:
            inds_in_label = np.where(np.array(labels)==label)[0] 
            features_in_label = features[inds_in_label]

            cluster_distances_from_label_to_embedded_labels = list()
            for embedded_label in self.embedded_set_labels:
                inds_in_embedded_label = np.where(np.array(self.embedded_labels)==embedded_label)[0]
                features_in_embedded_label = self.embedded_features[inds_in_embedded_label]

                cluster_distance_from_label_to_embedded_label \
                    = self.cluster_distance(features_in_label, features_in_embedded_label)
                cluster_distances_from_label_to_embedded_labels.append(\
                    cluster_distance_from_label_to_embedded_label)

            cluster_distances.append(cluster_distances_from_label_to_embedded_labels)
            cluster_distances_from_label_to_embedded_labels \
                = np.array(cluster_distances_from_label_to_embedded_labels)
            min_cluster_distances.append(cluster_distances_from_label_to_embedded_labels.min())
            argmin_cluster_distances.append(cluster_distances_from_label_to_embedded_labels.argmin())
        return cluster_distances, min_cluster_distances, argmin_cluster_distances

