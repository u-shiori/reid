import torch

cuda = torch.cuda.is_available()

import sklearn.base
import bhtsne
import scipy as sp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
import re
import logging

from datasets.getDataLoader import getDataLoader
from models.getModel import getModel
from distances.getDistance import getClusterDistance
from predictors.Predictor import Predictor

from _utils import getLogger, setLogLevel, setLogFile

pil_logger = getLogger('PIL')
pil_logger.setLevel(logging.INFO)
logger = getLogger(__name__)


def main(cfg_known_valid_path, cfg_known_inf_path, cfg_unknown_inf_path, weight_path, cluster_distance_name, thread, save=True):
    #load data
    pattern = ".*out(\d+).*"
    match = re.match(pattern, weight_path)
    num_out = int(match.group(1))

    model_names = ["triplet", "arcface", "improved_triplet", "quadruplet", "uir", "absolute_triplet", "absolute_triplet_hard_negative", "absolute_triplet_hard_negative_positive"]
    tmp_model_name = [split for split in weight_path.split("/") if split in model_names]
    assert len(tmp_model_name) == 1,\
        f""" {weight_path} から model名が検出できませんでした． """
    model_name = tmp_model_name[0]
    cluster_distances = ["Single", "Complete", "Average", "Centroid", "Ward"]
    assert cluster_distance_name in cluster_distances

    output_dir = f"../result/evals/{model_name}/{cluster_distance_name}/{weight_path.split('/')[-1].split('.')[0]}/{cfg_known_valid_path.split('/')[-2]}/"

    if model_name == "arcface":
        model_name = "Arcface"
    elif model_name == "triplet":
        model_name = "Triplet"
    elif model_name == "improved_triplet":
        model_name = "ImprovedTriplet"
    elif model_name == "quadruplet":
        model_name = "Quadruplet"
    elif model_name == "uir":
        model_name = "UIR"
    elif model_name == "absolute_triplet":
        model_name = "AbsoluteTriplet"
    elif model_name == "absolute_triplet_hard_negative_positive":
        model_name = "AbsoluteTripletHardNegativePositive"


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        while True:
            print(f"すでに{output_dir}は存在しますが続けますか？(y/n) : ", end="")
            inp = input()
            if inp == "y":
                break
            elif inp == "n":
                exit()
            else:
                print("'y' または 'n' をタイプしてください。")

    setLogFile(output_dir+'/eval.log')
    setLogLevel(10)
    logging.info(f"model : {model_name}")
    logging.info(f"num out : {num_out}")
    logging.info(f"cluster distance : {cluster_distance_name}")

    known_valid_dataloader, _, known_valid_class_num = getDataLoader(model_name, cfg_known_valid_path, batch_size=1, train=False, thread=thread, embedding_net=True)
    known_inf_dataloader, _, known_inf_class_num = getDataLoader(model_name, cfg_known_inf_path, batch_size=1, train=False, thread=thread, embedding_net=True)
    unknown_inf_dataloader, _, unknown_inf_class_num = getDataLoader(model_name, cfg_unknown_inf_path, batch_size=1, train=False, thread=thread, embedding_net=True)
    assert known_valid_class_num == known_inf_class_num

    #学習済みモデルをload,inference
    model = getModel(model_name, num_out, weight_path)

    #クラスタ間距離を取得
    cluster_distance = getClusterDistance(model_name, cluster_distance_name)

    #予測器を取得
    predictor = Predictor(model, cluster_distance, output_dir=output_dir)

    #埋め込み用のデータを特長空間上に埋め込む
    predictor.setEmbeddingSpace(known_valid_dataloader)




    #既知人物の予測
    known_preds, known_set_labels, known_cluster_distances, known_features, known_labels2\
        = predictor.predictID(known_inf_dataloader, eval=True)
    #既知人物の精度を出力
    known_labels = np.array(list(known_set_labels))
    known_cm = confusion_matrix(known_labels, known_preds)
    known_accuracy = accuracy_score(known_labels, known_preds)
    with open(f"{output_dir}/accuracy_known.csv", "w") as f:
        print(f"accuracy : {known_accuracy}", file=f)
        print("\nconfusion matrix", file=f)
        print(known_cm, file=f)
    with open(f"{output_dir}/actual_knwon.csv", "w") as f:
        for known_pred, known_label in zip(known_preds, known_labels):
            if known_pred != known_label:
                print(f"actual:{known_label} pred:{known_pred}", file=f)
    #未知人物の予測
    unknown_preds, unknown_set_labels, unknown_cluster_distances, unknown_features, unknown_labels2\
        = predictor.predictID(unknown_inf_dataloader, eval=True)
    unknown_labels = np.array([-1]*unknown_inf_class_num)
    #未知人物の精度を出力
    unknown_cm = confusion_matrix(unknown_labels, unknown_preds)
    unknown_accuracy = accuracy_score(unknown_labels, unknown_preds)
    with open(f"{output_dir}/accuracy_unknown.csv", "w") as f:
        print(f"accuracy : {unknown_accuracy}", file=f)
        print("\nconfusion matrix", file=f)
        print(unknown_cm, file=f)
    unknown_set_labels = list(unknown_set_labels)
    with open(f"{output_dir}/actual_unknwon.csv", "w") as f:
        for i, (unknown_pred, unknown_label) in enumerate(zip(unknown_preds, unknown_labels)):
            if unknown_pred != unknown_label:
                print(f"actual:{unknown_set_labels[i]} pred:{unknown_pred}", file=f)

    #既知人物と未知人物の合算精度を出力
    all_preds = np.concatenate([known_preds, unknown_preds])
    all_labels = np.concatenate([known_labels, unknown_labels])
    all_cm = confusion_matrix(all_labels, all_preds)
    all_accuracy = accuracy_score(all_labels, all_preds)
    with open(f"{output_dir}/accuracy_all.csv", "w") as f:
        print(f"accuracy : {all_accuracy}", file=f)
        print("\nconfusion matrix", file=f)
        print(all_cm, file=f)

    #ヒストグラムを出力
    for i, label in enumerate(predictor.embedded_set_labels):
        fig = plt.figure(figsize=(2.4,6))
        ax=fig.add_subplot(111)
        unknown_cluster_distances_for_i = [unknown_cluster_distances[j][i] for j in range(len(unknown_cluster_distances))]
        tmp = ax.hist([predictor.cluster_distances[i], known_cluster_distances[i], unknown_cluster_distances_for_i], bins=10, label=["embedded", "inf_known", "inf_unknown"])
        ax.legend()
        ax.axvline(x=predictor.thresholds[i], ymin=0, ymax=tmp[0][0].max(), color="black")
        ax.set_title(f"{cluster_distance_name} ({predictor.thresholds[i]:.02f})")
        fig.suptitle(f"label {label}  #{len(predictor.cluster_distances[i])+len(known_cluster_distances[i])+len(unknown_cluster_distances_for_i)}")
        fig.savefig(f"{output_dir}/hist_cluster_distances_label{label}.png", bbox_inches='tight')
        fig.clf()
        plt.close()
    #tSNEを出力
    plot_by_tsne(predictor.embedded_features, predictor.embedded_labels, known_features, known_labels2, unknown_features, unknown_labels2, output_dir, perplexity=30.0)


class BHTSNE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1):
        self.dimensions = dimensions
        self.perplexity = perplexity
        self.theta = theta
        self.rand_seed = rand_seed

    def fit_transform(self, X):
        return bhtsne.tsne(
            X.astype(sp.float64),
            dimensions=self.dimensions,
            perplexity=self.perplexity,
            theta=self.theta,
            rand_seed=self.rand_seed,
        )

def plot_by_tsne(embedded_ftr, embedded_label, known_ftr, known_label, unknown_ftr, unknown_label, output_dir, perplexity=30.0):
    #圧縮したい特徴を全て結合
    ftr_data = np.concatenate([embedded_ftr,known_ftr])
    ftr_data = np.concatenate([ftr_data,unknown_ftr])

    Bhtsne = BHTSNE(dimensions=2, perplexity=perplexity, theta=0.5, rand_seed=-1)

    st = time()
    #tSNEで次元削減
    low_dim_data = Bhtsne.fit_transform(ftr_data)
    logging.info(f"elapsed time(tsne) = {time() - st}")

    xmin = low_dim_data[:,0].min()
    xmax = low_dim_data[:,0].max()
    ymin = low_dim_data[:,1].min()
    ymax = low_dim_data[:,1].max()

    #次元削減した特徴を元どおりに分割（embedded, known, unknown）
    embedded_low_dim_data, valid_low_dim_data = np.split(low_dim_data,[len(embedded_ftr)])
    known_low_dim_data, unknown_low_dim_data = np.split(valid_low_dim_data,[len(known_ftr)])

    t_num = len(set(embedded_label))+len(set(unknown_label))
    cm = plt.cm.get_cmap("hsv")
    colors = [cm(i/t_num) for i in range(t_num)]
    plt.figure(figsize=(10,10))

    label_list = []
    #特徴抽出済みのみでplot
    for i, t in enumerate(set(embedded_label)):
        embedded_inds = np.where(np.array(embedded_label)==t)[0]
        plt.scatter(embedded_low_dim_data[embedded_inds,0], embedded_low_dim_data[embedded_inds,1], alpha=0.5, color=colors[i], marker=".")
        label_list.append(f"{t}(embedded)")
    plt.axis([xmin,xmax,ymin,ymax])
    plt.title("embedded='.'")
    # plt.legend(label_list)

    plt.savefig(f"{output_dir}/tsne(embedded).png")
    plt.clf()
    plt.close()


    #特徴抽出済みと既知人物でplot
    plt.figure(figsize=(10,10))
    label_list = list()
    for i, t in enumerate(set(embedded_label)):
        embedded_inds = np.where(np.array(embedded_label)==t)[0]
        known_inds = np.where(np.array(known_label)==t)[0]
        plt.scatter(embedded_low_dim_data[embedded_inds,0], embedded_low_dim_data[embedded_inds,1], alpha=0.1, color=colors[i], marker=".")
        plt.scatter(known_low_dim_data[known_inds,0], known_low_dim_data[known_inds,1], alpha=0.5, color=colors[i], marker="+")
        label_list.append(f"{t}(embedded)")
        label_list.append(f"{t}(known)")
    plt.axis([xmin,xmax,ymin,ymax])

    plt.title("embedded='.', known='+'")
    # plt.legend(label_list)

    plt.savefig(f"{output_dir}/tsne(embedded+known).png")
    plt.clf()
    plt.close()

    #特徴抽出済みと既知人物と未知人物でplot
    label_list = list()
    plt.figure(figsize=(10,10))
    for i, t in enumerate(set(embedded_label)):
        #特徴抽出済みと既知人物のplot
        embedded_inds = np.where(np.array(embedded_label)==t)[0]
        plt.scatter(embedded_low_dim_data[embedded_inds,0], embedded_low_dim_data[embedded_inds,1], alpha=0.1, color=colors[i], marker=".")
        label_list.append(f"{t}(embedded)")
        label_list.append(f"{t}(known)")
    for i, t in enumerate(set(unknown_label)):
        #未知人物のplot
        unknown_inds = np.where(np.array(unknown_label)==t)[0]
        plt.scatter(unknown_low_dim_data[unknown_inds,0], unknown_low_dim_data[unknown_inds,1], alpha=0.5, color=colors[len(set(embedded_label))+i], marker="*")
        label_list.append(f"{t}(unknown)")
    plt.axis([xmin,xmax,ymin,ymax])

    plt.title("embedded='.', unknown='*'")
    # plt.legend(label_list)

    plt.savefig(f"{output_dir}/tsne(embeddedunknwon).png")
    plt.clf()
    plt.close()




if __name__ == "__main__":
    cfg_known_valid_path = "../data/MOT17-04-FRCNN_f100_shuffleFalse_skipframe5_1/train.txt"
    cfg_known_inf_path   = "../data/MOT17-04-FRCNN_f100_shuffleFalse_skipframe5_2/train.txt"
    cfg_unknown_inf_path = "../data/MOT17-04-FRCNN_f100_shuffleFalse_skipframe5_3/train.txt"
    weight_paths = [
        "../result/checkpoints/absolute_triplet/MOT20-03_embeddingNet_out206_epoch29.pth",
        "../result/checkpoints/absolute_triplet_hard_negative/MOT20-03_embeddingNet_out206_epoch29.pth",
        ]
    cluster_distances = ["Single", "Complete", "Average", "Centroid", "Ward"]
    for weight_path in weight_paths:
        for cluster_distance in cluster_distances:
            main(cfg_known_valid_path, cfg_known_inf_path, cfg_unknown_inf_path, weight_path, cluster_distance, thread=0)