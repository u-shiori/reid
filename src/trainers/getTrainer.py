import torch
import logging

from trainers._metrics import euclidean_metric, cosin_metric
from trainers._MultipletTrainer import MultipletTrainer
from trainers._ArcfaceTrainer import ArcfaceTrainer
from trainers._UIRTrainer import UIRTrainer
from trainers._ClusterTripletTrainer import ClusterTripletTrainer
from distances.getDistance import getClusterDistance
from _utils import getLogger
logger = getLogger(__name__)

def getTrainer(model_name, sup_train_loader, semisup_train_loader, sup_valid_loader, semisup_valid_loader, sup_test_loader, semisup_test_loader, \
                model, margin_penalty, sup_train_loss_fn, semisup_train_loss_fn, test_loss_fn, cluster_distance_name=None):
    """Trainerを返す
        
    Parameters
    ----------
    model_name : str
        モデル名
    sup_train_loader : DataLoader
        教師あり学習用のtrain_loader
    semisup_train_loader : DataLoader 
        半教師あり学習用のtrain_loader (UIRで必須)
    sup_valid_loader : DataLoader
        教師あり学習用のtrain_loader
    semisup_valid_loader : DataLoader 
        半教師あり学習用のvalid_loader (UIRで必須)
    sup_test_loader : DataLoader
        教師あり学習用のtrain_loader
    semisup_test_loader : DataLoader 
        半教師あり学習用のtest_loader (UIRで必須)
    model : Model
        モデル
    margin_penalty : MarginPenalty
        Additive Angular Margin Penalty（ArcfaceとUIRで必須）
    sup_train_loss_fn : Loss
        教師あり学習用の損失関数
    semisup_train_loss_fn : Loss
        半教師あり学習用の損失関数（UIRで必須）
    test_loss_fn : Loss
        テスト用の損失関数（ArcfaceとUIRで必須）
    cluster_distance_name : str
        クラスタ間距離法の名前（ClusterTripletで必須）

    


    Returns
    -------
    trainer : Trainer

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #trainerの取得
    if model_name == "UIR":
        trainer = UIRTrainer(sup_train_loader, semisup_train_loader, sup_valid_loader, semisup_valid_loader, sup_test_loader, semisup_test_loader, \
                model, margin_penalty, sup_train_loss_fn, semisup_train_loss_fn, test_loss_fn, cosin_metric, device)
    elif model_name == "Arcface":
        trainer = ArcfaceTrainer(sup_train_loader, sup_valid_loader, sup_test_loader, model, margin_penalty, sup_train_loss_fn, test_loss_fn, cosin_metric, device)
    elif model_name == "Triplet" or model_name == "ImprovedTriplet" or model_name == "Quadruplet":
        trainer = MultipletTrainer(sup_train_loader, sup_valid_loader, sup_test_loader, model, sup_train_loss_fn, euclidean_metric, device)
    elif model_name=="ClusterTriplet":
        assert cluster_distance_name is not None
        cluster_distance = getClusterDistance("Triplet", cluster_distance_name)#TODO:
        trainer = ClusterTripletTrainer(sup_train_loader, sup_valid_loader, sup_test_loader, model, cluster_distance, sup_train_loss_fn, euclidean_metric, device)
    else:
        logging.error(f"trainer has not been set.")
        exit(1)
    return trainer


