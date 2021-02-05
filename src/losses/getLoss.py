import torch
import logging

from losses._ArcfaceLoss import ArcMarginPenalty, FocalLoss, UIRLoss
from losses._MultipletLoss import TripletLoss, QuadrupletLoss
from losses._ClusterTripletLoss import ClusterTripletLoss

from _utils import getLogger
logger = getLogger(__name__)


def getLoss(model_name, num_out, class_num, margin_a, margin_b=None, w=None):
    """Lossを返す
        
    Parameters
    ----------
    model_name : str
        モデル名
    num_out : int
        出力層のunit数
    class_num : int
        入力データのクラス数
    margin_a : float
        Tripletのマージンa
    margin_b : float
        ImprovedTripletとQuadrupletのマージンb（それ以外のときはNone）
    w : float
        UIRのlossの重みw（それ以外のときはNone)

    Returns
    -------
    margin_penalty .. Arcfaceの Angular Margin Penalty部分（ArcfaceとUIR以外はNone）
    sup_train_loss_fn .. 教師あり学習用のloss
    semisup_train_loss_fn .. 半教師あり学習用のloss(UIR以外はNone)
    test_loss_fn .. (ArcfaceとUIR以外はNone)
    
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "Arcface":
        margin_penalty = ArcMarginPenalty(num_out, class_num, s=30, m=0.5, easy_margin=False, device=device)
        sup_train_loss_fn = FocalLoss(gamma=2)
        semisup_train_loss_fn = None
        test_loss_fn = TripletLoss(margin_a, margin_b=margin_b)
    elif model_name == "UIR":
        margin_penalty = ArcMarginPenalty(num_out, class_num, s=30, m=0.5, easy_margin=False, device=device)
        sup_train_loss_fn = FocalLoss(gamma=2)
        semisup_train_loss_fn = UIRLoss(class_num, w)
        test_loss_fn = TripletLoss(margin_a, margin_b=margin_b)
    
    elif model_name=="Triplet" or model_name=="ImprovedTriplet":
        margin_penalty = None
        sup_train_loss_fn = TripletLoss(margin_a, margin_b=margin_b)
        semisup_train_loss_fn = None
        test_loss_fn = None
    elif model_name == "Quadruplet":
        margin_penalty = None
        sup_train_loss_fn = QuadrupletLoss(margin_a, margin_b)
        semisup_train_loss_fn = None
        test_loss_fn = None
    elif model_name=="ClusterTriplet":
        margin_penalty = None
        sup_train_loss_fn = ClusterTripletLoss(margin_a, margin_b)
        semisup_train_loss_fn = None
        test_loss_fn = None
    else:
        logging.error(f"loss_fn has not been set.")
        exit(1)



    return margin_penalty, sup_train_loss_fn, semisup_train_loss_fn, test_loss_fn
