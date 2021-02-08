import torch
from torchvision import transforms
import random
import numpy as np
import logging

from datasets._BaseDataset import BaseDataset
from datasets._TripletDataset import TripletDataset
from datasets._QuadrupletDataset import QuadrupletDataset
from datasets._ClusterTripletDataset import ClusterTripletDataset

from _utils import getLogger
logger = getLogger(__name__)


cuda = torch.cuda.is_available()

def getDataLoader(model_name, cfg_path, batch_size, train, thread=0, embedding_net=False):
    """DataLoaderを返す
        
    Parameters
    ----------
    model_name : str
        モデル名
    cfg_path : str
    batch_size : int
    train : bool
    thread : int
    embedding_net : bool
        どのモデルでもmultiplet型ではないdata_loaderを出力

    Returns
    -------
    sup_dataloader : dataloader
        教師あり学習用dataloader
    semisup_dataloader : dataloader
        半教師あり学習用dataloader(UIR以外はNone)
    class_num : int
        教師あり学習でのclass数

    """

    # データセット
    transform = transforms.ToTensor()

    kwargs = {'num_workers': thread, 'pin_memory': True} if cuda else {}
    
    if embedding_net:
        all_dataset = BaseDataset(model_name, cfg_path, transform=transform, imgpath=True)
        sup_dataloader = torch.utils.data.DataLoader(all_dataset, batch_size=batch_size, shuffle=train, **kwargs)
        semisup_dataloader = None
    else:
        all_dataset = BaseDataset(model_name, cfg_path, transform=transform,)

        # TripletかQuadrupletで学習データを出す様にする 
        if model_name == "UIR":
            labeled_dataset = BaseDataset(model_name, cfg_path, transform=transform,)
            unlabeled_dataset = BaseDataset(model_name, cfg_path, transform=transform,)
            random.seed(29)
            all_labels = all_dataset.labels
            all_images_path = all_dataset.images_path
            labels_set = set(all_labels)#全ラベル
            labeled_set = set(random.sample(list(labels_set), int(len(labels_set)*3/4)))#既知人物
            labeled_indexs = list()
            unlabeled_indexs = list()
            for i, label in enumerate(all_labels):
                if label in labeled_set:
                    labeled_indexs.append(i)
                else:#label in unlabeled_set
                    unlabeled_indexs.append(i)
            
            labeled_dataset.n_data = len(labeled_indexs)
            unlabeled_dataset.n_data = len(unlabeled_indexs)
            labeled_dataset.images_path = np.delete(all_images_path, unlabeled_indexs)
            unlabeled_dataset.images_path = np.delete(all_images_path, labeled_indexs)
            labeled_dataset.labels = np.delete(all_labels, unlabeled_indexs)
            unlabeled_dataset.labels = np.delete(all_labels, labeled_indexs)
            all_dataset.labels[unlabeled_indexs] = unlabeled_dataset.labels[0]
        else:
            unlabeled_dataset = None

            

        # Multipletで学習データを出す様にする
        if model_name == "Quadruplet":
            multiplet_dataset = QuadrupletDataset(all_dataset, train=train)
        elif model_name == "ClusterTriplet":
            multiplet_dataset = ClusterTripletDataset(all_dataset, train=train)
        else:
            multiplet_dataset = TripletDataset(all_dataset, train=train)
            # unlabeled_multiplet_dataset = TripletDataset(unlabeled_dataset, train=train)


        # ArcFaceとUIRの学習はTripletDataではなく単体を用いるが，test時とvalid時はtripletの形に揃える．
        if model_name == "Arcface":
            if train:
                sup_dataloader = torch.utils.data.DataLoader(all_dataset, batch_size=batch_size, shuffle=train, **kwargs)
                semisup_dataloader = None
            else:
                sup_dataloader = torch.utils.data.DataLoader(multiplet_dataset, batch_size=batch_size, shuffle=train, **kwargs)
                semisup_dataloader = None

        elif model_name == "UIR":
            sup_size = int(batch_size*3/4)
            semisup_size = batch_size - int(batch_size*3/4)
            if train:
                sup_dataloader = torch.utils.data.DataLoader(labeled_dataset, batch_size=sup_size, shuffle=train, **kwargs)
                semisup_dataloader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=semisup_size, shuffle=train, **kwargs)
            else:
                sup_dataloader = semisup_dataloader = torch.utils.data.DataLoader(multiplet_dataset, batch_size=sup_size, shuffle=train, **kwargs)
        elif model_name == "Triplet" or model_name == "ImprovedTriplet" or model_name == "Quadruplet" or model_name=="ClusterTriplet":
            sup_dataloader = torch.utils.data.DataLoader(multiplet_dataset, batch_size=batch_size, shuffle=train, **kwargs)
            semisup_dataloader = None
        else:
            logging.error(f"dataloader has not been set.")
            exit(1)

        
    class_num = len(set(all_dataset.labels))
    return sup_dataloader, semisup_dataloader, class_num