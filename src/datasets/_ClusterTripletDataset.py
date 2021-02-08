import numpy as np
from torch.utils.data import Dataset
import torch

class ClusterTripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, dataset, train=True):
        if dataset is None:
            return None

        self.dataset = dataset
        self.labels = self.dataset.labels
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                     for label in self.labels_set}
        self.label_to_outindices = {label: np.where(np.array(self.labels) != label)[0]
                                     for label in self.labels_set}
        self.train = train
        

        if not self.train:

            # generate fixed triplets for testing

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.labels[i]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.labels[i]]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.dataset))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.dataset[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2, label2 = self.dataset[positive_index]
            img3, label3 = self.dataset[negative_index]
            
            num=0
            anchor_imgs = list()
            while num!=3:#TODO
                index = np.random.choice(list(self.label_to_indices[label1]))
                imgs= self.dataset[index][0]#.unsqueeze(0)
                anchor_imgs.append(imgs)
                num+=1
            # anchor_imgs = torch.cat(anchor_imgs)

            negative_imgs = list()
            class_num = 0
            while class_num!=10:
                negative_label = np.random.choice(list(self.labels_set - set([label1])))
                imgs = list()
                num=0
                while num!=3:
                    index = np.random.choice(list(self.label_to_indices[negative_label]))
                    img = self.dataset[index][0]#.unsqueeze(0)
                    imgs.append(img)
                    num+=1
                # imgs = torch.cat(imgs)
                negative_imgs.append(imgs)
                class_num+=1

            return (img1, img2, img3), [label1, label2, label3], (anchor_imgs, negative_imgs)
        else:
            img1, label1 = self.dataset[self.test_triplets[index][0]]
            img2, label2 = self.dataset[self.test_triplets[index][1]]
            img3, label3 = self.dataset[self.test_triplets[index][2]]
            return (img1, img2, img3), [label1, label2, label3]

    def __len__(self):
        return len(self.dataset)