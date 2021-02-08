import numpy as np
from torch.utils.data import Dataset

class QuadrupletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative1 and negative2 samples
    Test: Creates fixed quadruplets for testing
    """

    def __init__(self, dataset, train=True):
        if dataset is None:
            return None

        self.dataset = dataset
        self.labels = self.dataset.labels
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                     for label in self.labels_set}
        self.train = train
        

        if not self.train:

            # generate fixed quadruplets for testing

            random_state = np.random.RandomState(29)
            quadruplets = []
            for i in range(len(self.dataset)):
                pos = random_state.choice(self.label_to_indices[self.labels[i]])
                neg1_label = np.random.choice(list(self.labels_set - set([self.labels[i]])))
                neg1 = random_state.choice(self.label_to_indices[neg1_label])
                neg2_label = np.random.choice(list(self.labels_set - set([self.labels[i], neg1_label])))
                neg2 = random_state.choice(self.label_to_indices[neg2_label])
                
                quadruplets.append([i, pos, neg1, neg2])
            self.test_quadruplets = quadruplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.dataset[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative1_label = np.random.choice(list(self.labels_set - set([label1])))
            negative1_index = np.random.choice(self.label_to_indices[negative1_label])
            negative2_label = np.random.choice(list(self.labels_set - set([label1, negative1_label])))
            negative2_index = np.random.choice(self.label_to_indices[negative2_label])
            img2, label2 = self.dataset[positive_index]
            img3, label3 = self.dataset[negative1_index]
            img4, label4 = self.dataset[negative2_index]
        else:
            img1, label1 = self.dataset[self.test_quadruplets[index][0]]
            img2, label2 = self.dataset[self.test_quadruplets[index][1]]
            img3, label3 = self.dataset[self.test_quadruplets[index][2]]
            img4, label4 = self.dataset[self.test_quadruplets[index][3]]

        
        return (img1, img2, img3, img4), [label1, label2, label3, label4]

    def __len__(self):
        return len(self.dataset)