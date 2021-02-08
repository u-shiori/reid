import numpy as np
import csv
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import os

class BaseDataset(Dataset):
    def __init__(self, model_name, cfg_file_path, transform=None, img_size=256, imgpath=False):
        assert model_name == "Triplet" or "Arcface" or "ImprovedTriplet" or "Quadruplet" or "UIR",\
            f"""model {model_name} is not supported. Choose model in [Triplet, Arcface, ImprovedTriplet, Quadruplet, UIR]."""

        with open(cfg_file_path, "r") as f:
            reader = csv.reader(f)
            # if "MOT" in cfg_file_path:
            #     data = np.array([[os.path.normpath(os.path.join(cfg_file_path,"../../../",row[0])), row[1]]\
            #             for row in reader])
            # elif "WLO" in cfg_file_path:
            data = np.array([[os.path.normpath(row[0]), row[1]]\
                    for row in reader])
        self.images_path = data[:, 0]
        self.labels = np.array(list(map(int, data[:, 1])))
        # if model_name=="Arcface" or model_name=="UIR":#XXX
        #     label_set = list(set(self.labels))
        #     label_to_idx = {label: idx for idx, label in enumerate(label_set)}
        #     self.labels = np.array([label_to_idx[label] for label in self.labels])

        self.img_size = img_size
        self.n_data = len(self.images_path)
        self.imgpath = imgpath

        self.transform = transform if transform else transforms.ToTensor()
    
    def pad_to_square(self, img, pad_value):
        c, h, w = img.shape
        dim_diff = np.abs(h - w)
        # (upper / left) padding and (lower / right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
        # Add padding
        img = F.pad(img, pad, "constant", value=pad_value)

        return img, pad

    def resize(self, image, size):
        image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
        return image

    def __getitem__(self, index):
        # tolist = lambda x: [x] if type(x) is int else x
        tolist = lambda x: x if type(x) is list else [int(x)]

        imgs = list()
        img_paths = list()
        labels = list()

        for i in tolist(index):
            img_path = self.images_path[i]
            label = self.labels[i]
            # Extract image as PyTorch tensor
            img = self.transform(Image.open(img_path).convert('RGB'))

            # Handle images with less than three channels
            if len(img.shape) != 3:
                img = img.unsqueeze(0)
                img = img.expand((3, img.shape[1:]))

            # Pad to square resolution
            img, pad = self.pad_to_square(img, 0)
            # Resize
            img = self.resize(img, self.img_size)
            imgs.append(img)
            img_paths.append(img_path)
            labels.append(label)


        fromlist = lambda x: x[0] if len(x)==1 else x
        if self.imgpath:
            return fromlist(img_paths), fromlist(imgs), fromlist(labels)
        else:
            return fromlist(imgs), fromlist(labels)


    def __len__(self):
        return self.n_data