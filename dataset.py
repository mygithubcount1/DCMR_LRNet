import os
import random

import pandas as pd
import numpy as np
from PIL import Image, ImageFile
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FGDataset(Dataset):

    def __init__(self, rawdata_root, anno, transform=None,is_train=True,target_transform=None):

        self.img_root = rawdata_root
        self.anno = anno
        self.imgs = pd.read_csv(anno, \
                           sep=" ", \
                           header=None, \
                           names=['Imagepath', 'label'])
        self.transform = transform
        self.is_train = is_train
        self.target_transform = target_transform

    def __len__(self):
        num_lines = sum(1 for line in open(self.anno))
        return num_lines

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.imgs.values[idx][0]
        original_path = path
        vein_path = f"vein2_{original_path}"


        original_path = self.img_root + '/' + original_path

        vein_path = self.img_root + '/' + vein_path


        target = int(self.imgs.values[idx][1]) -1

        with open(original_path, 'rb') as f:
            original_img = Image.open(f)
            original_img = original_img.convert('RGB')


        with open(vein_path, 'rb') as f:
            vein_img = Image.open(f)
            vein_img = vein_img.convert('L')

        if self.is_train:
            if self.transform:
                seed = random.getrandbits(32)
                torch.manual_seed(seed)
                np.random.seed(seed)
                original_img = self.transform['train'](original_img)
                torch.manual_seed(seed)
                np.random.seed(seed)
                vein_img = self.transform['train'](vein_img)
            original_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            original_img = original_normalize(original_img)
            return original_img, vein_img, target
        else:
            if self.transform:
                original_center_img = self.transform['val_center'](original_img)
                vein_center_img = self.transform['val_center'](vein_img)

                original_top_img = self.transform['val_top'](original_img)
                vein_top_img = self.transform['val_top'](vein_img)
                
                original_bottom_img = self.transform['val_bottom'](original_img)
                vein_bottom_img = self.transform['val_bottom'](vein_img)
            original_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            original_center_img = original_normalize(original_center_img)
            original_top_img = original_normalize(original_top_img)
            original_bottom_img = original_normalize(original_bottom_img)

            return original_center_img, vein_center_img, original_top_img, vein_top_img, original_bottom_img, vein_bottom_img, target

