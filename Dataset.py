from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import math


class PaintingDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.info_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx, self.idx_to_class = self.find_classes()

    def __len__(self):
        return len(self.info_file)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.info_file.at[idx, 'new_filename'])
        image = io.imread(img_name)
        style = self.info_file.at[idx, 'style']
        class_idx = self.class_to_idx[style]

        if self.transform:
            image = self.transform(image)

        return image, class_idx

    def split_train_val(self, val_percentile):
        """
        Split dataset in a training set and validation set
        :param val_percentile: percentage of validation data
        :return: training dataset and validation dataset
        """
        idx_val = math.ceil((1-val_percentile)*len(self))
        info_file_train = self.info_file.loc[0:idx_val-1, :]
        info_file_val = self.info_file.loc[idx_val:, :]
        info_file_train.to_csv('./info_file_train.csv', index=False)
        info_file_val.to_csv('./info_file_val.csv', index=False)

        data_train = PaintingDataset(csv_file='./info_file_train.csv',root_dir=self.root_dir, transform=self.transform)
        data_val = PaintingDataset(csv_file='./info_file_val.csv', root_dir=self.root_dir,transform=self.transform)

        return data_train, data_val

    def find_classes(self):
        df = self.info_file
        classes = list(df['style'].str.upper().unique())
        classes.sort()
        class_to_idx = {val: idx for (idx, val) in enumerate(classes)}
        idx_to_class = {idx: val for (idx, val) in enumerate(classes)}
        return classes, class_to_idx, idx_to_class

dataset = PaintingDataset(root_dir='train_reduced',csv_file='final_train_info.csv', transform=None)

test_idx = dataset[5]
dataset_train, dataset_val = dataset.split_train_val(0.3)


