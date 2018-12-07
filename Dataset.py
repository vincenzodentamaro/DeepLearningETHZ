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

    def split_train_val(self, val_percentile, filename='info_file'):
        """
        Split dataset in a training set and validation set
        :param val_percentile: percentage of validation data
        :return: training dataset and validation dataset
        """

        filename_train = './' + filename + '_train.csv'
        filename_val = './' + filename + '_val.csv'
        idx_val = math.ceil((1-val_percentile)*len(self))
        info_file_train = self.info_file.loc[0:idx_val-1, :]
        info_file_val = self.info_file.loc[idx_val:, :]
        info_file_train.to_csv(filename_train, index=False)
        info_file_val.to_csv(filename_val, index=False)

        data_train = PaintingDataset(csv_file=filename_train,root_dir=self.root_dir, transform=self.transform)
        data_val = PaintingDataset(csv_file=filename_val, root_dir=self.root_dir,transform=self.transform)

        return data_train, data_val

    def split_train_val_test(self, val_percentile, test_percentile, filename='info_file'):
        """
        Split dataset in a training set and validation set
        :param val_percentile: percentage of validation data
        :return: training dataset and validation dataset
        """
        filename_train = './'+filename+'_train.csv'
        filename_val = './'+filename+'_val.csv'
        filename_test = './'+filename+'_test.csv'
        idx_val = math.ceil((1-val_percentile-test_percentile)*len(self))
        idx_test = math.ceil((1-test_percentile)*len(self))
        info_file_train = self.info_file.loc[0:idx_val-1, :]
        info_file_val = self.info_file.loc[idx_val:idx_test-1, :]
        info_file_test = self.info_file.loc[idx_test:, :]
        info_file_train.to_csv(filename_train, index=False)
        info_file_val.to_csv(filename_val, index=False)
        info_file_test.to_csv(filename_test, index=False)

        data_train = PaintingDataset(csv_file=filename_train, root_dir=self.root_dir, transform=self.transform)
        data_val = PaintingDataset(csv_file=filename_val, root_dir=self.root_dir,transform=self.transform)
        data_test = PaintingDataset(csv_file=filename_test, root_dir=self.root_dir,transform=self.transform)

        return data_train, data_val, data_test

    def find_classes(self):
        df = self.info_file
        classes = list(df['style'].unique())
        classes.sort()
        class_to_idx = {val: idx for (idx, val) in enumerate(classes)}
        idx_to_class = {idx: val for (idx, val) in enumerate(classes)}
        return classes, class_to_idx, idx_to_class


class AugmentedPaintingDataset(Dataset):
    def __init__(self, main_dataset, extra_dataset):

        if not isinstance(main_dataset, PaintingDataset):
            raise TypeError('object of type PaintingDataset expected for main_dataset')
        if not isinstance(extra_dataset, PaintingDataset):
            raise TypeError('Object of type PaintingDataset expected for extra_dataset')

        self.main_dataset = main_dataset
        self.extra_dataset = extra_dataset
        self.len_main = len(self.main_dataset)
        self.len_extra = len(self.extra_dataset)
        self.assert_identical_classes()
        self.assert_identical_transform()

    def __len__(self):
        return self.len_main + self.len_extra

    def __getitem__(self, idx):
        if idx < self.len_main:
            return self.main_dataset[idx]
        else:
            idx = idx - self.len_main # the index of the extra dataset starts at zero again
            return self.extra_dataset[idx]

    def assert_identical_classes(self):
        if self.main_dataset.class_to_idx != self.extra_dataset.class_to_idx:
            raise ValueError('The classes of the main dataset and extra dataset are not identical. Main: {},'
                             'Extra: {}'.format(self.main_dataset.class_to_idx, self.extra_dataset.class_to_idx))
        pass

    def assert_identical_transform(self):
        if self.main_dataset.transform != self.extra_dataset.transform:
            raise ValueError('The transform function of the main dataset and extra dataset are not identical')
        pass

    def split_train_val_test(self, val_percentile, test_percentile):
        """
        Split dataset in a training set and validation set
        :param val_percentile: percentage of validation data
        :return: training dataset and validation dataset
        """
        data_train_main, data_val_main, data_test_main = self.main_dataset.split_train_val_test(
            val_percentile, test_percentile,'info_file_main')
        data_train_extra, data_val_extra, data_test_extra = self.extra_dataset.split_train_val_test(
            val_percentile, test_percentile, 'info_file_extra')


        data_train = AugmentedPaintingDataset(data_train_main, data_train_extra)
        data_val = AugmentedPaintingDataset(data_val_main, data_val_extra)
        data_test = AugmentedPaintingDataset(data_test_main, data_test_extra)

        return data_train, data_val, data_test

    def split_train_val(self, val_percentile):

        data_train_main, data_val_main = self.main_dataset.split_train_val(val_percentile, 'info_file_main')
        data_train_extra, data_val_extra = self.extra_dataset.split_train_val(val_percentile, 'info_file_extra')

        data_train = AugmentedPaintingDataset(data_train_main, data_train_extra)
        data_val = AugmentedPaintingDataset(data_val_main, data_val_extra)

        return data_train, data_val
