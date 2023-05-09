#!/usr/bin/env python

"""
Author: Vojtěch Čoupek
Description: MNIST dataset implemetnation
Project: Weight-Sharing of CNN - Diploma thesis FIT BUT 2023
"""

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch


class MnistDataset:
    """Class to encapsulate the MNIST dataset for easier usage.
    """

    def __init__(self, batch_size:int, dataset_path:str, val_split:float=0.5):
        """Inits the MnistDataset object, ensures the data (loads them if they are
        avaliable on the system or they are downloaded) and splits the into separate
        train, validation and test dataset.

        Args:
            batch_size (int): Is the datasets batch size.
            dataset_path (str): Is the folder, where the raw data are, or where they will be
                downloaded to.
            val_split (float, optional): Is the validation / Test dataset split (the data are splitted
                from the original MNIST valid subdataset). Defaults to 0.5.
        """
        
        self.batch_size = batch_size

        self.transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

        # download and create datasets
        self.train_dataset = datasets.MNIST(
            root=dataset_path, 
            train=True, 
            transform=self.transforms,
            download=True
        )

        valtest_dataset = datasets.MNIST(
            root=dataset_path, 
            train=False, 
            transform=self.transforms,
            download=True
        )

        # split validation to valid and test
        val_split_len = int(len(valtest_dataset) * val_split)
        self.valid_dataset, self.test_dataset = random_split(
            valtest_dataset, 
            [val_split_len, len(valtest_dataset) - val_split_len], 
            generator=torch.Generator().manual_seed(42)
        )

        # define the data loaders
        self.train_dl = DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )

        self.valid_dl = DataLoader(
            dataset=self.valid_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )

        self.test_dl = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )