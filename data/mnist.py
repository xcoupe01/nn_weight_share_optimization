from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch


class MnistDataset:

    def __init__(self, batch_size:int, dataset_path:str, val_split:float=0.5):
        
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