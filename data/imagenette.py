import sys
sys.path.append('../')

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import tarfile
import json
import os
import argparse

from data.utils.download import *

DOWNLOAD_URLS_DATASET = {
    'imagenette2': 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz',
    'imagewoof': 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz',
    'imagewang': 'https://s3.amazonaws.com/fast-ai-imageclas/imagewang.tgz', 
}
DOWLOAD_URL_CLASSES = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

class Imagenette(Dataset):
    """Implementation of the Imagenette dataset class.
    """

    def __init__(self, root_dir:str, classes:list, transform:transforms=None):
        """Inits the dataset and creates hooks to the needed files

        Args:
            root_dir (str): Is the folder, where the data are stored.
            classes (list): Is the lisf of tuples, where the first element is the 
                class id and the second is the class string. 
            transform (transforms, optional): Data transforms. Defaults to None.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.length = 0
        self.files:list[str] = []
        self.classes = classes

        for dir in os.listdir(root_dir):
            curr_dir_files = [[dir, x] for x in os.listdir(os.path.join(root_dir, dir))]
            self.files += curr_dir_files
            self.length += len(curr_dir_files)

    def __len__(self) -> int:
        """Tells the length of the dataset.

        Returns:
            int: The dataset length
        """

        return self.length

    def __getitem__(self, idx:int) -> tuple:
        """Returns the needed item of the dataset.

        Args:
            idx (int): Is the index of wanted dataset item.

        Returns:
            tuple: Tuple, where the first element is the data and the second
                is the data label.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.files[idx][0] ,self.files[idx][1])
        image = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        #print(image.shape, os.path.join(self.root_dir, self.files[idx][0] ,self.files[idx][1]))

        return (image, [i[0] for i in self.classes].index(self.files[idx][0]))

class ImagenetteDataset:
    """Class to encapsulate the Imagenette dataset for easier usage.
    """

    def __init__(self, batch_size:int, dataset_path:str, val_split:float=0.5, dataset_type:str = 'imagenette2'):
        """Inits the Dataset object, ensures the data (if they are avaliable, 
        they are loade, if not, they are downloaded), splits the into separate
        train, validation and test dataset.


        Args:
            batch_size (int): Is the datasets batch size.
            dataset_path (str): Is the folder, where the raw data are, or where they will be
                downloaded to.
            val_split (float, optional): Is the validation / Test dataset split (the data are splitted
                from the original Imagenette valid subdataset). Defaults to 0.5.
            dataset_type (str, optional): Specifies the dataset type, there are three types:
                `imagenette2`, `imagewoof` and `imagewang`. For more detail 
                look up https://github.com/fastai/imagenette. Defaults to 'imagenette2'.
        """
        
        self.batch_size = batch_size
        self.dataset_type = dataset_type

        # image transformation
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # download and process when not avalible    
        if not os.path.isdir(dataset_path):
            os.mkdir(dataset_path)
            filename = download_file(DOWNLOAD_URLS_DATASET[dataset_type], dataset_path)
            file = tarfile.open(filename)
            file.extractall(dataset_path)
            file.close()
            os.remove(filename)
        
        # ensure imagenet classes file
        if not os.path.isfile(os.path.join(dataset_path, 'imagenet_class_index.json')):
            download_file(DOWLOAD_URL_CLASSES, dataset_path)

        # load imagenet classes
        class_raw = json.load(open(os.path.join(dataset_path, 'imagenet_class_index.json')))
        self.classes = []
        for i in range(len(class_raw.keys())):
            self.classes.append(class_raw[f'{i}'])
        

        # create datasets
        self.train_dataset = Imagenette(
            os.path.join(dataset_path, dataset_type, 'train'),
            self.classes,
            self.transforms
        )

        valtest_dataset = Imagenette(
            os.path.join(dataset_path, dataset_type, 'val'),
            self.classes,
            self.transforms
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
            shuffle=True
        )

        self.test_dl = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def __str__(self):
        return f'{self.dataset_type}  dataset with: \n {len(self.train_dataset)} train datapoints, \n {len(self.valid_dataset)} valid datapoints \n {len(self.test_dataset)} test datapoints'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='imagenette.py', description='Imagenette and imagenet dataloader for WS-project. Dowloads the dataset')
    parser.add_argument('-p', '--path', help='Path where the dataset will be stored.')
    parser.add_argument('-t', '--type', choices=list(DOWNLOAD_URLS_DATASET.keys()), default=list(DOWNLOAD_URLS_DATASET.keys())[0], 
        help='The Imagenette dataset type')

    args = parser.parse_args()

    print(ImagenetteDataset(32, args.path, dataset_type=args.type))