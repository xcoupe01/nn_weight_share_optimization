import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import pandas as pd

DEF_FOLD_LIST = [
    '../../results/lenet_relu_compress_50'
]

DEF_FILE_LIST = [
    {'name': 'lenet_GA_save.csv', 'repres': 'chromosome'},
    {'name': 'lenet_PSO_save.csv', 'repres': 'representation'},
    {'name': 'lenet_BH_save.csv', 'repres': 'representation'},
    {'name': 'lenet_RND_save.csv', 'repres': 'representation'},
]

class PredictorData(Dataset):
    def __init__(self, folder_list:list[str], file_list:list[dict]):
        self.folder_list = folder_list
        self.file_list = file_list
        self.df = pd.DataFrame(columns=['representation', 'accuracy'])

        for folder in folder_list:
            for subfolder in [subf for subf in os.listdir(folder) if os.path.isdir(os.path.join(folder, subf))]:
                for file in os.listdir(os.path.join(folder, subfolder)):
                    if file not in [f['name'] for f in file_list]:
                        continue
                    repres = list(filter(lambda i: i['name'] == file, file_list))[0]['repres']
                    tmp_df = pd.read_csv(os.path.join(folder, subfolder, file))
                    tmp_df['representation'] = tmp_df[repres]
                    self.df = self.df.append(tmp_df[['representation', 'accuracy_loss']])
        
        self.df.reset_index()

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx:int):
        df_row = self.df.iloc[idx]
        return (torch.Tensor(eval(df_row['representation'])), torch.Tensor([df_row['accuracy_loss']]))

class PredictorDataset:

    def __init__(self, batch_size:int, folder_list:list[str], file_list:list[dict], train_split:float = 0.5, val_split:float = 0.5):

        self.batch_size = batch_size
        self.folder_list = folder_list
        self.file_list = file_list

        dataset = PredictorData(folder_list, file_list)

        # splitting
        train_split_len = int(len(dataset) * train_split)
        valtest_dataset, self.train_dataset = random_split(
            dataset,
            [train_split_len, len(dataset) - train_split_len],
            generator=torch.Generator().manual_seed(42)
        )

        val_split_len = int(len(valtest_dataset) * val_split)
        self.valid_dataset, self.test_dataset = random_split(
            valtest_dataset,
            [val_split_len, len(valtest_dataset) - val_split_len],
            generator=torch.Generator().manual_seed(42)
        )

        # define dataloaders
        self.train_dl = DataLoader(
            dataset = self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True
        )

        self.valid_dl = DataLoader(
            dataset = self.valid_dataset,
            batch_size = self.batch_size,
            shuffle = True
        )

        self.test_dl = DataLoader(
            dataset = self.test_dataset,
            batch_size = self.batch_size,
            shuffle = True
        )
