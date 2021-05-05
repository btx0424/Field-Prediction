from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
import numpy as np
import torch

class Structured(Dataset):
    @staticmethod
    def parse_name(name: str):
        aoa, mach = name[:-4].split('_')[-2:]
        return float(aoa), float(mach)

    def __init__(self, file_list, **kwargs):
        
        self.x_dim = kwargs.get('x_dim', [0, 1]) # x, y
        self.y_dim = kwargs.get('y_dim', [3, 4, 7]) # u, v, p

        self.file_list = [(file_name, *Structured.parse_name(file_name)) for file_name in file_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name, aoa, mach = self.file_list[idx]
        Z = np.load(file_name)
        x = torch.tensor(Z[..., 0, self.x_dim], dtype=torch.float).permute(2, 0, 1)
        y = torch.tensor(Z[..., 0, self.y_dim], dtype=torch.float).permute(2, 0, 1)
        c = torch.tensor([mach, aoa])
        return [x, c], y 

class StructuredModule(LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        import os
        file_list = []
        for file_name in os.listdir(self.data_dir):
            file_list.append(os.path.join(self.data_dir, file_name))
        
        dataset = Structured(file_list,) 
        train_len = int(len(file_list)*0.75) 
        self.train_set, self.test_set = random_split(dataset=dataset, lengths=[train_len, len(file_list)-train_len])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)