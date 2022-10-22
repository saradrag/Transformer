import numpy as np
import pandas as pd
from os import walk
import torch
from torch.utils.data import Dataset, DataLoader
from parameters import BATCH_SIZE


class SalesDataset(Dataset):
    def __init__(self):
        super(SalesDataset, self).__init__()
        for _,_,data in walk('data/train'):
            self.data_csv = data

    def __len__(self):
        return len(self.data_csv)*(1684-28)

    def __getitem__(self, item):
        n, i = divmod(item, (1684-28))
        df = pd.read_csv('data/train/'+self.data_csv[n])
        encoder_input = torch.tensor(df.iloc[i:i+15].values.astype(np.float32))
        decoder_input = torch.tensor(df.iloc[i+14:i+29].values.astype(np.float32))
        decoder_output = torch.tensor(df.iloc[i+14:i+29, -33:].values.astype(np.float32))
        return encoder_input, decoder_input, decoder_output


training_set = SalesDataset()

training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)