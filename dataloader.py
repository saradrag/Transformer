import numpy as np
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 64
N_WORDS_DECODER = 1000


class ArrayDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.data = np.loadtxt(data_path, dtype='i')
        self.labels = np.loadtxt(labels_path, dtype='i')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        labels = self.labels[item]
        shifted_labels = np.insert(labels, 0, N_WORDS_DECODER)[:-1]
        return self.data[item], shifted_labels, labels


training_set = ArrayDataset('data/training_arrays.txt', 'data/training_labels.txt')
validation_set = ArrayDataset('data/validation_arrays.txt', 'data/validation_labels.txt')
test_set = ArrayDataset('data/test_arrays.txt', 'data/test_labels.txt')

training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)