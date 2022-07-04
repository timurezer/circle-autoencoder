from torch.utils.data import Dataset
import numpy as np


class MyDataset(Dataset):
    '''
    custom dataset class for PyTorch dataloader
    '''
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class DataLoader:
    '''
    custom dataloader class for numpy autoencoder
    '''
    def __init__(self, X, batch_size):
        self.X = X.copy()
        self.N = X.shape[0]
        self.batch_size = batch_size
        self.idx = np.arange(X.shape[0])

    def __len__(self):
        return np.ceil(self.N / self.batch_size).astype(int)

    def shuffle(self):
        self.idx = np.random.permutation(self.N)
        self.X = self.X[self.idx]

    def __iter__(self):
        start, stop = 0, 0
        self.shuffle()
        while start < self.X.shape[0]:
            stop = start + self.batch_size
            yield self.X[start:stop]
            start = stop
