import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

from dataloaders import MyDataset
from data_generator import create_data
from plots import plot_results, print_examples_torch


class AutoEncoder(nn.Module):
    def __init__(self, h=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 2),
            # nn.Tanh()
        )

    def forward(self, x):
        y = self.encoder(x)
        z = self.decoder(y)
        return z

def single_pass(model, dataloader, loss_func, device, optim=None):
    loss_count = 0
    for i, data in enumerate(dataloader):
        data = data.float().to(device)
        pred = model.forward(data)
        loss = loss_func(data, pred)
        loss_count += loss.item()

        if optim is not None:
            loss.backward()
            optim.step()
            optim.zero_grad()
    return loss_count / len(dataloader)


def train_model(model, loss, optim, epochs, device, datas, batch_size, print_step=10):
    data_train, data_val = datas
    dataloader_train = DataLoader(MyDataset(data_train), batch_size, shuffle=True)
    dataloader_val = DataLoader(MyDataset(data_val), batch_size, shuffle=True)
    train_loss_all, val_loss_all = [], []
    # training loop
    for epoch in range(epochs):
        # train
        train_loss = single_pass(model, dataloader_train, loss, device, optim)

        if epoch % print_step == 0:
            # validation
            with torch.no_grad():
                val_loss = single_pass(model, dataloader_val, loss, device)
            print(
                f'epoch {epoch}, train_loss={train_loss}, val_loss={val_loss}')

            train_loss_all.append(train_loss)
            val_loss_all.append(val_loss)

    return train_loss_all, val_loss_all

def main(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = AutoEncoder()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    batch_size = 8
    epochs = 100
    lr = 5 * 1e-3
    optim = Adam(model.parameters(), lr=lr)
    loss = F.mse_loss

    # generate data and create data
    data_train = create_data(700)
    data_val = create_data(100)
    datas = [data_train, data_val]

    # train model
    train_loss, val_loss = train_model(model, loss, optim, epochs, device, datas, batch_size)

    # print loss graphs
    _, ax = plt.subplots()
    plot_results(ax, train_loss, val_loss, label='Loss')

    # look at examples
    print_examples_torch(model, device)

if __name__ == '__main__':
    main()
