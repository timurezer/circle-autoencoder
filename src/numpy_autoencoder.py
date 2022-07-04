import numpy as np
import random
import copy

from dataloaders import DataLoader
from data_generator import create_data
from examples import print_examples_np


class Module:
    # abstract class
    def forward(self, x):
        pass

    def backward(self, grad):
        pass

    def update(self, lr):
        pass


class Linear(Module):
    # class for linear layer as in PyTorch
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        xavier = 1 / np.sqrt(in_dim)
        self.W = np.random.uniform(-xavier, xavier, size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))
        self.dW = None
        self.db = None
        self.x = None

    def forward(self, x):
        # z = xW  + b, x is numpy row
        assert x.shape[-1] == self.in_dim
        self.x = x.copy()
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.x.T @ grad
        self.db = np.sum(grad, axis=0, keepdims=True)
        grad_ = grad @ self.W.T
        return grad_

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class ReLU(Module):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x.copy()
        return x * (x >= 0)

    def backward(self, grad):
        return grad * (self.x >= 0)


class MSE(Module):
    def __init__(self):
        self.x = None
        self.y = None
        self.N = None

    def forward(self, x, y):
        self.x = x.copy()
        self.y = y.copy()
        self.N = x.shape[0] * x.shape[1]
        return np.mean((x - y) ** 2)

    def backward(self):
        return 2 * (self.x - self.y) / self.N  # (batch, 2)


class AutoEncoder(Module):
    def __init__(self, h=6):
        self.encoder = [
            Linear(2, h),
            ReLU(),
            Linear(h, h),
            ReLU(),
            Linear(h, 1),
            ReLU()
        ]
        self.decoder = [
            Linear(1, h),
            ReLU(),
            Linear(h, h),
            ReLU(),
            Linear(h, 2),  # (h, 2)
        ]

    def forward(self, x):
        x_ = x.copy()
        for module in self.encoder:
            x_ = module.forward(x_)
        for module in self.decoder:
            x_ = module.forward(x_)
        return x_

    def backward(self, grad):
        grad_ = grad.copy()
        for module in self.decoder[::-1]:
            grad_ = module.backward(grad_)
        for module in self.encoder[::-1]:
            grad_ = module.backward(grad_)
        return grad_

    def update(self, lr):
        for module in self.encoder:
            module.update(lr)
        for module in self.decoder:
            module.update(lr)


def single_pass(model, dataloader, loss_func, lr, optim=True):
    loss_count = 0
    for i, data in enumerate(dataloader):
        data = data
        pred = model.forward(data)
        loss = loss_func.forward(pred, data)
        loss_count += loss
        grad = loss_func.backward()

        if optim:
            model.backward(grad)
            model.update(lr)
    return loss_count / len(dataloader)


def train_model(model, loss, lr, epochs, dataloaders, print_step=50):
    dataloader_train, dataloader_val = dataloaders
    train_loss_all, val_loss_all = [], []
    # training loop
    for epoch in range(epochs):
        # train
        train_loss = single_pass(model, dataloader_train, loss, lr)
        # validation
        if epoch % print_step == 0:
            val_loss = single_pass(model, dataloader_val, loss, lr, optim=False)
            print(
                f'epoch {epoch}, train_loss={train_loss}, val_loss={val_loss}')

            train_loss_all.append(train_loss)
            val_loss_all.append(val_loss)

    return train_loss_all, val_loss_all


def main(seed=42):
    # we fix seed to reproducibility of learning
    random.seed(seed)
    np.random.seed(seed)

    model = AutoEncoder()

    # parameters of learning
    batch_size = 8
    epochs = 100
    lr = 5 * 1e-3
    loss = MSE()

    # generate data and create data loaders
    data_train = create_data(700)  # 700
    data_val = create_data(100)

    dataloader_train = DataLoader(data_train, batch_size)
    # print(len(dataloader_train))
    dataloader_val = DataLoader(data_val, batch_size)
    dataloaders = [dataloader_train, dataloader_val]

    # train model
    train_loss, val_loss = train_model(model, loss, lr, epochs, dataloaders)

    # look at examples
    print_examples_np(model)


if __name__ == '__main__':
    main()
