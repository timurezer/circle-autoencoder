import torch
import numpy as np
import matplotlib.pyplot as plt
from data_generator import create_data

def plot_results(ax, train_loss, val_loss, label):
    epochs = np.arange(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label='train')
    ax.plot(epochs, val_loss, label='validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(label)
    ax.legend()

def print_examples_torch(model, device):
    data = create_data(100)
    with torch.no_grad():
        pred = model.forward(torch.tensor(data).float().to(device)).detach().cpu().numpy()
    x, y = data[:, 0], data[:, 1]
    x_pred, y_pred = pred[:, 0], pred[:, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, label='true')
    plt.scatter(x_pred, y_pred, label='pred')
    plt.legend()
    plt.show()


def print_examples_np(model):
    data = create_data(100)
    pred = model.forward(data)
    x, y = data[:, 0], data[:, 1]
    x_pred, y_pred = pred[:, 0], pred[:, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, label='true')
    plt.scatter(x_pred, y_pred, label='pred')
    plt.legend()
    plt.show()
