import torch
import matplotlib.pyplot as plt
from data_generator import create_data


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
