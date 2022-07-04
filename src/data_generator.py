import numpy as np

def create_data(n, noise=False, sigma=0.1):
    # creade set of points at circle with R=1
    phi = 2 * np.pi * np.arange(n) / n    # np.random.rand(n) # np.arange(n) / n
    x, y = np.cos(phi), np.sin(phi)
    if noise:
        x += np.random.normal(0, sigma, n)
        y += np.random.normal(0, sigma, n)
    data = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    return data
