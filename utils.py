import torch
import torchvision.transforms as T
from torchvision.datasets import MNIST
import numpy as np

def get_mnist_loaders(path="./data", batch_size=128):
    transform = T.Compose([T.ToTensor(), T.Lambda(lambda x: x.view(-1))])
    train_set = MNIST(path, train=True,  download=True, transform=transform)
    test_set  = MNIST(path, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# -------------- bruits / OOD --------------
def mnist_rotation(dataset, angle_range=15):
    """retourne un dataset MNIST rotatif"""
    def rotate(x):
        angle = np.random.uniform(-angle_range, angle_range)
        return T.functional.rotate(x.view(28,28).unsqueeze(0), angle).view(-1)
    return MNIST(dataset.root, train=False, download=False,
                 transform=T.Compose([T.ToTensor(), T.Lambda(rotate)]))

def mnist_gauss_noise(dataset, sigma=0.3):
    def add_noise(x):
        return x + sigma * torch.randn_like(x)
    return MNIST(dataset.root, train=False, download=False,
                 transform=T.Compose([T.ToTensor(), T.Lambda(add_noise)]))