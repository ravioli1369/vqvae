import torch
import torchvision.datasets as datasets

datasets.MNIST(root=".", train=True, download=True, transform=None)
