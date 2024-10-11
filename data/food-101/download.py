import torch
import torchvision.datasets as datasets

datasets.Food101(root=".", download=True, transform=None)
