import os
import time

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from datasets.block import BlockDataset, LatentBlockDataset
import matplotlib.pyplot as plt
import shutil

params = {
    "figure.figsize": [9, 6],
    "axes.labelsize": 14,
    "axes.titlesize": 18,
    "axes.titlepad": 15,
    "font.size": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "text.usetex": True if shutil.which("latex") else False,
    "font.family": "serif",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.top": True,
    "ytick.left": True,
    "ytick.right": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.minor.size": 2.5,
    "xtick.major.size": 5,
    "ytick.minor.size": 2.5,
    "ytick.major.size": 5,
    "axes.axisbelow": True,
    "figure.dpi": 200,
}
plt.rcParams.update(params)


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class AddPoissonNoise(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return torch.poisson(tensor.to(torch.float32)) / torch.max(tensor)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class AddSpeckleNoise(object):
    def __init__(self, std):
        self.std = std

    def __call__(self, tensor):
        return tensor + (torch.randn(1) * self.std) * torch.sqrt(tensor)

    def __repr__(self):
        return self.__class__.__name__ + "(std={0})".format(self.std)


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat channel
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

gaussian_noise_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        AddGaussianNoise(0.0, 0.1),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat channel
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

poisson_noise_transform = transforms.Compose(
    [
        transforms.PILToTensor(),
        AddPoissonNoise(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat channel
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

speckle_noise_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        AddSpeckleNoise(0.1),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat channel
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def load_codebooks(file_paths):
    codebooks = []
    for path in file_paths:
        if os.path.exists(path):
            codebook = np.load(path)
            codebooks.append(codebook)
        else:
            print(f"Error: File not found at {path}")
    return codebooks


def reduce_dimensionality_tsne(codebooks, n_components=2):
    tsne = TSNE(
        n_components=n_components, random_state=42, perplexity=30, max_iter=1000
    )
    reduced_codebooks = [tsne.fit_transform(codebook) for codebook in codebooks]
    return reduced_codebooks


def reduce_dimensionality_pca(codebooks, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_codebooks = [pca.fit_transform(codebook) for codebook in codebooks]
    return reduced_codebooks


def visualize_codebooks(reduced_codebooks, labels, title, method_name):
    plt.figure(figsize=(10, 8))
    for i, codebook in enumerate(reduced_codebooks):
        plt.scatter(
            codebook[:, 0],
            codebook[:, 1],
            label=labels[i],
            c=f"C{i}",
            alpha=0.6,
            edgecolors="w",
            s=100,
        )

    plt.title(f"{title} ({method_name})")
    plt.xlabel(f"{method_name} 1")
    plt.ylabel(f"{method_name} 2")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_codebooks_from_paths(file_paths, labels):
    codebooks = load_codebooks(file_paths)

    if len(codebooks) != len(labels):
        print("Error: The number of codebooks does not match the number of labels.")
        return

    reduced_codebooks_pca = reduce_dimensionality_pca(codebooks)
    visualize_codebooks(
        reduced_codebooks_pca, labels, "Visualization of Codebooks", "PCA"
    )

    reduced_codebooks_tsne = reduce_dimensionality_tsne(codebooks)
    visualize_codebooks(
        reduced_codebooks_tsne, labels, "Visualization of Codebooks", "t-SNE"
    )


def load_cifar():
    train = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    val = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    return train, val


def load_block():
    data_folder_path = os.getcwd()
    data_file_path = (
        data_folder_path
        + "/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy"
    )

    train = BlockDataset(
        data_file_path,
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    val = BlockDataset(
        data_file_path,
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    return train, val


def load_latent_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + "/data/latent_e_indices.npy"

    train = LatentBlockDataset(data_file_path, train=True, transform=None)

    val = LatentBlockDataset(data_file_path, train=False, transform=None)
    return train, val


def data_loaders(train_data, val_data, test_data, batch_size):

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    return train_loader, val_loader, test_loader


def load_data_and_data_loaders(dataset, batch_size):
    if dataset == "CIFAR10":
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size
        )
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset == "MNIST":
        data = datasets.MNIST(
            root="../data", train=True, download=False, transform=transform
        )
        training_data, validation_data = torch.utils.data.random_split(
            data, [50000, 10000]
        )
        test_data = datasets.MNIST(
            root="../data", train=False, download=False, transform=transform
        )
        training_loader, validation_loader, test_loader = data_loaders(
            training_data, validation_data, test_data, batch_size
        )
        x_train_var = torch.var(data.data / 255.0)

    elif dataset == "POISSON_MNIST":
        data = datasets.MNIST(
            root="../data",
            train=True,
            download=False,
            transform=poisson_noise_transform,
        )
        training_data, validation_data = torch.utils.data.random_split(
            data, [50000, 10000]
        )
        test_data = datasets.MNIST(
            root="../data",
            train=False,
            download=False,
            transform=poisson_noise_transform,
        )
        training_loader, validation_loader, test_loader = data_loaders(
            training_data, validation_data, test_data, batch_size
        )
        x_train_var = torch.var(data.data / 255.0)

    elif dataset == "GAUSSIAN_MNIST":
        data = datasets.MNIST(
            root="../data",
            train=True,
            download=False,
            transform=gaussian_noise_transform,
        )
        training_data, validation_data = torch.utils.data.random_split(
            data, [50000, 10000]
        )
        test_data = datasets.MNIST(
            root="../data",
            train=False,
            download=False,
            transform=gaussian_noise_transform,
        )
        training_loader, validation_loader, test_loader = data_loaders(
            training_data, validation_data, test_data, batch_size
        )
        x_train_var = torch.var(data.data / 255.0)

    elif dataset == "SPECKLE_MNIST":
        data = datasets.MNIST(
            root="../data",
            train=True,
            download=False,
            transform=speckle_noise_transform,
        )
        training_data, validation_data = torch.utils.data.random_split(
            data, [50000, 10000]
        )
        test_data = datasets.MNIST(
            root="../data",
            train=False,
            download=False,
            transform=speckle_noise_transform,
        )
        training_loader, validation_loader, test_loader = data_loaders(
            training_data, validation_data, test_data, batch_size
        )
        x_train_var = torch.var(data.data / 255.0)

    elif dataset == "BLOCK":
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size
        )

        x_train_var = np.var(training_data.data / 255.0)
    elif dataset == "LATENT_BLOCK":
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size
        )

        x_train_var = np.var(training_data.data)

    else:
        raise ValueError(
            "Invalid dataset: only CIFAR10 and BLOCK datasets are supported."
        )

    return (
        training_data,
        validation_data,
        training_loader,
        validation_loader,
        test_loader,
        x_train_var,
    )


def readable_timestamp():
    return time.ctime().replace("  ", " ").replace(" ", "_").replace(":", "_").lower()


def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = os.getcwd() + "/"

    os.makedirs(SAVE_MODEL_PATH + os.path.dirname(timestamp), exist_ok=True)
    results_to_save = {
        "model": model.state_dict(),
        "results": results,
        "hyperparameters": hyperparameters,
    }

    torch.save(results_to_save, SAVE_MODEL_PATH + timestamp + ".pth")
