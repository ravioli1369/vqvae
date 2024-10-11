import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

import utils
from models.vqvae import VQVAE

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()


def extract_codebook(model_path, model):
    model.load_state_dict(torch.load(model_path)["model"])
    model.eval()
    return model.vector_quantization.embedding.weight.detach().cpu().numpy()


parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=5000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=8)
parser.add_argument("--n_embeddings", type=int, default=64)
parser.add_argument("--beta", type=float, default=0.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--dataset", type=str, default="CIFAR10")

# whether or not to save model
parser.add_argument("-save", action="store_true")
parser.add_argument("--filename", type=str, default=timestamp)

# testing
parser.add_argument("-test", action="store_true")
parser.add_argument("--model_path", type=str, default=None)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print("Results will be saved in ./" + args.filename + ".pth")


model = VQVAE(
    args.n_hiddens,
    args.n_residual_hiddens,
    args.n_residual_layers,
    args.n_embeddings,
    args.embedding_dim,
    args.beta,
).to(device)


if __name__ == "__main__":
    codebook = extract_codebook("../results/poisson/poisson.pth", model)
    print(codebook.shape)
    np.save("../results/poisson/poisson.npy", codebook)
