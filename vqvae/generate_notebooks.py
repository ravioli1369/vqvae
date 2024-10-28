import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from codebooks import extract_codebook
import utils
from models.vqvae import VQVAE

"""
Load data and define batch data loaders
"""


def train(
    filename,
    training_loader,
    model,
    optimizer,
    x_train_var,
    log_interval,
):
    for i in range(2000):
        (x, _) = next(iter(training_loader))
        x = x.to(device)
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = torch.mean((x_hat - x) ** 2) / x_train_var
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        if i % log_interval == 0:
            """
            save model and print values
            """
            if True:
                hyperparameters = args.__dict__
                utils.save_model_and_results(model, results, hyperparameters, filename)

            print(
                "Update #",
                i,
                "Recon Error:",
                np.mean(results["recon_errors"][-log_interval:]),
                "Loss",
                np.mean(results["loss_vals"][-log_interval:]),
                "Perplexity:",
                np.mean(results["perplexities"][-log_interval:]),
            )


def test(model_path, test_loader):
    model.load_state_dict(torch.load(model_path)["model"])
    model.eval()
    with torch.no_grad():
        (x, _) = next(iter(test_loader))
        x = x.to(device)
        embedding_loss, x_hat, perplexity = model(x)
        for i in range(10):
            fig, ax = plt.subplots(1, 2, figsize=(9, 5))
            ax[0].imshow(x[i].cpu().permute(1, 2, 0).numpy() + 1)
            ax[1].imshow(x_hat[i].cpu().permute(1, 2, 0).numpy() + 1)
            ax[0].set_title("Original", fontsize=15)
            ax[1].set_title("Reconstructed", fontsize=15)
            fig.suptitle(f"{os.path.basename(model_path).split('.')[0]}", fontsize=17)
            fig.tight_layout()
            fig.savefig(
                f'{os.path.dirname(model_path)}/{os.path.basename(model_path).split(".")[0]}_{i}.png'
            )
        recon_loss = torch.mean((x_hat - x) ** 2) / x_train_var
        loss = recon_loss + embedding_loss

        print(
            "Recon Error:",
            recon_loss.cpu().detach().numpy(),
            "Loss",
            loss.cpu().detach().numpy(),
            "Perplexity:",
            perplexity.cpu().detach().numpy(),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VQ-VAE", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--directory", type=str, default="../models")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    # Access the arguments
    directory = args.directory
    log_interval = args.log_interval

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    codebook_dims = np.array([8, 16, 32, 64])
    n_embeddings = np.array([32, 64, 128, 256])
    batch_size = 64
    datasets = ["GAUSSIAN_MNIST", "POISSON_MNIST", "SPECKLE_MNIST", "MNIST"]
    folders = ["gaussian", "poisson", "speckle", "original"]

    for dataset, folder in zip(datasets, folders):
        for dim in codebook_dims:
            for n_embed in n_embeddings:
                (
                    *_,
                    training_loader,
                    _,
                    test_loader,
                    x_train_var,
                ) = utils.load_data_and_data_loaders(dataset, batch_size)
                filename = (
                    directory
                    + "/"
                    + folder
                    + "/"
                    + f"notebook_dim_{dim}_n_embed_{n_embed}"
                )
                model = VQVAE(
                    128,
                    32,
                    2,
                    n_embed,
                    dim,
                    0.25,
                ).to(device)
                optimizer = optim.Adam(model.parameters(), lr=3e-4, amsgrad=True)
                results = {
                    "n_updates": 0,
                    "recon_errors": [],
                    "loss_vals": [],
                    "perplexities": [],
                }

                if args.test:
                    model_path = f"{filename}.pth"
                    test(model_path, test_loader)
                else:
                    train(
                        filename,
                        training_loader,
                        model,
                        optimizer,
                        x_train_var,
                        log_interval,
                    )
                    codebook = extract_codebook(f"{filename}.pth", model)
                    np.save(f"{filename}.npy", codebook)
