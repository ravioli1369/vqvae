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

# generate video of training
parser.add_argument("-video", action="store_true")
parser.add_argument("--video_directory", type=str, default=None)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print("Results will be saved in ./" + args.filename + ".pth")

"""
Load data and define batch data loaders
"""

(
    training_data,
    validation_data,
    training_loader,
    validation_loader,
    test_loader,
    x_train_var,
) = utils.load_data_and_data_loaders(args.dataset, args.batch_size)
"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

model = VQVAE(
    args.n_hiddens,
    args.n_residual_hiddens,
    args.n_residual_layers,
    args.n_embeddings,
    args.embedding_dim,
    args.beta,
).to(device)

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

model.train()

results = {
    "n_updates": 0,
    "recon_errors": [],
    "loss_vals": [],
    "perplexities": [],
}


def train():
    for i in range(args.n_updates):
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

        if args.video:
            model.eval()
            codebook = [
                model.vector_quantization.embedding.weight.detach().cpu().numpy()
            ]
            reduced_codebooks_pca = utils.reduce_dimensionality_pca(codebook)
            utils.visualize_codebooks(
                reduced_codebooks_pca,
                ["Embedding Vectors"],
                "Visualization of Codebooks during Training",
                "PCA",
                os.path.join(args.video_directory, f"pca/pca_{i}.png"),
                ([-3, 3], [-3, 3]),
            )
            reduced_codebooks_tsne = utils.reduce_dimensionality_tsne(codebook)
            utils.visualize_codebooks(
                reduced_codebooks_tsne,
                ["Embedding Vectors"],
                "Visualization of Codebooks during Training",
                "t-SNE",
                os.path.join(args.video_directory, f"tsne/tsne_{i}.png"),
                ([-10, 10], [-10, 10]),
            )
            reduced_codebooks_mds = utils.reduce_dimensionality_mds(codebook)
            utils.visualize_codebooks(
                reduced_codebooks_mds,
                ["Embedding Vectors"],
                "Visualization of Codebooks during Training",
                "MDS",
                os.path.join(args.video_directory, f"mds/mds_{i}.png"),
                ([-3, 3], [-3, 3]),
            )
            reduced_codebooks_isomap = utils.reduce_dimensionality_isomap(codebook)
            utils.visualize_codebooks(
                reduced_codebooks_isomap,
                ["Embedding Vectors"],
                "Visualization of Codebooks during Training",
                "Isomap",
                os.path.join(args.video_directory, f"isomap/isomap_{i}.png"),
                ([-3, 3], [-3, 3]),
            )
            reduced_codebooks_lle = utils.reduce_dimensionality_lle(codebook)
            utils.visualize_codebooks(
                reduced_codebooks_lle,
                ["Embedding Vectors"],
                "Visualization of Codebooks during Training",
                "Modified LLE",
                os.path.join(args.video_directory, f"lle/lle_{i}.png"),
                ([-3, 3], [-3, 3]),
            )
            model.train()

        if i % args.log_interval == 0:
            """
            save model and print values
            """
            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(
                    model, results, hyperparameters, args.filename
                )

            print(
                "Update #",
                i,
                "Recon Error:",
                np.mean(results["recon_errors"][-args.log_interval :]),
                "Loss",
                np.mean(results["loss_vals"][-args.log_interval :]),
                "Perplexity:",
                np.mean(results["perplexities"][-args.log_interval :]),
            )


def test(model_path):
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
    if args.test:
        test(args.model_path)
    else:
        train()
