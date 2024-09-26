import numpy as np
import torch
import os
import torch.optim as optim
import argparse
import utils
from models.vqvae import VQVAE
import matplotlib.pyplot as plt

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
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--dataset",  type=str, default='CIFAR10')

# whether or not to save model
parser.add_argument("-save", action="store_true")
parser.add_argument("--filename",  type=str, default=timestamp)
parser.add_argument("--noise", type=str, default='gaussian')
parser.add_argument("--noise_std", type=float, default=0.1)
#testing
parser.add_argument("-test", action="store_true")
parser.add_argument("--model_path", type=str, default=None)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print('Results will be saved in ./' + args.filename + '.pth')

"""
Load data and define batch data loaders
"""

training_data, validation_data, training_loader, validation_loader, test_loader, x_train_var = utils.load_data_and_data_loaders(
    args.dataset, args.batch_size)
"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}


def train():
    for i in range(args.n_updates):

        (x, _) = next(iter(training_loader))
        x = x.to(device)
        if args.noise == 'gaussian':
            noisy_x = x + args.noise_std * torch.randn_like(x)
        elif args.noise == 'poisson':
            transformed_x = (x + 1) * 255
            noisy_x = torch.poisson(transformed_x) / 255 - 1
        elif args.noise == 'speckle':
            noisy_x = x + (torch.randn(1) * args.noise_std)*torch.sqrt(x)
        
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity = model(noisy_x)
        recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        if i % args.log_interval == 0:
            """
            save model and print values
            """
            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(
                    model, results, hyperparameters, args.filename)

            print('Update #', i, 'Recon Error:',
                  np.mean(results["recon_errors"][-args.log_interval:]),
                  'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                  'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]))


def test(model_path):
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    with torch.no_grad():
        (x, _) = next(iter(test_loader))
        if args.noise == 'gaussian':
            noisy_x = x + args.noise_std * torch.randn_like(x)
        elif args.noise == 'poisson':
            noisy_x = torch.poisson(x)
        elif args.noise == 'speckle':
            noisy_x = x * torch.randn_like(x)
        x = x.to(device)
        noisy_x = noisy_x.to(device)
        embedding_loss, x_hat, perplexity = model(noisy_x)
        #show original, noisy and reconstructed image with appropriate titles
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(x[0].cpu().permute(1, 2, 0).numpy())
        ax[0].set_title('Original')
        ax[1].imshow(noisy_x[0].cpu().permute(1, 2, 0).numpy())
        ax[1].set_title('Noisy')
        ax[2].imshow(x_hat[0].cpu().permute(1, 2, 0).numpy())
        ax[2].set_title('Reconstructed')
        fig.savefig('Denoising_noise' + str(args.noise) + '.png')

        recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        loss = recon_loss + embedding_loss

        print('Recon Error:', recon_loss.cpu().detach().numpy(),
            'Loss', loss.cpu().detach().numpy(),
            'Perplexity:', perplexity.cpu().detach().numpy())

if __name__ == "__main__":
    if args.test:
        test(args.model_path)
    else:
        train()
