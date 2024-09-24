import torch
from vqvae import VQVAE  # Assuming the VQ-VAE class is named `VQVAE`

# Load the trained VQ-VAE model from a checkpoint
model = VQVAE()
checkpoint = torch.load('path_to_model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set model to evaluation mode
