import os
import utils
import numpy as np
from glob import glob

gaussian_codebooks = np.sort(
    glob("/home/exouser/Documents/vqvae/results/gaussian/codebooks/*.npy")
)
epochs = [int(file.split("_")[-1].split(".")[0]) for file in gaussian_codebooks]
sort = np.argsort(epochs)
gaussian_codebooks = gaussian_codebooks[sort]
poisson_codebooks = np.sort(
    glob("/home/exouser/Documents/vqvae/results/poisson/codebooks/*.npy")
)
epochs = [int(file.split("_")[-1].split(".")[0]) for file in poisson_codebooks]
sort = np.argsort(epochs)
poisson_codebooks = poisson_codebooks[sort]
speckle_codebooks = np.sort(
    glob("/home/exouser/Documents/vqvae/results/speckle/codebooks/*.npy")
)
epochs = [int(file.split("_")[-1].split(".")[0]) for file in speckle_codebooks]
sort = np.argsort(epochs)
speckle_codebooks = speckle_codebooks[sort]
original_codebooks = np.sort(
    glob("/home/exouser/Documents/vqvae/results/original/codebooks/*.npy")
)
epochs = [int(file.split("_")[-1].split(".")[0]) for file in original_codebooks]
sort = np.argsort(epochs)
original_codebooks = original_codebooks[sort]

epochs = np.sort(epochs)

codebooks = {
    "gaussian": gaussian_codebooks,
    "poisson": poisson_codebooks,
    "speckle": speckle_codebooks,
    "original": original_codebooks,
}
distributions = ["gaussian", "poisson", "speckle", "original"]
metrics = ["pca", "tsne", "mds", "isomap", "lle", "kde"]

for dist1 in distributions:
    for dist2 in distributions:
        if dist1 == dist2:
            continue
        dist1_codebooks = codebooks[dist1]
        dist2_codebooks = codebooks[dist2]
        for dist1_codebook, dist2_codebook, epoch in zip(
            dist1_codebooks, dist2_codebooks, epochs
        ):
            filename = dist1_codebook.split("/")[-1].split(".")[0]
            save_paths = [
                f"/home/exouser/Documents/vqvae/video/{dist1}_{dist2}/{metric}/{epoch}.png"
                for metric in metrics
            ]
            for path in save_paths:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            if (
                os.path.exists(save_paths[0])
                & os.path.exists(save_paths[1])
                & os.path.exists(save_paths[2])
                & os.path.exists(save_paths[3])
                & os.path.exists(save_paths[4])
                & os.path.exists(save_paths[5])
            ):
                continue
            else:
                utils.visualize_codebooks_from_paths(
                    [dist1_codebook, dist2_codebook],
                    [dist1, dist2],
                    f"Epoch = {epoch}",
                    save_paths,
                )
