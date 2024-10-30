import os
import utils
import numpy as np
from glob import glob

gaussian_codebooks = np.sort(
    glob("/home/exouser/Documents/vqvae/models_and_codebooks/gaussian/*.npy")
)
poisson_codebooks = np.sort(
    glob("/home/exouser/Documents/vqvae/models_and_codebooks/poisson/*.npy")
)
speckle_codebooks = np.sort(
    glob("/home/exouser/Documents/vqvae/models_and_codebooks/speckle/*.npy")
)
original_codebooks = np.sort(
    glob("/home/exouser/Documents/vqvae/models_and_codebooks/original/*.npy")
)
codebooks = {
    "gaussian": gaussian_codebooks,
    "poisson": poisson_codebooks,
    "speckle": speckle_codebooks,
    "original": original_codebooks,
}
distributions = ["gaussian", "poisson", "speckle", "original"]
metrics = ["pca", "tsne", "mds", "isomap", "lle", "kde"]
save_paths = [
    f"/home/exouser/Documents/vqvae/models_and_codebooks/{dist}/{metric}/"
    for dist, metric in zip(distributions, metrics)
]

for dist1 in distributions:
    for dist2 in distributions:
        if dist1 == dist2:
            continue
        dist1_codebooks = codebooks[dist1]
        dist2_codebooks = codebooks[dist2]
        for dist1_codebook, dist2_codebook in zip(dist1_codebooks, dist2_codebooks):
            filename = dist1_codebook.split("/")[-1].split(".")[0]
            save_paths = [
                f"/home/exouser/Documents/vqvae/models_and_codebooks/{dist1}/{filename}_{dist2}_{metric}.png"
                for metric in metrics
            ]
            utils.visualize_codebooks_from_paths(
                [dist1_codebook, dist2_codebook],
                [dist1, dist2],
                f"Visualizing {dist1} and {dist2} codebooks",
                save_paths,
            )
