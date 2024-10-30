import imageio
import os

dist = ["gaussian", "poisson", "speckle", "original"]
metrics = ["pca", "tsne", "mds", "isomap", "lle", "kde"]
for d1 in dist:
    for d2 in dist:
        for m in metrics:
            if d1 == d2:
                continue
            image_folder = f"/home/exouser/Documents/vqvae/video/{d1}_{d2}/{m}/"
            output_video_path = (
                f"/home/exouser/Documents/vqvae/video/{d1}_{d2}/{d1}_{d2}_{m}.mp4"
            )

            images = []
            for i in range(0, 2000, 10):
                filename = os.path.join(image_folder, f"{i}.png")
                if os.path.exists(filename):
                    images.append(imageio.imread(filename))

            imageio.mimsave(output_video_path, images, fps=15)
