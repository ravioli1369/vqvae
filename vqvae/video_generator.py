import imageio
import os

dist = ["speckle"]
metrics = ["pca", "lle", "mds", "isomap", "tsne"]

for d in dist:
    for m in metrics:
        print(f"Generating video for {m} with {d} distribution")
        image_folder = f"../results/video/og_{d}/{m}"
        output_video_path = f"../results/video/og_{d}/output_video_{m}_{d}.mp4"

        images = []
        for i in range(0, 2000, 10):
            filename = os.path.join(image_folder, f"{i}.png")
            if os.path.exists(filename):
                images.append(imageio.imread(filename))

        imageio.mimsave(output_video_path, images, fps=15)
