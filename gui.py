import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import imageio

# Create the main application window
root = tk.Tk()
root.title("VQVAE Demo")
root.geometry("575x125")

# Initialize variables
selected_option = tk.StringVar(value="Select option")


# Function to handle the logic for image display
def display_metrics():
    dist1 = selected_options[0].get().lower()
    dist2 = selected_options[1].get().lower()
    metric = selected_options[2].get().lower().replace("-", "")
    n_embeddings = selected_options_hyper[0].get()
    n_dim = selected_options_hyper[1].get()
    if dist1 not in ["original", "gaussian", "poisson", "speckle"]:
        messagebox.showerror(
            "Invalid Selection", "Please select a valid distribution for codebook 1"
        )
        return
    if dist2 not in ["original", "gaussian", "poisson", "speckle"]:
        messagebox.showerror(
            "Invalid Selection", "Please select a valid distribution for codebook 2"
        )
        return
    if dist1 == dist2:
        messagebox.showerror(
            "Invalid Selection",
            "Please select different distributions for codebooks 1 and 2",
        )
        return
    if metric not in ["isomap", "pca", "tsne", "mds", "kde"]:
        messagebox.showerror(
            "Invalid Selection", "Please select a valid metric for visualization"
        )
        return
    if int(n_embeddings) not in [32, 64, 128, 256]:
        messagebox.showerror(
            "Invalid Selection", "Please select a valid number of embeddings"
        )
        return
    if int(n_dim) not in [8, 16, 32, 64]:
        messagebox.showerror(
            "Invalid Selection", "Please select a valid embedding dimension"
        )
        return
    file_path = f"./models_and_codebooks/{dist1}/codebook_dim_{n_dim}_n_embed_{n_embeddings}_{dist2}_{metric}.png"
    # Create a new pop-up window
    popup = tk.Toplevel(root)
    popup.title("Image Display")
    popup.geometry("800x600")

    img = Image.open(file_path)
    img = img.resize((800, 600))  # Resize image to fit window
    img_tk = ImageTk.PhotoImage(img)

    img_label = tk.Label(popup, image=img_tk)
    img_label.image = img_tk  # Reference to avoid garbage collection
    img_label.pack()


def display_video():
    dist1 = selected_options[0].get().lower()
    dist2 = selected_options[1].get().lower()
    metric = selected_options[2].get().lower().replace("-", "")
    if dist1 not in ["original", "gaussian", "poisson", "speckle"]:
        messagebox.showerror(
            "Invalid Selection", "Please select a valid distribution for codebook 1"
        )
        return
    if dist2 not in ["original", "gaussian", "poisson", "speckle"]:
        messagebox.showerror(
            "Invalid Selection", "Please select a valid distribution for codebook 2"
        )
        return
    if dist1 == dist2:
        messagebox.showerror(
            "Invalid Selection",
            "Please select different distributions for codebooks 1 and 2",
        )
        return
    if metric not in ["isomap", "pca", "tsne", "mds", "kde"]:
        messagebox.showerror(
            "Invalid Selection", "Please select a valid metric for visualization"
        )
        return

    video_reader = imageio.get_reader(
        f"./video/{dist1}_{dist2}/{dist1}_{dist2}_{metric}.mp4"
    )
    # Create a new pop-up window
    popup = tk.Toplevel(root)
    popup.title("Video Display")
    popup.geometry("800x600")

    def play_video():
        try:
            frame = video_reader.get_next_data()
            frame = Image.fromarray(frame)
            frame = frame.resize((800, 600))  # Resize video frame
            img_tk = ImageTk.PhotoImage(frame)
            vid_label.configure(image=img_tk)
            vid_label.image = img_tk  # Reference to avoid garbage collection
            popup.after(70, play_video)  # Update frame every 70 ms (approx. 15 fps)
        except Exception as e:
            print("End of video or error:", e)

    # Label to display video frames in the pop-up window
    vid_label = tk.Label(popup)
    vid_label.pack()

    play_video()


# Define the options for dropdown to select which kind of codebook to display
codebook_options = [
    ["Codebook 1", "Original", "Gaussian", "Poisson", "Speckle"],
    ["Codebook 2", "Original", "Gaussian", "Poisson", "Speckle"],
    ["Metric", "Isomap", "PCA", "T-SNE", "MDS", "KDE"],
]

# Create a list to hold the selected options
selected_options = [tk.StringVar() for _ in codebook_options]

# Set up each dropdown menu
for i, options in enumerate(codebook_options):
    selected_options[i].set(options[0])
    option_menu = ttk.OptionMenu(root, selected_options[i], *options)
    option_menu.grid(row=0, column=i, padx=10, pady=10)

# Define options for selecting the hyperparameter of the selected codebook
hyperparameter_options = [
    ["No. of embeddings", 32, 64, 128, 256],
    ["Embedding dimension", 8, 16, 32, 64],
]
selected_options_hyper = [tk.StringVar() for _ in hyperparameter_options]

# Set up each dropdown menu for hyperparameters
for i, options in enumerate(hyperparameter_options):
    label = tk.Label(root, text=options[0])
    label.grid(row=i + 1, column=0, padx=10, pady=2, sticky="w")
    selected_options_hyper[i].set(options[0])
    option_menu = ttk.OptionMenu(root, selected_options_hyper[i], *options)
    option_menu.grid(row=i + 1, column=1, padx=10, pady=2, columnspan=2)


# Set up action button for plotting metrics
plot_button = ttk.Button(root, text="Display Plot", command=display_metrics)
plot_button.grid(row=0, column=4, padx=10, pady=10)

# Set up action button for showing the training video
plot_button = ttk.Button(root, text="Display Video", command=display_video)
plot_button.grid(row=1, column=4, padx=10, pady=5)


# Run the application
root.mainloop()
