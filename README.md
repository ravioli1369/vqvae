# IE643 Project
## Vector Quantized Variational Autoencoders

The gui is built using `tkinter` and can be run using `python gui.py` directly. The options to choose from are self explanatory and also discussed in our [demo video](https://drive.google.com/file/d/16SqoMDf2q39U-l-3loyNmZQvghYzb7pu/view?usp=drive_link).

We train the VQVAE model using the files present in the `vqvae/` folder. The folder contains a README from the repo we had cloned for the PyTorch implementation of VQVAE. 
The key files to note are:

- `main.py`: Trains the model and saves the final `.pth` file in the specified directory. The code can be run using the sample command

  ```bash
  python main.py --dataset IMAGENET -save --filename ../models_and_codebooks/tinyimagenet/tinyimagenet
  ```
  We can specify other parameters by running `main.py -h`, which prints the help screen.
  Sample output:
  ```
  > python main.py -h
    usage: main.py [-h] [--batch_size BATCH_SIZE] [--n_updates N_UPDATES] [--n_hiddens N_HIDDENS]
                   [--n_residual_hiddens N_RESIDUAL_HIDDENS] [--n_residual_layers N_RESIDUAL_LAYERS]
                   [--embedding_dim EMBEDDING_DIM] [--n_embeddings N_EMBEDDINGS] [--beta BETA]
                   [--learning_rate LEARNING_RATE] [--log_interval LOG_INTERVAL] [--dataset DATASET] [-save]
                   [--filename FILENAME] [-test] [--model_path MODEL_PATH] [--video_directory VIDEO_DIRECTORY]
    
    options:
      -h, --help            show this help message and exit
      --batch_size BATCH_SIZE
      --n_updates N_UPDATES
      --n_hiddens N_HIDDENS
      --n_residual_hiddens N_RESIDUAL_HIDDENS
      --n_residual_layers N_RESIDUAL_LAYERS
      --embedding_dim EMBEDDING_DIM
      --n_embeddings N_EMBEDDINGS
      --beta BETA
      --learning_rate LEARNING_RATE
      --log_interval LOG_INTERVAL
      --dataset DATASET
      -save
      --filename FILENAME
      -test
  --model_path MODEL_PATH
  --video_directory VIDEO_DIRECTORY
  ```
- `codebooks.py`: Used to extract codebooks from the trained models, and saves them in the specified directory as `.npz` array files.
  Sample command:
  ```bash
  python codebooks.py --filename ../models_and_codebooks/tinyimagenet/tinyimagenet.pth --model_path ../models_and_codebooks/tinyimagenet/tinyimagenet.pth
  ```
- `plot_final_epoch_metrics.py`, `plot_metrics_for_video.py` and `video_generator.py` are files used by us to generate the final video and plots for the GUI.
- Almost all the notebooks contain documentation of each metric we used, along with the appropriate plots produced by them.


Our results are stored in `models_and_codebooks/` with appropriate subdirectories for each result. The GUI code displays the plots from this folder. We also have a `video/` folder which contains the epochwise plots for each metric and combination of noises. The GUI uses this folder as well for displaying the videos.
