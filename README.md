# Chess-GPU-Hackathon-Strong-Compute

This is the codebase in which we developed a chess bot that combines convolutional neural networks with self-attention and squeeze-excitation blocks. Unlike traditional chess engines relying on reinforcement learning, this approach treats board evaluation as a computer vision problem, processing 8×8 chess boards with piece encoding to produce a scalar position score. Trained on 350,000 Grandmaster games using a 48-GPU cluster, this innovative model won us and my team a $10,000 credit in a hackathon.

# Team: THETA HAT
### Gopi Maguluri
### Venkatachalam Subramanian Periya Subbu


### [Download the model from here](https://drive.google.com/file/d/1YGw2ALPADgDksUJsQrTEXxSHVNf-L3EN/view?usp=sharing)

### [Here is a demo video](https://drive.google.com/file/d/1NjxraVBCrHRyEOtKoU4VkW28or7WkpUj/view?usp=drive_link)

# Model architecture
![model architecture](Theta_Hat_Model_Architecture.png)

# Project Structure

## Root Directory
- **train_chessVision.py**: Main training script for the chess vision model with distributed training capabilities
- **model.py**: Core model implementation with CNN, self-attention, and squeeze-excitation blocks
- **model_config.yaml**: Configuration file for model hyperparameters
- **chess_gameplay.py**: Script for playing chess using the trained model
- **requirements.txt**: Dependencies needed to run the project
- **chessVision.isc**: Configuration file for the vision system

## Directories

### ```theta_hat_submission/```
Contains the final submission version of the project:
- **chess_gameplay.py**: Game playing script
- **model.py**: Model implementation
- **model_config.yaml**: Simplified configuration
- **pre_submission_val.py**: Validation script for pre-submission
- **checkpoint.pt**: Saved model weights

### ```utils/```
Utility functions and preprocessing tools:
- **train_utils.py**: Utilities for training
- **transformer_utils.py**: Transformer-related utility functions
- **datasets.py**: Dataset loaders and processors
- **optimizers.py**: Custom optimizers (including Lamb optimizer)
- **data_utils.py**: Data handling utilities
- **constants.py**: Project constants
- **stockfish**: Stockfish chess engine binary for evaluation
- **data_preprocessing/**: Notebooks and scripts for data preparation
  - **eval_preproc.ipynb**: Preprocessing for evaluation data
  - **gm_preproc.ipynb**: Preprocessing for Grandmaster games
  - **lc_preproc.ipynb**: Preprocessing for LichessChess data
  - **preproc.isc**: Preprocessing configuration
  - **preproc_boardeval.py**: Board evaluation preprocessing

### ```logs/```
Training logs from distributed training:
- **rank_0_1st_training.txt**: Logs from first training run
- **rank_0_2nd_training.txt**: Logs from second training run

### ```eval_notebook/```
Contains evaluation notebooks:
- **gameplay.ipynb**: Interactive notebook to evaluate and play with the model

### ```sample_demo/```
Demo images:
- **demo.png**: Sample visualization
- **demo_2.png**: Additional visualization
