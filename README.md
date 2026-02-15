# Duality AI Challenge - Offroad Semantic Segmentation

This project implements a state-of-the-art (SOTA) semantic segmentation model for offroad environments using DINOv2 as the backbone (fine-tuned on a custom dataset). The model segments images into 10 classes: Background, Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Logs, Rocks, Landscape, and Sky.

Key features:
- **Backbone**: DINOv2 Large (unfrozen for fine-tuning).
- **Decoder Head**: Deep ConvNeXt-style for robust feature decoding.
- **Loss**: Hybrid (Dice + CrossEntropy) for better segmentation accuracy.
- **Augmentations**: Heavy color jitter to improve robustness to lighting variations.
- **Evaluation**: Mean IoU (mIoU) and per-class IoU on validation/test sets.
- **Visualization**: Generates colored masks and side-by-side comparisons.

The project includes scripts for training/fine-tuning and testing/inference. It is optimized for NVIDIA GPUs and headless execution (e.g., via `nohup`).

## Prerequisites

- Python 3.8+ (tested on 3.12.3).
- NVIDIA GPU with CUDA support (for faster training/inference; CPU fallback available but slow).
- At least 16GB VRAM recommended for the "large" backbone.
- Windows OS (for the provided `create_and_install.bat`; adapt for Linux/Mac if needed).

## Setup Instructions

1. **Clone or Download the Project**:
   - Download the project files: `train_segmentation_finetune.py`, `test.py`, and `create_and_install.bat`.
   - Place them in a root project directory (e.g., `offroad_segmentation_project`).

2. **Create Virtual Environment and Install Dependencies**:
   - Run the provided batch script to set up a virtual environment and install required packages:
     ```
     create_and_install.bat
     ```
   - This will:
     - Create a virtual environment named `venv`.
     - Activate it.
     - Upgrade pip.
     - Install core packages: `torch`, `torchvision`, `numpy`, `matplotlib`, `pillow`, `opencv-python`.
   - If you're on Linux/Mac, manually run:
     ```
     python -m venv venv
     source venv/bin/activate  # or venv\Scripts\activate on Windows
     pip install --upgrade pip
     pip install torch torchvision numpy matplotlib pillow opencv-python
     ```
   - Note: Torch will automatically install with CUDA support if available. Verify with `python -c "import torch; print(torch.cuda.is_available())"`.

3. **Download Required Resources**:
   - **Pre-trained Model (`best_segmentation_model.pth`)**: Download from [this Google Drive link](https://drive.google.com/file/d/1d7SBJ1QpKa4XLaaYyhyaYlRO1kkF_t69/view?usp=sharing). Place it in the root project directory (next to `test.py`).
   - **Training/Validation Data**: Download or prepare the `Offroad_Segmentation_Training_Dataset` folder. It should contain `train` and `val` subfolders, each with `Color_Images` (RGB images) and `Segmentation` (mask images). Place this folder in the root project directory.
   - **Test Data**: Download or prepare the `Offroad_Segmentation_testImages` folder. It should contain a `Color_Images` subfolder (RGB images) and optionally a `Segmentation` subfolder (for metrics calculation). Place this folder in the root project directory.
   - **Sample Output Images**: For reference, view generated test results (masks, colored masks, comparisons) from a sample run at [this Google Drive folder](https://drive.google.com/drive/folders/1lYoZwMcMZ5TjodcnzLkdTGyOGdibf3HH?usp=sharing).
   - **Project Report**: Detailed report available at [xyz.com](xyz.com).

## Directory Structure

Organize your project root directory as follows:

```
offroad_segmentation_project/
├── train_segmentation_finetune.py       # Training script
├── test.py                              # Testing/inference script
├── create_and_install.bat               # Setup script (Windows)
├── best_segmentation_model.pth          # Downloaded pre-trained model
├── Offroad_Segmentation_Training_Dataset/  # Training data folder
│   ├── train/
│   │   ├── Color_Images/                # RGB training images (.png/.jpg)
│   │   └── Segmentation/                # Corresponding mask images (.png)
│   └── val/
│       ├── Color_Images/                # RGB validation images (.png/.jpg)
│       └── Segmentation/                # Corresponding mask images (.png)
├── Offroad_Segmentation_testImages/     # Test data folder
│   ├── Color_Images/                    # RGB test images (.png/.jpg)
│   └── Segmentation/                    # Optional: Mask images for metrics (.png)
└── test_results/                        # Auto-generated during testing
    ├── masks/                           # Raw prediction masks (0-9 values)
    ├── masks_color/                     # Colored prediction masks
    └── comparisons/                     # Side-by-side input/truth/prediction images
```

- The `test_results/` folder will be created automatically when running `test.py`.
- Ensure image and mask filenames match exactly (e.g., `image1.png` in `Color_Images` pairs with `image1.png` in `Segmentation`).
- Masks are expected to be single-channel images with raw pixel values (e.g., 0, 100, 200, etc.), which are mapped to class IDs 0-9 during processing.

## How to Run

### 1. Training/Fine-Tuning
- Ensure the `Offroad_Segmentation_Training_Dataset` folder is in the root directory.
- Activate the virtual environment (if not already): `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac).
- Run the training script:
  ```
  python train_segmentation_finetune.py
  ```
- **Details**:
  - Downloads DINOv2 automatically if not cached.
  - Trains for 19 epochs (configurable in script).
  - Uses gradient accumulation for effective batch size of 8.
  - Saves the best model as `best_segmentation_model.pth` based on validation IoU.
  - Logs progress to console (suitable for `nohup python train_segmentation_finetune.py > train.log &` in headless mode).
  - Expected runtime: ~1-2 hours per epoch on a high-end GPU (e.g., RTX 3090).
- After training, the best model is saved in the root directory for use in testing.

### 2. Testing/Inference
- Ensure the `Offroad_Segmentation_testImages` folder and `best_segmentation_model.pth` are in the root directory.
- Activate the virtual environment (as above).
- Run the testing script:
  ```
  python test.py
  ```
- **Details**:
  - Loads the model from `best_segmentation_model.pth`.
  - Processes all images in `Offroad_Segmentation_testImages/Color_Images`.
  - If masks are provided in `Segmentation`, calculates mIoU and per-class IoU.
  - Generates outputs in `test_results/`:
    - Raw masks (grayscale, values 0-9).
    - Colored masks (using a predefined palette).
    - Comparisons (input + ground truth + prediction) for the first 50 images.
  - Prints a final evaluation report to console.
  - Expected runtime: ~1-5 minutes for 100 images on GPU.

## Troubleshooting

- **Out of Memory (OOM)**: Reduce `BATCH_SIZE` to 1 or use a smaller backbone (change `BACKBONE_SIZE` to "base" or "small" in scripts).
- **Missing Directories**: Ensure data folders exist and are correctly named/structured.
- **No GPU Detected**: Check CUDA installation. Fall back to CPU by setting `device = 'cpu'` in scripts (slow).
- **Package Issues**: Re-run the setup script or manually install missing packages (e.g., `pip install facebookresearch/dinov2` if DINOv2 fails to load).
- **Mask Conversion Errors**: Ensure masks are single-channel PNGs with the exact raw values from `value_map` (0, 100, 200, etc.).

## Results and Report

- Sample test outputs (masks and comparisons) from a run on desktop: [Google Drive Folder](https://drive.google.com/drive/folders/1lYoZwMcMZ5TjodcnzLkdTGyOGdibf3HH?usp=sharing).
- Full project report (including metrics, architecture details, and ablation studies): [xyz.com](xyz.com).

For questions or improvements, contact the project maintainer.