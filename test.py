"""
Duality AI Challenge - FINAL TESTING & INFERENCE SCRIPT
Optimized for NVIDIA GPU & PyTorch 2.6+

- Loads 'best_segmentation_model.pth' automatically.
- Generates colored predictions.
- Calculates Mean IoU and Per-Class IoU.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
import time

# Force matplotlib to not use any Xwindow backend (safe for nohup)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
BACKBONE_SIZE = "large"  # MUST MATCH TRAINING (small/base/large)
BATCH_SIZE = 2           # Keep small for safety
NUM_CLASSES = 10

# Class names for reporting
CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

# Color palette for visualization (RGB)
COLOR_PALETTE = np.array([
    [0, 0, 0],        # 0: Background - Black
    [34, 139, 34],    # 1: Trees - Forest Green
    [0, 255, 0],      # 2: Lush Bushes - Lime
    [210, 180, 140],  # 3: Dry Grass - Tan
    [139, 90, 43],    # 4: Dry Bushes - Brown
    [128, 128, 0],    # 5: Ground Clutter - Olive
    [139, 69, 19],    # 6: Logs - Saddle Brown
    [128, 128, 128],  # 7: Rocks - Gray
    [160, 82, 45],    # 8: Landscape - Sienna
    [135, 206, 235],  # 9: Sky - Sky Blue
], dtype=np.uint8)

# ============================================================================
# 2. UTILS
# ============================================================================
# Mapping from raw pixel values to new class IDs (Same as training)
value_map = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 700:6, 800:7, 7100:8, 10000:9}

def convert_mask(mask):
    """Converts 16-bit/8-bit ID masks to class indices 0-9"""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

def mask_to_color(mask):
    """Convert a class mask (0-9) to a colored RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(NUM_CLASSES):
        color_mask[mask == class_id] = COLOR_PALETTE[class_id]
    return color_mask

def save_comparison(img_tensor, gt_mask, pred_mask, save_path, name):
    """Save side-by-side comparison: Input | Truth | Prediction"""
    # Denormalize Image
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std + mean) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Colorize Masks
    gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title(f'Input: {name}')
    axes[0].axis('off')

    axes[1].imshow(gt_color)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(pred_color)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ============================================================================
# 3. DATASET
# ============================================================================
class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Check if directories exist
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory missing: {self.image_dir}")
        if not os.path.exists(self.masks_dir):
            print(f"WARNING: Mask directory missing: {self.masks_dir}. Metrics cannot be calculated.")
            self.masks_dir = None
            
        self.data_ids = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg'))]

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        # Handle Ground Truth (if available)
        if self.masks_dir:
            mask_path = os.path.join(self.masks_dir, data_id)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
                mask = convert_mask(mask)
                if self.mask_transform:
                    mask = self.mask_transform(mask) * 255
            else:
                mask = torch.zeros((image.shape[1], image.shape[2])) # Dummy mask
        else:
            mask = torch.zeros((image.shape[1], image.shape[2]))

        return image, mask, data_id

# ============================================================================
# 4. MODEL ARCHITECTURE (Must match training)
# ============================================================================
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.project = nn.Conv2d(in_channels, 512, kernel_size=1)
        self.decode = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.project(x)
        return self.decode(x)

# ============================================================================
# 5. METRIC UTILS
# ============================================================================
def compute_iou_batch(preds, labels, num_classes=10):
    """Compute IoU for a batch of predictions"""
    iou_per_class = []
    preds = preds.view(-1)
    labels = labels.view(-1)
    
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        
        if union == 0:
            iou_per_class.append(float('nan')) # Ignore absent classes
        else:
            iou_per_class.append(intersection / union)
            
    return iou_per_class

# ============================================================================
# 6. MAIN FUNCTION
# ============================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ---------------- ARGUMENTS ----------------
    # Default: looks for 'best_segmentation_model.pth' in current folder
    model_path = os.path.join(script_dir, "best_segmentation_model.pth")
    
    # Default: looks for 'test_images' folder one level up
    test_data_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_testImages')
    
    output_dir = os.path.join(script_dir, "test_results")
    
    # ---------------- SETUP ----------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{time.strftime('%H:%M:%S')}] Starting Inference on {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks_color"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "comparisons"), exist_ok=True)

    # Dimensions
    w, h = 952, 532 

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])

    # Load Dataset
    print(f"[{time.strftime('%H:%M:%S')}] Loading images from: {test_data_dir}")
    try:
        test_dataset = TestDataset(test_data_dir, transform=transform, mask_transform=mask_transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Found {len(test_dataset)} images.")

    # ---------------- LOAD MODEL ----------------
    print(f"[{time.strftime('%H:%M:%S')}] Loading DINOv2 ({BACKBONE_SIZE})...")
    backbone_archs = {"small": "vits14", "base": "vitb14_reg", "large": "vitl14_reg"}
    backbone = torch.hub.load("facebookresearch/dinov2", f"dinov2_{backbone_archs[BACKBONE_SIZE]}")
    backbone.to(device)
    backbone.eval()

    # Get embedding dim
    with torch.no_grad():
        dummy = torch.randn(1, 3, h, w).to(device)
        emb_dim = backbone.forward_features(dummy)["x_norm_patchtokens"].shape[2]

    head = SegmentationHead(emb_dim, NUM_CLASSES, w//14, h//14).to(device)

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return
        
    print(f"[{time.strftime('%H:%M:%S')}] Loading weights from {model_path}...")
    # Fix for PyTorch 2.6 weights_only error
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'head' in checkpoint:
        head.load_state_dict(checkpoint['head'])
    else:
        head.load_state_dict(checkpoint) # Support older save formats
        
    head.eval()

    # ---------------- INFERENCE LOOP ----------------
    print(f"[{time.strftime('%H:%M:%S')}] Running inference...")
    
    total_iou_per_class = [[] for _ in range(NUM_CLASSES)]
    
    with torch.no_grad():
        for i, (imgs, true_masks, names) in enumerate(test_loader):
            imgs = imgs.to(device)
            true_masks = true_masks.squeeze(1).long().to(device)

            # 1. Forward Pass
            features = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = head(features)
            logits_up = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            preds = torch.argmax(logits_up, dim=1)

            # 2. Calculate Metrics (if ground truth exists)
            batch_ious = compute_iou_batch(preds, true_masks, NUM_CLASSES)
            for cls_idx, iou in enumerate(batch_ious):
                if not np.isnan(iou):
                    total_iou_per_class[cls_idx].append(iou)

            # 3. Save Visualizations
            for j in range(len(imgs)):
                img_name = names[j]
                base_name = os.path.splitext(img_name)[0]
                
                # Save Raw Mask (0-9)
                pred_np = preds[j].cpu().numpy().astype(np.uint8)
                Image.fromarray(pred_np).save(os.path.join(output_dir, "masks", f"{base_name}.png"))
                
                # Save Color Mask
                color_np = mask_to_color(pred_np)
                Image.fromarray(color_np).save(os.path.join(output_dir, "masks_color", f"{base_name}.png"))
                
                # Save Comparison (First 50 only to save space)
                if i * BATCH_SIZE + j < 50:
                    save_comparison(imgs[j], true_masks[j], preds[j], 
                                  os.path.join(output_dir, "comparisons", f"{base_name}_comp.png"),
                                  img_name)
                                  
            if i % 10 == 0:
                print(f"Processed batch {i}/{len(test_loader)}")

    # ---------------- FINAL REPORT ----------------
    print("\n" + "="*40)
    print("FINAL EVALUATION RESULTS")
    print("="*40)
    
    final_class_ious = []
    print(f"{'Class Name':<20} | {'IoU':<10}")
    print("-" * 33)
    
    for cls_idx, iou_list in enumerate(total_iou_per_class):
        if len(iou_list) > 0:
            mean_cls_iou = np.mean(iou_list)
            final_class_ious.append(mean_cls_iou)
            print(f"{CLASS_NAMES[cls_idx]:<20} | {mean_cls_iou:.4f}")
        else:
            print(f"{CLASS_NAMES[cls_idx]:<20} | N/A")
            
    print("-" * 33)
    if len(final_class_ious) > 0:
        mean_iou = np.mean(final_class_ious)
        print(f"MEAN IoU (mIoU)      | {mean_iou:.4f}")
    else:
        print("Mean IoU: N/A (No ground truth found)")
    print("="*40)
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()