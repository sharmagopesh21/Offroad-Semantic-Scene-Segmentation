"""
Duality AI Challenge - SOTA Fine-Tuning Script
Optimized for NVIDIA GPU & Headless Execution (nohup)

- Model: DINOv2 Large (Unfrozen)
- Head: Deep ConvNeXt-style Decoder
- Loss: Hybrid (Dice + CrossEntropy)
- Augmentation: Heavy Color Jitter for Robustness
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import time

# Force matplotlib to not use any Xwindow backend.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
BACKBONE_SIZE = "large"           # "small", "base", "large" (Use "large" for best results)
BATCH_SIZE = 2                    # Keep small for Large backbone to avoid OOM
GRAD_ACCUM_STEPS = 4              # Effective batch size = 2 * 4 = 8
LR_BACKBONE = 5e-6                # Slow learning for the pre-trained brain
LR_HEAD = 5e-4                    # Fast learning for the new head
EPOCHS = 19
PRINT_FREQ = 20                   # Print log every 20 batches (keeps nohup clean)

# ============================================================================
# 2. UTILS & LOSS
# ============================================================================
# Map raw mask values to class IDs (0-9)
value_map = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 700:6, 800:7, 7100:8, 10000:9}
n_classes = len(value_map)

def convert_mask(mask):
    """Converts 16-bit/8-bit ID masks to class indices 0-9"""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, inputs, target):
        inputs = torch.softmax(inputs, dim=1)
        # Create one-hot encoding for target
        target_one_hot = F.one_hot(target, num_classes=self.n_classes).permute(0, 3, 1, 2).float()
        
        dims = (2, 3)
        intersection = (inputs * target_one_hot).sum(dim=dims)
        union = inputs.sum(dim=dims) + target_one_hot.sum(dim=dims)
        
        # Compute Dice score (smooth to avoid div by zero)
        dice = 2.0 * (intersection + 1e-8) / (union + 1e-8)
        
        # Loss is 1 - Dice (average over classes and batch)
        return 1 - dice.mean()

# ============================================================================
# 3. DATASET
# ============================================================================
class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Directory not found: {self.image_dir}")
            
        self.data_ids = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg'))]

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        # Load Image and Mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        if self.transform:
            image = self.transform(image)
            # Resize mask with Nearest Neighbor to preserve class IDs
            mask = self.mask_transform(mask) * 255

        return image, mask

# ============================================================================
# 4. MODEL
# ============================================================================
class SegmentationHead(nn.Module):
    """
    Robust decoding head to interpret DINOv2 features.
    Projects high-dim embeddings down to class logits.
    """
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        
        # Project embedding to reasonable dimension
        self.project = nn.Conv2d(in_channels, 512, kernel_size=1)
        
        # Decoding layers
        self.decode = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1), # Added dropout for regularization
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # DINOv2 outputs (Batch, Tokens, Channels) -> Reshape to (Batch, Channels, H, W)
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.project(x)
        return self.decode(x)

# ============================================================================
# 5. MAIN TRAINING LOOP
# ============================================================================
def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{time.strftime('%H:%M:%S')}] Starting Training on {device}")
    print(f"[{time.strftime('%H:%M:%S')}] Config: Backbone={BACKBONE_SIZE}, Epochs={EPOCHS}")

    # Dimensions (Must be divisible by 14 for DINOv2)
    # 960x540 is the target, closest multiple of 14 is 952x532
    w, h = 952, 532 

    # --- ROBUST TRANSFORMS ---
    # Training: Add noise/color jitter to force model to learn shapes, not just colors
    train_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation: Clean resize only
    val_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])

    # --- DATA LOADING ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming standard folder structure: root -> code_folder, root -> Offroad_Segmentation_Training_Dataset
    data_root = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset')
    
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')

    print(f"[{time.strftime('%H:%M:%S')}] Loading data from {data_root}...")
    
    trainset = MaskDataset(train_dir, transform=train_transform, mask_transform=mask_transform)
    valset = MaskDataset(val_dir, transform=val_transform, mask_transform=mask_transform)
    
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"[{time.strftime('%H:%M:%S')}] Train Samples: {len(trainset)} | Val Samples: {len(valset)}")

    # --- MODEL SETUP ---
    print(f"[{time.strftime('%H:%M:%S')}] Loading DINOv2...")
    backbone_archs = {"small": "vits14", "base": "vitb14_reg", "large": "vitl14_reg"}
    backbone = torch.hub.load("facebookresearch/dinov2", f"dinov2_{backbone_archs[BACKBONE_SIZE]}")
    backbone.to(device)
    
    # UNFREEZE BACKBONE (Crucial for fine-tuning)
    backbone.train()

    # Get dynamic embedding size
    with torch.no_grad():
        dummy = torch.randn(1, 3, h, w).to(device)
        output = backbone.forward_features(dummy)["x_norm_patchtokens"]
        emb_dim = output.shape[2]
    
    print(f"[{time.strftime('%H:%M:%S')}] Embedding Dim: {emb_dim}")

    head = SegmentationHead(emb_dim, n_classes, w//14, h//14).to(device)

    # --- OPTIMIZER ---
    # Differential Learning Rates: Low for backbone, High for head
    optimizer = optim.AdamW([
        {'params': backbone.parameters(), 'lr': LR_BACKBONE},
        {'params': head.parameters(), 'lr': LR_HEAD}
    ], weight_decay=1e-4)

    # Hybrid Loss
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes)

    # --- TRAINING LOOP ---
    best_val_iou = 0.0
    
    print(f"[{time.strftime('%H:%M:%S')}] Starting Training Loop...")
    print("=" * 60)

    for epoch in range(EPOCHS):
        # 1. TRAIN
        backbone.train()
        head.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for i, (imgs, masks) in enumerate(train_loader):
            imgs = imgs.to(device)
            masks = masks.squeeze(1).long().to(device)

            # Forward
            features = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = head(features)
            
            # Upsample to full size for loss calculation
            logits_up = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            
            # Calculate Loss
            loss_ce = criterion_ce(logits_up, masks)
            loss_dice = criterion_dice(logits_up, masks)
            loss = (0.5 * loss_ce) + (0.5 * loss_dice)
            
            # Gradient Accumulation
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            current_loss = loss.item() * GRAD_ACCUM_STEPS
            epoch_loss += current_loss

            # Nohup-friendly logging
            if i % PRINT_FREQ == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{i}/{len(train_loader)}] Loss: {current_loss:.4f}")

        avg_train_loss = epoch_loss / len(train_loader)

        # 2. VALIDATION
        backbone.eval()
        head.eval()
        val_iou_list = []
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.squeeze(1).long().to(device)

                features = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = head(features)
                logits_up = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
                
                # Get predictions
                preds = torch.argmax(logits_up, dim=1)
                
                # Simple IoU calculation
                intersection = (preds == masks).float().sum()
                union = preds.numel() + masks.numel() - intersection
                iou = (intersection / union).item()
                val_iou_list.append(iou)

        avg_val_iou = np.mean(val_iou_list)

        print("-" * 60)
        print(f"END EPOCH {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val IoU: {avg_val_iou:.4f}")
        
        # Save Best Model
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            save_path = os.path.join(script_dir, "best_segmentation_model.pth")
            torch.save({
                'backbone': backbone.state_dict(),
                'head': head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'iou': best_val_iou
            }, save_path)
            print(f"*** New Best Model Saved to {save_path} ***")
        print("-" * 60)
        sys.stdout.flush() # Ensure logs appear immediately in nohup.out

if __name__ == "__main__":
    main()