"""
Visualization utilities for segmentation results
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import cv2


def visualize_prediction(image, mask, pred_mask, save_path=None, alpha=0.5):
    """
    Visualize image, ground truth mask, and predicted mask
    
    Args:
        image: Input image (H, W, 3) or (3, H, W)
        mask: Ground truth mask (H, W) or (1, H, W)
        pred_mask: Predicted mask (H, W) or (1, H, W)
        save_path: Path to save the visualization
        alpha: Transparency for overlay
    """
    # Convert to numpy arrays and proper format
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    
    # Ensure proper dimensions
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
        pred_mask = pred_mask.squeeze(0)
    
    # Normalize image to 0-1 range
    if image.max() > 1:
        image = image / 255.0
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(image)
    axes[1].imshow(mask, alpha=alpha, cmap='Reds')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Predicted mask
    axes[2].imshow(image)
    axes[2].imshow(pred_mask, alpha=alpha, cmap='Blues')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, save_path=None):
    """
    Plot training and validation curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(val_losses, label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU curves
    if 'iou' in train_metrics and 'iou' in val_metrics:
        axes[0, 1].plot(train_metrics['iou'], label='Train IoU')
        axes[0, 1].plot(val_metrics['iou'], label='Val IoU')
        axes[0, 1].set_title('Training and Validation IoU')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Dice curves
    if 'dice' in train_metrics and 'dice' in val_metrics:
        axes[1, 0].plot(train_metrics['dice'], label='Train Dice')
        axes[1, 0].plot(val_metrics['dice'], label='Val Dice')
        axes[1, 0].set_title('Training and Validation Dice')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Accuracy curves
    if 'accuracy' in train_metrics and 'accuracy' in val_metrics:
        axes[1, 1].plot(train_metrics['accuracy'], label='Train Accuracy')
        axes[1, 1].plot(val_metrics['accuracy'], label='Val Accuracy')
        axes[1, 1].set_title('Training and Validation Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_mask_overlay(image, mask, color=[255, 0, 0], alpha=0.5):
    """
    Create a colored overlay of mask on image
    
    Args:
        image: Input image (H, W, 3)
        mask: Binary mask (H, W)
        color: RGB color for mask overlay
        alpha: Transparency factor
    
    Returns:
        Overlayed image
    """
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    
    # Blend with original image
    overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    
    return overlay


def save_prediction_comparison(image, gt_mask, pred_mask, save_path, threshold=0.5):
    """
    Save a comparison image showing original, ground truth, and prediction
    
    Args:
        image: Input image
        gt_mask: Ground truth mask
        pred_mask: Predicted mask
        save_path: Path to save the comparison
        threshold: Threshold for binary prediction
    """
    # Convert predictions to binary
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = (pred_mask > threshold).float().cpu().numpy()
    
    # Ensure proper format
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.squeeze(0)
    
    # Create visualization
    visualize_prediction(image, gt_mask, pred_mask, save_path)
