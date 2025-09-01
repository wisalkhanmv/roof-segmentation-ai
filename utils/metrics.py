"""
Segmentation metrics utilities
"""
import torch
import numpy as np
from sklearn.metrics import jaccard_score


def calculate_iou(pred, target, threshold=0.5):
    """
    Calculate Intersection over Union (IoU)
    
    Args:
        pred: Predicted masks (B, 1, H, W) - can be tensor or numpy array
        target: Ground truth masks (B, 1, H, W) - can be tensor or numpy array
        threshold: Threshold for binary segmentation
    
    Returns:
        IoU score
    """
    # Convert to numpy if needed
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = (target > threshold).astype(np.float32)
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection / union)


def calculate_dice(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calculate Dice coefficient
    
    Args:
        pred: Predicted masks (B, 1, H, W) - can be tensor or numpy array
        target: Ground truth masks (B, 1, H, W) - can be tensor or numpy array
        threshold: Threshold for binary segmentation
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice score
    """
    # Convert to numpy if needed
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = (target > threshold).astype(np.float32)
    
    intersection = (pred_binary * target_binary).sum()
    dice = (2 * intersection + smooth) / (pred_binary.sum() + target_binary.sum() + smooth)
    
    return float(dice)


def calculate_accuracy(pred, target, threshold=0.5):
    """
    Calculate pixel accuracy
    
    Args:
        pred: Predicted masks (B, 1, H, W) - can be tensor or numpy array
        target: Ground truth masks (B, 1, H, W) - can be tensor or numpy array
        threshold: Threshold for binary segmentation
    
    Returns:
        Accuracy score
    """
    # Convert to numpy if needed
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = (target > threshold).astype(np.float32)
    
    correct = (pred_binary == target_binary).astype(np.float32).sum()
    total = target_binary.size
    
    return float(correct / total)


def calculate_precision_recall(pred, target, threshold=0.5):
    """
    Calculate precision and recall
    
    Args:
        pred: Predicted masks (B, 1, H, W) - can be tensor or numpy array
        target: Ground truth masks (B, 1, H, W) - can be tensor or numpy array
        threshold: Threshold for binary segmentation
    
    Returns:
        Tuple of (precision, recall)
    """
    # Convert to numpy if needed
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = (target > threshold).astype(np.float32)
    
    tp = (pred_binary * target_binary).sum()
    fp = (pred_binary * (1 - target_binary)).sum()
    fn = ((1 - pred_binary) * target_binary).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return float(precision), float(recall)


def calculate_all_metrics(pred, target, threshold=0.5):
    """
    Calculate all metrics at once
    
    Args:
        pred: Predicted masks (B, 1, H, W)
        target: Ground truth masks (B, 1, H, W)
        threshold: Threshold for binary segmentation
    
    Returns:
        Dictionary of all metrics
    """
    iou = calculate_iou(pred, target, threshold)
    dice = calculate_dice(pred, target, threshold)
    accuracy = calculate_accuracy(pred, target, threshold)
    precision, recall = calculate_precision_recall(pred, target, threshold)
    
    return {
        'iou': iou,
        'dice': dice,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }
