"""
Configuration file for Roof Segmentation Pipeline
"""
import os
from pathlib import Path

class Config:
    # Dataset paths - updated for full INRIA dataset
    INRIA_ROOT = "/home/wisalkhanmv/stuff/services/roof-trainer/AerialImageDataset"
    TRAIN_IMAGES = os.path.join(INRIA_ROOT, "train", "images")
    TRAIN_LABELS = os.path.join(INRIA_ROOT, "train", "gt")
    TEST_IMAGES = os.path.join(INRIA_ROOT, "test", "images")
    
    # Model configuration
    MODEL_NAME = "unet"  # "unet" or "deeplabv3plus"
    BACKBONE = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 3
    CLASSES = 1  # Binary segmentation
    
    # Training parameters - optimized for full dataset
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    IMAGE_SIZE = 512
    VAL_SPLIT = 0.2
    NUM_WORKERS = 4
    
    # Augmentation parameters
    CROP_SIZE = 512
    FLIP_PROB = 0.5
    ROTATION_LIMIT = 30
    BRIGHTNESS_LIMIT = 0.2
    CONTRAST_LIMIT = 0.2
    
    # Inference parameters
    TILE_SIZE = 512
    OVERLAP = 64
    BATCH_SIZE_INFERENCE = 4
    
    # Paths
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    OUTPUT_DIR = "outputs"
    
    # Device
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu"
    
    # Logging
    LOGGER = "tensorboard"  # "tensorboard" or "wandb"
    WANDB_PROJECT = "roof-segmentation"
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        for dir_path in [cls.CHECKPOINT_DIR, cls.LOG_DIR, cls.OUTPUT_DIR]:
            Path(dir_path).mkdir(exist_ok=True)
    
    @classmethod
    def get_model_config(cls):
        """Get model configuration dictionary"""
        return {
            "model_name": cls.MODEL_NAME,
            "backbone": cls.BACKBONE,
            "encoder_weights": cls.ENCODER_WEIGHTS,
            "in_channels": cls.IN_CHANNELS,
            "classes": cls.CLASSES,
        }
