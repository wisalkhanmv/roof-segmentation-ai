"""
Segmentation Models using segmentation_models.pytorch
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SegmentationModel(nn.Module):
    """
    Wrapper for segmentation models from segmentation_models.pytorch
    """
    
    def __init__(self, 
                 model_name='unet',
                 backbone='resnet34',
                 encoder_weights='imagenet',
                 in_channels=3,
                 classes=1,
                 activation=None):
        """
        Initialize segmentation model
        
        Args:
            model_name: Name of the model ('unet', 'deeplabv3plus', 'fpn', 'pspnet')
            backbone: Backbone architecture ('resnet34', 'resnet50', 'efficientnet-b0', etc.)
            encoder_weights: Pre-trained weights ('imagenet', 'ssl', 'swsl', None)
            in_channels: Number of input channels
            classes: Number of output classes
            activation: Activation function for output
        """
        super().__init__()
        
        self.model_name = model_name
        self.backbone = backbone
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
        
        # Create model based on name
        if model_name == 'unet':
            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation
            )
        elif model_name == 'deeplabv3plus':
            self.model = smp.DeepLabV3Plus(
                encoder_name=backbone,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation
            )
        elif model_name == 'fpn':
            self.model = smp.FPN(
                encoder_name=backbone,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation
            )
        elif model_name == 'pspnet':
            self.model = smp.PSPNet(
                encoder_name=backbone,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation
            )
        elif model_name == 'linknet':
            self.model = smp.Linknet(
                encoder_name=backbone,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor (B, classes, H, W)
        """
        return self.model(x)
    
    def get_encoder(self):
        """Get the encoder part of the model"""
        return self.model.encoder
    
    def get_decoder(self):
        """Get the decoder part of the model"""
        return self.model.decoder


class RoofSegmentationModel(nn.Module):
    """
    Specialized roof segmentation model with additional features
    """
    
    def __init__(self, 
                 model_name='unet',
                 backbone='resnet34',
                 encoder_weights='imagenet',
                 in_channels=3,
                 classes=1,
                 dropout=0.1,
                 use_attention=True):
        """
        Initialize roof segmentation model
        
        Args:
            model_name: Base model architecture
            backbone: Backbone architecture
            encoder_weights: Pre-trained weights
            in_channels: Number of input channels
            classes: Number of output classes
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        
        # Base segmentation model
        self.base_model = SegmentationModel(
            model_name=model_name,
            backbone=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None
        )
        
        # Additional layers for roof segmentation
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        if use_attention:
            # Simple spatial attention mechanism
            self.attention = nn.Sequential(
                nn.Conv2d(classes, 1, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.attention = nn.Identity()
        
        # Final activation (none for binary segmentation when using BCEWithLogitsLoss)
        self.activation = nn.Identity() if classes == 1 else nn.Softmax(dim=1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor (B, classes, H, W)
        """
        # Get base prediction
        base_output = self.base_model(x)
        
        # Apply dropout
        output = self.dropout(base_output)
        
        # Apply attention if enabled
        if isinstance(self.attention, nn.Sequential):
            attention_weights = self.attention(output)
            output = output * attention_weights
        
        # Apply final activation
        output = self.activation(output)
        
        return output


def create_model(model_config):
    """
    Factory function to create segmentation model
    
    Args:
        model_config: Dictionary containing model configuration
        
    Returns:
        Segmentation model instance
    """
    model_name = model_config.get('model_name', 'unet')
    backbone = model_config.get('backbone', 'resnet34')
    encoder_weights = model_config.get('encoder_weights', 'imagenet')
    in_channels = model_config.get('in_channels', 3)
    classes = model_config.get('classes', 1)
    
    # Create specialized roof segmentation model
    model = RoofSegmentationModel(
        model_name=model_name,
        backbone=backbone,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        dropout=0.1,
        use_attention=True
    )
    
    return model


def get_model_summary(model, input_size=(3, 512, 512)):
    """
    Get model summary and parameter count
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
        
    Returns:
        Model summary string
    """
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = OrderedDict()
            summary[m_key]["module_name"] = m_key
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["output_shape"] = list(output.size())
            summary[m_key]["num_parameters"] = sum(p.numel() for p in module.parameters())
            
        hooks.append(module.register_forward_hook(hook))
    
    # Create hooks
    summary = OrderedDict()
    hooks = []
    
    # Register hooks
    model.apply(register_hook)
    
    # Make a forward pass
    x = torch.zeros(1, *input_size)
    model(x)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Create summary string
    summary_str = f"Model Summary:\n"
    summary_str += f"{'='*50}\n"
    summary_str += f"Total Parameters: {total_params:,}\n"
    summary_str += f"Trainable Parameters: {trainable_params:,}\n"
    summary_str += f"{'='*50}\n"
    
    for layer in summary.values():
        summary_str += f"{layer['module_name']:<25} {str(layer['input_shape']):<20} {str(layer['output_shape']):<20} {layer['num_parameters']:,}\n"
    
    return summary_str


# Import OrderedDict for model summary
from collections import OrderedDict
