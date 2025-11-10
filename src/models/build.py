"""Model building utilities with timm support."""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    timm = None


class FeatureExtractor:
    """Hook to capture intermediate features from a model."""
    
    def __init__(self):
        self.features = None
        self.hook = None
    
    def __call__(self, module, input, output):
        self.features = output.detach()
    
    def attach(self, model, layer_name='layer4'):
        """Attach hook to specified layer."""
        for name, module in model.named_modules():
            if name == layer_name:
                self.hook = module.register_forward_hook(self)
                return
        raise ValueError(f"Layer {layer_name} not found in model")
    
    def detach(self):
        """Remove the hook."""
        if self.hook is not None:
            self.hook.remove()
            self.hook = None


def build_model(
    architecture: str,
    num_classes: int = 1,
    pretrained: bool = False,
    **kwargs
) -> nn.Module:
    """
    Build a model using timm library.
    
    Args:
        architecture: Model architecture name (e.g., 'resnet18', 'efficientnet_b0')
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
        **kwargs: Additional arguments passed to timm.create_model
    
    Returns:
        PyTorch model ready for training/inference
    
    Raises:
        ImportError: If timm is not installed
        ValueError: If architecture is not supported
    """
    if not TIMM_AVAILABLE:
        raise ImportError(
            "timm library is required for model building. "
            "Install with: pip install timm"
        )
    
    architecture = architecture.lower()
    
    # Map common names
    arch_map = {
        'resnet18': 'resnet18',
        'resnet34': 'resnet34',
        'resnet50': 'resnet50',
        'efficientnet_b0': 'efficientnet_b0',
        'efficientnet_b1': 'efficientnet_b1',
        'vit_base_patch16_224': 'vit_base_patch16_224',
        'vit_small_patch16_224': 'vit_small_patch16_224',
    }
    
    if architecture not in arch_map:
        raise ValueError(
            f"Architecture '{architecture}' not supported. "
            f"Supported: {list(arch_map.keys())}"
        )
    
    timm_name = arch_map[architecture]
    
    try:
        # Create model with timm
        model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=num_classes,
            **kwargs
        )
    except Exception as e:
        raise ValueError(f"Failed to create model '{architecture}': {e}")
    
    # Attach feature extractor for models with layer4 (ResNets)
    if hasattr(model, 'layer4'):
        feature_extractor = FeatureExtractor()
        try:
            feature_extractor.attach(model, 'layer4')
            model.feature_extractor = feature_extractor
        except ValueError:
            # If layer4 doesn't exist, just skip
            pass
    
    # Set to eval mode by default
    model.eval()
    
    return model