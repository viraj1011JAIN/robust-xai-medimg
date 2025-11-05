import pytest
import torch
from torchvision.models import resnet18

from src.xai.gradcam import GradCAM


def test_generate_two_args_first_not_module_raises_typeerror():
    m = resnet18(weights=None).eval()
    gc = GradCAM(m, target_layer_name="layer4")
    x = torch.randn(1, 3, 8, 8)
    # Call generate with two args where the first is NOT a Module -> hit the TypeError path
    with pytest.raises(TypeError):
        gc.generate("not_a_module", x)  # triggers the branch around line 52
