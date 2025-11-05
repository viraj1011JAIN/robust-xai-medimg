import pytest
import torch

from src.xai.gradcam import GradCAM


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.head = torch.nn.Linear(4 * 8 * 8, 2)

    def forward(self, x):
        f = self.layer4(x)
        # produce 1-D logits so GradCAM hits "unsqueeze if 1-D" branch
        logits = self.head(f.flatten(1)).max(dim=-1).values  # shape [B]
        return logits, f


def test_gradcam_bad_layer_raises():
    m = Toy().eval()
    with pytest.raises(ValueError):
        _ = GradCAM(m, target_layer_name="not_there")


def test_gradcam_unsqueeze_and_generate_then_remove():
    m = Toy().eval()
    cam = GradCAM(m, target_layer_name="layer4")
    x = torch.rand(1, 3, 8, 8)
    out = cam.generate(m, x)
    assert out.shape == (8, 8)
    cam.remove()  # covers remove hooks
