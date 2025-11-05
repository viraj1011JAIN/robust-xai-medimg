# tests/test_gradcam_signatures_and_targets.py
import torch
import torch.nn as nn

from src.xai.gradcam import GradCAM


class TinyBackbone(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.layer4 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, k)

    def forward(self, x):
        z = self.layer4(x)  # activation hook lives here
        g = self.avg(z).squeeze(-1).squeeze(-1)
        logits = self.fc(g)
        # Return tuple to cover "model returns (logits, aux)" branch
        return logits, {"aux": z.sum().detach()}


def test_generate_single_arg_and_class_idx_and_squeeze():
    m = TinyBackbone(k=5).eval()
    cam = GradCAM(m, target_layer_name="layer4")

    x = torch.rand(1, 3, 16, 16)
    # single-arg call; no class_idx -> sum
    out1 = cam.generate(x, squeeze=True)
    assert out1.ndim == 2 and out1.shape == (16, 16)

    # targeted class_idx path (just ensure it runs)
    out2 = cam.generate(x, class_idx=2, squeeze=True)
    assert out2.ndim == 2 and out2.shape == (16, 16)

    # two-arg signature (model, x) to hit that branch
    out3 = cam.generate(m, x, class_idx=4, squeeze=True)
    assert out3.ndim == 2 and out3.shape == (16, 16)


def test_generate_batch_and_no_squeeze():
    m = TinyBackbone(k=2).eval()
    cam = GradCAM(m, target_layer_name="layer4")
    x = torch.rand(2, 3, 12, 12)
    out = cam.generate(x, squeeze=False)
    assert out.ndim == 3 and out.shape == (2, 12, 12)
