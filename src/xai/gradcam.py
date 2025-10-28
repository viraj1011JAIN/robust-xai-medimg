import torch
import torch.nn.functional as F


class GradCAM:
    """
    Minimal Grad-CAM for ResNet-like models.
    Uses first-order gradients (memory-friendly).
    """

    def __init__(self, model, target_layer_name="layer4"):
        self.model = model
        modules = dict(model.named_modules())
        if target_layer_name not in modules:
            raise ValueError(
                f"Layer {target_layer_name} not found. Available: {list(modules.keys())[:20]} ..."
            )
        self.layer = modules[target_layer_name]
        self.activations = None
        self.gradients = None
        self.fh = self.layer.register_forward_hook(self._fwd)
        self.bh = self.layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.activations = out.detach()

    def _bwd(self, module, gin, gout):
        self.gradients = gout[0].detach()

    def generate(self, x):
        self.model.eval()
        x = x.detach().requires_grad_(True)
        logits = self.model(x).squeeze()
        if logits.ndim == 0:
            logits = logits.unsqueeze(0)
        logits.sum().backward()

        acts = self.activations  # [B, C, h, w]
        grads = self.gradients  # [B, C, h, w]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)  # [B, h, w]
        cam = F.relu(cam)
        # Normalize to [0,1]
        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam = F.interpolate(
            cam.unsqueeze(1), size=x.shape[2:], mode="bilinear", align_corners=False
        ).squeeze(1)
        return cam

    def remove_hooks(self):
        self.fh.remove()
        self.bh.remove()
