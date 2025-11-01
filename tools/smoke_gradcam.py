import argparse

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from torchvision.models import ResNet18_Weights, resnet18


def pick_device(requested: str | None) -> str:
    if requested:
        if requested == "cuda" and not torch.cuda.is_available():
            print("Requested CUDA but PyTorch was built without CUDA; falling back to CPU.")
            return "cpu"
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--device", choices=["cpu", "cuda"], default=None, help="Force device (default: cuda if available else cpu)"
    )
    args = p.parse_args()
    device = pick_device(args.device)

    # Load a small ImageNet model
    try:
        weights = ResNet18_Weights.IMAGENET1K_V1
    except Exception:
        weights = "IMAGENET1K_V1"
    model = resnet18(weights=weights).eval().to(device)

    # Random RGB image in [0,1], 224x224
    rgb_img = np.random.rand(224, 224, 3).astype(np.float32)
    input_tensor = preprocess_image(
        rgb_img.copy(),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ).to(device)

    target_layer = model.layer4[-1]

    # New API: no use_cuda kwarg, and prefer context manager
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor)[0]

    vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cv2.imwrite("gradcam_smoke.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved: gradcam_smoke.png | device: {device}")


if __name__ == "__main__":
    main()
