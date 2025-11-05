from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
from torch import nn

__all__ = ["build_model", "main"]


@dataclass
class ModelCfg:
    name: Literal["mlp", "resnet18", "resnet50", "vit_b16"] = "mlp"
    in_ch: int = 3
    img_size: int = 224
    num_classes: int = 2
    hidden: int = 128


class TinyMLP(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        img_size: int = 224,
        hidden: int = 128,
        num_classes: int = 2,
    ):
        super().__init__()
        flat = in_ch * img_size * img_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(
    name: Literal["mlp", "resnet18", "resnet50", "vit_b16"] = "mlp",
    in_ch: int = 3,
    img_size: int = 224,
    num_classes: int = 2,
    hidden: int = 128,
    *,
    num_out: Optional[int] = None,
) -> nn.Module:
    """
    Returns a model instance. For Phase 0 + smoke, default is a tiny MLP.
    `num_out` is accepted as an alias for `num_classes` (tests use this).
    """
    out_ch = int(num_out) if num_out is not None else int(num_classes)
    name = (name or "mlp").lower()

    if name == "mlp":
        return TinyMLP(in_ch=in_ch, img_size=img_size, hidden=hidden, num_classes=out_ch)

    if name == "resnet18":
        try:
            from torchvision.models import resnet18

            m = resnet18(weights=None)
            if in_ch != 3:
                m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
                nn.init.kaiming_normal_(m.conv1.weight, mode="fan_out", nonlinearity="relu")
            if out_ch != 1000:
                m.fc = nn.Linear(m.fc.in_features, out_ch)
            return m
        except Exception as e:
            raise RuntimeError("torchvision not available or failed to import resnet18") from e

    if name == "resnet50":
        try:
            from torchvision.models import resnet50

            m = resnet50(weights=None)
            if in_ch != 3:
                m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
                nn.init.kaiming_normal_(m.conv1.weight, mode="fan_out", nonlinearity="relu")
            if out_ch != 1000:
                m.fc = nn.Linear(m.fc.in_features, out_ch)
            return m
        except Exception as e:
            raise RuntimeError("torchvision not available or failed to import resnet50") from e

    if name == "vit_b16":
        try:
            from torchvision.models import vit_b_16

            m = vit_b_16(weights=None)
            if out_ch != 1000:
                m.heads.head = nn.Linear(m.heads.head.in_features, out_ch)
            return m
        except Exception as e:
            raise RuntimeError("torchvision ViT not available; use 'mlp' for smoke.") from e

    raise ValueError(f"Unknown model name: {name!r}")


def train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
    model.train(True)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.0)
    loss_fn = nn.CrossEntropyLoss()

    opt.zero_grad(set_to_none=True)
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    opt.step()
    acc = (logits.argmax(dim=-1) == y).float().mean().item()
    return loss.detach(), acc


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Baseline trainer (smoke-safe).")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny CPU-only smoke test and exit 0 on success.",
    )
    p.add_argument(
        "--model",
        default="mlp",
        choices=["mlp", "resnet18", "resnet50", "vit_b16"],
        help="Model name.",
    )
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--in-ch", type=int, default=3)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    if args.smoke:
        torch.manual_seed(0)
        model = build_model(
            name="mlp",
            in_ch=args.in_ch,
            img_size=args.img_size,
            num_classes=args.num_classes,
        )
        x = torch.rand(4, args.in_ch, args.img_size, args.img_size)
        y = torch.randint(0, args.num_classes, (4,))
        loss, acc = train_step(model, x, y)
        print(f"[SMOKE] loss={float(loss):.4f} acc={acc:.3f}")
        return 0

    print(
        "Baseline runner is ready. Use --smoke for a quick check, or Phase 1 configs for real training."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
