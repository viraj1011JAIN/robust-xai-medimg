from __future__ import annotations

import argparse
import builtins
import html
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Tuple

import torch
from torch import nn

# --- expose __import__ so tests can patch src.train.baseline.__import__ ---
__import__ = builtins.__import__

__all__ = [
    "ModelCfg",
    "TinyMLP",
    "build_model",
    "_build_torchvision",
    "train_step",
    "_write_smoke_html",
    "_open_in_edge",
    "_parse_args",
    "main",
]


@dataclass
class ModelCfg:
    name: Literal["mlp"] = "mlp"
    in_ch: int = 3
    img_size: int = 224
    num_classes: int = 2
    hidden: int = 128


class TinyMLP(nn.Module):
    def __init__(
        self, in_ch: int = 3, img_size: int = 224, hidden: int = 128, num_classes: int = 2
    ):
        super().__init__()
        flat = int(in_ch) * int(img_size) * int(img_size)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _replace_first_conv_resnet(model: nn.Module, in_ch: int) -> None:
    conv1: nn.Conv2d = model.conv1  # type: ignore[attr-defined]
    if conv1.in_channels == in_ch:
        return
    new_conv = nn.Conv2d(
        in_ch,
        conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=(conv1.bias is not None),
    )
    if in_ch == 1 and conv1.weight.shape[1] == 3:
        with torch.no_grad():
            new_conv.weight.copy_(conv1.weight.mean(dim=1, keepdim=True))
            if conv1.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(conv1.bias)
    else:
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)
    model.conv1 = new_conv  # type: ignore[attr-defined]


def _replace_vit_head(m: nn.Module, num_classes: int) -> None:
    """
    Make torchvision ViT heads consistent across versions:
    - sometimes m.heads is nn.Linear
    - sometimes m.heads is nn.Sequential(head=nn.Linear)
    - fallback to m.hidden_dim (usually 768)
    """
    heads = getattr(m, "heads", None)

    if isinstance(heads, nn.Linear):
        in_feats = heads.in_features
        m.heads = nn.Linear(in_feats, num_classes)
        return

    if isinstance(heads, nn.Sequential):
        last_linear = None
        for mod in reversed(list(heads.children())):
            if isinstance(mod, nn.Linear):
                last_linear = mod
                break
        if last_linear is not None:
            m.heads = nn.Linear(last_linear.in_features, num_classes)
            return

    in_feats = getattr(m, "hidden_dim", 768)
    m.heads = nn.Linear(in_feats, num_classes)


def _build_torchvision(name: str, in_ch: int, num_classes: int) -> nn.Module:
    # Use the *module-level* __import__ so tests can patch it.
    try:
        tvm = __import__("torchvision.models", fromlist=["models"])
    except Exception as e:
        raise RuntimeError("torchvision is not available") from e

    n = (name or "").lower()
    if n == "resnet18":
        m = tvm.resnet18(weights=None)
        _replace_first_conv_resnet(m, in_ch)
        m.fc = nn.Linear(m.fc.in_features, num_classes)  # type: ignore[attr-defined]
        return m
    if n == "resnet50":
        m = tvm.resnet50(weights=None)
        _replace_first_conv_resnet(m, in_ch)
        m.fc = nn.Linear(m.fc.in_features, num_classes)  # type: ignore[attr-defined]
        return m
    if n == "vit_b16":
        m = tvm.vit_b_16(weights=None)
        if in_ch != 3:
            raise ValueError("vit_b16 only supported for in_ch=3 in this lightweight builder.")
        _replace_vit_head(m, num_classes)
        return m

    raise ValueError("Unknown torchvision model")


def build_model(
    name: Optional[str] = "mlp",
    in_ch: int = 3,
    img_size: int = 224,
    num_classes: int = 2,
    hidden: int = 128,
    num_out: Optional[int] = None,
) -> nn.Module:
    n = (name or "").lower()
    out = int(num_out) if num_out is not None else int(num_classes)

    if n in ("", "mlp"):
        return TinyMLP(in_ch=in_ch, img_size=img_size, hidden=hidden, num_classes=out)

    try:
        return _build_torchvision(n, in_ch=in_ch, num_classes=out)
    except RuntimeError:
        # Fallback when torchvision is unavailable
        return TinyMLP(in_ch=in_ch, img_size=img_size, hidden=hidden, num_classes=out)


@torch.no_grad()
def train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
    model.train(False)
    logits = model(x)
    if logits.ndim == 2 and logits.size(1) == 1:
        loss = nn.BCEWithLogitsLoss()(logits.squeeze(1), y.float())
        pred = (logits.squeeze(1) > 0).long()
    else:
        loss = nn.CrossEntropyLoss()(logits, y.long())
        pred = logits.argmax(dim=1)
    acc = (pred == y.long()).float().mean().item()
    return loss, float(acc)


def _write_smoke_html(path: str | os.PathLike, title: str, loss: float, acc: float) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    html_body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title></head>
<body>
<h1>{html.escape(title)}</h1>
<p><strong>loss</strong> = {loss:.4f}</p>
<p><strong>acc</strong> = {acc:.4f}</p>
</body></html>
"""
    p.write_text(html_body, encoding="utf-8")


def _open_in_edge(path: str) -> None:
    try:
        if os.name == "nt":
            os.system(f'start msedge "{path}"')
    except Exception:
        pass


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Run a tiny smoke forward")
    p.add_argument("--model", default="mlp", help="Model name (mlp/resnet18/resnet50/vit_b16)")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--in-ch", type=int, default=3)
    p.add_argument("--smoke-html", default="", help="Write smoke HTML report to this path")
    p.add_argument("--open-edge", action="store_true", help="Try to open the HTML in Edge")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    if args.smoke:
        torch.manual_seed(0)
        model = build_model(
            args.model,
            in_ch=args.in_ch,
            img_size=args.img_size,
            num_classes=args.num_classes,
            hidden=16,
        )
        x = torch.randn(4, args.in_ch, args.img_size, args.img_size)
        y = torch.randint(0, max(2, int(args.num_classes)), (4,))
        loss, acc = train_step(model, x, y)

        print(f"[SMOKE] loss={float(loss):.4f} acc={acc:.4f}")

        if args.smoke_html:
            _write_smoke_html(args.smoke_html, "Smoke Report", float(loss), float(acc))
            if args.open_edge:
                _open_in_edge(args.smoke_html)
        return 0
    else:
        print("[baseline] nothing to do (run with --smoke).")
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
