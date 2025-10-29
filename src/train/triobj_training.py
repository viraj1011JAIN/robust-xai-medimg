from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # pip install pytorch-msssim
    from pytorch_msssim import ssim

    _HAS_SSIM = True
except Exception:
    _HAS_SSIM = False


class TriObjectiveLoss:
    """
    L_total = L_task + λ_rob * L_rob + λ_expl * L_expl

    - L_task: BCEWithLogits on clean inputs
    - L_rob : BCEWithLogits on adversarial inputs (via attacker(model, x, y))
    - L_expl: 1 - SSIM(CAM_clean, CAM_adv), computed sparsely to save memory
    """

    def __init__(
        self,
        lambda_rob: float = 1.0,
        lambda_expl: float = 0.5,
        attacker: Optional[object] = None,
        gradcam: Optional[object] = None,
        expl_freq: int = 10,  # compute explanation loss every N steps
        expl_subsample: float = 0.25,  # probability to compute when freq hits
    ) -> None:
        self.lambda_rob = float(lambda_rob)
        self.lambda_expl = float(lambda_expl)

        self.attacker = attacker
        self.gradcam = gradcam

        self.expl_freq = max(int(expl_freq), 1)
        self.expl_subsample = float(expl_subsample)

        self.task_loss_fn = nn.BCEWithLogitsLoss()
        self._step = 0

    def _maybe_expl_loss(self, x_clean: torch.Tensor, x_adv: Optional[torch.Tensor]) -> torch.Tensor:
        if self.gradcam is None or x_adv is None:
            return x_clean.new_zeros(())

        # frequency + subsample gate (saves memory on small GPUs)
        self._step += 1
        if (self._step % self.expl_freq) != 0:
            return x_clean.new_zeros(())
        if torch.rand(1, device=x_clean.device).item() > self.expl_subsample:
            return x_clean.new_zeros(())

        # Produce CAMs WITHOUT tracking grads to keep memory low
        with torch.no_grad():
            # Expect [B, H, W] in [0, 1]
            cam_clean = self.gradcam.generate(x_clean.detach())
            cam_adv = self.gradcam.generate(x_adv.detach())

        if not _HAS_SSIM:
            # Fallback: simple L1 distance between heatmaps
            return F.l1_loss(cam_clean, cam_adv)

        # SSIM expects [B, 1, H, W], range [0, 1]
        ssim_val = ssim(
            cam_clean.unsqueeze(1),
            cam_adv.unsqueeze(1),
            data_range=1.0,
            size_average=True,
        )
        return 1.0 - ssim_val  # higher is worse → minimize

    def __call__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor):
        """
        Args:
            model: nn.Module
            x:     [B, 3, H, W] in [0, 1]
            y:     [B] {0,1}

        Returns:
            (loss_total: torch.Tensor, metrics: dict)
        """
        metrics = {}

        # 1) Task loss (clean)
        logits_clean = model(x).squeeze()
        loss_task = self.task_loss_fn(logits_clean, y.float())
        metrics["loss_task"] = float(loss_task.detach())

        # 2) Robust loss (adversarial)
        x_adv = None
        if self.attacker is not None:
            # Attacker must do FP32 internally; returns [0, 1]
            x_adv = self.attacker(model, x, y).detach()
            logits_adv = model(x_adv).squeeze()
            loss_rob = self.task_loss_fn(logits_adv, y.float())
            metrics["loss_rob"] = float(loss_rob.detach())
        else:
            loss_rob = x.new_zeros(())

        # 3) Explanation stability (optional, sparse)
        loss_expl = self._maybe_expl_loss(x, x_adv)
        metrics["loss_expl"] = float(loss_expl.detach())

        loss_total = loss_task + self.lambda_rob * loss_rob + self.lambda_expl * loss_expl
        metrics["loss_total"] = float(loss_total.detach())
        return loss_total, metrics
