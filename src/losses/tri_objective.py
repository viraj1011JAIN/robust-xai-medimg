# src/losses/tri_objective.py
import torch
import torch.nn as nn
from pytorch_msssim import ssim


class TriObjectiveLoss:
    """
    L_total = L_task + λ_rob * L_rob + λ_expl * L_expl

    - L_task: BCEWithLogits on clean inputs
    - L_rob:  same loss on adversarial inputs (FGSM/PGD), if attacker is provided
    - L_expl: 1 - SSIM(CAM_clean, CAM_adv) computed sparsely for memory
    """

    def __init__(
        self,
        lambda_rob: float = 1.0,
        lambda_expl: float = 0.5,
        attacker=None,  # callable(model, x, y) -> x_adv
        gradcam=None,  # object with .generate(x) -> [B,H,W] in [0,1]
        expl_freq: int = 10,  # compute explanation loss every N steps
        expl_subsample: float = 0.25,  # probability to apply when step matches
    ):
        self.lambda_rob = float(lambda_rob)
        self.lambda_expl = float(lambda_expl)
        self.attacker = attacker
        self.gradcam = gradcam
        self.expl_freq = int(expl_freq)
        self.expl_subsample = float(expl_subsample)
        self.task_loss_fn = nn.BCEWithLogitsLoss()
        self._step = 0

    @torch.no_grad()
    def _cams(self, x, x_adv):
        cam_clean = self.gradcam.generate(x).clamp(0, 1)
        cam_adv = self.gradcam.generate(x_adv).clamp(0, 1)
        return cam_clean, cam_adv

    def __call__(self, model, x, y):
        """
        Args:
            model: nn.Module
            x: float32 images in [0,1], shape [B,C,H,W] on correct device
            y: labels, shape [B] or [B,] (binary 0/1)
        Returns:
            loss_total, metrics dict
        """
        self._step += 1
        metrics = {}

        # 1) Task loss (clean)
        logits_clean = model(x).squeeze()
        yf = y.float()
        loss_task = self.task_loss_fn(logits_clean, yf)
        metrics["loss_task"] = float(loss_task.detach())

        # 2) Robust loss (adversarial)
        if self.attacker is not None:
            x_adv = self.attacker(model, x, y)
            logits_adv = model(x_adv).squeeze()
            loss_rob = self.task_loss_fn(logits_adv, yf)
            metrics["loss_rob"] = float(loss_rob.detach())
        else:
            x_adv = None
            loss_rob = x.new_tensor(0.0)
            metrics["loss_rob"] = 0.0

        # 3) Explanation stability (sparse to save VRAM/compute)
        loss_expl = x.new_tensor(0.0)
        do_expl = (
            self.gradcam is not None
            and self._step % max(self.expl_freq, 1) == 0
            and torch.rand(1, device=x.device).item()
            < max(min(self.expl_subsample, 1.0), 0.0)
            and x_adv is not None
        )
        if do_expl:
            with torch.no_grad():
                cam_clean, cam_adv = self._cams(x.detach(), x_adv.detach())
                # SSIM over [B,1,H,W]
                ssim_val = ssim(
                    cam_clean.unsqueeze(1),
                    cam_adv.unsqueeze(1),
                    data_range=1.0,
                    size_average=True,
                )
            loss_expl = 1.0 - ssim_val
            metrics["ssim"] = float(ssim_val.detach())
        metrics["loss_expl"] = float(loss_expl.detach())

        loss_total = (
            loss_task + self.lambda_rob * loss_rob + self.lambda_expl * loss_expl
        )
        metrics["loss_total"] = float(loss_total.detach())
        return loss_total, metrics
