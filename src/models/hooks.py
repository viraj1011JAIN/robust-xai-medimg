from __future__ import annotations


class FeatureExtractor:
    def __init__(self, target_module):
        self.target_module = target_module
        self.activations = None
        self.gradients = None
        self.fwd_hook = target_module.register_forward_hook(self._save_activation)
        self.bwd_hook = target_module.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def close(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()
