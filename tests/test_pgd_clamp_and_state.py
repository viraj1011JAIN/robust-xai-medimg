import torch

from src.attacks import pgd as pgd


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3 * 8 * 8, 2)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))


def test_pgd_clamp_and_state_restore():
    torch.manual_seed(0)
    model = Tiny()
    model.train(True)  # force training mode to check restore
    x = torch.rand(2, 3, 8, 8)
    y_onehot = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)

    # Use clamp to cover the optional clamping branch
    x_adv = pgd.pgd_attack(
        model,
        x,
        y_onehot,  # one-hot → should be converted via argmax
        eps=2 / 255,
        alpha=1 / 255,
        steps=3,
        random_start=True,
        clamp=(0.0, 1.0),  # cover clamp path
    )

    # shape preserved
    assert x_adv.shape == x.shape
    # model state restored to training mode
    assert model.training is True
