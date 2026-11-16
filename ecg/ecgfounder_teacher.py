"""ECGFounder teacher wrapper for KD experiments.

Net1D backbone: 1024-D pooled features (`deep_features` in net1d.py).
12-lead variant: 12 leads @ 500Hz.
1-lead variant : 1 lead  @ 500Hz.

Usage:
    teacher = ECGFounderTeacher(n_leads=12, device='cuda')
    z = teacher(x)   # (B, 1024) float32, no_grad, half-precision internal
"""
from __future__ import annotations
import sys, os, torch, torch.nn as nn

# Make ECGFounder net1d importable.
_ECGF_DIR = "/home/xiaoma/foundation_model/ecg/ECGFounder"
if _ECGF_DIR not in sys.path:
    sys.path.insert(0, _ECGF_DIR)
from net1d import Net1D  # noqa: E402

CKPT_12LEAD = "/home/xiaoma/foundation_model/ecg/ckpts/ecgfounder/12_lead_ECGFounder.pth"
CKPT_1LEAD  = "/home/xiaoma/foundation_model/ecg/ckpts/ecgfounder/1_lead_ECGFounder.pth"

ECGFOUNDER_DIM = 1024  # last filter_list element

def _build_net1d(in_channels: int, n_classes: int = 150) -> Net1D:
    """Mirror ft_*_ECGFounder construction (use_bn=False, use_do=False)."""
    return Net1D(
        in_channels=in_channels,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=n_classes,
        return_features=True,
    )


class ECGFounderTeacher(nn.Module):
    """Frozen ECGFounder, returns 1024-D pooled features."""

    def __init__(self, n_leads: int = 12, device: str = "cuda", half: bool = True):
        super().__init__()
        self.n_leads = n_leads
        self.device = device
        self.half = half
        ckpt_path = CKPT_12LEAD if n_leads == 12 else CKPT_1LEAD
        assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"

        # ECGFounder pretrained head is 150-class.
        self.model = _build_net1d(in_channels=n_leads, n_classes=150)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        # Strip 'module.' if present
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if unexpected:
            # Filter out final-classifier mismatches we don't need.
            unexpected = [k for k in unexpected if not k.startswith("dense.")]
        print(f"[ECGFounderTeacher] loaded {ckpt_path}  missing={len(missing)} unexpected={len(unexpected)}")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        if self.half:
            self.model = self.model.half()
        self.model.to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_leads, T) float32 → (B, 1024) float32."""
        x = x.to(self.device, non_blocking=True)
        if self.half:
            x = x.half()
        out, deep_features = self.model(x)
        return deep_features.float()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


if __name__ == "__main__":
    t = ECGFounderTeacher(n_leads=12, device="cuda")
    print(f"params={t.n_params()/1e6:.2f}M")
    z = t(torch.randn(4, 12, 2500))
    print(f"output: {z.shape}, dtype={z.dtype}")
