from __future__ import annotations
from dataclasses import dataclass

import torch, torch.nn as nn
from torchvision.models import resnet50
from reben_publication.BigEarthNetv2_0_ImageClassifier import (
    BigEarthNetv2_0_ImageClassifier,
)

# ── config dataclass ────────────────────────────────────────────
@dataclass
class CNNCfg:
    lstm:           bool  = False
    lstm_hidden:    int   = 128
    lstm_layers:    int   = 1
    freeze_backbone:bool  = True
    ckpt:           str   = "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0"

# ── model ───────────────────────────────────────────────────────
class ResNetYieldRegressor(nn.Module):
    """
    Input  | (B,10,32,32)   or (B,T,10,32,32) if lstm=True
    Output | (B,) predicted yield
    """

    def __init__(self, cfg: CNNCfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = self._build_backbone(cfg)
        self.seq_mode = cfg.lstm

        # LSTM pooling (optional)
        if cfg.lstm:
            self.temporal_pool = nn.LSTM(
                input_size=2048,            # resnet50 global-feat dim
                hidden_size=cfg.lstm_hidden,
                num_layers=cfg.lstm_layers,
                batch_first=True,
            )
            head_in = cfg.lstm_hidden
        else:
            self.temporal_pool = None
            head_in = 2048

        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # nothing to re-enable; if you still want a small trainable block,
            # un-freeze just `conv1` and its BatchNorm:
            for n, p in self.backbone.named_parameters():
                if n.startswith("0.conv1") or n.startswith("0.bn1"):
                    p.requires_grad = True


    # ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.seq_mode:                     # (B,T,C,H,W)
            b, t, c, h, w = x.shape
            feats = self.backbone(x.view(b * t, c, h, w))  # (B*T,2048)
            feats = feats.view(b, t, -1)
            feats, _ = self.temporal_pool(feats)
            feats = feats[:, -1]              # last time-step
        else:                                 # (B,C,H,W)
            feats = self.backbone(x)
        return self.head(feats).squeeze(1)

    # ----------------------------------------------------------
    @staticmethod
    def _build_backbone(cfg: CNNCfg) -> nn.Module:
        """
        Locate the ResNet-50 inside BigEarthNet's ConfigILM wrapper and
        return a backbone that outputs a flat 2048-D vector.
        """
        wrapper = BigEarthNetv2_0_ImageClassifier.from_pretrained(cfg.ckpt)

        # ❶ grab the first timm ResNet module anywhere in the wrapper tree
        from timm.models.resnet import ResNet
        resnet = next(m for m in wrapper.modules() if isinstance(m, ResNet))

        # ❷ drop whatever classifier head the checkpoint contains
        for head_name in ("classifier", "fc", "head"):
            if hasattr(resnet, head_name):
                setattr(resnet, head_name, nn.Identity())
                print(f"ResNet head '{head_name}' replaced with Identity")

        # ❸ helper guarantees (B,2048) output regardless of timm version
        class _ResNetFeat(nn.Module):
            def __init__(self, net: ResNet) -> None:
                super().__init__()
                self.net = net

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # timm >=0.9 provides forward_features; older versions do not
                if hasattr(self.net, "forward_features"):
                    feats = self.net.forward_features(x)   # (B,2048,1,1) or (B,2048)
                else:                                     # fallback to regular forward
                    feats = self.net(x)                   # (B,2048)
                # squeeze global-avg-pool if still 4-D
                if feats.ndim == 4:
                    feats = feats.flatten(1)
                return feats                              # (B,2048)

        backbone = _ResNetFeat(resnet)
        return backbone



