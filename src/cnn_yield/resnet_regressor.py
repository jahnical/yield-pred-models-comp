from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet50
from data_util import _veg_indices
from reben_publication.BigEarthNetv2_0_ImageClassifier import (
    BigEarthNetv2_0_ImageClassifier,
)

@dataclass
class CNNCfg:
    lstm: bool = False
    lstm_hidden: int = 128
    lstm_layers: int = 3
    freeze_backbone: bool = True
    ckpt: str = "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0"
    head_hidden: int = 256
    dropout: float = 0.2

    @classmethod
    def from_optuna_trial(cls, trial) -> "CNNCfg":
        return cls(
            lstm=True,
            lstm_hidden=trial.suggest_int("lstm_hidden", 64, 512),
            lstm_layers=trial.suggest_int("lstm_layers", 1, 3),
            freeze_backbone=trial.suggest_categorical("freeze_backbone", [True, False]),
            head_hidden=trial.suggest_int("head_hidden", 64, 512),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            # ckpt can also be made tunable if needed
        )

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
                input_size=2048,
                hidden_size=cfg.lstm_hidden,
                num_layers=cfg.lstm_layers,
                batch_first=True,
                dropout=cfg.dropout,
            )
            head_in = cfg.lstm_hidden + 5
        else:
            self.temporal_pool = None
            head_in = 2048 + 5

        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for n, p in self.backbone.named_parameters():
                if n.startswith("0.conv1") or n.startswith("0.bn1"):
                    p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.seq_mode:
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
            feats = self.backbone(x_flat).view(B, T, -1)
            feats, _ = self.temporal_pool(feats)
            feats = feats[:, -1]
            idx = _veg_indices(x[:, -1])
        else:
            feats = self.backbone(x)
            idx = _veg_indices(x)
        feats = torch.cat([feats, idx], dim=1)
        return self.head(feats).squeeze(1)

    @staticmethod
    def _build_backbone(cfg: CNNCfg) -> nn.Module:
        wrapper = BigEarthNetv2_0_ImageClassifier.from_pretrained(cfg.ckpt)
        from timm.models.resnet import ResNet
        resnet = next(m for m in wrapper.modules() if isinstance(m, ResNet))
        for head_name in ("classifier", "fc", "head"):
            if hasattr(resnet, head_name):
                setattr(resnet, head_name, nn.Identity())
                print(f"ResNet head '{head_name}' replaced with Identity")
        class _ResNetFeat(nn.Module):
            def __init__(self, net: ResNet) -> None:
                super().__init__()
                self.net = net
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if hasattr(self.net, "forward_features"):
                    feats = self.net.forward_features(x)
                else:
                    feats = self.net(x)
                if feats.ndim == 4:
                    feats = feats.flatten(1)
                return feats
        backbone = _ResNetFeat(resnet)
        return backbone
