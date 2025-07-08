from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from data_util import _veg_indices
from reben_publication.BigEarthNetv2_0_ImageClassifier import (
    BigEarthNetv2_0_ImageClassifier,
)

# ---------------------------------------------------------------------
# ⚙️  CONFIG (easy to tweak from notebooks / Optuna)
# ---------------------------------------------------------------------
@dataclass
class VitConfig:
    img_size: int = 32
    in_chans: int = 10
    patch_size: int = 8
    lstm: bool = False
    lstm_hidden: int = 128
    lstm_layers: int = 3
    freeze_backbone: bool = True
    dropout: float = 0.2
    ckpt: str = "BIFOLD-BigEarthNetv2-0/vit_base_patch8_224-s2-v0.2.0"


# ---------------------------------------------------------------------
# 🧩  MODEL
# ---------------------------------------------------------------------
class ViTYieldRegressor(nn.Module):
    """Vision-Transformer backbone + optional LSTM head for yield regression.

    Input shape:
        * without LSTM –  (B, 10, 32, 32)
        * with    LSTM –  (B, T, 10, 32, 32)
    Output: (B, 1) predicted yield.
    """

    def __init__(self, cfg: VitConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = self._build_backbone(cfg)
        self.seq_mode = cfg.lstm

        emb_dim = self.backbone.embed_dim
        if cfg.lstm:
            self.temporal_pool = nn.LSTM(
                input_size=emb_dim,
                hidden_size=cfg.lstm_hidden,
                num_layers=cfg.lstm_layers,
                batch_first=True,
                dropout=cfg.dropout,
            )
            head_in = cfg.lstm_hidden + 5        # ← + five indices
        else:
            self.temporal_pool = None
            head_in = emb_dim + 5                # ← + five indices

        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 256),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 1),
        )

        # freeze
        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # always train patch-embed (channel-adapter)
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad = True

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.seq_mode:                                 # (B,T,C,H,W)
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)

            feats = self.backbone(x_flat)                 # (B*T,emb)
            feats = feats.view(B, T, -1)                  # (B,T,emb)
            feats, _ = self.temporal_pool(feats)          # keep hidden seq
            feats = feats[:, -1]                          # last timestep  (B,hidden)

            # vegetation indices from last timestep frame
            idx = _veg_indices(x[:, -1])                  # (B,5)
        else:                                             # (B,C,H,W)
            feats = self.backbone(x)                      # (B,emb)
            idx   = _veg_indices(x)                       # (B,5)

        feats = torch.cat([feats, idx], dim=1)            # ← concat indices
        return self.head(feats).squeeze(1)

    # -----------------------------------------------------------------
    # helper: build + weight-transfer
    # -----------------------------------------------------------------
    @staticmethod
    def _build_backbone(cfg: VitConfig) -> VisionTransformer:
        """Create a 32×32 / 10-channel ViT and copy pretrained weights."""
        # 1️⃣ load checkpoint wrapper and grab the timm ViT
        wrapper = BigEarthNetv2_0_ImageClassifier.from_pretrained(cfg.ckpt)
        old_vit = next(m for m in wrapper.modules() if isinstance(m, VisionTransformer))

        # 2️⃣ instantiate new ViT
        new_vit = VisionTransformer(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=old_vit.embed_dim,
            depth=len(old_vit.blocks),
            num_heads=old_vit.blocks[0].attn.num_heads,
            mlp_ratio=old_vit.mlp_ratio if hasattr(old_vit, "mlp_ratio") else 4,
            qkv_bias=old_vit.blocks[0].attn.qkv.bias is not None,
            num_classes=0,
        )

        # 3️⃣ copy everything except pos-embed & first-layer kernel
        state = {k: v for k, v in old_vit.state_dict().items()
                 if not k.startswith(("pos_embed", "patch_embed.proj.weight"))}
        new_vit.load_state_dict(state, strict=False)

        # resize positional-embedding
        with torch.no_grad():
            cls_tok, grid = old_vit.pos_embed[:, :1], old_vit.pos_embed[:, 1:]
            side_old = int(grid.shape[1] ** 0.5)
            side_new = cfg.img_size // cfg.patch_size
            grid = grid.reshape(1, side_old, side_old, -1).permute(0, 3, 1, 2)
            grid = nn.functional.interpolate(grid, size=(side_new, side_new), mode="bilinear")
            grid = grid.permute(0, 2, 3, 1).reshape(1, side_new * side_new, -1)
            new_vit.pos_embed.copy_(torch.cat([cls_tok, grid], dim=1))
            # copy first 10 channels of patch-embed kernel
            new_vit.patch_embed.proj.weight.copy_(old_vit.patch_embed.proj.weight[:, : cfg.in_chans])

        return new_vit