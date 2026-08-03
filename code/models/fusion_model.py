"""
Multi-modal fusion model for SND1 inhibitor discovery.

Architecture:
  MolCLR(512)   ──→ Proj(256) + LayerNorm ──┐
                                              ├─ SEBlockFusion ──→ Classifier
  Uni-Mol2(768)  ──→ Proj(256) + LayerNorm ──┘

SE-Block: per-dimension gating (not scalar), ~8K extra params.
Contrastive auxiliary loss aligns feature spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── Components ─────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + x)


class SEBlockFusion(nn.Module):
    """
    Per-dimension squeeze-excitation fusion.

    For each dimension d, learns a gate σ_d ∈ [0,1]:
      fused_d = σ_d * molclr_d + (1-σ_d) * unimol_d
      σ = sigmoid(FC(ReLU(FC([molclr; unimol]))))

    ~8K params at embed_dim=256 (reduction=4).
    """

    def __init__(self, embed_dim: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim // reduction),
            nn.ReLU(),
            nn.Linear(embed_dim // reduction, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, m_embed, u_embed):
        combined = torch.cat([m_embed, u_embed], dim=1)
        gate = self.fc(combined)           # (B, D)
        fused = gate * m_embed + (1 - gate) * u_embed
        return fused, gate                 # gate → 1 = molclr, 0 = unimol


class ScalarGateFusion(nn.Module):
    """Original scalar gate: α = sigmoid(MLP([m; u])), fused = α*m + (1-α)*u."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, m_embed, u_embed):
        combined = torch.cat([m_embed, u_embed], dim=1)
        alpha = self.gate(combined)        # (B, 1)
        return alpha * m_embed + (1 - alpha) * u_embed, alpha


# ── Loss helpers ───────────────────────────────────────────────────

def contrastive_alignment_loss(m_embed, u_embed):
    """Encourage projected embeddings of the same molecule to align."""
    cos = F.cosine_similarity(m_embed, u_embed, dim=1)
    return (1.0 - cos).mean()


def mixup_features(x_m, x_u, y, alpha=0.4):
    """Feature-level mixup: convex combination of random pairs."""
    if alpha <= 0:
        return x_m, x_u, y
    batch_size = x_m.size(0)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam = max(lam, 1.0 - lam)  # ensure minority class dominates mix
    idx = torch.randperm(batch_size, device=x_m.device)
    x_m_mix = lam * x_m + (1 - lam) * x_m[idx]
    x_u_mix = lam * x_u + (1 - lam) * x_u[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return x_m_mix, x_u_mix, y_mix


# ── Main Model ─────────────────────────────────────────────────────

class FusionModel(nn.Module):
    """
    SE-Block gated fusion + classifier for MolCLR + Uni-Mol2 features.

    Parameters
    ----------
    molclr_dim : int (512)
    unimol_dim : int (1536)
    embed_dim : int (256)
    dropout_rate : float
    fusion_type : 'se_block' | 'scalar_gate'
    """

    def __init__(
        self,
        molclr_dim: int = 512,
        unimol_dim: int = 1536,
        embed_dim: int = 256,
        output_dim: int = 1,
        dropout_rate: float = 0.2,
        fusion_type: str = "se_block",
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.molclr_proj = nn.Sequential(
            nn.Linear(molclr_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.unimol_proj = nn.Sequential(
            nn.Linear(unimol_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

        if fusion_type == "se_block":
            self.fusion = SEBlockFusion(embed_dim)
        else:
            self.fusion = ScalarGateFusion(embed_dim)

        half_dim = embed_dim // 2
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, half_dim),
            nn.BatchNorm1d(half_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(half_dim, half_dim),
            nn.BatchNorm1d(half_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(half_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, molclr_vec, unimol_vec, return_embeds=False):
        m_embed = self.molclr_proj(molclr_vec)
        u_embed = self.unimol_proj(unimol_vec)
        fused, gate = self.fusion(m_embed, u_embed)
        pred = self.classifier(fused)
        if return_embeds:
            return pred, gate, m_embed, u_embed
        return pred, gate


# ── Ablation Variants ────────────────────────────────────────────

def build_ablation(variant: str, **kwargs) -> nn.Module:
    molclr_dim = kwargs.get("molclr_dim", 512)
    unimol_dim = kwargs.get("unimol_dim", 1536)
    embed_dim = kwargs.get("embed_dim", 256)
    output_dim = kwargs.get("output_dim", 1)
    dropout = kwargs.get("dropout_rate", 0.2)
    fusion_type = kwargs.get("fusion_type", "se_block")

    if variant == "full":
        return FusionModel(**kwargs)

    if variant == "scalar_gate":
        return FusionModel(molclr_dim=molclr_dim, unimol_dim=unimol_dim,
                          embed_dim=embed_dim, output_dim=output_dim,
                          dropout_rate=dropout, fusion_type="scalar_gate")

    if variant == "simple_concat":
        return _SimpleConcat(molclr_dim, unimol_dim, embed_dim, output_dim, dropout)

    if variant == "molclr_only":
        return _SingleModality(molclr_dim, embed_dim, output_dim, dropout)

    if variant == "unimol_only":
        return _SingleModality(unimol_dim, embed_dim, output_dim, dropout)

    if variant == "no_gate":
        return _NoGate(molclr_dim, unimol_dim, embed_dim, output_dim, dropout)

    raise ValueError(f"Unknown variant: {variant}")


class _SimpleConcat(nn.Module):
    def __init__(self, molclr_dim, unimol_dim, embed_dim, output_dim, dropout):
        super().__init__()
        half_dim = embed_dim // 2
        self.net = nn.Sequential(
            nn.Linear(molclr_dim + unimol_dim, embed_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, half_dim), nn.BatchNorm1d(half_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(half_dim, output_dim), nn.Sigmoid(),
        )

    def forward(self, m, u):
        return self.net(torch.cat([m, u], dim=1)), None


class _SingleModality(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, dropout):
        super().__init__()
        half_dim = embed_dim // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, half_dim), nn.BatchNorm1d(half_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(half_dim, output_dim), nn.Sigmoid(),
        )

    def forward(self, x, _=None):
        return self.net(x), None


class _NoGate(nn.Module):
    def __init__(self, molclr_dim, unimol_dim, embed_dim, output_dim, dropout):
        super().__init__()
        self.molclr_proj = nn.Sequential(
            nn.Linear(molclr_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        self.unimol_proj = nn.Sequential(
            nn.Linear(unimol_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        half_dim = embed_dim // 2
        self.cls = nn.Sequential(
            nn.Linear(embed_dim, half_dim), nn.BatchNorm1d(half_dim), nn.ReLU(),
            ResidualBlock(half_dim, dropout), nn.Dropout(dropout),
            nn.Linear(half_dim, 1), nn.Sigmoid(),
        )

    def forward(self, m, u):
        fused = self.molclr_proj(m) + self.unimol_proj(u)
        return self.cls(fused), None
