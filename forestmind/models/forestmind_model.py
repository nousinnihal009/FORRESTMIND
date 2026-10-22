"""
ForestMind — Full Model Assembly.

Assembles all components into a single end-to-end model:
  1. Foundation Model (encoder) — per-timestep feature extraction
  2. Multimodal Fusion — enriches embeddings with context
  3. Temporal Transformer — captures degradation dynamics
  4. Prediction Heads — health, attribution, trajectory

The model processes multi-temporal satellite data and produces
comprehensive forest state assessment and future predictions.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor, nn

from forestmind.models.foundation.base import FoundationModelBase
from forestmind.models.foundation.custom_vit import CustomViTEncoder
from forestmind.models.fusion.multimodal import MultimodalFusion
from forestmind.models.heads.attribution import CauseAttributionHead
from forestmind.models.heads.health import ForestHealthHead
from forestmind.models.heads.trajectory import TrajectoryHead
from forestmind.models.temporal.transformer import TemporalTransformer

logger = logging.getLogger(__name__)


def _build_foundation_model(config: dict) -> FoundationModelBase:
    """Factory: create the specified foundation model from config."""
    name = config.get("name", "custom_vit")

    if name == "custom_vit":
        vit_cfg = config.get("custom_vit", {})
        return CustomViTEncoder(
            in_channels=14,  # 12 optical + 2 SAR
            embed_dim=config.get("embedding_dim", 256),
            patch_size=vit_cfg.get("patch_embed_size", 16),
            depth=vit_cfg.get("depth", 6),
            num_heads=vit_cfg.get("num_heads", 8),
            mlp_ratio=vit_cfg.get("mlp_ratio", 4.0),
            drop_rate=vit_cfg.get("drop_rate", 0.1),
        )
    elif name == "prithvi":
        from forestmind.models.foundation.prithvi import PrithviAdapter
        prithvi_cfg = config.get("prithvi", {})
        return PrithviAdapter(
            model_name=prithvi_cfg.get("model_name", "prithvi_eo_v2_300"),
            target_dim=config.get("embedding_dim", 256),
            freeze_backbone=prithvi_cfg.get("freeze", True),
            use_proxy=prithvi_cfg.get("use_proxy", False),
            pretrained=prithvi_cfg.get("pretrained", True),
        )
    elif name == "terramind":
        from forestmind.models.foundation.terramind import TerraMindAdapter
        terra_cfg = config.get("terramind", {})
        return TerraMindAdapter(
            model_name=terra_cfg.get("model_name", "terramind_v1_base"),
            target_dim=config.get("embedding_dim", 256),
            freeze_backbone=terra_cfg.get("freeze", True),
            use_proxy=terra_cfg.get("use_proxy", False),
            pretrained=terra_cfg.get("pretrained", True),
            sar_modality=terra_cfg.get("sar_modality", "S1GRD"),
        )
    else:
        raise ValueError(f"Unknown foundation model: {name}")


class ForestMindModel(nn.Module):
    """End-to-end ForestMind model.
    
    Pipeline:
      Input: (B, T, C, H, W) multi-temporal satellite data
               + (B, T) timestamps
      
      → Foundation Model: per-timestep encoding
        (B, T, C, H, W) → (B, T, D)
      
      → Multimodal Fusion: context enrichment
        (B, T, D) → (B, T, D)
      
      → Temporal Transformer: temporal modeling
        (B, T, D) → sequence (B, T, D), summary (B, D)
      
      → Prediction Heads:
        Health:      summary → {state, score}
        Attribution: summary → {drivers, strengths}
        Trajectory:  sequence + summary → {future scores, uncertainty}
    
    Args:
        config: Full configuration dictionary.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        foundation_cfg = config.get("foundation", {})
        fusion_cfg = config.get("fusion", {})
        temporal_cfg = config.get("temporal", {})
        heads_cfg = config.get("heads", {})

        embed_dim = foundation_cfg.get("embedding_dim", 256)

        # ─── Foundation Model ──────────────────────────────
        self.foundation = _build_foundation_model(foundation_cfg)
        logger.info(
            f"Foundation model: {foundation_cfg.get('name', 'custom_vit')} "
            f"({self.foundation.num_parameters:,} params, "
            f"{self.foundation.num_trainable_parameters:,} trainable)"
        )

        # ─── Multimodal Fusion ─────────────────────────────
        self.fusion = MultimodalFusion(
            embed_dim=embed_dim,
            method=fusion_cfg.get("method", "cross_attention"),
            num_heads=fusion_cfg.get("num_heads", 4),
            num_layers=fusion_cfg.get("num_layers", 2),
            dropout=fusion_cfg.get("dropout", 0.1),
            climate_dim=fusion_cfg.get("climate_dim"),
            activity_dim=fusion_cfg.get("activity_dim"),
        )

        # ─── Temporal Transformer ──────────────────────────
        self.temporal = TemporalTransformer(
            embed_dim=embed_dim,
            depth=temporal_cfg.get("depth", 4),
            num_heads=temporal_cfg.get("num_heads", 8),
            dim_feedforward=temporal_cfg.get("dim_feedforward", 512),
            dropout=temporal_cfg.get("dropout", 0.1),
            max_seq_len=temporal_cfg.get("max_seq_len", 48),
            use_continuous_pe=temporal_cfg.get("use_continuous_pe", True),
            causal=temporal_cfg.get("causal", False),
        )

        # ─── Prediction Heads ─────────────────────────────
        health_cfg = heads_cfg.get("health", {})
        self.health_head = ForestHealthHead(
            embed_dim=embed_dim,
            num_classes=health_cfg.get("num_classes", 5),
            predict_continuous=health_cfg.get("continuous_score", True),
        )

        attr_cfg = heads_cfg.get("attribution", {})
        self.attribution_head = CauseAttributionHead(
            embed_dim=embed_dim,
            num_drivers=attr_cfg.get("num_drivers", 8),
            predict_strength=True,
        )

        traj_cfg = heads_cfg.get("trajectory", {})
        self.trajectory_head = TrajectoryHead(
            embed_dim=embed_dim,
            num_horizons=len(traj_cfg.get("horizons", [30, 180, 365, 1095, 1825])),
            horizon_days=traj_cfg.get("horizons", [30, 180, 365, 1095, 1825]),
            predict_uncertainty=traj_cfg.get("predict_uncertainty", True),
        )

        # Log total parameters
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"ForestMind total: {total:,} params ({trainable:,} trainable)")

    def forward(
        self,
        optical: Tensor,
        sar: Tensor,
        timestamps: Tensor,
        observation_mask: Tensor | None = None,
        optical_mask: Tensor | None = None,
        sar_mask: Tensor | None = None,
        quality_mask: Tensor | None = None,
        climate_context: Tensor | None = None,
        activity_context: Tensor | None = None,
        climate_mask: Tensor | None = None,
        activity_mask: Tensor | None = None,
        sar_quality_mask: Tensor | None = None,
    ) -> dict[str, dict[str, Tensor]]:
        """Full forward pass through the ForestMind pipeline.
        
        Args:
            optical: (B, T, 12, H, W) multi-spectral optical data.
            sar: (B, T, 2, H, W) SAR backscatter.
            timestamps: (B, T) day offsets.
            
        Returns:
            Dictionary with outputs from each prediction head:
            {
                "health": {logits, probs, state, score},
                "attribution": {logits, probs, active, strength},
                "trajectory": {mean, score_100, uncertainty},
            }
        """
        if optical is None and sar is None:
            raise ValueError("At least one satellite modality is required")
        reference = optical if optical is not None else sar
        if reference.ndim != 5:
            raise ValueError("Satellite input must have shape (B, T, C, H, W)")
        B, T, _, H, W = reference.shape
        if optical is None:
            optical = reference.new_zeros(B, T, 12, H, W)
            optical_mask = torch.zeros(B, T, dtype=torch.bool, device=reference.device)
        if sar is None:
            sar = reference.new_zeros(B, T, 2, H, W)
            sar_mask = torch.zeros(B, T, dtype=torch.bool, device=reference.device)
        if optical.shape != (B, T, 12, H, W) or sar.shape != (B, T, 2, H, W):
            raise ValueError("Expected matching optical (12 channels) and SAR (2 channels) grids")
        masks = []
        for name, mask in (("optical_mask", optical_mask), ("sar_mask", sar_mask)):
            if mask is None:
                mask = torch.ones(B, T, dtype=torch.bool, device=reference.device)
            if mask.shape != (B, T):
                raise ValueError(f"{name} must have shape (B, T)")
            masks.append(mask.to(device=reference.device, dtype=torch.bool))
        optical_mask, sar_mask = masks
        valid = optical_mask | sar_mask
        if observation_mask is not None:
            if observation_mask.shape != (B, T):
                raise ValueError("observation_mask must have shape (B, T)")
            valid = valid & observation_mask.to(device=reference.device, dtype=torch.bool)
        if self.foundation.supports_modalities == ["optical"]:
            valid = valid & optical_mask
        if not valid.any(dim=1).all():
            raise ValueError("Every sample needs at least one observed timestep supported by the encoder")
        optical = optical.masked_fill(~(optical_mask & valid)[..., None, None, None], 0)
        sar = sar.masked_fill(~(sar_mask & valid)[..., None, None, None], 0)
        if quality_mask is not None:
            if quality_mask.shape != (B, T, H, W):
                raise ValueError("quality_mask must have shape (B, T, H, W)")
            optical = optical.masked_fill(~quality_mask[:, :, None].bool(), 0)
        if sar_quality_mask is not None:
            if sar_quality_mask.shape != (B, T, H, W) or sar_quality_mask.dtype != torch.bool:
                raise ValueError("sar_quality_mask must be boolean with shape (B, T, H, W)")
            sar = sar.masked_fill(~sar_quality_mask[:, :, None], 0)
        if not torch.isfinite(optical).all() or not torch.isfinite(sar).all():
            raise ValueError("Observed satellite pixels must be finite; mask missing data explicitly")
        C_opt = optical.shape[2]
        C_sar = sar.shape[2]

        # ─── Per-Timestep Encoding ─────────────────────────
        # Concatenate optical + SAR along channel dim
        # (B, T, C_opt+C_sar, H, W) → reshape to (B*T, C, H, W)
        combined = torch.cat([optical, sar], dim=2)
        combined_flat = combined.view(B * T, C_opt + C_sar, H, W)

        # Foundation model encoding
        # Exclude padded observations from the encoder as well as attention;
        # otherwise padding can contaminate BatchNorm statistics in proxy CNNs.
        flat_valid = valid.reshape(-1)
        if hasattr(self.foundation, "encode_modalities"):
            encoded = self.foundation.encode_modalities(
                combined_flat[flat_valid, :12], combined_flat[flat_valid, 12:],
                optical_mask.reshape(-1)[flat_valid], sar_mask.reshape(-1)[flat_valid],
            )
        else:
            encoded = self.foundation(combined_flat[flat_valid])
        embeddings_flat = encoded.new_zeros(B * T, self.foundation.embedding_dim)
        embeddings_flat[flat_valid] = encoded
        embeddings = embeddings_flat.view(B, T, -1)  # (B, T, D)

        # ─── Multimodal Fusion (per-timestep) ──────────────
        fused_embeddings = []
        for t in range(T):
            fused_t = self.fusion(
                embeddings[:, t],
                climate_context[:, t] if climate_context is not None else None,
                activity_context[:, t] if activity_context is not None else None,
                climate_mask[:, t] if climate_mask is not None else None,
                activity_mask[:, t] if activity_mask is not None else None,
            )
            fused_embeddings.append(fused_t)
        fused_embeddings = torch.stack(fused_embeddings, dim=1)  # (B, T, D)

        # ─── Temporal Modeling ─────────────────────────────
        sequence, summary = self.temporal(fused_embeddings, timestamps, valid)

        # ─── Prediction Heads ─────────────────────────────
        health_out = self.health_head(summary)
        attribution_out = self.attribution_head(summary)
        trajectory_out = self.trajectory_head(sequence, summary, valid)

        return {
            "health": health_out,
            "attribution": attribution_out,
            "trajectory": trajectory_out,
        }

    @torch.no_grad()
    def predict(
        self,
        optical: Tensor,
        sar: Tensor,
        timestamps: Tensor,
        **kwargs,
    ) -> dict:
        """Inference-mode prediction with human-readable outputs.
        
        Same as forward() but with evaluation mode and no gradients.
        """
        self.eval()
        outputs = self.forward(optical, sar, timestamps, **kwargs)

        health_states = ["Healthy", "Stressed", "Degraded", "Severely Degraded", "Cleared"]
        driver_names = [
            "Agriculture", "Cattle Pasture", "Mining", "Logging",
            "Road Construction", "Wildfire", "Drought", "Urban Expansion",
        ]

        # Human-readable predictions (first sample in batch)
        health = outputs["health"]
        attr = outputs["attribution"]
        traj = outputs["trajectory"]

        result = {
            "health_state_index": health["state"][0].item(),
            "health_state": health_states[health["state"][0].item()],
            "health_score": round(health["score"][0].item(), 1),
            "health_confidence": round(health["probs"][0].max().item() * 100, 1),
            "active_drivers": [
                {
                    "name": driver_names[i],
                    "confidence": round(attr["probs"][0, i].item() * 100, 1),
                    "strength": round(attr["strength"][0, i].item() * 100, 1),
                }
                for i in range(len(driver_names))
                if attr["active"][0, i].item() > 0
            ],
            "trajectory": {
                "horizons": [f"{days} days" for days in self.trajectory_head.horizon_days],
                "health_scores": [round(v, 1) for v in (traj["score_100"][0]).tolist()],
            },
            "driver_strengths": attr["strength"][0].tolist(),
            "probabilities_calibrated": False,
        }

        if "uncertainty" in traj:
            result["trajectory"]["uncertainty"] = [
                round(v * 100, 1) for v in traj["uncertainty"][0].tolist()
            ]

        return result
