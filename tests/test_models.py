"""Tests for ForestMind model architecture."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest
import yaml

from forestmind.models.foundation.custom_vit import CustomViTEncoder
from forestmind.models.fusion.multimodal import MultimodalFusion, CrossAttentionFusion, GatedFusion, ConcatFusion
from forestmind.models.temporal.transformer import TemporalTransformer, ContinuousTemporalPE
from forestmind.models.heads.health import ForestHealthHead
from forestmind.models.heads.attribution import CauseAttributionHead
from forestmind.models.heads.trajectory import TrajectoryHead
from forestmind.models.forestmind_model import ForestMindModel


# Small config for fast testing
TEST_CONFIG = {
    "foundation": {
        "name": "custom_vit",
        "embedding_dim": 64,
        "custom_vit": {
            "patch_embed_size": 16,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "drop_rate": 0.0,
        },
    },
    "fusion": {
        "method": "cross_attention",
        "num_heads": 4,
        "num_layers": 1,
        "dropout": 0.0,
    },
    "temporal": {
        "depth": 2,
        "num_heads": 4,
        "dim_feedforward": 128,
        "dropout": 0.0,
        "max_seq_len": 24,
        "use_continuous_pe": True,
    },
    "heads": {
        "health": {"num_classes": 5, "continuous_score": True},
        "attribution": {"num_drivers": 8},
        "trajectory": {
            "horizons": [30, 180, 365, 1095, 1825],
            "predict_uncertainty": True,
        },
    },
}

BATCH_SIZE = 2
TIME_STEPS = 4
IMG_SIZE = 64


class TestCustomViT:
    """Test CustomViT encoder."""

    def test_output_shape(self):
        model = CustomViTEncoder(in_channels=14, embed_dim=64, depth=2, num_heads=4, img_size=64)
        x = torch.randn(BATCH_SIZE, 14, IMG_SIZE, IMG_SIZE)
        out = model.encode(x)
        assert out.shape == (BATCH_SIZE, 64)

    def test_forward_equals_encode(self):
        model = CustomViTEncoder(in_channels=14, embed_dim=64, depth=2, num_heads=4, img_size=64)
        model.eval()  # deterministic (no dropout)
        x = torch.randn(BATCH_SIZE, 14, IMG_SIZE, IMG_SIZE)
        assert torch.equal(model(x), model.encode(x))

    def test_freeze_unfreeze(self):
        model = CustomViTEncoder(in_channels=14, embed_dim=64, depth=2, num_heads=4, img_size=64)
        model.freeze()
        assert model.num_trainable_parameters == 0
        model.unfreeze()
        assert model.num_trainable_parameters > 0

    def test_supports_modalities(self):
        model = CustomViTEncoder()
        assert "optical" in model.supports_modalities
        assert "sar" in model.supports_modalities


class TestFusion:
    """Test multimodal fusion modules."""

    @pytest.mark.parametrize("method", ["cross_attention", "gated", "concat"])
    def test_fusion_output_shape(self, method):
        fusion = MultimodalFusion(embed_dim=64, method=method, num_heads=4, num_layers=1)
        x = torch.randn(BATCH_SIZE, 64)
        out = fusion(x)
        assert out.shape == (BATCH_SIZE, 64)

    def test_cross_attention_no_context(self):
        fusion = CrossAttentionFusion(embed_dim=64)
        x = torch.randn(BATCH_SIZE, 64)
        out = fusion(x, None)
        assert torch.equal(out, x)


class TestTemporalTransformer:
    """Test temporal transformer."""

    def test_output_shapes(self):
        model = TemporalTransformer(embed_dim=64, depth=2, num_heads=4)
        embeddings = torch.randn(BATCH_SIZE, TIME_STEPS, 64)
        timestamps = torch.arange(TIME_STEPS).unsqueeze(0).expand(BATCH_SIZE, -1).float() * 30

        sequence, summary = model(embeddings, timestamps)
        assert sequence.shape == (BATCH_SIZE, TIME_STEPS, 64)
        assert summary.shape == (BATCH_SIZE, 64)

    def test_without_timestamps(self):
        model = TemporalTransformer(embed_dim=64, depth=2, num_heads=4)
        embeddings = torch.randn(BATCH_SIZE, TIME_STEPS, 64)
        sequence, summary = model(embeddings, None)
        assert sequence.shape == (BATCH_SIZE, TIME_STEPS, 64)

    def test_continuous_pe(self):
        pe = ContinuousTemporalPE(embed_dim=64)
        timestamps = torch.tensor([[0, 30, 60, 90]], dtype=torch.float32)
        out = pe(timestamps)
        assert out.shape == (1, 4, 64)


class TestPredictionHeads:
    """Test prediction heads."""

    def test_health_head(self):
        head = ForestHealthHead(embed_dim=64, num_classes=5)
        x = torch.randn(BATCH_SIZE, 64)
        out = head(x)

        assert out["logits"].shape == (BATCH_SIZE, 5)
        assert out["probs"].shape == (BATCH_SIZE, 5)
        assert out["state"].shape == (BATCH_SIZE,)
        assert out["score"].shape == (BATCH_SIZE,)
        assert torch.all(out["score"] >= 0) and torch.all(out["score"] <= 100)

    def test_attribution_head(self):
        head = CauseAttributionHead(embed_dim=64, num_drivers=8)
        x = torch.randn(BATCH_SIZE, 64)
        out = head(x)

        assert out["logits"].shape == (BATCH_SIZE, 8)
        assert out["probs"].shape == (BATCH_SIZE, 8)
        assert out["active"].shape == (BATCH_SIZE, 8)
        assert out["strength"].shape == (BATCH_SIZE, 8)

    def test_trajectory_head(self):
        head = TrajectoryHead(embed_dim=64, num_horizons=5, predict_uncertainty=True)
        seq = torch.randn(BATCH_SIZE, TIME_STEPS, 64)
        summary = torch.randn(BATCH_SIZE, 64)
        out = head(seq, summary)

        assert out["mean"].shape == (BATCH_SIZE, 5)
        assert out["score_100"].shape == (BATCH_SIZE, 5)
        assert out["logvar"].shape == (BATCH_SIZE, 5)
        assert out["uncertainty"].shape == (BATCH_SIZE, 5)


class TestForestMindModel:
    """Test full model assembly."""

    def test_forward_pass(self):
        model = ForestMindModel(TEST_CONFIG)
        optical = torch.randn(BATCH_SIZE, TIME_STEPS, 12, IMG_SIZE, IMG_SIZE)
        sar = torch.randn(BATCH_SIZE, TIME_STEPS, 2, IMG_SIZE, IMG_SIZE)
        timestamps = torch.arange(TIME_STEPS).unsqueeze(0).expand(BATCH_SIZE, -1).float() * 30

        outputs = model(optical, sar, timestamps)

        assert "health" in outputs
        assert "attribution" in outputs
        assert "trajectory" in outputs
        assert outputs["health"]["logits"].shape == (BATCH_SIZE, 5)
        assert outputs["attribution"]["logits"].shape == (BATCH_SIZE, 8)
        assert outputs["trajectory"]["mean"].shape == (BATCH_SIZE, 5)

    def test_predict(self):
        model = ForestMindModel(TEST_CONFIG)
        optical = torch.randn(1, TIME_STEPS, 12, IMG_SIZE, IMG_SIZE)
        sar = torch.randn(1, TIME_STEPS, 2, IMG_SIZE, IMG_SIZE)
        timestamps = torch.arange(TIME_STEPS).unsqueeze(0).float() * 30

        result = model.predict(optical, sar, timestamps)

        assert "health_state" in result
        assert "health_score" in result
        assert "active_drivers" in result
        assert "trajectory" in result
        assert isinstance(result["health_state"], str)
        assert isinstance(result["health_score"], float)

    def test_gradient_flow(self):
        model = ForestMindModel(TEST_CONFIG)
        optical = torch.randn(BATCH_SIZE, TIME_STEPS, 12, IMG_SIZE, IMG_SIZE)
        sar = torch.randn(BATCH_SIZE, TIME_STEPS, 2, IMG_SIZE, IMG_SIZE)
        timestamps = torch.arange(TIME_STEPS).unsqueeze(0).expand(BATCH_SIZE, -1).float() * 30

        outputs = model(optical, sar, timestamps)
        loss = outputs["health"]["logits"].sum() + outputs["trajectory"]["mean"].sum()
        loss.backward()

        # Check gradients exist
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
        assert has_grad, "No gradients flowing through model"
