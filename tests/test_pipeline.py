"""Tests for the full ForestMind pipeline (data → model → risk engine)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import pytest

from forestmind.data.synthetic import SyntheticForestGenerator
from forestmind.data.transforms import Normalize, ToTensor, Compose, get_train_transforms, get_eval_transforms
from forestmind.data.dataset import ForestMindDataset
from forestmind.models.forestmind_model import ForestMindModel
from forestmind.training.losses import ForestMindLoss
from forestmind.training.metrics import ForestMindMetrics
from forestmind.risk.engine import EcologicalRiskEngine


# Minimal config for fast pipeline testing
MINI_CONFIG = {
    "foundation": {
        "name": "custom_vit",
        "embedding_dim": 32,
        "custom_vit": {
            "patch_embed_size": 16, "depth": 1, "num_heads": 4,
            "mlp_ratio": 2.0, "drop_rate": 0.0,
        },
    },
    "fusion": {"method": "gated", "num_heads": 4, "num_layers": 1, "dropout": 0.0},
    "temporal": {
        "depth": 1, "num_heads": 4, "dim_feedforward": 64,
        "dropout": 0.0, "max_seq_len": 12, "use_continuous_pe": True,
    },
    "heads": {
        "health": {"num_classes": 5, "continuous_score": True},
        "attribution": {"num_drivers": 8},
        "trajectory": {"horizons": [30, 180, 365, 1095, 1825], "predict_uncertainty": True},
    },
    "data": {"patch_size": 64, "num_bands": 12, "sar_channels": 2, "time_steps": 4},
    "seed": 42,
}


class TestDataPipeline:
    """Test data pipeline from generation through transforms."""

    def test_train_transforms(self):
        gen = SyntheticForestGenerator(patch_size=64, time_steps=4, seed=42)
        sample = gen.generate_sample()
        
        transform = get_train_transforms()
        transformed = transform(sample)

        assert isinstance(transformed["optical"], torch.Tensor)
        assert isinstance(transformed["health_state"], torch.Tensor)
        assert transformed["optical"].dtype == torch.float32

    def test_eval_transforms(self):
        gen = SyntheticForestGenerator(patch_size=64, time_steps=4, seed=42)
        sample = gen.generate_sample()
        
        transform = get_eval_transforms()
        transformed = transform(sample)

        assert isinstance(transformed["optical"], torch.Tensor)

    def test_dataset_indexing(self):
        dataset = ForestMindDataset(
            source="synthetic", num_samples=5,
            transform=get_eval_transforms(),
            config=MINI_CONFIG["data"], seed=42,
        )

        assert len(dataset) == 5
        sample = dataset[0]
        assert isinstance(sample["optical"], torch.Tensor)


class TestEndToEndPipeline:
    """Test the full pipeline: data → model → loss → metrics → risk engine."""

    def test_training_step(self):
        """One training step completes without error."""
        model = ForestMindModel(MINI_CONFIG)
        criterion = ForestMindLoss()

        dataset = ForestMindDataset(
            source="synthetic", num_samples=4,
            transform=get_eval_transforms(),
            config=MINI_CONFIG["data"], seed=42,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)

        model.train()
        batch = next(iter(loader))

        outputs = model(batch["optical"], batch["sar"], batch["timestamps"])
        losses = criterion(outputs, batch)

        assert "total" in losses
        assert losses["total"].requires_grad

        losses["total"].backward()

    def test_metrics_accumulation(self):
        """Metrics accumulate correctly across batches."""
        model = ForestMindModel(MINI_CONFIG)
        metrics = ForestMindMetrics()

        dataset = ForestMindDataset(
            source="synthetic", num_samples=8,
            transform=get_eval_transforms(),
            config=MINI_CONFIG["data"], seed=42,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        model.eval()
        with torch.no_grad():
            for batch in loader:
                outputs = model(batch["optical"], batch["sar"], batch["timestamps"])
                metrics.update(outputs, batch)

        result = metrics.compute()
        assert 0 <= result.health_accuracy <= 1
        assert result.trajectory_mae >= 0
        summary = metrics.summary_str(result)
        assert "Health Accuracy" in summary

    def test_model_to_risk_engine(self):
        """Model predictions feed correctly into risk engine."""
        model = ForestMindModel(MINI_CONFIG)
        gen = SyntheticForestGenerator(patch_size=64, time_steps=4, seed=42)
        sample = gen.generate_sample()

        transform = Compose([Normalize(), ToTensor()])
        t_sample = transform(sample)

        optical = t_sample["optical"].unsqueeze(0)
        sar = t_sample["sar"].unsqueeze(0)
        timestamps = t_sample["timestamps"].unsqueeze(0)

        prediction = model.predict(optical, sar, timestamps)

        # Feed into risk engine
        engine = EcologicalRiskEngine()
        report = engine.analyze(
            health_history=sample["health_history"],
            health_state=t_sample["health_state"].item(),
            health_score=prediction["health_score"],
            health_confidence=prediction["health_confidence"],
            active_drivers=prediction["active_drivers"],
            driver_strengths=sample["driver_strengths"],
            trajectory_scores=np.array(prediction["trajectory"]["health_scores"]),
        )

        assert report.health_score == prediction["health_score"]
        assert report.tipping_point is not None
        assert report.interventions is not None
        summary = report.summary()
        assert "FORESTMIND" in summary
