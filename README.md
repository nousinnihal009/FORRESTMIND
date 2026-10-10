# 🌲 ForestMind — Multimodal Forest Intelligence & Tipping-Point Early Warning

> **"Is this ecosystem entering a trajectory toward irreversible degradation, why is it happening, and what is likely to happen next?"**

ForestMind is a research prototype for predicting forest degradation trajectories, attributing degradation drivers, and estimating ecological tipping-point risk — going far beyond binary "deforestation yes/no" classification.

## Architecture

```
SATELLITE DATA (Sentinel-2, Sentinel-1, Landsat)
         │
         ↓
Earth Observation Foundation Model ──── Swappable:
  (CustomViT / Prithvi / TerraMind)     • CustomViT  (~5M params, CPU)
         │                               • Prithvi-EO-2.0 (GPU)
         ↓                               • TerraMind (GPU)
Multimodal Fusion
  (Cross-Attention / Gated / Concat)
         │
         ↓
Temporal Transformer
  (Continuous positional encoding)
         │
    ┌────┼────┐
    ↓    ↓    ↓
 Health  Cause  Future
 State   Attribution  Trajectory
    │    │    │
    └────┼────┘
         ↓
Ecological Risk Engine
  • Critical Slowing Down
  • Tipping-Point Estimation
  • Intervention Simulation
         │
         ↓
EARLY WARNING SYSTEM
```

## Key Features

### 🏥 Forest Health Assessment
- **5-level classification**: Healthy → Stressed → Degraded → Severely Degraded → Cleared
- **Continuous score**: Fine-grained 0–100 health metric

### 🔍 Cause Attribution
Multi-label detection of 8 degradation drivers:
Agriculture, Cattle Pasture, Mining, Logging, Road Construction, Wildfire, Drought, Urban Expansion

### 📈 Future Trajectory Prediction
Health projections at 5 horizons: 30 days, 6 months, 1 year, 3 years, 5 years — with calibrated uncertainty.

### ⚠️ Tipping-Point Risk Assessment
Real ecological indicators from dynamical systems theory:
- **Critical Slowing Down** (AR(1) coefficient, variance amplification)
- **Vegetation resilience** estimation
- **Spatial fragmentation** analysis
- **Drought sensitivity** assessment
- **Time-to-tipping** estimation

### 🛡️ Intervention Simulation
Counterfactual "digital twin" modeling:
- 7 intervention scenarios (conservation corridors, fire management, full protection, etc.)
- Cost-effectiveness analysis
- Comparative trajectory visualization

## Quick Start

### Install

```bash
cd forestmind
pip install -e ".[dev]"
```

### Generate Synthetic Data

```bash
python scripts/generate_synthetic_data.py --num-train 2000 --num-val 400
```

### Train

```bash
python scripts/train.py --config configs/default.yaml
```

### Run Demo (no training required)

```bash
python scripts/demo_prediction.py
```

This runs the full pipeline on a synthetic sample with randomly initialized weights — demonstrating the complete data → model → risk analysis → intervention simulation pipeline.

### Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
forestmind/
├── configs/default.yaml          # Training configuration
├── forestmind/
│   ├── data/                     # Data pipeline
│   │   ├── synthetic.py          #   Synthetic satellite data generator
│   │   ├── transforms.py         #   EO-specific augmentations
│   │   └── dataset.py            #   Unified dataset interface
│   ├── models/
│   │   ├── foundation/           # Swappable foundation models
│   │   │   ├── base.py           #   Abstract interface
│   │   │   ├── custom_vit.py     #   Lightweight ViT (CPU-friendly)
│   │   │   ├── prithvi.py        #   Prithvi-EO-2.0 adapter
│   │   │   └── terramind.py      #   TerraMind adapter
│   │   ├── fusion/multimodal.py  # Cross-attention/gated/concat fusion
│   │   ├── temporal/transformer.py # Temporal transformer (continuous PE)
│   │   ├── heads/                # Prediction heads
│   │   │   ├── health.py         #   5-class + continuous health
│   │   │   ├── attribution.py    #   8-driver multi-label
│   │   │   └── trajectory.py     #   Multi-horizon + uncertainty
│   │   └── forestmind_model.py   # Full model assembly
│   ├── risk/                     # Ecological Risk Engine
│   │   ├── critical_slowing.py   #   CSD indicator analysis
│   │   ├── tipping_point.py      #   Tipping-point estimation
│   │   ├── intervention.py       #   Counterfactual simulation
│   │   └── engine.py             #   Risk orchestration + reporting
│   ├── training/                 # Training infrastructure
│   │   ├── losses.py             #   Multi-task losses
│   │   ├── metrics.py            #   Per-task evaluation metrics
│   │   └── trainer.py            #   Training loop
│   └── utils/
│       ├── geo.py                # Geospatial utilities
│       └── visualization.py     # Publication-quality plots
├── scripts/                      # User-facing entry points
│   ├── generate_synthetic_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── demo_prediction.py
└── tests/                        # Test suite
```

## Foundation Model Strategy

ForestMind abstracts the foundation model behind a clean interface (`FoundationModelBase`). This lets you:

| Model | Status | Hardware | Best For |
|-------|--------|----------|----------|
| **CustomViT** | ✅ Ready | CPU | Development, prototyping |
| **Prithvi-EO-2.0** | 🔌 Adapter ready | GPU (≥16GB) | Optical time-series (HLS) |
| **TerraMind** | 🔌 Proxy mode | GPU (≥16GB) | Multimodal (optical + SAR + DEM) |

Switch models by changing one line in `configs/default.yaml`:
```yaml
foundation:
  name: custom_vit  # → prithvi | terramind
```

## References

- Scheffer et al. (2009) "Early-warning signals for critical transitions" *Nature*
- Dakos et al. (2012) "Methods for Detecting Early Warnings" *PLoS ONE*
- Verbesselt et al. (2016) "Remotely sensed resilience of tropical forests" *Nature Climate Change*
- Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model (IEEE TGRS 2025)
- TerraMind: Large-Scale Generative Multimodality for Earth Observation (arXiv 2024)

## License

Apache 2.0
