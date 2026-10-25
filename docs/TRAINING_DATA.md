# Preparing real training data

`scripts/prepare_training_data.py` joins existing raw observation sequences to independently supplied labels. It supports partially labeled examples, protected geographic and chronological partitions, explicit label uncertainty, training-only normalization and fixed-length batches. It never generates a health label, driver, future condition or negative event from an unlabeled satellite image.

No independent health/driver/trajectory dataset has been supplied with this release. The templates are deliberately incomplete and cannot pass the audit until real observations and references replace the placeholders. Software fixture tests do not establish prediction accuracy.

## What to provide

Place a versioned JSON manifest alongside a `sequences/` directory containing raw NPZ outputs from `scripts/prepare_sequence.py`. Optional aligned SAR/climate can already be attached to those sequences. Relative sequence paths must stay inside the manifest directory, and each entry pins the exact NPZ SHA256. Keep the original provenance sidecars with the source files.

For each example provide its unique ID, region ID, exact WGS84 rectangle, partition, prediction cutoff, feature-availability date and observed labels. A label includes the observed date, source record/URL, product or survey version, uncertainty, actual use terms, when the reference became available and a reviewed declaration that it is independent of the model's input-derived target heuristics. A declaration is not proof of scientific quality.

Supported labels are:

| Task | Values and meaning | Missing labels |
| --- | --- | --- |
| `health_state` | `healthy`, `stressed`, `degraded`, `severely_degraded`, `cleared`; use a justified biome-specific definition | Omit the task |
| `health_score` | Measured condition score from 0 to 100; document endpoints and protocol | Omit the task |
| `drivers` | Named independently observed presence/absence, exactly 0 or 1 | Omit each unknown driver; missing is not absence |
| `driver_strengths` | Named measured strength from 0 to 1; only where such a measurement is justified | Omit each unknown strength |
| `trajectory` | Future **condition** from 0 to 1, keyed by exact horizon days | Omit each unobserved horizon |

Canonical drivers are `agriculture`, `cattle_pasture`, `mining`, `logging`, `road_construction`, `wildfire`, `drought` and `urban_expansion`. Multiple drivers may be observed together. Their labels express the chosen reference protocol; prediction does not by itself demonstrate causal attribution.

Each supplied task needs a frozen `target_spec` containing its definition, units and reference protocol. Units are `canonical_five_state`, `score_0_100`, `binary_presence`, `strength_0_1` and `condition_0_1`, respectively. Canopy-loss fraction is a different outcome and must not be placed in the health trajectory head. Use the separate canopy-loss baseline manifest described below.

For example, a current observation can include only a health-state label. A fire example can supply only `drivers.wildfire`; other drivers remain unknown. A longitudinal study can supply only a 365-day future condition measurement. All three are supported without filling in invented targets. A task/channel appearing in held-out labels must have at least one observed training target for that same channel; one observation is a software minimum, not adequate scientific training.

## Prepare, audit and train

Start from [`templates/training-manifest.template.json`](templates/training-manifest.template.json). Add examples to its empty `examples` list and remove the helper blocks. The template's buffer and embargo are examples, not ecological recommendations; choose them for the study and freeze them before examining held-out outcomes.

```powershell
python scripts/prepare_training_data.py --manifest data/my-study/study.json --check
python scripts/prepare_training_data.py --manifest data/my-study/study.json --output data/my-study-prepared-v1
python scripts/train.py --config data/my-study-prepared-v1/training-config.yaml
python scripts/evaluate.py --checkpoint data/my-study-prepared-v1/checkpoints/best_model.pt --split test --output artifacts/my-study-evaluation.json
```

`--check` reads and audits inputs, including normalization support, without creating a dataset. Preparation creates a new output directory and refuses existing paths. Outputs include `train/`, `val/`, `test/`, optional `calibration/`, an exact source manifest copy, `dataset-report.json` and a configurable `training-config.yaml`. The report records source and output hashes, counts for each target/driver/horizon and split, normalization evidence, and a code fingerprint. Paths in the copied manifest remain relative to its original location, which the report records. The generated config pins the report's SHA256; loading rechecks the copied manifest, all split files (including unused calibration files), normalization, supervised-head support, units and horizon definitions. Changed or extra files fail before training.

The generated config starts a small CPU `custom_vit` experiment. It is a reproducible starting configuration, not a tuned model or an approved operational checkpoint. It includes the actual `data.supervised_targets` support map; unobserved output heads must remain unavailable in user-facing reports. Select model settings using training/validation data only. A separate calibration partition may fit probability/interval calibration after model selection. It must never participate in training or checkpoint selection. Keep the test partition untouched until the planned final comparison. Operational model registration still requires independent evaluation and acceptance.

## What the audit enforces

- Nonempty train, validation and test partitions; optional distinct calibration partition. Each region identity belongs to one partition. Cross-partition footprints must satisfy a positive spatial buffer using conservative spherical bounding circles, not just centroid distance.
- Chronological order is train → validation → calibration (if present) → test. All earlier labels must have become available before the later cutoffs, with the declared embargo. Future target dates must match the configured horizon exactly; reference publication/availability cannot predate the target.
- Actual feature acquisition timestamps, physical units and matching footprints. Future optical or SAR observations, mismatched day offsets, source hash changes, unknown physical climate feature definitions, object arrays and nonfinite samples are rejected. Historical product revision and publication claims still require an external audit; a current ERA5 download is not proof of the vintage available at an earlier cutoff.
- Current labels must fall within the explicit age tolerance before their prediction cutoff. The default is the same calendar date. Feature staleness has a separately declared limit (31 days by default).
- At least one known target per example. Unknown targets receive numeric storage placeholders with **false boolean masks**: `health_state_mask`, `health_score_mask`, `drivers_mask`, `driver_strengths_mask`, `trajectory_mask`. Masks, not placeholder values, determine loss and metric participation.
- Fixed declared grid and sequence length. Grid dimensions must be multiples of 16 for the generated encoder; its positional capacity is configured for the actual grid. Shorter sequences receive trailing masked padding with the last actual timestamp repeated; observed scenes are never silently dropped. SAR has a separate pixel quality mask. All optional climate fields have canonical shapes even when absent for an example.
- Optical, SAR and climate normalization uses only valid observed **training** samples. Invalid/cloudy pixels and false feature masks never enter statistics. Absent modalities receive neutral 0/1 transforms and false masks; a modality observed only in held-out data is rejected. Observed constant features use a standard-deviation floor of `1e-6`, recorded in the report. Native pretrained checkpoints may require fixed documented pretraining normalization instead; do not assume these fitted statistics are compatible with them.

Local bounds are 10,000 examples, 600 steps, 16 horizons, up to 512 MiB uncompressed per NPZ, 2 GiB total loaded input arrays and one million cross-partition footprint comparisons. Larger studies need a separately designed streaming/block pipeline. Keep original inputs and choose a new output directory/version after a failed or revised conversion; partial writes do not produce a successful final training config.

## Public reference products and their limits

These are candidates to prepare independently reviewed labels. They have not been downloaded, converted into training labels or used to establish model accuracy in this implementation.

- **Canopy loss:** [Hansen/UMD Global Forest Change 2025 v1.13](https://storage.googleapis.com/earthenginepartners-hansen/GFC-2025-v1.13/download.html) supplies annual loss-year rasters at approximately 30 m at the equator. Its stated license is CC BY 4.0; preserve attribution and record transformations. Loss denotes a stand-replacement disturbance, not necessarily permanent deforestation or ecological health. Product revisions and changes in sensor sensitivity limit comparisons across years. Annual labels do not provide an exact event day. Use [`canopy-loss-references.template.json`](templates/canopy-loss-references.template.json) with `scripts/benchmark_references.py` for a separately defined annual continuous outcome. Do not invent a daily event date to calculate warning lead time.
- **Burned area:** [NASA MCD64A1 Collection 6.1](https://www.earthdata.nasa.gov/data/catalog/lpcloud-mcd64a1-061) provides monthly burned-area mapping at 500 m with burn date and quality layers. Match the reference scale and inspect quality/uncertainty before labeling a much smaller optical patch. Burned area is a disturbance observation, not a five-state forest-health label. [NASA's data-use guidance](https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy) says unrestricted NASA-led mission data use CC0 unless otherwise marked; keep product-specific terms and citation with each reference.
- **Active fire:** [NASA FIRMS API documentation](https://firms.modaps.eosdis.nasa.gov/content/academy/data_api/firms_api_use.html) describes bounded area/date queries and a free user MAP_KEY. Hotspot detections can contribute independent fire evidence; they are not equivalent to a burned-area perimeter or proof of a causal driver. A user-owned key is needed for API access; do not put it in a manifest, source code, report or chat.

Health state, recovery, long-horizon condition and causal driver labels generally need suitable field surveys, longitudinal reference observations or reviewed annotations. Public canopy-loss/fire products alone cannot fill all those targets. Field data should include coordinates or polygons, observation dates, measurement definitions, uncertainty and permission to use it; anonymize sensitive observer or location information where appropriate.
