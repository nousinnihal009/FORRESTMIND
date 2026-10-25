# Research protocol

## Testable question

Do measured optical/radar/climate features, pretrained EO representations and resilience diagnostics improve prediction of a predefined future degradation outcome beyond simple baselines?

Freeze the outcome, spatial unit, observation cutoff, horizons and evaluation rules before comparison. A vegetation-anomaly target supports a vegetation-anomaly claim, not an irreversible ecosystem-transition claim.

## Data and references

1. Select a biome with a documented seasonal cycle, consistently gridded patches and versioned acquisitions/composites.
2. Define independent reference outcomes: confirmed canopy loss, field-supported persistent deterioration, or another justified target. Record label uncertainty and distinguish temporary stress from clearing.
3. Use only measurements available by each prediction cutoff. Audit compositing, gap filling, anomaly baselines and normalization for future leakage.
4. Preserve missingness, sensor identity, spatial footprint, dates and quality. A missing modality is not a negative label.
5. Document weak labels and reference disagreements. Reproducing the same threshold that generated a label does not demonstrate ecological insight.

## Partitions

Separate training, validation/calibration and untouched test regions with spatial buffers appropriate to footprints and ecological autocorrelation. Reserve a later temporal test. Purge examples whose target horizon crosses a test boundary; keep nearby/duplicate observations within one split.

The code requires explicit disk split directories and rejects duplicated location IDs or exact bounding boxes when provided. It cannot automatically detect all geographic overlap, near-duplicates or temporal leakage. Maintain and audit a versioned split manifest.

## Comparisons

| Model | What it tests |
| --- | --- |
| Persistence and seasonal naive | Minimum benchmarks; implemented for the monitoring index |
| Indices + tabular model | Conventional measured-feature baseline |
| Pretrained optical embeddings + head | Representation benefit |
| Optical/radar/context + temporal model | Multimodal/history benefit |
| Same temporal model + resilience features | Incremental early-warning benefit |

Keep outcomes, held-out examples, horizons and tuning budgets comparable. Include missing-modality, sensor-transfer and observation-quality ablations. Test whether cloud coverage alone explains apparent gains.

## Implemented rolling-origin backtest

`backtest_prepared` predicts from history through each cutoff only. Seasonal references never exceed the cutoff. Targets must exist in the exact future calendar month; missing/unusable targets are skipped and counted.

Exports retain input hashes, cutoff/training boundaries, nominal forecast dates, actual target dates, predictions and errors. Reports include 1/6/12-month MAE/RMSE, counts and empirical range coverage where bounds exist. Overlapping origins are dependent, so independent-sample confidence intervals are not reported. Nominal 30/180/365-day forecasts are compared with actual dated records in the target calendar month.

The target remains the same uncalibrated spectral index. Synthetic data checks software behavior, not ecological accuracy. Use this baseline workflow before testing separately defined outcomes on independent field/reference labels.

## Metrics and calibration

For continuous targets report per-horizon/region MAE/RMSE, bias and improvement over persistence. For rare events report precision-recall AUC, precision/recall at a preregistered threshold, useful lead time, false alerts per region-year and detection delay. For probabilities report Brier score and reliability curves; calibrate only on a separate calibration partition.

Use justified spatial/temporal block resampling. Examine sensitivity to labels, record length, seasonal adjustment and window size. A neural variance head is not calibrated merely because it outputs a standard deviation.

## Advancement criteria

- Monitoring: verified acquisitions → aligned grid → quality mask → true dated monthly observations → traceable report.
- Multimodal research: actual checkpoint inference on documented inputs, justified labels and reproducible baseline comparisons.
- Early warning: useful lead time and predictive gain on untouched regions/time periods, with calibration and false-alert accounting.
- Scenarios: declared sensitivity assumptions first; causal claims only with intervention data, an appropriate comparison design or validated process model.

Archive data/model versions, configurations, split manifests, evaluation code and all outputs, including failed ablations and untested conditions. Advance claims only when the experiment supports them.

## Connected analysis and context boundaries

The selected-area worker now freezes the exact geometry, observation cutoff, observation hash, accepted acquisition manifest, code hashes, contextual layers and registered model version into an immutable snapshot. Refreshing observations creates another snapshot. Export the displayed manifest ID; a saved scenario retains that same manifest even after a later refresh. GET endpoints only read saved computations; raster work, backtesting, network collection and inference run on a serialized worker. Each database has its own `<database-stem>-workflow` artifacts directory. Current-view freshness checks cover observation/context changes, software/code versions, and source/checkpoint/evaluation file size/modification fingerprints. Fingerprints are cheap change hints, not cryptographic verification; execution rechecks the accepted-source and registered-model SHA256 hashes. Historical exports remain unchanged even when the current view becomes stale.

Analysis persistence is bounded: at most four active jobs, 10000 retained job records, 2048 snapshots per workspace, 256 per area, 256 MiB aggregate analysis metadata, 1 GiB sequence artifacts and 16 MiB per record. Metadata accounting includes contexts, bindings, snapshots, jobs, saved scenarios and model reports, including the legacy scenario/model import APIs. Pending jobs reserve space for an 8 KiB final job record so a quota failure can still be recorded. `GET /api/v1/workflow/storage` reports usage and limits. Capacity refusal preserves saved evidence and requests operator backup followed by a new workspace database; it never silently deletes old snapshots. An existing database that already exceeds a limit remains readable but cannot accept further bounded work until the operator moves to a new workspace.

Observed NDVI/NDMI/NBR spatial deltas use only pixels valid in both first and last selected acquisitions. Seasonal differences, processing effects and remaining cloud/illumination errors can explain change. These are **observed spectral differences**, not classified clearing or calibrated maps of future degradation. The map export retains both dates, masks via null values, grid extent, source hashes and units.

Recovery diagnostics freeze the first 24 consecutive usable months as a calendar-month NDVI baseline. Subsequent negative episodes use a declared 0.1 NDVI threshold; return requires two consecutive observations within 0.05 of baseline. Missing-month gaps and unrecovered final episodes are censored. This is retrospective index return, not an ecological recovery time or evidence of critical slowing down. Cross-episode ecological comparisons still require independent disturbance severity, climate and ecosystem controls.

Optional contextual collection is actual provider data, with its own dated provenance:

- ERA5 daily temperature, rainfall, surface soil moisture and reference evapotranspiration are retrieved at the AOI centroid, using the fixed Open-Meteo archive endpoint. Monthly counts retain null days; sums may be partial. The coarse reanalysis grid is not a forest-only observation or a model embedding. The actual three-day Tapajós smoke artifact is `artifacts/climate-context-pilot.json`. See the [provider documentation](https://open-meteo.com/en/docs/historical-weather-api) for its reanalysis provenance, coverage, latency and hosted usage terms.
- ESA WorldCover provides historical classified tree/mangrove cover, not an authoritative forest boundary. Native map sampling and chosen output grid spacing differ. Annual reference, publication and retrieval dates are retained. A map published after an historical prediction cutoff is a retrospective reference, not a feature available to that historical forecast. These maps never enter model inputs or rolling-origin backtests. Version 100 (2020) and version 200 (2021) have different algorithms; their difference is not reported as forest loss. Patch areas, internal edge density and four-neighbour topology are descriptive at the selected sampling scale. Species-specific habitat connectivity is not measured.
- Sentinel-1 RTC context uses a bounded, same-platform/relative-orbit/pass pair and positive calibrated linear gamma-naught samples. Pixelwise conversion is `10*log10(power)` with no clipping or min-max scaling. Differences use the common valid mask. Radar moisture, terrain, speckle and acquisition-geometry effects remain confounders. Two observations are not a seasonal radar baseline. Sequence alignment can select a prior matching scene within an explicit maximum age, retaining independent radar masks and acquisition dates.

Context source/mask arrays are included in the snapshot export. A failed recollection preserves the previous successful evidence and reports a separate collection error. Adding a contextual layer does not establish that the deployed model is multimodal. Aligned inputs retain timing rules, feature units, masks and hashes; the registered model must pin the same contract and minimum coverage. Optical-only models receive the original optical inputs with auxiliary modalities excluded. ERA5 features use trailing windows and a lag, but are explicitly retrospective reanalysis without verified historical vintage. See [MULTIMODAL_INPUTS.md](MULTIMODAL_INPUTS.md).

## Trusted model registration

Normal application inference never executes an HTTP-supplied checkpoint path, executable upload or synthetic/proxy checkpoint. An operator may set `FORESTMIND_MODEL_REGISTRY` to a locally reviewed JSON file containing a `models` array. Paths in that file resolve relative to the registry; only the operator can configure them. Every entry needs:

```json
{
  "id": "reviewed-study-model",
  "version": "study-version",
  "path": "checkpoints/model.pt",
  "sha256": "64-character-checkpoint-sha256",
  "trained_on_synthetic": false,
  "license": "Actual applicable license",
  "target": "Predefined measurable outcome, units and horizons",
  "training_through": "YYYY-MM-DD",
  "scope_bbox": [-56, -5, -54, -2],
  "input": {
    "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
    "units": "surface_reflectance_0_1",
    "grid_shape": [64, 64],
    "pixel_size_m_range": [10, 50],
    "sampling": "individual_acquisitions",
    "min_steps": 12,
    "max_steps": 48,
    "required_modalities": ["optical"]
  },
  "evaluation": {
    "path": "evidence/held-out-evaluation.json",
    "sha256": "64-character-evaluation-sha256",
    "independence_audited": true,
    "acceptance_record": "Reviewed target, buffered split, metrics, failure modes and acceptance decision"
  }
}
```

The strings, study bounds and scales above illustrate the schema; they are **not an approved model**. The evaluation must be a real-data `test` report for that exact checkpoint. The loaded checkpoint must itself declare real training (`data.synthetic: false` and a nonsynthetic `data.source`). Physical bands, actual AOI pixel spacing, shape, sampling, geographic scope, cutoff and missing modalities must match. All four normalization arrays must be configured in the checkpoint; measured context requires its own normalization and exact feature contract. Automatic execution accepts local `custom_vit` and audited native configurations with matching pinned runtime and complete local state. Development proxies remain refused. No eligible real-trained model is supplied. Registry declarations and a local audit record do not independently authenticate scientific quality.

For independent neural labels, use the [training manifest converter](TRAINING_DATA.md). It freezes source/config/sample hashes, buffered geography, temporal embargoes, input cutoffs and train-only moments. Partial target masks exclude unknown labels from losses and metrics. Each output's support is recorded from observed training labels and enabled loss terms; inference withholds unsupported outputs. A controlled converter/train/evaluate test verifies plumbing only. Source authenticity, ecological definitions, uncertainty, test custody and enough representative examples remain study responsibilities.

Eligible execution prepares a masked sequence from accepted acquisition files, runs on the local CPU and stores a model report plus the shared snapshot manifest. Ineligible execution records concrete reasons; it does not substitute the repository's synthetic demonstration checkpoint.

## Reproducible real-reference baseline experiment

`scripts/benchmark_references.py` supplies the missing conventional comparison harness without manufacturing reference labels. Prepare a versioned JSON manifest with:

- `schema_version: 1`, `dataset_id`, `version`, `synthetic: false` and a fixed `feature_names` list.
- `target`: `name`, `kind` (`continuous` or `event`), `definition`, `units`, `intended_use`, and one positive `horizon_days`. Use one frozen manifest per horizon/outcome.
- `split_policy`: positive `spatial_buffer_km`, nonnegative `temporal_embargo_days`, `protected_test: true`.
- `examples`: unique `id`, `region_id`, WGS84 `bbox`, `split` (`train`, `calibration`, `test`), `cutoff`, `features_available_at`, `target_date` exactly the frozen horizon after cutoff, `features` keyed by every feature name with finite numbers or explicit null, and numeric `target`.
- Each example also requires independent `label_source` metadata (`reference`, `version`, `uncertainty`, `independent: true`), finite `persistence` and `seasonal` predictions, their `persistence_available_at` and `seasonal_available_at` dates, and `quality_group` (`high`, `partial`, `low`). A supplied `season` label is optional; otherwise calendar quarters are reported without claiming biome-specific seasons.
- Event outcomes are 0/1 and baseline scores lie in [0,1]. Positive references additionally require `event_id` and an `event_date` strictly after the cutoff and no later than the horizon. Every repeated `event_id` must identify the same region, event date and protected split; unrelated events cannot be merged for favorable lead-time or detection counts. These support unique-event missed-detection and first-warning lead-time summaries.

At least 10 training, 5 calibration and 5 test examples are required to execute software; those counts are not scientifically adequate sample-size recommendations. Training and calibration event partitions must each contain both classes. Every partition uses disjoint region IDs and conservatively buffered footprints. A spherical bounding-circle lower bound audits spatial separation; it can reject large elongated footprints whose nearest corners would pass. Reduce spatial units or document a separately reviewed spatial-block design rather than silently disabling buffers. Training target windows must end before calibration cutoffs plus the embargo; calibration targets must end before test cutoffs plus the embargo.

```powershell
python scripts/benchmark_references.py freeze --input data/references/study.json --protocol artifacts/study-protocol.json
python scripts/benchmark_references.py run --input data/references/study.json --protocol artifacts/study-protocol.json --output artifacts/study-reference-comparison.json
```

Freeze before inspecting test outcomes. Protocol and result writers refuse to overwrite existing files. The protocol pins the entire data manifest, target, feature list, split policy, model settings, audit result, experiment implementation and NumPy/scikit-learn versions. Changing any test labels, target, features or thresholds afterward makes the existing protocol invalid. A changed implementation/runtime must be restored to the frozen version before execution; it cannot silently change an experiment after its protocol is frozen. These guards cannot enforce human custody or prove that a researcher has never inspected a test label.

The harness compares persistence, seasonal predictions, regularized features (Ridge for continuous targets, logistic regression for event targets), a fixed random forest and fixed gradient-boosted trees on **identical test IDs**. Training-only medians and explicit missingness flags prevent preprocessing leakage. Event sigmoid calibration is fitted on the separate calibration partition; raw and calibrated metrics are both retained. Continuous residual intervals use calibration quantiles and explicitly lack guaranteed coverage under transfer.

The boosted baseline uses scikit-learn's [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) for continuous outcomes and [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) for binary events. Its frozen settings are 100 stages, learning rate 0.05, maximum tree depth 2, at least 3 training samples per leaf, full-sample fitting and seed 17. Loss is squared error or binary log loss respectively. Early stopping is disabled: no calibration/test observations select the tree count or any other model setting. The same training-only imputed features and missingness indicators feed the random-forest and boosted baselines; scaling for regularized features is fitted only on training examples. Changing the boosted settings invalidates an existing frozen protocol. Adding this implementation does not authorize rerunning or retuning an already inspected protected study; preserve its original implementation/runtime or preregister a genuinely new study.

Outputs retain raw predictions/labels, label references, input/protocol/implementation hashes, runtime versions, MAE/RMSE/bias or Brier/average-precision/precision/recall/F1/reliability, event missed detections/lead time, false alert origins per unioned region-year exposure and region/quality/season subgroup counts. Subgroups below five examples withhold their metrics. False alert origins are not deduplicated ecological episodes. Negative improvements over persistence remain in the result. Dependence-aware uncertainty/significance, recurrent/pretrained/multimodal/resilience ablations and authenticated field-label quality still require further experiments; no naive confidence interval is reported. The boosted comparison is implemented, but its accuracy on independent real forest references remains untested.

All harness tests use explicitly documented controlled software fixtures. No real-reference benchmark has been run: an independent versioned reference dataset, protected splits and research acceptance decision are still missing. Successful tests demonstrate audit/model/calibration plumbing, not forest predictive performance.
