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
