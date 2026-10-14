# Completing ForestMind

The application and research code are working foundations. Completion of the proposed ecological early-warning system requires measured evidence at each stage; passing software tests cannot substitute for this evidence.

## Current implementation

The local application supports observation imports, baseline analyses, alerts, sensitivity scenarios, historical backtests and persisted model reports. The research path supports satellite caches, explicit masked sequences, checkpoint inference, supervised training and auditable evaluation outputs. The training demonstration remains synthetic. Current hardware has CPU-only PyTorch and no CUDA device.

## Remaining milestones and acceptance evidence

| Milestone | Work required | Evidence needed to advance |
| --- | --- | --- |
| Verified monitoring pilot | Select one study AOI; acquire and quality-check a sufficiently long optical/radar/climate history; retain provider and processing versions | Actual acquisitions produce dated observations, maps and exports with coverage and missingness accounted for; no fabricated context |
| Reference dataset | Define one measurable future outcome, reference labels and their uncertainty; create buffered spatial and temporal partitions | Versioned label and split manifests, label review and leakage audit; protected test regions and time periods |
| Real model comparison | Verify a native pretrained checkpoint's inputs and offline loading; extract representations; train small heads before considering fine-tuning | Identical held-out examples compared across persistence, seasonal, conventional feature and pretrained-model baselines |
| Useful early warning | Measure lead time and missed/false alerts; calibrate only on a separate development/calibration partition | Untouched-test performance, calibration curves, error analysis, subgroup/region results and uncertainty estimates that account for dependence |
| Intervention evidence | Decide whether scenarios remain sensitivity tools or become causal estimates; obtain suitable intervention/comparison data if claiming effects | Defensible study design and validated outcome comparison; otherwise keep hypothetical assumptions explicit |
| Hosted operational pilot | Choose users and hosting; add authentication, authorization, migrations, recoverable jobs, backups, restore tests, health monitoring and deployment controls | End-to-end access tests, restart/failure recovery, restore drill and documented operating limits |

## Recommended next experiment

Start with a single, explicitly defined degradation or disturbance outcome in one study area. Freeze the target, forecast horizon, reference data, split manifest and evaluation metrics before model tuning. Use the acquisition and evaluation artifacts to investigate whether learned representations improve on the simplest baselines. Expand geography or model complexity only after that comparison supports it.

Real pretrained validation requires the selected checkpoint and a compatible runtime; large experiments require compute appropriate to the measured workload. No paid compute or hosted deployment has been provisioned. A scientific finding of no predictive improvement must be retained and reported, not hidden by changing labels or thresholds on the test set.

## What remains a research claim

Predicting a vegetation anomaly, detecting clearing, assigning a driver score, estimating resilience statistics and establishing an irreversible ecosystem tipping point are different tasks. The current software does not establish the last claim. Operational alerts require a defined user response and an acceptable false-alert rate, both evaluated against independent observations.

See `RESEARCH_PROTOCOL.md` for experimental controls and `VALIDATION.md` for what has actually been exercised in this workspace.
