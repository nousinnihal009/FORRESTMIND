# ForestMind research implementation plan — 2 November 2026

## Audit and preserved behavior

The source, configurations, tests, dataset preparation, checkpoint evidence, SQLite schema and original O01–O59 requirements were reviewed before implementation. The existing local FastAPI/SQLite/plain-JavaScript application remains the foundation. Preserve accepted acquisitions, immutable exports, partial-data masks, model eligibility, alert history, bounded workers, local access controls and all existing user observations.

Real evidence includes Sentinel-2 observations and selected radar, ERA5 and WorldCover context. The current Mudumalai pilot has nine usable months, longest consecutive run eight, no radar context and no model run. Its history cannot satisfy the 12/36-month forecast/resilience rules. Supplied trained checkpoints use synthetic examples; native runtime checks used random states. Dashboard forecasts are historical baselines, and intervention coefficients are assumptions. None establishes predictive validity. The user confirmed that no independently verified reference data is available.

## Ordered milestones and acceptance

| Priority | Deliverable and rationale | Files / interfaces | Dependencies | Acceptance and validation |
| --- | --- | --- | --- | --- |
| 1 | Define an annual gross tree-cover-loss target and auditable positive/negative/uncertain reference decisions. Establish what a model predicts before fitting it. | New reference-review module, CLI, empty template and contribution guide; existing training/reference harness | Dated independent evidence and human review for real labels | Reject missing/contradictory evidence, interval/geometry mismatch, invented event days, future feature availability and unreviewed labels. Preserve source hashes and uncertain examples. |
| 2 | Produce reproducible candidate datasets and protected partitions using existing measured histories and reference records. | Reference-review converter; `data/training_data.py`, history/sequence tools | Longer histories, geographic breadth, reviewed labels, documented source rights | Exact dates/units/footprints; no synthetic fill; disjoint buffered geography; chronological holdouts; immutable output and complete audit. |
| 3 | Extend the conventional comparison with historical-trend inputs, complete event metrics and dependence-aware uncertainty. | `training/reference_experiment.py`, metric helpers, comparison tests | Same reviewed examples; explicitly dated baseline predictions | Identical held-out IDs; train-only transforms; separate calibration; undefined metrics withheld; region-block resampling; negative results retained. |
| 4 | Compare pretrained optical, multimodal temporal and resilience-augmented models using the same protected examples. | Existing foundation/fusion/temporal models, training and evaluation | Licensed actual pretrained weights, adequate labels and compute; outcome-compatible event head | Reproducible training/configuration; model selection on validation only; held-out metrics, calibration and ablations. No random-weight or synthetic performance claims. |
| 5 | Complete climate/fire/human-pressure histories and spatial event probabilities. | Climate diagnostics, product adapters, model/event/raster contracts | Long frozen climate baseline; product access; aligned spatial labels and independently calibrated predictor | Past-only standardized anomalies; provenance/quality/caching; no sparse-history drought diagnosis; spatial calibration and geographic evaluation. |
| 6 | Extend scenarios with explicit assumption ranges, then ecological/process mechanisms where evidence permits. | `monitoring/projections.py`, scenario API/UI | User-defined ranges for arithmetic sensitivity; intervention/control evidence for causal claims | Central trajectories unchanged; signed-coefficient bounds; ranges clearly distinct from confidence intervals; immutable manifest; no invented treatment effects. |
| 7 | Improve research dashboards and hosted operations after scientific interfaces stabilize. | Existing UI/API and operational controls | Reviewed model artifacts; deployment/account decisions | Traceable model metrics and source dates; accessibility; authentication/ownership/load/recovery acceptance before hosting. |

Each software milestone receives focused tests and the complete regression suite after significant changes. Fixture tests remain explicitly software tests. Scientific milestones stay blocked until real evidence is supplied and evaluated.

## Mapping all original sections

The authoritative detailed checklist remains `REQUIREMENTS_MATRIX.md` (O01–O59). Status here describes the full requested capability, not just an interface.

| Original section | Status at audit | Next milestone / remaining dependency |
| --- | --- | --- |
| 1 Problem and four intelligence layers | Incomplete | 1–6; real predictive and intervention evidence |
| 2 Modular architecture | Incomplete | 4–5; trained, evaluated components integrated into working orchestration |
| 3 Data sources and preparation | Incomplete | 2, 5; radar history, Landsat UI/harmonization, fire and human-pressure sources |
| 4 Foundation models | Blocked | 4; actual licensed pretrained weights and independently trained/evaluated heads |
| 5 Multimodal fusion | Incomplete | 4–5; human inputs and real same-example ablations |
| 6 Forest states | Blocked | 1, 4; independent biome-appropriate condition references |
| 7 Attribution | Blocked | 4–5; verified multilabel disturbance references and temporal evidence |
| 8 Temporal intelligence | Blocked | 2–4; sufficient real histories, references and comparative evaluation |
| 9 Forecasting | Incomplete | 1–5; defined target, real training, calibration, useful held-out forecasts |
| 10 Resilience and tipping points | Incomplete | 2, 4–6; long histories, comparable disturbances and longitudinal ecological targets |
| 11 Risk engine | Incomplete | 3–5; spatial event model, calibration, ecological consequence/utility decisions |
| 12 Interventions | Incomplete | 6; explicit sensitivity ranges then intervention/process evidence |
| 13 Software architecture | Complete for documented local workflow | Preserve local stack; hosted operation is separate milestone 7 |
| 14 Dashboard | Incomplete | 5, 7; genuine spatial future risk and calibrated uncertainty/evaluation displays |
| 15 Milestones and A–D comparisons | Incomplete | 1–4; all models on identical protected real examples |
| 16 Beyond segmentation | Incomplete | 4–5; demonstrated forecasting contribution beyond observed cover |
| 17 Scientific contribution | Blocked | 3–4; useful improvement on independent real references |

## Current execution boundary

Begin with target/reference intake and evaluation improvements. No real model training is authorized by a fabricated reference, a source-name declaration, or a passed test. Public reference products can supply candidate annotations; independent review and source limitations remain explicit. Subsequent work must update this plan, the detailed matrix and validation record with actual results. This plan does not mark the overall project complete.

## Continuation on November 4

Target/reference intake, source-verified packets, independent baseline-mask evidence, effective review availability and guarded partitions are implemented. Valid continuous targets remain eligible when only the derived significant-loss classification is ambiguous. The baseline harness adds historical trend, class/calibration metrics and bootstrap resampling that merges nearby test footprints. Scenario input ranges now produce signed-coefficient sensitivity bands and tables, with stale-request protection and persisted exports. Standalone climate tools freeze sufficiently long, same-source references and calculate complete-month departures with explicit missingness. These are software increments; real training/evaluation remains blocked by the user's missing verified references, and actual climate departures need a suitable historical dataset. Executed checks and limits belong in `VALIDATION.md`.

The November 4 complete regression finished successfully: 1,057 tests passed in 200.87 seconds, with no skips or failures. JavaScript syntax and whitespace checks passed. The remaining priorities are still the real historical/reference dataset and protected baseline study, followed by eligible pretrained/multimodal comparisons; fire/human-pressure integrations, spatial future-risk modeling, ecological intervention mechanisms and hosted operation remain incomplete as recorded in the matrix.

## Continuation on November 5

Public reference acquisition and a real experiment have replaced the blanket assumption that no external labels can be obtained. A pinned 229,591-byte Zenodo archive supplies 10,339 interpreted point histories. The separate, explicitly retrospective study freezes geography, years, code and settings before fitting logistic, random-forest and boosted baselines. It retains 1,422 test examples and all results, including zero recall at the fixed threshold; it does not replace the 100 ha outcome with a point target. Original source metadata/checksums, unknown transitions, duplicate-coordinate exclusions, protocol and predictions are saved.

Mudumalai now has a real 30-year ERA5 reference and five later observed years. An immutable, resumable builder verifies provider/grid matching, source hashes, budgets and derived artifacts. The primary annual-area outcome also has its previously missing optional neural head, scalar masks, loss, target metadata, checkpoint support, independent metrics and inference withholding. Disabled configuration preserves legacy model parameter keys. Full regression and actual experimental metrics are recorded in `VALIDATION.md`.

The next predictive experiment needs measured optical/radar features and area-compatible independent labels, with newly protected evaluation data. Current results cannot be tuned on the already inspected test set. Additional native pretrained comparisons, fire/human-activity integration, spatial risk, causal interventions and hosted operation remain explicit implementation/research work.

Final November 5 verification: 1,125 full-suite tests passed in 157.50 seconds, plus an overlapping 21-test final preview/import check. The initial full-run Windows progress-file replacement failure is retained in the validation log; bounded write retries now pass fault-injection tests. Actual reference and climate files, source-code snapshots, frozen protocol and test predictions are preserved. No observation-database migration or model registration was performed.
