# ForestMind

**Forest observation, evidence, and early-warning research in one local workspace.**

ForestMind v0.7 connects worldwide place search, confirmed small-area selection, real satellite acquisition and continuing monitoring in one local application. It now prepares dated optical, radar and climate inputs, converts independently supplied references into protected training datasets, and supports offline loading of complete native model checkpoints. Inspect measured change, context, exploratory forecasts, alerts and hypothetical scenarios. No independently validated real-trained model is bundled.

The included Amazon regions are **synthetic demonstrations**. The dashboard's condition index, forecast envelopes and scenario coefficients are transparent, uncalibrated baselines. They do not establish ecological tipping points, causal drivers or intervention effectiveness. Imported measurements retain their source and an unverified-provenance label.

## Run the application

Python 3.10+ is required. From the repository root:

```powershell
python -m pip install -e ".[web,real-data,dev]"
python scripts/serve.py
```

Open **http://127.0.0.1:8000**. Interactive API documentation: **http://127.0.0.1:8000/docs**.

The server enforces loopback access. Regions, analyses, alerts, schedules, scenarios and jobs persist in `data/forestmind.db`; source artifacts live in adjacent `forestmind-acquisition` and `forestmind-workflow` directories. This is a **single-process, single-user local application with no authentication**. Public hosting is disabled pending authentication and user isolation. Use `--port 8001` if needed; `--reload` enables development reloading. Stop with Ctrl+C. Local schedules run only while the service is running.

Choose a separate study database in PowerShell:

```powershell
$env:FORESTMIND_DB = "data/my-study.db"
python scripts/serve.py
```

The frontend is packaged HTML/CSS/JavaScript served by FastAPI; no Node build is needed. The overview uses schematic footprints; Find a forest uses attributed OpenStreetMap tiles. Place search uses Photon/OSM and does not establish a verified forest boundary. Optional Google Fonts have system-font fallbacks.

## Search, collect and keep monitoring

1. Open **Find a forest**. Search a forest/place, enter coordinates, or draw a small rectangle. Review the country, exact bounds and area. Confirm the rectangle separately from the named-place result.
2. Choose dates, cloud/valid-pixel settings and a download budget. Start collection; progress, missing months, cancellation and explicit resume stay visible. No missing observations are invented.
3. Open the saved area after its first usable observation. **Data coverage** explains gaps and history requirements. **Connected analysis** shows spatial optical changes and can collect real ERA5 climate, ESA WorldCover and orbit-matched Sentinel-1 context for the same geometry/cutoff.
4. In **Monitoring updates**, choose a history start/end and bounded download budget to collect absent months, including older history for the same area, or configure a local schedule. Existing months are preserved. Review dated alerts, acknowledge or resolve them, and export reproducible evidence.

Normal monitoring needs no Python commands or manual model-report import. A registered model runs only if its checkpoint, independent evaluation, input compatibility and supported geography pass checks. No eligible real-trained model is bundled. Real observations are never run through the synthetic demonstration model as an ecological prediction.

The saved [Mudumalai pilot in Tamil Nadu](docs/pilots/MUDUMALAI.md) documents a small real-data study area, its public sources and the user's ₹0 operating constraint. It is a local pilot, not a claim of complete reserve coverage. The app remembers the selected area across reloads and prefers an observed area when no selection is stored.

Limits include a 25 km² maximum rectangular AOI, 60 months per acquisition request, a single worker and bounded downloads/storage. Forecasts need at least 12 consecutive usable months; descriptive autocorrelation/variance needs 36. These are software eligibility rules, not scientific validation. See the [operator guide](docs/OPERATOR_GUIDE.md), [acquisition limits and recovery](docs/ACQUISITION_OPERATIONS.md), [requirement-by-requirement status](docs/REQUIREMENTS_MATRIX.md) and [actual verification evidence](docs/VALIDATION.md).

## Implemented capabilities

| Area | Behavior |
| --- | --- |
| Worldwide entry | Explicit attributed place search, interactive map, drawing/coordinates and confirmed bounded area |
| Connected jobs | Durable acquisition, bounded retries, cancellation, restart recovery, resume and transactional saved-area updates |
| Multimodal inputs | Past-only, grid-matched Sentinel-1 RTC and trailing ERA5 features aligned to optical dates, separate pixel/feature masks, unit and coverage checks, and registered model compatibility |
| Reference datasets | Independent label manifests, spatial buffers and temporal embargoes, train-only normalization, partial supervision and artifact integrity checks |
| Spatial evidence | Computable common-valid spectral change; classified-cover patches, internal edges, complete-boundary compactness and centroid connectivity |
| Continuing monitoring | Skip existing months, local schedules, freshness diagnostics, configurable dated alert episodes and review history |
| Monitoring | Four reproducible demo regions; strict JSON import for dated monthly measurements |
| Spatial overview | Selectable WGS84 footprints, condition/coverage/alert layers and zoom |
| Evidence | NDVI/NDMI/NBR history, candidate-driver evidence, coverage and dated alert explanations |
| History coverage | Monthly quality calendar, explicit gaps, consecutive usable runs, date-window checks and JSON export |
| Forecasting | 30/180/365-day baseline with uncalibrated residual envelopes; withheld for inadequate histories |
| Resilience | Descriptive season-adjusted autocorrelation and variance, with validity checks |
| Backtesting | Persistence, seasonal naive and damped trend at historical cutoffs; auditable per-horizon exports |
| Scenarios | Saved pressure/fire/restoration/climate comparisons and five-year hypothetical index sensitivity; no causal effect claims |
| API/jobs | REST endpoints, SQLite persistence, background jobs and interrupted-job recovery |
| Export | Region JSON, observation CSV, scenario JSON and backtesting JSON with provenance |
| Satellite pipeline | Sentinel-2/Sentinel-1/Landsat adapters, common-grid caches and optical-cache import preparation |
| Repeatable acquisition | Frozen monthly Sentinel-2 candidates, bounded downloads, resumable progress and accepted-only exports |
| Neural research | Custom ViT, pinned offline Prithvi/TerraMind checkpoint loading, measured-context fusion, temporal model and task heads; outputs without training support withheld |
| Model runs | Validated, persisted inference reports with checkpoint/input hashes, class/driver scores and configured horizons |
| Reproducible evaluation | JSON metrics, raw predictions and targets, split identity, artifact/code hashes and per-horizon errors |
| Real-reference experiments | Frozen targets/datasets, buffered holdouts, identical-example persistence/seasonal/regularized/random-forest/boosted baselines, calibration and event metrics; independent real reference data must be supplied |
| Operations | Local HTTP boundary, diagnostics, migrations, database snapshots, verified full workspace backup/restore and opt-in signed webhook outbox |

## Import observations

Choose **Import observations** in the dashboard. **Use example template** loads a valid synthetic example; retain `synthetic: true` for fabricated data. The template is also available at `GET /api/v1/import-example`.

```json
{
  "name": "My monitoring AOI",
  "country": "Brazil",
  "source": "Sentinel-2 composites; document provider and processing version",
  "synthetic": false,
  "notes": "Describe masking, aggregation and measurement units here.",
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[-55.1,-3.5],[-55.0,-3.5],[-55.0,-3.4],[-55.1,-3.4],[-55.1,-3.5]]]
  },
  "observations": [
    {"date":"2025-01-20","ndvi":0.78,"ndmi":0.42,"nbr":0.61,"valid_fraction":0.85}
  ]
}
```

Dates must be strictly increasing, unique, and at most one per month. Accepts 1–600 records and a simple closed WGS84 polygon without holes or antimeridian crossing. Missing optional measurements may be null or omitted; do not use zero for missing data. Indices lie in `[-1,1]`; valid fraction and forest cover in `[0,1]`. Optional context: precipitation anomaly (standard deviations), temperature anomaly (°C), SAR backscatter change (dB), forest cover and independently supplied fire anomaly. Invalid numeric types, nonfinite values, duplicate dates and malformed polygons are rejected.

A usable condition score requires all three spectral indices and at least 40% valid-pixel samples. Forecasting requires 12 consecutive usable months and a usable latest observation. Dashboard resilience diagnostics require at least 36 consecutive usable months. These are screening choices, not ecological standards.

## Advanced satellite commands

These commands remain available for research and migration. Use **Find a forest** for the connected application workflow above.

```powershell
python -m pip install -e ".[web,real-data]"
python scripts/download_data.py --sensors s2 --region=-55.10,-3.50,-55.08,-3.48 --start 2023-01-01 --end 2026-01-01 --max-scenes 100 --cache-dir data/real
python scripts/prepare_observations.py --cache-dir data/real --sensor sentinel2 --name "My Sentinel-2 AOI" --country Brazil --region=-55.10,-3.50,-55.08,-3.48 --output data/my-region.json
```

Import `data/my-region.json`. Keep `data/my-region.provenance.json`: it records acquisition timestamps, cache hashes, provider metadata and aggregation quality counts. Monthly indices use valid pixel samples only. Empty months and auxiliary climate/fire/cover data are never invented.

For Sentinel-2, an explicit alternate provider is available through [Element 84 Earth Search](https://github.com/Element84/earth-search). This small pilot command requests one scene from its `sentinel-2-c1-l2a` collection:

```powershell
python scripts/download_data.py --provider earth-search --sensors s2 --region=-55.10,-3.50,-55.08,-3.48 --start 2024-07-01 --end 2024-07-31 --max-cloud 25 --max-scenes 1 --patch-size 64 --timeout 30 --cache-dir data/real-pilot/earth-search
python scripts/prepare_observations.py --cache-dir data/real-pilot/earth-search --sensor sentinel2 --name "Tapajos real Sentinel-2 pilot" --country Brazil --region=-55.10,-3.50,-55.08,-3.48 --output artifacts/real-pilot-region.json
```

The default provider remains `planetary-computer`; failures never silently switch providers. Earth Search currently supports Sentinel-2 only. It requires explicit per-band reflectance scaling metadata and uses all twelve optical bands plus SCL cloud masking. Provider and collection identities are retained in the cache and provenance. The scene cap applies to the provider's returned subset, which is then sorted locally; it does not guarantee the earliest acquisitions. A one-scene pilot establishes data plumbing, not a history sufficient for forecasting.

The bridge requires current, explicitly georeferenced cache metadata; regenerate legacy caches. Bands are reprojected to the same WGS84 AOI grid. **Output resolution is AOI extent divided by patch size**, not an assumed 10 m resolution. A large bbox squeezed into 128 pixels is a coarse sample. Choose appropriate spatial units and inspect source quality.

Downloads require provider availability and the optional raster/STAC stack. Network requests have bounded timeouts (`--timeout`, in seconds); failed or partial downloads return a nonzero exit status. Numerical tests validate local grid alignment and data contracts; they do not prove all remote collections are accessible or establish ecological validity.

Current caches use **`forestmind-grid-v3`**. Sentinel-2 reflectance processing now accounts for the product processing baseline, including the offset needed for baseline 04.00 and later when asset metadata omit scaling. Regenerate v2 caches before preparing observations or inference sequences. See the [provider's baseline-change example](https://github.com/microsoft/PlanetaryComputerExamples/blob/main/datasets/sentinel-2-l2a/baseline-change.ipynb).

### Build a repeatable monthly history

Freeze a monthly acquisition plan before downloading. This example searches twelve months for the same small AOI. Each execution may attempt at most two new downloads and six candidates overall:

```powershell
python scripts/build_history.py plan --plan artifacts/history-plan.json --provider earth-search --region=-55.10,-3.50,-55.08,-3.48 --start 2024-07-01 --end 2025-06-30 --patch-size 64 --candidates-per-month 3 --search-limit 30 --max-attempts 6 --max-downloads 2 --min-valid-fraction 0.6 --timeout 30
python scripts/build_history.py execute --plan artifacts/history-plan.json --report artifacts/history-progress.json --cache-dir data/real-pilot/earth-search --output artifacts/history-batch-1.json --name "Tapajos monthly satellite pilot" --country Brazil
python scripts/build_history.py execute --plan artifacts/history-plan.json --report artifacts/history-progress.json --cache-dir data/real-pilot/earth-search --resume --output artifacts/history-batch-2.json --name "Tapajos monthly satellite pilot" --country Brazil
```

The plan records the exact UTC month windows, provider, processing settings and unsigned candidate metadata. Candidates are ranked by scene cloud cover, acquisition time and scene ID within the bounded catalog subset. Execution checks one candidate per missing month each round, starting with months having fewer previous attempts, including after resume. It checks actual AOI quality and accepts at most one acquisition per month. This sampling rule differs from compositing every scene in a month; it can introduce sampling bias and requires scientific review for a study.

Download budgets count attempted acquisitions, including failed requests, and reset on each explicit execution. Completed cache files are checked again when resuming; modified or missing files are never silently accepted. Progress is saved atomically around each attempt. Plan/report hashes detect changes to saved artifacts, not source authenticity. A failed or incomplete run returns exit code **2**, retains its progress, and can still export accepted months. Resume continues untried candidates and interrupted attempts; rejected or failed candidates are not silently retried. A new plan is needed to change provider, dates, search bounds or candidate selection.

Use a new output filename on each run; output and provenance files are never overwritten. Keep the plan, progress report, cache and export sidecar together. The export includes only accepted scenes, excluding cloudy or rejected alternatives that remain in the cache. Missing months, SAR, climate context and labels are never filled in. Import the latest batch as a new region snapshot; previous imports remain available for comparison.

In **Data coverage**, set the desired study period to see missing and unusable months, the longest consecutive usable run, and whether observations reach the requested ending month. For collected areas, the initial period covers all requested study dates, including failed backfills and gaps outside the actual observation span. Imported and demonstration records default to their observed span. Inspecting coverage starts no downloads. Readiness means the current software's minimum history requirements are met; it does not establish predictive accuracy or a tipping-point warning.

## Backtest before interpreting forecasts

The dashboard's **Backtesting** tab and CLI share one evaluation module:

```powershell
python scripts/backtest_monitoring.py --input data/my-region.json --output artifacts/my-region-backtest.json
```

Predictions use only history through each cutoff. Reports include MAE/RMSE, counts, skipped targets and empirical range coverage where bounds exist. Nominal 30/180/365-day forecasts are compared with actual dated observations in the matching 1/6/12 calendar months; both dates are retained.

Targets remain a fixed spectral index, **not independent field labels**. Overlapping origins are dependent. Synthetic backtesting checks software behavior only. See [the research protocol](docs/RESEARCH_PROTOCOL.md).

## Neural research workflow

```powershell
python scripts/train.py --config configs/quick_test.yaml
python scripts/evaluate.py --checkpoint checkpoints/quick_test/best_model.pt --split test --output artifacts/quick-test-evaluation.json
python scripts/demo_prediction.py
```

The quick test uses a tiny synthetic dataset and creates a new checkpoint. The no-checkpoint demo uses random weights and says so. Neither synthetic training nor a successful forward pass establishes ecological performance.

Real satellite caches contain observations, not health/driver/future labels. Use the [training data guide](docs/TRAINING_DATA.md) and its empty reference templates to attach independent, dated labels with source/version and uncertainty. The converter audits spatial buffers, temporal embargoes, input cutoffs and source hashes, then writes protected `train/`, `val/`, optional `calibration/` and `test/` partitions, training-only normalization, an audit report and a runnable configuration:

```powershell
python scripts/prepare_training_data.py --manifest data/my-training-manifest.json --check
python scripts/prepare_training_data.py --manifest data/my-training-manifest.json --output data/my-training-dataset
python scripts/train.py --config data/my-training-dataset/training-config.yaml
```

The paths above are placeholders for your references. Missing targets remain masked: an unknown driver is not a negative example, and a model output without training support is withheld. Partial supervision can cover any supported head without inventing the other labels. The trajectory head predicts condition on a declared 0–1 scale; canopy loss is a separate event target for the conventional reference experiment. Declaring a source independent does not establish its authenticity. Legacy manually prepared disk datasets remain supported, but lack the converter's stronger split and integrity audit.

### Pretrained encoders

```powershell
python -m pip install -e ".[foundation]"
```

Choose `foundation.name: prithvi` or `terramind`. Offline prediction/evaluation constructs a pinned native architecture with `pretrained=False`, then requires a complete, finite local ForestMind state with exact keys, shapes and dtypes. Missing weights or an incompatible runtime fail clearly. `use_proxy: true` explicitly selects an **untrained development encoder**; a proxy is never silently substituted and is refused by normal connected analysis.

Native training/evaluation requires explicit `data.normalization` optical/SAR means and positive standard deviations. Match the checkpoint's documented units, band order, normalization and resolution. TerraMind uses native modality keys; Prithvi maps narrow NIR to Sentinel-2 B8A. Landsat does not measure Sentinel-2 red-edge/B8A bands, so unavailable channels are not fabricated. Verify compatibility before using a harmonized stack.

Actual Prithvi and TerraMind CPU construction, strict checkpoint roundtrip and forward execution were verified with random temporary states and network calls blocked. This verifies execution, not pretrained benefit or ecological accuracy. See the [native model audit](docs/NATIVE_MODEL_AUDIT.md) for exact versions, supported variants and evidence. Licensed pretrained weights, real-trained heads, GPU execution and independently labeled benchmarks still need their own validation. Older architecture checkpoints may require retraining.

### Connect a trained checkpoint to the dashboard

The normal **Connected analysis** workflow automatically prepares accepted acquisitions and runs an eligible operator-registered checkpoint. Configure the trusted local `FORESTMIND_MODEL_REGISTRY` using [the research protocol](docs/RESEARCH_PROTOCOL.md). The browser never executes uploaded checkpoints. **Model runs** also supports portable reports from `scripts/predict.py` for advanced research; those imported outputs are explicitly user-provided and unverified. Observation-based scores and alerts retain their own methods.

First prepare an optical sequence from current Sentinel-2 caches:

```powershell
python scripts/prepare_sequence.py --cache-dir data/real --region=-55.10,-3.50,-55.08,-3.48 --through 2025-12-31 --time-steps 12 --output data/my-sequence.npz
```

This selects up to twelve actual acquisitions at or before the cutoff; it does not create monthly composites or fill missing dates. Poor-coverage acquisitions are skipped and listed. The NPZ contains raw reflectance, exact acquisition timestamps, masks, and no labels. Without optional context, SAR is absent and masked. The [multimodal input guide](docs/MULTIMODAL_INPUTS.md) covers radar/climate arguments and exact matching rules. Connected analysis applies the same alignment to saved valid context and reports missing coverage. ERA5 remains retrospective reanalysis, without proof it was available at a historical forecast time. A provenance sidecar records sources, units, matching policy and dates. A registered model receives only the modalities and coverage its contract accepts.

With a compatible trained checkpoint, use the region ID shown in **Model runs** and the last actual acquisition date printed by the preparation command:

```powershell
python scripts/predict.py --checkpoint checkpoints/my-study/best_model.pt --input data/my-sequence.npz --region-id YOUR_REGION_ID --observed-through YYYY-MM-DD --output artifacts/my-model-run.json
```

The study checkpoint and date above are placeholders for your own artifacts. Real inputs require all four normalization arrays (`optical_mean`, `optical_std`, `sar_mean`, `sar_std`) in the checkpoint's `data.normalization`. Synthetic inputs require `--synthetic-input`; embedded metadata must agree. Missing masks, inconsistent dates, unsafe object arrays, unsupported dimensions and nonfinite values are rejected. No synthetic imagery is generated by inference.

The offline prediction command supports local `custom_vit`, explicit development proxies and audited native Prithvi/TerraMind configurations with complete local checkpoints and a matching pinned runtime. Native support does not relax the connected workflow's independent evaluation, geography, units, input-quality or real-training requirements.

Reports retain the canonical health/driver names and configured horizons. Unsupported outputs are null; an unsupported health-class head has no class scores. Supported scores remain uncalibrated. Forecast standard deviations are model outputs, not validated confidence intervals. A report hash identifies an artifact; it does not authenticate a source or establish ecological accuracy.

### Evaluation artifacts

`evaluate.py --output` saves metrics together with every example's predictions, labels and target masks, checkpoint/effective-config hashes, input and target hashes, source manifests, implementation hashes, runtime versions, configured horizon errors, and losses weighted by observed labels. Unobserved metrics are null with zero observed counts. The report distinguishes development (`val`) and held-out (`test`) partitions. A config override cannot silently relabel trained forecast horizons. Common credential fields and signed URL queries are redacted from the shareable config record.

This output supports subsequent comparisons and calibration experiments; the command does not fit a calibrator or establish model validity. Synthetic evaluation is explicitly identified as a software check.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/v1/status` | Service/method status |
| GET | `/api/v1/overview` | Region summaries, coverage and counts |
| POST | `/api/v1/regions/import` | Validate, save and analyze measurements |
| GET | `/api/v1/regions/{id}` | Full evidence record |
| GET | `/api/v1/regions/{id}/health` | Condition, components, quality |
| GET | `/api/v1/regions/{id}/forecast` | Forecasts and limitations |
| GET | `/api/v1/regions/{id}/drivers` | Candidate-driver evidence |
| GET | `/api/v1/regions/{id}/alerts` | Dated region alerts |
| GET | `/api/v1/regions/{id}/evaluation` | Historical baseline comparison |
| GET | `/api/v1/regions/{id}/coverage?start=YYYY-MM-DD&end=YYYY-MM-DD` | Calendar coverage and method readiness for optional inclusive date bounds |
| POST | `/api/v1/regions/{id}/model-runs` | Validate and persist a portable inference report |
| GET | `/api/v1/regions/{id}/model-runs` | Saved neural inference reports for a region |
| GET | `/api/v1/regions/{id}/model-runs/{run_id}` | One stored report and import provenance |
| POST | `/api/v1/analysis` | Start a job: `{ "region_id": "xingu" }` |
| GET | `/api/v1/analysis/{job_id}` | Persisted job status/result |
| POST | `/api/v1/scenarios` | Save a sensitivity comparison |
| GET | `/api/v1/scenarios/{id}` | Retrieve comparison |
| GET | `/api/v1/regions/{id}/export?format=json` | JSON report; `csv` exports measurements |

## Verification

```powershell
python -m pytest tests -q
node --check forestmind/web/app.js
node --check forestmind/web/model-runs.js
node --check forestmind/web/coverage.js
```

Install `[web,dev,real-data]` for optional raster tests; without Rasterio they skip explicitly. Tests cover input validation, SQLite/job recovery, missingness, temporal information boundaries, scenario invariants, reprojection and controlled pretrained-adapter contracts. Mocked encoder tests are not real checkpoint benchmarks.

## Architecture

```text
Search/map → confirmed area → durable bounded acquisition queue
    → provider scenes → aligned masked cache → transactional SQLite observations
    → immutable geometry/cutoff/source/code/model manifest
    → spatial change / context / eligible model / diagnostics / hypothetical scenarios
    → FastAPI → dashboard / exports / alerts / refresh and local schedules

Separate research path:
observations + independent labels + explicit data partitions
    → normalization → EO encoder → measured-context fusion
    → masked temporal model → condition / driver / trajectory heads
    → exploratory risk diagnostics and scenario sensitivity
```

`api`, `monitoring` and `web` implement the application. `data/adapters`, `models`, `training` and `risk` implement satellite/ML research. Workflows live in `scripts`, with regression tests in `tests`.

See [methodology and limitations](docs/METHODOLOGY.md), [the research protocol](docs/RESEARCH_PROTOCOL.md), and [the remaining completion milestones](docs/COMPLETION_PLAN.md) before interpreting results.

## References

- [TerraMind official implementation](https://github.com/IBM/terramind) and [TerraTorch integration](https://github.com/torchgeo/terratorch/blob/main/docs/guide/terramind.md).
- [Prithvi-EO-2.0 model card](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL).
- [Remotely sensed resilience of tropical forests](https://www.nature.com/articles/nclimate3108).
- [Emerging signals of declining forest resilience under climate change](https://www.nature.com/articles/s41586-022-04959-9).
- [Rasterio: normalizing images to a consistent grid](https://rasterio.readthedocs.io/en/stable/topics/virtual-warping.html).

License: Apache-2.0.
