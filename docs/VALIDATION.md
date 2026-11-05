# Build verification

## Worldwide landscape planning — November 5, 2026

Product scope is forests worldwide; Mudumalai remains one saved pilot. The new landscape service stores versioned study-rectangle plans in SQLite, partitions them into acquisition-compatible areas, handles international-date-line crossings and pages very large plans without creating all tiles or download jobs. The dashboard connects a chosen tile to the existing acquisition form. Exact coordinates survive selection, Fit and Apply; new selections invalidate boundary confirmation. Slow list/page responses cannot replace a later user selection. Coverage reports saved nonsynthetic observation presence for exact rectangular footprints, not inferred ecological condition or completeness.

- **41 backend tests passed in 7.78 seconds**, including Amazon, Congo, Borneo, India, Europe, Canada, Siberia, Australia, Chile, date-line crossings and both supported latitude limits. Generated areas pass the existing acquisition and polygon validators. Tests cover adjoining boundaries/area conservation, idempotency, plan integrity, SQLite backup/reload, capacity limits, synthetic/empty/overlap exclusion, duplicate footprints and offline acquisition → partial → resume integration. A world-sized 1 km² plan defines over 100 million areas while storing less than 2,500 bytes and generating only the requested final three tiles. These are controlled geometry/workflow checks, not observations from those forests. An initial sandboxed focused run stalled at the in-process HTTP client; the escalated isolated run completed.
- **61 UI checks passed in 7.79 seconds**, including 19 new executable planner checks plus existing plain-language and operations tests. They cover exact coordinate retention, confirmation reset, API response contracts, pagination, saved selection, stale responses, errors, escaped names, storage denial and export scope. Counts overlap the full suite and are not added to it.
- **Complete regression: 1,186 passed in 468.58 seconds**, no failures or skips, exit 0. Command: `python -m pytest tests -q --tb=short -p no:cacheprovider --basetemp=artifacts/pytest-worldwide-final-20261105`, with `PYTHONPATH=artifacts/geo-validation`. Output is saved in `artifacts/regression-worldwide-final-20261105.txt`. All packaged web JavaScript files passed `node --check`; `git diff --check` passed.
- **Actual browser check:** the available in-app browser successfully opened an isolated localhost preview on port 8012, using `artifacts/global-browser-20261105/preview.db`. A Congo-coordinate rectangle `[21,-1,21.2,-0.8]` saved as 20 monitoring areas across 494.51 km². Selecting its first area populated the acquisition form with the correct name, location and bounds; boundary confirmation remained unchecked. No collection was started.
- A crossing rectangle `[179.9,-17,-179.8,-16.7]` saved as 56 areas across 1,065.01 km², with ordinary tiles on both sides of 180°. Browser pagination showed areas 21–40; reloading retained the selected saved plan. The downloaded JSON plan correctly reported zero observed areas and all 56 missing. A copy is retained at `artifacts/global-browser-20261105/exported-forest-plan.json`. The queue remained at zero saved collection jobs. The actual narrow browser layout was visually inspected and no browser console errors were returned. Other viewport sizes were not separately exercised in this run.

These preview rectangles are UI test plans, not verified forest boundaries or acquired observations. The original monitoring database and earlier public experiment were not modified. No new provider acquisition, operational checkpoint, paid service or deployment was introduced. Global planning does not establish a worldwide forest inventory, automatic acquisition of every tile, complete local history, calibrated global forecasts or ecological tipping-point validity. See [GLOBAL_MONITORING.md](GLOBAL_MONITORING.md).

## Public evidence, actual baseline experiment and annual-loss model — November 5, 2026

### Actual data and experiment

- Downloaded the pinned, published CC BY 4.0 global forest-reference archive from Zenodo record 6403863: **10,339 interpreted points**, 229,591 ZIP bytes, provider MD5 `e56c8251ca391693674bf00a9a1bfb77`. Original metadata, archive and SHA256 manifest are retained in `data/public-references/global-forest-v1.1/`. The decoder preserves unknown transitions and the original attributes; these are published visual interpretations, not fabricated fixture labels or a new field campaign. [PUBLIC_REFERENCES.md](PUBLIC_REFERENCES.md) records the source, definitions and limitations.
- An initial identity audit refused duplicate coordinates before fitting or test-performance inspection. The final fixed rule excludes all 112 records at duplicated coordinates without selecting among their labels; 446 points are excluded by the cell boundary buffer. The study retains 24,567 eligible point/year examples: **21,679 train, 527 validation, 939 calibration, 1,422 test**. Nearest cross-partition centre distance is **20.7677 km**. Earlier/later calendar windows and disjoint geographic cells are explicit; interpretation publication dates are never backdated.
- The frozen protocol is `artifacts/public-reference-study-20261105/protocol.json`; actual fits, metrics, bootstrap intervals and all test predictions are in `results.json` in that directory. The run exited 0; CLI output is retained in `run.log`. Logistic regression, random forest and gradient boosting used training-only predictors/transforms, with optional separate calibration. Validation/test did not select model settings. No deployable model was exported.
- Test data contains **11 positive events across 1,422 origins and 77 geographic cells**. Every method has **zero positive recall at the fixed 0.5 threshold**. Small Brier changes do not establish useful warning performance. The sampling strata and retained histories are not a population prevalence sample; scores and calibration apply to this selected retrospective benchmark only. Inputs are past interpreted history and coordinates, not observed satellite features or pretrained embeddings. This is separate from the primary 100 ha annual-area target.

| Method | Test Brier (lower is better) | AUROC | Average precision |
| --- | ---: | ---: | ---: |
| No new loss / last-year event / past event rate | 0.00773558 | 0.50000 | 0.00773558 |
| Training event rate | 0.00767927 | 0.50000 | 0.00773558 |
| Logistic | 0.00770235 | 0.58598 | 0.02118197 |
| Logistic, calibrated | 0.00766580 | 0.58598 | 0.02118197 |
| Random forest | 0.00773447 | 0.53254 | 0.00906374 |
| Random forest, calibrated | 0.00769186 | 0.53254 | 0.00906374 |
| Gradient boosting | 0.00768471 | 0.63063 | 0.01161677 |
| Gradient boosting, calibrated | 0.00767785 | 0.63063 | 0.01161677 |

These are observed results, not selected targets or guarantees. For example, the calibrated logistic Brier improvement interval over no-new-loss is approximately **[-0.00001226, 0.00017028]** and includes zero. The raw logistic and constant training-rate comparisons retain their positive conditional intervals as well; they still have zero event recall at the fixed threshold. All intervals are saved, use 1,000 paired geographic-cell replicates, and exclude fitting uncertainty. Annual labels cannot support exact-day lead times. Preserve this completed study; do not retune or refreeze its inspected test set.

- Real Mudumalai ERA5 acquisition completed all four source windows: **1991–2020 reference plus 2021–2025 observations**. The packet `data/research-climate-mudumalai-20261105/` contains raw provider contexts, immutable reference and departure artifacts, plan and progress. All **48** reference feature/month pairs and **240** later departures across **60 months** are available. A completed resume made zero new requests and reproduced the saved products. These climate data do not fill missing optical observations or supply forest outcomes.
- The primary annual-area neural contract now has a separate optional head, explicit units/horizon/footprint checks, masked training, actual supervised support counts, MAE/RMSE, checkpoint evaluation and inference withholding. The saved-run UI displays area fraction separately. No real trained annual-area model is claimed: the public point references do not measure the required areas. [ANNUAL_LOSS_MODEL.md](ANNUAL_LOSS_MODEL.md) documents that boundary.

### Software verification

- **141 focused tests passed in 49.12 seconds**, covering the annual-loss chain, public references/benchmark and existing conversion, partial-supervision and inference contracts. A separate executable JavaScript annual-loss display test passed. These tests use controlled fixtures; only the separately recorded public experiment uses real reference data.
- The first complete suite finished with **1,121 passed and one failure in 186.97 seconds**. Log: `artifacts/regression-public-annual-20261105.txt`. The failure was Windows access denial during atomic climate-progress replacement. A bounded metadata-write retry and regression tests were added; persistent denial preserves prior progress and cannot repeat a provider download. The exact builder that obtained the real sources is preserved in `artifacts/climate-history-builder-used-20261105.py` and matches its original plan fingerprint. A new read-only completed-packet verifier preserves that history without altering the resume code pin.
- Final complete regression: **1,125 passed in 157.50 seconds**, no failures or skips, exit 0. Command: `python -m pytest tests -q --tb=short -p no:cacheprovider --basetemp=artifacts/pytest-public-annual-final-20261105`, with `PYTHONPATH=artifacts/geo-validation`. Log: `artifacts/regression-public-annual-final-20261105.txt`. This follows **70 passing climate checks in 21.18 seconds** for the write fix and completed-packet audit.
- The final small annual-only import-preview adjustment passed **21 UI/import checks in 1.50 seconds**, including the new case; log command uses `tests/test_annual_loss_presentation.py tests/test_model_file_preview.py tests/test_plain_language_ui.py` and `artifacts/pytest-annual-presentation-final-20261105`. This focused run overlaps the full suite; counts are not added. Every web JavaScript file passed syntax checks, and the standard `git diff --check` passed.
- The final read-only real climate audit exited 0 with `status: verified`, all four windows and zero requests. It reproduced the existing 48 reference feature/month pairs and 240 observed departures without rewriting sources or provenance. Frozen public-experiment implementation copies are archived under `artifacts/public-reference-study-20261105/implementation/`; both file hashes match the executed protocol.

The original monitoring database was not targeted by this work. No paid compute, public deployment, external notification, field-data fabrication or operational model registration occurred. Existing native pretrained comparisons, annual-area reference evidence, optical/radar coverage, spatial calibrated risk and ecological intervention/tipping validation remain incomplete. Actual browser layout was not reverified; UI execution tests and final JavaScript syntax checks are distinct from browser QA.

## Reference studies, frozen climate diagnostics and scenario ranges — November 4, 2026

This increment defines the annual gross tree-cover-loss intake and review contract, preserves copied evidence and review availability, extends protected baseline comparisons and event metrics, adds spatial-block improvement intervals, freezes sufficiently long climate references, and displays explicit scenario assumption ranges. The existing FastAPI/SQLite/plain-JavaScript architecture is retained. Scope and remaining requirements are recorded in `BUILD_PLAN_20261102.md`, `REQUIREMENTS_MATRIX.md`, `DATA_CONTRIBUTION.md` and `CLIMATE_BASELINE.md`.

- Full regression: **1,057 passed in 200.87 seconds**, no failures or skips, exit status 0. The run used `PYTHONPATH=artifacts/geo-validation` and `python -m pytest tests -q --tb=short -p no:cacheprovider --basetemp=artifacts/pytest-research-full-20261104`. Complete log: `artifacts/regression-research-20261104.txt`. Existing geospatial dependencies and isolated local API tests ran outside the Windows sandbox using the workspace temporary directory.
- Focused reference verification: **66 passed in 58.91 seconds**, covering `test_reference_review.py`, `test_reference_statistics.py` and `test_reference_experiment.py`. The run used `artifacts/pytest-reference-reviewer-20261104` as its temporary directory. Focused checks overlap the full suite; their counts are not added to its total.
- Focused climate verification: **48 passed in 4.75 seconds**, covering `test_climatology.py` with `artifacts/pytest-climatology-final-20261104`. Rehashed references with annual samples outside their declared source windows are rejected; adjacent windows covering a full month remain valid.
- Every web JavaScript file passed `node --check`; `git diff --check` found no whitespace errors. The matrix contains 59 unique O01–O59 rows with explicit Complete/Incomplete/Blocked statuses.
- The actual empty reference template produced `artifacts/reference-readiness-20261104/intake-report.json`: zero submitted/eligible labels, `ready_for_baseline_experiment: false`, and no `references.json`. This verifies refusal to invent training evidence; it is not a populated scientific dataset.

Tests use controlled software fixtures, including generated climate histories and labels, and establish software behavior only. Source hashes and reviewer declarations cannot authenticate ecological truth. The available pilot climate history remains insufficient for the frozen reference. No real model was trained or evaluated, long climate dataset downloaded, existing observation records changed, or service deployed for this increment. Scenario ranges are arithmetic sensitivity to stated inputs, not calibrated uncertainty or causal effects. New UI behavior is covered by executable JavaScript and isolated API tests; actual browser layout was not reverified during this increment. Previously running services and packaged wheels may predate the current source.

## Download reliability and historical coverage — October 31, 2026

- **935 tests passed in 216.33 seconds**, no failures or skips, with the existing raster/STAC dependency directory enabled. Command: `python -m pytest tests -q --tb=short -p no:cacheprovider --basetemp=artifacts/pytest-reliability-full-20261031`. Log: `artifacts/regression-reliability-20261031.txt`. Every web JavaScript file also passed `node --check`; `git diff --check` was clean.
- Regression cases cover transient versus permanent failures, wrapped HTTP causes, shared persisted retry budgets, exact-source/grid partial-band reuse and memory bounds, cancellation, masks/scaling, no partial cache writes, fair attempts across months and resumes, historical backfill preservation, failed requested periods, and larger bounded collection budgets.
- Browser interaction tests cover invalid periods, navigation or form replacement while submission is pending, fast terminal jobs before polling, requested versus observed coverage periods, stale responses and preservation of manually selected inspection dates.
- A read-only preview of the real database showed Mudumalai's nine observations over the requested July 2025–October 2026 period: **9/16 usable months, 7 missing months, longest consecutive usable run 8, and no eligible forecast**. Monitoring updates exposed historical start/end and the budget, explained its future-default behavior, and rejected reversed dates before enqueueing. Browser console errors were empty; page width and scroll width both measured 608px. The preview's collection/notification startup workers and all write requests were disabled, and the preview was stopped after review.
- Original and copied `regions`, `acquisition_jobs`, `acquisition_areas`, `acquisition_schedules` and `acquisition_snapshots` rows matched after verification. All ten regions and all nine pilot dates remain present. Evidence: `artifacts/reliability-pilot-verification-20261031.json`.

The suite initially encountered the documented Windows sandbox temporary-directory and TestClient socket restrictions. The final successful run used a workspace temporary directory and execution outside the sandbox. No imagery was downloaded for this update. Retry behavior was verified with controlled local failures, not a new live-provider benchmark; the precise cause of the earlier pilot failures remains unproven. No pilot model was trained, independent reference outcomes obtained, or public deployment performed. These changes are in current source; previously running servers and packaged wheels may predate them.

## Mudumalai pilot and acquisition reliability — October 29, 2026

The source preview now retains successful area selection across reloads, prefers an observed area for a fresh selection, and handles blocked storage and stale/failed requests. Terminal collection status derives accurate accepted/missing months and explicit catalog, budget, cancellation and acquisition-failure reasons from preserved evidence. New Sentinel-2 diagnostics identify the failed band and bounded cause chain with access details redacted. This changes diagnostics and presentation, not optical preprocessing or accepted cache values.

- **877 tests passed in158.58 seconds**, no failures or skips. Full log: `artifacts/regression-mudumalai-20261029.txt`. All web JavaScript files passed syntax checks. New cases include selection precedence, absent storage, stale selection, stopped retries, already-saved months, byte-preserved historical reports, band/SCL failures, cyclic causes and access-query redaction.
- Real original-workspace pilot `area-2b3262c5a5e5` has nine optical observations over October12,2025–August30,2026, a dated WorldCover2021 reference, and eleven ERA5 climate monthly summaries. Seven of nine optical dates have complete trailing climate features. June/July are missing inside the observed span; latest requested months also remain missing. No radar or eligible trained model is claimed.
- Historical collection accepted eight new months after one bounded resume. The six-attempt September–October refresh accepted none: September provider reads failed and October remained untried when the budget stopped. All jobs are terminal. The updated API exposes these reasons while the preserved underlying interrupted report still retains its resumable state.
- Six acquisition plan/report hashes match before and after restart and status reads. All ten saved regions remain present. Verification: `artifacts/mudumalai-pilot-verification-20261029.json`; complete export: `artifacts/mudumalai-analysis-20261029.json`, SHA256 `f58861ed7ff9b04918837f330f62ffe1214c63dcde9b9149f98a42fbc9779807`.
- Browser reload retained Mudumalai. Connected analysis showed nine dated observations, climate and forest-cover context, seven of nine climate matches and no model prediction. At the actual601px pane, page width equals scroll width, no raw code blocks are present, the global error is hidden and browser error logs are empty. Screenshot: `artifacts/mudumalai-pilot-20261029.png`; audit: `artifacts/mudumalai-browser-audit-20261029.json`.

The original workspace is served at port8001 by source process7096 at verification, using `artifacts/geo-validation` dependencies. Existing packaged wheels predate these changes. Full-run approval initially failed because of the account usage limit; after the user resumed, the complete run succeeded. These results validate software behavior and real-data plumbing, not ecological prediction accuracy. The user's budget remains ₹0, with no verified labels or research partner yet; see `pilots/MUDUMALAI.md`.

## Readable interface cleanup — October 28, 2026

The source preview replaces raw JSON result panels with readable cards, tables, dates and unavailable states. Observation, forest-cover and model-report imports have file previews rather than editable code. Acquisition and service status, notification messages and validation errors use plain language. Downloaded records retain the complete technical data.

- Full regression: **857 passed in 132.30 seconds**, no failures or skips. Evidence: `artifacts/regression-ui-20261027.txt`. All web JavaScript files also passed `node --check` on October 28.
- Regression coverage includes rejected/replaced/stale file selections, calendar dates, exact forest-cover geometry independent of property order, changed-region import guards, synthetic provenance, actual zero versus missing values, readable API/native errors, empty five-year comparisons, independent service failures, existing monitoring intervals and original download payloads.
- Browser verification used the copied database `artifacts/research-preview-20261025.db`. For handoff, `http://127.0.0.1:8001/` has been restored to the original `data/forestmind.db` (hidden process PID11728; logs `artifacts/ui-workspace-20261028.*`). This section describes the updated source preview; the packaged v0.7.0 wheel recorded below predates these UI changes.

Browser review at desktop 1280×900 and mobile 390×844 expanded every connected-analysis disclosure. Both views had no raw `pre`, `code` or `textarea` elements, and document widths equaled scroll widths (1265/1265 and 375/375). Screenshots are saved in `artifacts/ui-readable-measurements-20261028.png` and `artifacts/ui-mobile-measurements-20261028.png`. The browser-downloaded record, `artifacts/ui-workflow-export-20261028.json`, has SHA256 `e9880c34b0c12a5fdff2bc152047539a3cf54d2db6a9b03eac7ae7719b4b918d`, exactly matching the previously saved record.

The final browser review identified and fixed a scenario error-state issue: an insufficient-history response previously used the global error area and left stale scenario results visible. Failed comparisons now clear old results and exports and show a local **Comparison unavailable** panel. After this adjustment, **33 focused tests passed in 4.04 seconds**, covering `test_plain_language_ui.py`, `test_model_file_preview.py`, `test_readable_workflow.py` and `test_operations_presentation.py`; app JavaScript syntax passed again. This focused run overlaps the 857-test full run, which predates the last adjustment. Browser verification confirmed that a real four-month history cannot produce a five-year forecast and now explains this in the scenario panel with the global error hidden.

Mobile observation-file preview accepted the synthetic 48-record template and enabled submission. Replacing it with a malformed file cleared the preview and disabled submission. Monitoring-region search for Tamil Nadu, observed-data filters and readable collection-job status were checked. Mobile Find a forest had no raw data blocks and no page overflow (375/375). Browser error logs were empty, and the viewport override was reset.

Original-workspace browser smoke confirmed all nine regions and a readable connected report. The saved October 21 analysis was stale relative to current methods, so **Run selected-area analysis** refreshed the derived report from existing cached inputs. The stale warning cleared. Four optical observations, one of four radar matches, one of four climate matches and no model remained explicit. No collection action was triggered.

The before/after fingerprints (`artifacts/ui-original-workspace-before-20261028.json` and `artifacts/ui-original-workspace-after-20261028.json`) show **18 tables exactly unchanged**. Only workflow jobs increased from 9 to 10 and snapshots from 13 to 14. All nine prior jobs and thirteen prior snapshots individually match their original row digests: no historical record was replaced or removed. The original database therefore contains the intentional new derived analysis while preserving the checked source and historical records. Final evidence: `artifacts/ui-original-workspace-20261028.png` and `artifacts/ui-browser-audit-20261028.json`. Final browser error logs were empty.

No new training evidence is supplied by this UI work. The user has no independently verified reference data, and no real-trained, independently validated forest model or validated tipping-point forecast is claimed.

## v0.7 research bridge — October 26, 2026

This build adds measured optical/radar/climate input alignment, a protected independent-reference dataset converter, partial target supervision, unsupported-output withholding and pinned offline native checkpoint execution. The user confirmed on October 26 that **no verified reference data is available yet**. No real-trained model, ecological forecast accuracy, causal attribution or tipping-point prediction is claimed. Remaining implementation and evidence dependencies are listed in `COMPLETION_PLAN.md`.

### Executed checks

- Full regression: **814 passed in 134.70 seconds**, no skips, with raster dependencies enabled. `artifacts/regression-0.7.0-final.txt`. The command used `python -m pytest tests -q --tb=short -p no:cacheprovider` with a unique workspace temporary directory. Windows local API socket and hardlink checks required execution outside the sandbox.
- Final radar wording and responsive presentation changes were followed by **42 passing checks in 13.42 seconds**, including served assets and saved-record immutability. `artifacts/presentation-0.7.0-final.txt`. This is an overlapping focused run, not an additional total. All seven JavaScript files passed syntax checks.
- Three controlled converter → generated configuration → one-epoch training → checkpoint → masked evaluation runs passed: optical-only, measured-climate and disabled-attribution-loss cases. Checkpoint support records reflect actual observed targets and enabled losses. The tests also cover altered report/manifest/sample hashes, extra samples, unit/horizon changes, buffers, embargoes, future inputs, unknown labels, source immutability and rectangular augmentation. Their fixture labels are fabricated software inputs, not independent ecological references.
- Actual native CPU diagnostics used TerraTorch 1.2.13, PyTorch 2.10.0, torchvision 0.25.0 and timm 1.0.30. TerraMind tiny (6,238,153 parameters) and Prithvi v1 100M encoder configuration (86,471,305 parameters including ForestMind) completed strict state roundtrips and repeated equal forward outputs on a preserved real optical sequence. Empty caches and network interception recorded **zero attempted network calls**. The states were random and untrained, then removed. See `artifacts/native-terramind-runtime-20261025.json`, `artifacts/native-prithvi-runtime-20261025.json` and `NATIVE_MODEL_AUDIT.md`. Native RTC forward, other sizes and GPU execution remain unverified.
- Final wheel: `artifacts/dist/forestmind-0.7.0-py3-none-any.whl`, **320,111 bytes**, SHA256 `161a8f12e5d7ad9da97e53f65c11e13ae50afdcd10ee66d483862470aa7f6a33`. All **87** packaged source/assets matched both workspace files and the separately installed copy. The installed package's HTML, status, overview and updated JavaScript returned HTTP200. `artifacts/package-verification-0.7.0.json` retains the imported location, package/runtime inventory and comparisons.

Earlier integration failures were corrected before these checks: scalar NumPy targets gained an unwanted dimension during tensor conversion, and two new integration test fixtures had setup mistakes. `artifacts/regression-research-build-20261025.txt` preserves the earlier 774-pass/one-failure run; it is not counted as a pass. Windows sandbox temporary-directory failures during packaging were resolved by using a dedicated workspace temporary directory; the successful wheel was subsequently verified independently.

### Saved real data and browser verification

Verification used `artifacts/research-preview-20261025.db`, a SQLite backup copy of the user's workspace. No provider downloads, external notifications or public deployment were requested. Original nine regions, four contexts, thirteen workflow snapshots, three source bindings and the existing model run remained intact; all four accepted optical source hashes matched. The user's Tamil Nadu area and older imports were preserved. Schedules remained disabled and notifications unconfigured.

The copied Tapajós workflow retained four real optical observations through October12,2024. Existing saved radar and daily ERA5 records passed matching and hash checks. **One of four image dates had a usable radar match; one of four had complete climate features.** Missing inputs stayed masked. No eligible model existed, so no model prediction was produced. The earlier isolated alignment smoke used the July radar pilot; the copied workspace uses its separately saved August radar pair. These are distinct measured artifacts, not contradictory date records.

The browser ran selected-area analysis, displayed the coverage counts and timing assumptions, and exported `artifacts/browser-research-export-20261026.json`. The final snapshot was not stale. A separate explicitly synthetic random-weight fixture was imported only into the copied demo area to verify missing-output presentation: health state/score, seven drivers and the unsupported horizon showed **Unavailable**, while supported fixture channels retained their values. This proves display behavior, not training success or ecological accuracy.

At the default desktop size and a 390×844 viewport, the page had no horizontal overflow (document/client widths 1265/1265 and375/375). Wide tables scroll internally, the coverage description wraps on desktop, and the browser error log was empty. The temporary viewport override was reset. Screenshots: `artifacts/forestmind-aligned-inputs-20261026.png`, `forestmind-aligned-mobile-20261026.png`, `forestmind-partial-outputs-20261026.png`, and `forestmind-partial-mobile-20261026.png`.

### Remaining acceptance gates

G1's earlier bounded local-workflow acceptance is retained, with the new research paths verified as above. **G2 remains open** for hosted authentication, ownership isolation and deployment operations. **G3 remains open** for independent labeled references, licensed pretrained artifacts and trained heads, protected comparisons, separate calibration and ecological usefulness/transfer. Seasonal radar histories, UI Landsat integration, automatic fire/activity histories, recurrent/ablation comparisons, calibrated spatial risk and causal/process simulation also remain unfinished engineering or research; see the matrix for exact scope.

## v0.6 connected application — October 22, 2026

This release continues the existing FastAPI/SQLite/buildless dashboard. It connects worldwide search, small confirmed AOIs, actual acquisition, durable jobs, saved observations, optional real context, reproducible analyses, refresh and local schedules. The full original scientific proposal is **not complete**; the [requirements matrix](REQUIREMENTS_MATRIX.md) maps all 59 original-proposal and 40 expanded acceptance items, including incomplete implementations and missing evidence.

### Executed software checks

- Final integrated regression: **671 passed in51.69 seconds**, with geospatial dependencies enabled. Command: `python -m pytest tests -q --tb=short --basetemp=artifacts/pytest-release-20261022`. Log: `artifacts/regression-0.6.0-release.txt`. All tests passed; none were skipped.
- Earlier failures remain recorded: the initial v0.6 run had611 passes and three WinError5 atomic-save failures (`artifacts/regression-0.6.0.txt`). The bounded Windows retry repair then passed a623-test full run. After the additional diagnostics, an overlapping run had670 passes and one duplicate-ID error in a newly added test fixture (`artifacts/regression-0.6.0-20261022.txt`); that fixture was corrected before the final671-test run. Neither earlier failure is represented as a pass.
- Focused acquisition/history/WorldCover/radar checks:144 passed before that repair. Workflow/reference/API/model-import checks:70 passed; final additional capacity/freshness checks:13 passed; reference-experiment checks:15 passed. Native inference/adapter checks:83 passed. Notifications/search/local-security/full-backup checks:18 passed. Frontend contracts:14 passed. These runs overlap and must not be added together as a total.
- All seven dashboard JavaScript files passed `node --check`.
- Final focused additions:14 notification tests,20 reference-experiment tests,38 climate/workflow/frontend tests,9 spectral-availability tests and46 spatial/workflow/landcover tests passed. These overlap the full suite. The Windows repair also passed an actual non-delete-sharing file-handle test.
- Packaging: rebuilt `artifacts/dist/forestmind-0.6.0-py3-none-any.whl`,287,363 bytes, SHA256 `c8b95a0e2f6595228a18ca698ee4e99fc31e69b5d737630877ece0c54e788d13`. All82 packaged source/assets match current files byte-for-byte (`artifacts/package-verification-0.6.0.json`). Installed the wheel into a separate local directory and loaded its actual package; its empty-workspace API, HTML and updated JavaScript returned200 (`artifacts/wheel-smoke-0.6.0.json`).
- Runtime inventory: `artifacts/validated-runtime-0.6.0.json`, Windows/Python3.13.7, CPU PyTorch, with optional geospatial dependencies in `artifacts/geo-validation`. This is a record of tested versions, not a universal dependency lockfile or security audit.

### Actual browser workflow and provider results

The in-app browser performed the application actions below. No synthetic observations were substituted, no public deployment occurred, and no external notification was sent.

| Check | Observed result |
| --- | --- |
| Worldwide search | Explicit `Tapajos Brazil` query returned eight disambiguated Photon/OSM results with location and attribution. A place result was selected, then a small independent study rectangle was confirmed. |
| Confirm and collect | `Tapajos connected real pilot`, bbox `[-55.10,-3.50,-55.08,-3.48]`, July–August2024,64×64 grid, two-download cap. Job `acq-d74c8ccee2f14153` accepted July17 and August26,100% valid samples, and saved area `area-858ffd6a4f28`. |
| Refresh/cancel/resume | UI refresh job `acq-aead355d7b794983` was cancelled during planning, confirmed cancelled with saved observations unchanged, then explicitly resumed. It accepted September2 and October12 using exactly two downloads and retained the original July/August records. The broader requested window remains partial with explicit gaps. |
| Local schedule | Enabled a15-day interval in the UI, observed a next-run date, then disabled it after the check. Restart retained `enabled:false`, `interval_days:15`; no unattended test schedule remains enabled for this area. |
| Additional geography | Existing user area `Tamil Nadu, India` was preserved: November10,2025,95.9229% valid samples. New `Black Forest real pipeline check`, bbox `[8.20,48.55,8.22,48.57]`, saved August11,2024 at100% validity; July had no eligible candidate in the bounded plan. These are pipeline checks, not ecological geographic-transfer validation. |
| No data | A small Pacific Ocean test rectangle returned no eligible observations. The job is failed with month-level reasons and zero downloads; no saved region or fabricated measurement was created. |
| Climate context | UI ERA5 collection saved July17–August26,2024 centroid reanalysis, actual units/counts and partial-month date ranges. Standalone three-day smoke also succeeded (`artifacts/climate-context-pilot.json`). |
| Land-cover context | Actual ESA WorldCover2021 v200 crop,64×64,4096 valid samples, class10 throughout this small Tapajós area. UI computed one patch and zero internal edge; this is historical classified woody cover, not independent current forest verification. `artifacts/worldcover-real-smoke.json`. |
| Radar context | UI collected S1A/orbit39/descending RTC acquisitions August4 and16,2024,100% common valid samples; mean VV change≈+0.211dB and VH≈+0.451dB. A separate July11→23 smoke is retained in `artifacts/radar-real-smoke.json`. Dates/polarizations/units/orbit and masks are exported; the pairs do not establish disturbance cause. |
| Spatial/keyboard | Real NDVI/NDMI/NBR difference rasters render. ArrowRight changes optical and radar pixel readouts. Final optical example reads pixel2,1≈+0.00397 NDVI with area mean≈−0.064. It is measured spectral change, not loss or future-risk probability. |
| Mobile |390×844 viewport: document client/scroll widths both375px; no page overflow in real-region list or connected analysis. Spatial grid, labels and keyboard focus remained usable. Wide tables have internal scrolling. Screenshot `artifacts/forestmind-connected-mobile.jpg`. |
| Desktop |1365×900 breakpoint checked, selected-area identity/cutoff and connected controls visible. Screenshot `artifacts/forestmind-connected-desktop.jpg`. Final browser error log was empty after reconnecting to the running service. Temporary viewport overrides were reset. |
| Browser export | **Export reproducible record** downloaded the actual selected-area JSON. Workspace copy `artifacts/browser-connected-export.json`:2,327,832bytes, SHA256 `037b0a7e42f2f3baa5eae95cc809aa020ca06e49a68583995e3418381a7aaa00`; geometry/cutoff2024-10-12/four observations/context hashes/version0.6.0 retained, model null. |
| Persistence/immutability | After restart, all nine areas remained (four synthetic/five real records), including all four refreshed Tapajós observations. Export of the older exact snapshot ID compared equal to its saved pre-restart payload. Rerunning after code changes produced a fresh non-stale snapshot. `artifacts/restart-verification-0.6.0.json`. |
| Operations | UI showed local service and database integrity `ok`, available disk space, and created verified DB snapshot `forestmind-20261020T112518-8a22d2.db`. Notifications displayed disabled/unconfigured with zero outbox entries. |
| October22 connected rerun | After restarting with the final code, clicked **Run selected-area analysis** for the real Tapajós area. A new non-stale manifest `f52622afdec675a60940be1889023894d1d32fb233afb2ed26f7a3f2edbec98b` retained geometry, cutoff and four observations. Existing raw climate values/hash were unchanged; no additional provider collection was requested. |
| Paired climate diagnostic | Browser showed July17–31 precipitation minus reference ET0 **−73.34mm**,15/15 paired days, and August1–26 **−111.10mm**,26/26 paired days. Both are partial calendar months, explicitly distinct from validated drought or forest water use. `artifacts/forestmind-climate-20261022.jpg`. |
| Patch shape | Actual WorldCover crop has one patch reaching the AOI boundary. Browser correctly displayed0 complete/1 clipped-or-unknown boundaries, with null perimeter/compactness. Expanded table retained the4.9366km² classified area and explanation. Analytic complete-shape behavior is tested with controlled fixtures; no real complete-patch ecological validation is claimed. |
| Final responsive check | New climate/shape sections checked at desktop and390×844 mobile. Document client/scroll widths were1350/1350px and375/375px respectively; tables scroll internally. No browser console errors were recorded. Temporary viewport override was reset. `artifacts/forestmind-shape-mobile-20261022.jpg` and `artifacts/forestmind-release-20261022.jpg`. |
| Final browser export | Downloaded `artifacts/browser-connected-export-20261022.json`,2,335,851 bytes, SHA256 `f6771484df2a11691fe67756e8b0732f2988546307b26741d22ed1fe0af54148`. Geometry and observations equal the earlier export; the older exact manifest still returns its original payload. Detailed checks: `artifacts/final-browser-verification-20261022.json`. |

`artifacts/connected-final-jobs.json` retains the five acquisition jobs and their actual settings/outcomes. The original single-scene and eight-month imports remain unchanged. The real histories are still too short or discontinuous for the current forecast/resilience minimums, and the application withholds those outputs. Trained model and five-year ecological forecast validation were not demonstrated on these records.

### Recovery and scientific gates

Full workspace backups were created and hash/integrity verified at `data/backups/workspace-20261019` (11files) and `data/backups/workspace-20261020` (22files), including SQLite and both artifact trees. Automated restore tests recover the original job/source references and reject tampering, path traversal and overwrites. Database-only UI snapshots are deliberately distinguished from full source backups. Schedule concurrency fixes preserve user edits during enqueueing and prevent duplicate due ticks. The [operator guide](OPERATOR_GUIDE.md) documents original-path restore, quotas and rollback.

**G1 achieved for the documented bounded local monitoring workflow:** real browser search/acquisition, progress/cancel/resume, saved observations, connected context and diagnostics, refresh/schedule, failure/persistence/export, responsive checks, regression and installed-package smoke have passed. This acceptance does not cover every original research capability or guarantee worldwide data availability. **G2:** public/hosted deployment is not ready: no authentication, area/job ownership isolation, hosted supervision/load testing or real webhook-delivery acceptance. Local access is enforced. **G3:** not achieved: independent reference labels, real-trained accepted models, protected experiments and ecological predictive/causal evidence are missing. The five-baseline reference-experiment harness is tested with fixtures; it has not run an independent real-data benchmark. Native pretrained runtime/checkpoints remain unavailable and offline loading is blocked, as detailed in [NATIVE_MODEL_AUDIT.md](NATIVE_MODEL_AUDIT.md). Other incomplete original capabilities remain explicit in the matrix and completion plan.

## Earlier v0.5 evidence (retained historical record)

ForestMind v0.5 was verified locally on Windows on2026-10-16, with its monthly-history artifact and persistence audit on October17. The first real single-scene pilot was verified during v0.4 on October14. The counts and limitations below describe that earlier build.

## Completed checks

- Full test suite: **472 passed** in 64.61 seconds, including raster reprojection, strict inference and API contracts, acquisition-to-sequence conversion, evaluation provenance, explicit satellite providers, resumable monthly acquisition, report integrity, coverage/readiness, download failures, persistence, monitoring and saved plots.
- Frontend syntax: `app.js`, `model-runs.js` and `coverage.js` passed `node --check`.
- Browser: checked region selection, alerts and observations, JSON import validation, scenario calculation and invalidation after input changes, historical backtesting, and responsive layouts. No browser console errors were observed in the final preview.
- CPU training and evaluation: the small synthetic configuration completed and produced `checkpoints/quick_test/best_model.pt`.
- Checkpoint demo: completed successfully with the configured optical input shape of 4 time steps by 12 bands by 32 by 32 pixels.
- Inference: the actual quick-test checkpoint produced `artifacts/demo-inference-report.json` from the explicitly synthetic `artifacts/demo-inference-input.npz`. The report preserves synthetic input and training provenance and all five configured forecast horizons.
- Model-run browser flow: imported that report into the Tapajós demo region, verified malformed JSON feedback, reloaded the page and confirmed persistence. Narrow-screen layout had no page overflow, and the browser console had no errors. Screenshot: `artifacts/forestmind-model-runs.jpg`.
- Real-observation browser flow: imported `artifacts/real-pilot-region.json` as **Tapajos real satellite pilot** (`import-9e78ebc0e30c`). Verified the acquisition date, spectral indices, 100% valid samples and Earth Search source. Forecasts and resilience diagnostics correctly remain unavailable for the one-observation record. No browser console errors. Screenshot: `artifacts/forestmind-real-pilot.jpg`.
- Coverage browser flow: selected a twelve-month window for the saved single observation and verified one usable month, eleven missing months and no usable run at the requested period end. Checked JSON export, reversed-date feedback and clearing stale results when dates change. At a 390-pixel viewport there was no horizontal page overflow. No browser console errors.
- Monthly-history browser flow: imported eight actual acquisitions as **Tapajos monthly satellite pilot** (`import-2973ec681f83`). After a server restart, verified persistence and the July 2024–June 2025 coverage window: eight usable months, four missing months, longest usable run of seven and a run of one at the period end. Forecast and resilience results remain unavailable. The final preview had no browser console errors. Screenshot: `artifacts/forestmind-history-coverage.jpg`; API response: `artifacts/history-coverage.json`.
- Evaluation: `artifacts/quick-test-evaluation.json` contains raw predictions and reference targets for two synthetic test examples, configured horizon errors and reproducibility hashes. These sample metrics are not real-world validation.
- Packaging: built `artifacts/dist/forestmind-0.5.0-py3-none-any.whl` and verified that it contains the history acquisition and coverage modules, coverage JavaScript, inference runner, sequence preparation and model-results JavaScript. SHA256: `63d599837dbd60df468fc8afca679eae40bcc40b3a00ef1d2509c8c9d0660fc6`.

The optional raster dependencies used for the full suite were installed under the ignored local directory `artifacts/geo-validation`. They are not bundled in the package. The test invocation was:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'artifacts/geo-validation'
python -m pytest tests -q --tb=short
```

## Run locally

From the repository root, after installing the dependencies described in the README:

```powershell
python scripts/serve.py --port 8000
```

Open <http://127.0.0.1:8000>. The preview server used for verification is temporary; start this command for a new session.

## Live-data verification

The public Planetary Computer catalog returned the actual scene `S2B_MSIL2A_20240707T140709_R110_T21MYS_20240707T174548`, acquired on 2024-07-07, for the small demonstration AOI `[-55.10,-3.50,-55.08,-3.48]`.

Planetary Computer raster ingestion did not complete. Bounded reads returned incomplete HTTP 206 byte ranges. A small 16 KiB range request succeeded, but serial GDAL requests still produced truncated raster tiles. The speculative serial-request change was reverted. Failed reads return a nonzero status and do not produce a usable cache. No real observations or model assessments were fabricated from these failures.

`artifacts/real-pilot-range-diagnostic.json` records the Planetary Computer range-read diagnostic. Download logs redact signed access queries, including truncated source strings emitted by the raster decoder.

The provider check exposed a separate reflectance-processing issue: newer processing baselines require a nonzero offset when asset metadata omit scaling. This is corrected in cache version `forestmind-grid-v3`; older caches are rejected by both observation and sequence preparation. Tests cover baseline-based scaling, including reprocessed historical scenes.

### Successful Earth Search pilot

An explicitly selected alternate provider, Element 84 Earth Search, completed the ordinary download CLI in 99.0 seconds. It returned **`S2A_T21MYS_20240729T140134_L2A`**, acquired **2024-07-29T14:03:30.961Z**, from `sentinel-2-c1-l2a` for the same AOI. All twelve optical bands and the SCL mask were read and aligned to a 64 × 64 grid. All **4,096 sampled pixels** passed the masks. The scene-wide cloud metadata was 1.466749%; that is distinct from validity within this small AOI. Each band's explicit metadata specified `DN × 0.0001 − 0.1`, applied once.

Prepared observations were NDVI **0.852677**, NDMI **0.291797** and NBR **0.631346**. The import JSON and its provenance are saved as `artifacts/real-pilot-region.json` and `artifacts/real-pilot-region.provenance.json`. The dashboard displays these real measurements alongside four clearly identified synthetic demo regions.

`artifacts/real-pilot-sequence.npz` and its provenance sidecar contain one actual optical acquisition, its masks, timestamps and source hashes. The cutoff was July 31; the last actual observation remains July 29. SAR and labels are absent, and no dates were padded. This sequence was not passed through the synthetic demonstration checkpoint as a real ecological forecast.

This verifies a real provider → aligned cache → observation preparation → dashboard path and a cache → inference-input path. It does not verify a long monitoring history, field labels or predictive skill. The exact pilot commands are in the README; the full download and preparation evidence is in `artifacts/real-pilot-verification.json`.

### Resumable monthly-history pilot

The monthly planner searched **2024-07-01 through 2025-06-30** for the same small AOI and explicit Earth Search collection. The frozen plan uses a 64 × 64 grid, at most three candidates per month from a bounded search of thirty, a minimum valid-pixel fraction of 0.6, and per-invocation limits of six attempts and two new downloads. Candidate ranking and the scene-cloud filter are recorded in the plan; this is a selected sample, not an exhaustive history of every available acquisition.

Four execution invocations, with `--resume` after the first, each made exactly **two attempts and two downloads**. They produced eight accepted observations. Earlier accepted caches were retained and verified during resume. February–May 2025 had no eligible candidates in the frozen search results, so the final report correctly remains **incomplete** with `candidates_exhausted`; absent months were not filled. This does not establish that no satellite observations exist for those months under other search criteria.

| Actual acquisition date | Valid pixel samples |
| --- | ---: |
| 2024-07-17 | 100% |
| 2024-08-26 | 100% |
| 2024-09-02 | 100% |
| 2024-10-12 | 99.707% |
| 2024-11-29 | 100% |
| 2024-12-01 | 64.087% |
| 2025-01-23 | 76.953% |
| 2025-06-16 | 100% |

The final audit validated the plan and report hashes, all eight accepted cache files, their source identities, processing grids, bands and quality masks. Recomputing observations from the accepted files reproduced `artifacts/history-region.json` exactly and matched the deterministic provenance fields. The earlier July 29 single-scene pilot is excluded from this separate monthly export. A mask passing the software checks is not field verification of cloud-free or ecologically representative measurements.

Saved evidence:

- `artifacts/history-plan.json`: frozen search results, settings and plan hash.
- `artifacts/history-progress.json`: schema-2 report, four runs, accepted source hashes and missing-month outcomes.
- `artifacts/history-region.json` and `artifacts/history-region.provenance.json`: eight actual observations and their source audit trail.
- `artifacts/history-coverage.json`: persisted region's API response for the complete requested year.
- `artifacts/forestmind-history-coverage.jpg`: final dashboard coverage preview.

The README documents the reproducible `build_history.py plan` and `execute --resume` workflow. This pilot establishes bounded downloads, resume integrity and an honest gap report for real data. It has only seven consecutive usable months and does not meet the current software minimums of twelve for a baseline forecast or thirty-six for descriptive resilience diagnostics.

## Research limits

These checks establish software behavior, not ecological predictive validity. The four seeded monitoring regions and quick training run use synthetic data. The two imported real pilots contain one observation and eight monthly observations respectively; the monthly pilot has four gaps in its requested year and remains insufficient for forecasts or resilience diagnostics. Native pretrained checkpoints and operational deployment were not exercised. The new offline inference command rejects native backbone constructors that would download weights; only local custom/proxy checkpoints are currently supported there. Monitoring scores and scenario outputs remain uncalibrated research indicators. Validation on independent real observations and events, geographic and temporal holdouts, and uncertainty calibration remain necessary before scientific or operational claims.

See `METHODOLOGY.md`, `RESEARCH_PROTOCOL.md` and `COMPLETION_PLAN.md` for the assumptions and remaining milestones.
