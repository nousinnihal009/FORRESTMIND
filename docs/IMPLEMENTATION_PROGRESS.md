# Continuation record — 2026-10-19

## Latest continuation checkpoint

- Version is now 0.6.0. README updated with normal UI flow; `docs/OPERATOR_GUIDE.md` now covers setup, quotas, backups, restore, local rollout/rollback, notifications and troubleshooting. Final validation/completion docs and wheel/environment record still pending.
- Full actual workspace backup completed and verified: `data/backups/workspace-20261019`, 11 files (database + source/workflow trees). Do not overwrite it. Existing user data preserved.
- Real radar smoke succeeded: `artifacts/radar-real-smoke.json`, Tapajós July 11 → July 23 2024, S1A/orbit39/descending, 4096 common-valid samples; VV change −0.244568 dB, VH −0.466682 dB. This is measured backscatter context, not disturbance attribution or a forecast.
- Browser climate collection succeeded for July17–Aug26 2024. Actual climate and WorldCover contexts persisted through restart. Browser reconnect verified all eight original saved regions.
- Notifications now bind authorization to destination and cap retained events at 10,000; no actual destination configured or messages sent. Two added tests pending inclusion in final full suite.
- Final UI changes: collapsed capability cards, real radar grid keyboard inspection, compact partial-month climate coverage, notification cap/reauthorization state, arbitrary 1–365 day schedule values, ARIA tabs/panel/focus. Frontend14tests passed and all seven JS syntax checks passed.
- Acquisition audit fixed due-schedule claim race and cancellation timestamps;144 focused tests passed in12.88s. Restart needed after these edits. Native pretrained offline path is undergoing a separate implementation feasibility audit.
- Research reference experiment harness exists with frozen outcomes/datasets/splits, buffered geography/embargo, identical baseline comparisons, calibration/events/subgroups. Earlier focused39tests passed. Final workflow storage/freshness audit currently running.
- Current server at this checkpoint: PID17792, exec session64980, `python scripts/serve.py --port 8000` with artifacts/geo-validation on PYTHONPATH. Verify current PID/command and active jobs before restarting. Browser tab1 in IAB, handle forestTab in current CUA session; it is collecting radar context for area-858ffd6a4f28. UI verifies outcome next. Context/session handles can reset after usage interruptions.
- Still to do: complete browser radar/refresh/schedule/cancel-resume/failure/mobile/export evidence, final full tests, packaging and accurate three-gate handoff. Preserve user's Tamil Nadu area.

The user requests the full original proposal plus the worldwide search → real acquisition → monitoring workflow. Do not interpret the local application milestone as scientific completion. Original proposal: attachment `2356b8c3-679f-4302-9adf-414b4a571fcc/Pasted text.txt`; current expanded request: attachment `044b2d52-e600-4756-bd72-97e7c1ad4605/Pasted text.txt`.

## Work in progress

- Parent: application integration, Photon place search, local HTTP boundary, migrations/backups, alert lifecycle, regression/browser/live-data verification and operator documentation.
- Acquisition agent: persistent single-worker jobs, frozen history integration, cancellation/recovery/resume, budgets, saved area merge and schedules.
- Web agent: existing visual language plus worldwide map/search/area selection, job controls, refresh/schedules, connected results and alert handling.
- Analysis agent: immutable evidence manifests, automatic eligible local checkpoint execution, spatial change/recovery/fragmentation computations, requirements matrix.

The code at the beginning was v0.5 with 472 recorded tests. The first integrated regression run in this continuation passed **528 tests in 78.22s** before subsequent WorldCover/scenario/backup/notification additions. Focused acquisition/history/API tests passed **129 in 30.24s**. Search/security/alerts/backup/API checks initially passed **36 in 12.83s**. A subsequent focused run passed 63 tests and found one Windows backup file-handle leak; fixed using explicit connection closure, then both complete workspace restore tests passed. Final full-suite results are still pending after the additional changes.

No paid services, external notification delivery or deployment are authorized or performed. Notifications are now opt-in code with mocked tests; keep delivery off in this workspace. Existing real pilot and synthetic examples remain distinct.

## Actual browser and provider evidence so far

- Existing default database and user-created Tamil Nadu area were preserved. Tamil Nadu has one real November 2025 observation, explicit gaps and a partial two-download job.
- In the actual browser, explicit Photon query `Tapajos Brazil` returned disambiguated Brazilian places with OSM/Photon attribution. Selected a place, then confirmed the established small study rectangle `[-55.10,-3.50,-55.08,-3.48]` with an explicit July–August 2024 window and two-download cap.
- `Tapajos connected real pilot` was acquired through the UI and saved with **two real observations**, July 17 and August 26, 2024. Browser coverage showed 2/2 usable months and correctly withheld short-history forecasts/resilience. Job records saved in `artifacts/connected-jobs-before-restart.json`. Full connected-result/restart/refresh/schedule browser verification is still pending.
- Real ERA5 climate smoke succeeded for July 1–3, 2024; artifact `artifacts/climate-context-pilot.json`.
- Real ESA WorldCover 2021 64×64 crop succeeded for Tapajós, 4096 valid samples, all mapped tree-cover class10; `artifacts/worldcover-real-smoke.json`. This is an annual land-cover classification, not verified current ecological forest or land-loss evidence.
- Server started as process22896 on loopback port8000. Inspect current process command line and active jobs before restarting; do not assume this PID remains current. The browser tab has been preserved across usage interruptions.

## Environment note

Windows sandbox execution can hang while FastAPI TestClient constructs an asyncio socket pair. The focused and full test commands worked outside the sandbox after approval. This was diagnosed in faulthandler as socket.accept, before app code. Raster/STAC extras are in ignored `artifacts/geo-validation`; use its path as PYTHONPATH for complete raster tests and the local server. Avoid leaving stalled duplicate test/server processes.

## Provider choices and bounds

Photon explicit search, server cache, single outstanding request and at most one request per 1.1 seconds. Photon/OSM attribution; named places are not forest boundaries. Geocoder can be changed through operator configuration. Public demo service has no SLA.

Acquisition uses explicit Earth Search Sentinel-2, strict small WGS84 rectangles and recorded quality settings. Failed requests and missing observations remain visible. Network destinations restricted in web worker context. Local HTTP service rejects remote peers/unknown Host and cross-origin writes; public hosting remains disabled.

## Handoff rule

Read current files and this record before continuing; do not repeat completed work. Advance three separate gates: working local app, deployment readiness, scientific validity. Record exact unresolved code work as well as external reference-label/model/compute dependencies. Tests and synthetic demonstrations never establish ecological predictive validity.
