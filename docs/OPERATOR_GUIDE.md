# Local operation and recovery

ForestMind v0.6 is a local, single-user research application. The ordinary search → confirmed area → collection → evidence → refresh/schedule workflow runs in the dashboard. No public deployment, account provisioning or external notification delivery is included in this release.

## Install and start

Use Python 3.10 or later and a dedicated virtual environment. The verification environment is Windows with Python 3.13 and CPU PyTorch. Rasterio needs a compatible binary wheel; avoid a source build unless you maintain a GDAL toolchain.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[web,real-data,dev]"
python scripts/serve.py --port 8000
```

Open http://127.0.0.1:8000. Keep the terminal running for jobs and schedules. Stop with Ctrl+C. Port conflicts can be resolved with `--port 8001`; use that same port in the browser. Do not run multiple production workers or expose a reverse proxy to this unauthenticated service. Both the launch script and HTTP boundary enforce local access; authentication and user isolation are separate unfinished deployment work.

Optional `FORESTMIND_DB` selects a different database before startup. Each database has separate acquisition and analysis artifact directories. The default database is `data/forestmind.db`; its directories are `data/forestmind-acquisition/` and `data/forestmind-workflow/`. Four synthetic examples are seeded and visibly identified. Existing real areas are retained across restarts.

## Everyday monitoring

Find a forest supports explicit place search, map drawing and keyboard coordinate entry. Search results are names and locations, not validated forest boundaries. Confirm a small rectangle before collecting. The saved area and actual forest cover within it are distinct. Overlapping areas remain independent studies; summed footprint area in the overview can double-count overlap.

Start with a short period and a small download budget. Each attempted scene, including a failed request, consumes budget. Cancel stops cooperatively between bounded requests. Resume is an explicit new budget allocation against the frozen plan. Keep partial records: a missing month, a cloud-masked observation and no detected change are different findings. The acquisition guide describes exact bounds and retry behavior.

After the first accepted scene, the area is saved and available in Monitoring regions. Connected analysis uses accepted cache files only. Optical change needs two compatible acquisitions. Radar context needs two quality-passing RTC scenes with the same platform, orbit and pass. Climate is centroid reanalysis, and WorldCover is a dated classification. Context layers are not silently promoted to trained-model features. Source units, masks, dates and grid metadata are retained in exports.

Monitoring updates can acquire absent months and enable a local schedule. The service must be running. After downtime, one current refresh is queued; missed intervals are not replayed. Ordinary imported records have no acquisition configuration and cannot automatically refresh. Create an area through Find a forest to enable this path.

Alerts include dated supporting observations and quality, configurable thresholds, acknowledgement, resolution and review notes. Freshness/ingestion issues are separate from ecological screening signals. Acknowledging an alert does not change measurements or establish that the underlying issue is resolved.

## Health, storage and diagnostics

Data & methods shows database health, disk availability, schema migrations and optional notification status. `/api/v1/operations` returns the same diagnostic record. Acquisition capabilities expose worker state, request limits and storage used; persisted job records explain per-month failure, retries, attempts and saved results. There are no credentials in these API responses.

Acquisition has a 2 GiB artifact cap, one worker and twelve queued/running/retrying jobs. The HTTP body limit is 4 MiB; the dashboard's classified-cover form separately limits JSON to 2 MiB, and the API validates a maximum 128 × 128 grid. Database snapshots stop after twenty files. Notification history stops accepting new events at 10,000 records; existing delivery records are preserved for deduplication. These are local safety bounds, not a capacity-tested hosted service.

Connected analysis has one worker and at most four queued/running jobs. It retains at most 10,000 job records, 2,048 snapshots per workspace and 256 snapshots per area. Its shared metadata budget is 256 MiB across contexts, source bindings, snapshots, jobs, saved scenarios and model reports; source-sequence artifacts have a separate 1 GiB cap, and each metadata record is limited to 16 MiB. Pending jobs reserve completion space. Inspect `GET /api/v1/workflow/storage` for current usage and limits. At capacity, new work is refused while saved evidence remains readable/exportable. Back up the study and select a new workspace database; there is no automatic deletion of old evidence.

Archive only reviewed completed work with the service stopped and a verified backup. Do not delete a running job directory or lock file. Removing a referenced raster makes future cache-dependent analysis/resume unavailable; it does not erase saved observation reports. Preserve model registry, checkpoints and evaluation artifacts separately if they live outside the workspace backup trees.

## Back up and restore

The dashboard's **Create database snapshot** produces an integrity-checked SQLite snapshot while the service is running. It contains saved measurements, reports, jobs and schedules, but not source rasters. It is useful for inspection and observation recovery; it is not a full reproducibility backup.

For a complete backup, stop the server, use a new destination each time, then verify it:

```powershell
python scripts/manage_workspace.py backup --db data/forestmind.db --destination data/backups/study-20261019
python scripts/manage_workspace.py verify --source data/backups/study-20261019
```

The command takes the worker's exclusive lease, snapshots SQLite, copies both artifact trees and writes a SHA256/size manifest. It refuses symlinks, a destination within the source artifact tree, an existing backup directory or a workspace above 3 GiB. A failed attempt can leave a partial destination; preserve it for diagnosis and use a new destination on retry. `verify` must succeed before considering a backup usable. Keep another copy on independently protected storage under your own backup policy.

Restore is deliberately non-overwriting and requires the **original absolute database path** because acquisition provenance references source paths. With the service stopped, move the current database and its two artifact trees into a separate recovery directory inside the workspace, preserving all originals. Check the resolved paths before moving them. Then run:

```powershell
python scripts/manage_workspace.py restore --source data/backups/study-20261019 --db data/forestmind.db
python scripts/serve.py --port 8000
```

The restore verifies hashes, path containment, SQLite integrity and absence of conflicting targets before copying. Do not merge backup files into an active workspace. Check saved areas, source-dependent analysis and schedule settings after restart. The restore tests exercise preservation of jobs and absolute source references plus rejection of tampered/path-traversing backups.

## Upgrade and rollback

1. Stop the service and create/verify a full workspace backup. Preserve the current source revision or wheel and dependency environment.
2. Install the new release in a separate virtual environment; run its regression checks against a disposable database. Startup applies additive schema migrations.
3. Start the new release on the existing database, inspect health and a real saved area, then resume ordinary work.
4. If rollback is needed, stop the new version. Preserve its database/artifacts in a recovery directory. Restore the pre-upgrade backup to the original path and run the preserved previous source/wheel/environment. Do not downgrade code against a newly migrated database or discard post-upgrade observations without preserving them.

The package can be built locally with `python -m pip wheel . --no-deps --no-build-isolation --wheel-dir artifacts/dist`. Installation from source includes the dashboard assets. An operator should retain exact installed package versions with each deployment; the validation record identifies the environment actually tested. No hosted deployment is approved by this local rollback procedure.

## Optional notifications

Delivery is disabled by default. An operator may configure `FORESTMIND_WEBHOOK_URL` (public HTTPS, port 443) and `FORESTMIND_WEBHOOK_SECRET` outside source control, restart, then explicitly authorize delivery in Data & methods. Enabling sends current active non-synthetic alert episodes and subsequent new episodes. Payloads include region identity and supporting observations; choose a receiver you intend to receive that information. Changing the configured destination requires fresh authorization. Disabling stops future dispatch attempts; a request already in flight can finish.

The durable outbox reserves each attempt before network delivery and attempts delivery at most three times, including across restarts. An interrupted attempt can have reached the receiver even though its outcome was not saved. With attempts remaining, recovery retries it; after the third attempt, it stops and reports the uncertain outcome for receiver-side inspection. Recipients must deduplicate the `event_id` / `X-ForestMind-Event` identifier: retries can duplicate delivery. Verify `X-ForestMind-Signature` as HMAC-SHA256 over the exact received body, using the shared secret. Redirects and non-public DNS destinations are rejected. Endpoint queries, secrets and provider response bodies are not returned to the browser. Tests use a mocked receiver; no real notifications were sent during verification.

## Providers and troubleshooting

Photon/OSM search is explicit, cached and throttled; public demo availability is not guaranteed. `FORESTMIND_GEOCODER_URL` can select an operator-run HTTPS Photon-compatible endpoint. The browser map uses ordinary attributed OSM tiles. Sentinel-2 acquisition uses explicitly selected Earth Search. Radar uses Planetary Computer Sentinel-1 RTC. Annual WorldCover uses the ESA public COG archive. ERA5 context uses Open-Meteo's historical endpoint, whose public service is for noncommercial use. Respect each provider's current terms and quotas; the source-specific guides retain attribution and scientific limits.

| Symptom | Action |
| --- | --- |
| Connection refused | Start the local service and use its selected port; inspect startup errors. |
| Optional raster dependency missing | Install `[real-data]` in the same interpreter used by `serve.py`. |
| Search unavailable | Use coordinates; retry later or configure your own Photon endpoint. |
| No candidates / low quality | Inspect month-level evidence, adjust a new plan deliberately, or choose another period. Do not infer no forest change. |
| Cancel appears delayed | Wait for the current bounded provider read to return or time out. |
| Job interrupted | Acquisition recovers its persisted plan automatically on startup; interrupted analysis can be rerun without losing prior snapshots. |
| Analysis record stale | Rerun it after data, method, context or model configuration changes. Older immutable exports remain valid historical records. |
| Forecast unavailable | Inspect consecutive usable history and latest-month quality; downloading more scenes does not guarantee eligibility. |
| Model unavailable | Inspect exact checkpoint/evaluation/scope/input reasons. Do not relabel a synthetic checkpoint as real. |
| Radar / climate / cover collection failed | Earlier dated context remains available; retry the relevant action after reviewing the failure. No substitute data is created. |
| Windows tests stall before app startup | This workspace's restricted runner can block asyncio's local socket pair. Run approved tests in an ordinary local terminal; use a fresh pytest temporary directory if needed. |

See [requirements and acceptance gates](REQUIREMENTS_MATRIX.md), [verified evidence](VALIDATION.md), [research protocol](RESEARCH_PROTOCOL.md), [acquisition](ACQUISITION_OPERATIONS.md), [land cover](LANDCOVER_CONTEXT.md) and [radar](RADAR_CONTEXT.md).
