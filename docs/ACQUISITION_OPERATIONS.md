# Local satellite acquisition

The dashboard submits work to a durable SQLite queue. A single worker performs catalog searches, raster reads, masked index preparation and a transactional merge into the selected saved area. HTTP requests only validate and enqueue work. The worker starts with the application and stops with it; local schedules require the service to remain running.

## Supported requests and limits

- Explicit provider: Element 84 Earth Search, collection `sentinel-2-c1-l2a`, Sentinel-2 Level-2A optical measurements. There is no fallback to synthetic observations or another provider.
- Confirmed WGS84 rectangular AOI, between 0.0001 and 25 km², below 85° absolute latitude, with at most 1° span on either axis. Antimeridian crossings are rejected. A named place or drawn rectangle is not verified forest cover.
- Dates: June 23, 2015 through today in UTC, at most 60 calendar months per submitted job. Collection availability can be narrower.
- Output grid: 32–256 pixels per side, default 64. Grid spacing follows the selected geographic extent; resampling does not increase native source detail.
- At most 5 candidates per month from 50 catalog results, at most 72 candidate attempts and 24 network acquisition attempts per explicit batch. Defaults are 3 candidates, 30 search results, 12 candidate attempts and 6 network attempts. The dashboard can choose a smaller initial budget.
- Twelve active jobs maximum, one acquisition worker, 2 GiB of acquisition artifacts. Quotas apply to the acquisition directory associated with the database. Pipeline/model artifacts outside that directory require their own retention policy.
- Each catalog/raster request has a timeout of at most 60 seconds. Transient provider failures have at most two automatic retries, with interruptible exponential backoff. Raster retries consume the same persisted download budget, including failures.

The current pipeline selects at most one quality-passing acquisition per UTC month; it does not create exhaustive composites, fill missing months, infer unmeasured modalities, or label the AOI as entirely forest. Scene-wide cloud cover is a catalog filter; actual valid pixel samples determine accepted AOI quality.

## Job lifecycle and recovery

`queued` work is persisted before the request returns. `running` jobs report planning, acquiring or retrying phases. `completed` means the requested months are present in the merged area. `partial` means usable data or resumable progress exists but gaps remain. `failed` preserves the reason and per-month evidence. `cancelled` work stops without changing existing observations.

Cancellation is cooperative between bounded catalog and raster requests. An in-flight request must return or time out. Acquired caches and checkpointed progress remain on disk. Cancelling before a merge leaves saved observations unchanged; an already committed merge remains valid.

The worker uses an OS file lock so only one process executes acquisition for a database. A secondary local API instance may enqueue work for the existing lock owner; it does not perform restart recovery or start another worker. This is still a local single-user service, not a hosted multi-worker deployment design.

On restart, interrupted work returns to the queue. Completed catalog windows are reused, frozen plans are checked, and accepted cache hashes and processing contracts are revalidated. Persisted download counters never reset automatically. An explicit **Resume** authorizes another batch equal to the original download budget. Resume continues remaining/interrupted candidates from the frozen plan; previously failed or rejected candidates are not silently selected again. If all candidates are exhausted, submit a new plan with a different period or quality selection and inspect the resulting sampling differences.

Refresh uses the same saved geometry and processing settings, skips months already stored, and merges only absent months. Existing observations are not overwritten. Automatic refresh considers at most the latest 60 months when a saved history becomes older than that limit. This rolling acquisition window does not delete earlier saved observations. The area can retain up to 600 monthly observations.

## Files and provenance

For database `data/forestmind.db`, acquisition artifacts live under `data/forestmind-acquisition/`. Each `acq-…` subdirectory contains a frozen `plan.json`, atomically checkpointed `progress.json`, a `cache/` tree and immutable `provenance-….json` snapshots.

Snapshots include accepted source paths, acquisition timestamps, provider/collection, source hashes, valid pixel counts, grid and processing metadata, selected settings and the accepted-only prepared payload. The corresponding snapshot, monthly records and derived analysis are committed together in SQLite. An artifact left by interruption before a database commit is not a saved monitoring result. Hashes detect file changes; they do not independently authenticate provider or ecological claims.

Keep the database and its acquisition directory together when making a backup. A database-only restore retains observations and reports, but cannot verify or resume absent raster caches. Do not delete active job folders or the `worker.lock` file. For manual storage reclamation, stop the service, back up the database and artifact tree, and archive completed job folders only when their cache-dependent analyses and resume capability are no longer needed. Existing observation exports remain in SQLite, but archived/missing caches make further cache-dependent computations unavailable until restored.

## API

All routes begin with `/api/v1`.

| Method | Route | Purpose |
|---|---|---|
| GET | `/acquisition/capabilities` | Provider, limits, storage and worker state |
| POST | `/acquisition/jobs` | Validate and enqueue a bounded AOI/history request |
| GET | `/acquisition/jobs` | Latest 100 persisted jobs |
| GET | `/acquisition/jobs/{job_id}` | Current counters and per-month progress |
| POST | `/acquisition/jobs/{job_id}/cancel` | Request cooperative cancellation |
| POST | `/acquisition/jobs/{job_id}/resume` | Authorize one additional bounded batch |
| GET | `/regions/{region_id}/acquisition` | Saved configuration, jobs, schedule and provenance snapshots |
| POST | `/regions/{region_id}/refresh` | Acquire absent months, optionally bounded by start/end/download count |
| GET/PUT | `/regions/{region_id}/schedule` | Local schedule, enabled flag and 1–365 day interval |

Acquisition submission requires `name`, `bbox: [west,south,east,north]`, `provider: "earth-search"`, `start` and `end`; optional quality and resource settings use the names shown by the API documentation. Exact duplicate active requests return the existing job. Different overlapping updates to the same saved area return HTTP 409 until its active job finishes.

## Troubleshooting

- **No usable observations:** inspect each month's search and quality reason. A provider search failure differs from no eligible scenes or a mask rejection. New parameters create a new plan; old evidence is retained.
- **Download budget exhausted:** resume explicitly for another bounded batch. Failed requests count toward the budget.
- **Storage quota reached:** archive artifacts after stopping and backing up the service. Increasing a display/grid size does not fix missing provider data.
- **Hash/plan mismatch:** restore intact plan, report and caches from backup. The worker refuses silent reuse of changed artifacts.
- **Saved area cannot refresh:** ordinary observation imports lack an acquisition configuration. Create the area through the satellite acquisition workflow first.
- **Insufficient forecast/resilience history:** examine combined coverage. Acquisition success alone does not satisfy consecutive-history requirements or establish predictive validity.
- **Schedule did not run while the computer was off:** start the service. A due schedule enqueues one current refresh and advances its next run; it does not replay every missed interval.

The acquisition service restricts browser-triggered outbound catalog/raster requests to documented public provider hosts. This restriction, input validation and local binding do not provide multi-user authentication, operational authorization or scientific validation.
