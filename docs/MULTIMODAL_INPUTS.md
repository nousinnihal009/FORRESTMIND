# Aligned measured inputs

`forestmind.data.sequence.prepare_sequence` can now attach existing Sentinel-1 RTC
and ERA5 evidence to actual Sentinel-2 acquisitions. It does not download data or
manufacture observations, labels or model performance. The unchanged default is
an optical sequence with absent radar explicitly masked.

`add_sequence_context(prepared, ...)` provides the same operation on an already
prepared optical sequence, returning a new object without modifying the original
arrays, context JSON or provenance. Apply it once; to change the context or timing
policy, begin again from the optical sources. Earlier saved sequences remain
independent evidence artifacts.

## Preparing a sequence

Use original saved acquisition-context JSON, with the same AOI and grid as the
optical caches. Context files are limited to 16 MiB by the CLI. The command performs
no network requests and refuses to overwrite an existing output or sidecar.

```powershell
python scripts/prepare_sequence.py --cache-dir data/my-study --through 2025-12-31 --region=-55.10,-3.50,-55.08,-3.48 --radar-context data/my-radar-context.json --climate-context data/my-climate-context.json --radar-max-age-days 12 --climate-window-days 30 --climate-lag-days 5 --output data/my-study-sequence.npz
```

The paths above are placeholders for actual artifacts. Either context is optional.
The result remains a pickle-free NPZ plus a JSON provenance sidecar. Normalization
must subsequently use the training partition's statistics and the checkpoint's
documented feature contract. Neither this command nor the alignment module fits
statistics or trains a model.

## Radar timing and measurement contract

Only calibrated `sentinel-1-rtc` positive linear gamma-naught power is accepted.
The context must use the exact optical bounding box, grid shape, grid spacing and
EPSG:4326 CRS; this step never resamples implicitly. All scenes must share satellite
platform, relative orbit, ascending/descending pass, IW instrument mode and VV/VH
polarizations. Duplicate scene IDs and times are rejected. When provided, geometry,
sample and whole-context hashes are checked.

For each optical acquisition, select the latest sufficiently valid radar scene
whose timestamp is no later than that optical acquisition and whose age is at most
`radar_max_age_days` (default 12). Same-day future hours are excluded. If the latest
candidate fails the configured valid-pixel fraction, an earlier eligible scene can
be used. The default minimum valid fraction is 0.4 of the whole sample grid. A
nonrectangular AOI with large masked portions can therefore be conservatively
withheld. Future or stale radar is never carried backward or interpolated.

The original positive power is converted pixelwise with `10 * log10(power)`.
No clipping or min-max normalization occurs. Radar retains a separate pixel mask,
so a cloud-invalid optical pixel does not erase an independently valid radar
measurement. RTC gamma0 is **not** interchangeable with a model trained on
GRD/sigma0. Checkpoints must explicitly support these units and preprocessing.

| NPZ field | Shape / meaning |
| --- | --- |
| `sar` | `(T, 2, H, W)` float32, VV/VH in dB gamma0 |
| `sar_mask` | `(T,)` boolean, an eligible radar scene exists |
| `sar_quality_mask` | `(T, H, W)` boolean, separately valid radar pixels |
| `sar_acquisition_dates` | `(T,)` UTC timestamps; empty string when missing |
| `sar_age_days` | `(T,)` exact fractional age; `-1` when missing |
| `sar_units` | Scalar string `gamma0_db` |

Values under false masks are stored as zero solely for numerical safety. They
must never be interpreted as observed zero backscatter.

## Climate features and missingness

ERA5 inputs must retain `raw_response.daily`, matching UTC dates, exact documented
units and the requested optical-AOI centroid. Monthly aggregates alone are not
sufficient. A fixed trailing window ends `climate_lag_days` before each optical UTC
date (default 30-day window, 5-day lag). At least one lag day is required because a
full-day aggregate on the acquisition date would include future hours.

The four raw features have this fixed order:

| Feature | Unit | Aggregation |
| --- | --- | --- |
| `temperature_mean_c` | `degC` | Mean of daily mean 2 m temperature |
| `precipitation_sum_mm` | `mm` | Sum of daily precipitation |
| `reference_et0_sum_mm` | `mm` | Sum of daily FAO reference ET0 |
| `soil_moisture_mean_m3_m3` | `m3/m3` | Mean of daily surface soil moisture |

`climate_context`, `climate_mask` and `climate_coverage` each have shape `(T, 4)`.
Feature names and units are stored as string arrays. A climate value is usable only
when its own valid-day fraction meets `climate_min_coverage` (default 1.0).
Unreported dates and null measurements both reduce coverage. An absent feature
does not erase another valid feature, and genuine zero rainfall is valid data.

If a study explicitly lowers minimum coverage, precipitation/ET0 sums remain
partial sums of measured days; they are never scaled to invent the missing days.
Low-coverage features remain zero with false masks. Coverage, exact window dates
and valid-day counts are retained in the manifest.

The lag is an explicit modeling assumption, **not verification that the same ERA5
version existed at a historical forecast cutoff**. Reanalysis may be revised.
Historical-vintage availability remains marked unverified. The values are coarse
reanalysis at a centroid, not in-situ forest measurements, standardized drought
indices, causal effects or learned embeddings. Independent predictive evaluation
must account for this limitation.

## Reproducibility and acceptance

Schema-2 sequence provenance adds `available_modalities`, `alignment`, optional
`sar_units`, and climate feature names/units. `alignment.parameters` records every
timing and coverage setting. Each source context, matched radar sample and raw
climate response has a computed SHA256; a final `alignment_sha256` also appears in
the NPZ. Hashes establish local artifact consistency, not authenticity of a data
provider or ecological correctness. A model's registered contract must match the
alignment parameters as well as units, bands, grid and normalization.

Use original Python-written acquisition context when verifying existing hashes.
Browser reserialization can change numeric spellings (`1.0` to `1`) and invalidate
an older canonical JSON digest despite visually similar data. A hash mismatch is
reported; it is not silently accepted or repaired.

`tests/test_context_alignment.py` covers future-data exclusion, same-day radar
ordering, stale scenes, independent pixel masks, units/orbit/grid rejection,
feature-specific missingness, partial sums, policy validation, source immutability
and CLI serialization. The existing optical sequence tests continue to pass.

An actual local smoke reused the four accepted Tapajós optical scenes, the original
real RTC pilot and saved daily ERA5 evidence without downloads or source changes.
The July 17 optical acquisition matched July 11 radar; the August 26 acquisition
had a complete trailing climate window. All other missing modality/time pairs
remained masked. The evidence is
`artifacts/context-alignment-real-smoke-20261023.json`. This verifies measured-input
plumbing, not a trained multimodal forecast or scientific predictive gain.
