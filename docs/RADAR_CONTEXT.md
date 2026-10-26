# Sentinel-1 RTC contextual measurements

**Collect radar context** uses the Planetary Computer `sentinel-1-rtc` collection. A bounded catalog request returns at most 30 candidates from the most recent 180 days of the requested interval. Selection requires two acquisitions at least one day apart with the same satellite platform, relative orbit, pass direction, IW mode and VV/VH polarizations. Exactly two scenes and four cropped band reads are attempted. No alternate provider, synthetic image or unmatched orbital pair is substituted.

The provider supplies terrain-corrected gamma-naught calibrated power. Positive finite samples are aligned to the exact selected WGS84 AOI using bilinear interpolation in linear power, then converted with `10 × log10(power)`. There is no clipping or arbitrary normalization. Exported paired differences use the same valid pixels in both dates, with at least 40% common coverage required. [Official collection metadata and methodology](https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-1-rtc).

The provider grid has approximately 10 m spacing; the source's range and azimuth resolving scales are separately declared, typically 20 m and 22 m. Those quantities and the native projection/affine are retained instead of claiming that resampling creates 10 m instrument detail. [Microsoft's example and item metadata](https://github.com/microsoft/PlanetaryComputerExamples/blob/main/datasets/sentinel-1-rtc/sentinel-1-rtc-example.ipynb).

Results include actual timestamps, orbit identity, unrounded aligned linear-power samples, dB maps, masks, sample/source-metadata hashes, source URLs, units, search bounds and the CC BY 4.0 license. Unsigned STAC metadata is retained. SAS tokens remain in memory, and errors redact signed access queries. Only documented catalog and asset hosts are permitted. Full remote scenes are not downloaded or represented as locally hashed full files.

The paired comparison remains descriptive and does not modify monthly optical index observations. Original radar scenes can also be aligned to optical acquisition dates under an explicit grid, unit, age and coverage contract; only a compatible registered model receives those inputs. See [MULTIMODAL_INPUTS.md](MULTIMODAL_INPUTS.md). Radar changes may reflect moisture, canopy/ground structure, terrain, speckle or registration effects; they do not identify forest loss or its cause. Two dates are not a seasonal radar model or calibrated uncertainty. Nodata and positive-power masks do not independently validate local layover/shadow correction or ecological reliability.

## Operations

Install the real-data dependencies. The existing bounded network client obtains public temporary SAS access when needed. Current public access was verified without an account; older collection prose still refers to accounts. Microsoft documented removing that requirement in May 2024. [Provider clarification](https://github.com/microsoft/PlanetaryComputer/discussions/351).

An area larger than 25 km², an invalid date range, incompatible pairing or less than 40% common coverage produces an explicit unavailable result. Radar requests use at most 128 × 128 sample grids and a maximum 60-second timeout per catalog/raster request. The default grid is 64 × 64 and timeout is 30 seconds. A returned catalog subset may lack a compatible pair even if one exists elsewhere in the archive. A short one-date study interval may also be insufficient.

## Actual verification

The Tapajós AOI `[-55.10,-3.50,-55.08,-3.48]`, requested July 1–31, 2024, returned two Sentinel-1A scenes on July 11 and July 23, both relative orbit 39, descending. The bounded 64 × 64 read completed in 10.32 seconds with all 4,096 common samples valid. Mean paired changes were −0.244568 dB VV and −0.466682 dB VH. They are measured contextual differences, not a diagnosis. `artifacts/radar-real-smoke.json` retains the evidence.

Focused tests cover matching and incompatible tracks/platforms, time cutoffs, signed URL sanitization, resource/response bounds, positive-power conversion without clipping, common masks and low-coverage abstention. A real source check verifies the provider path for this AOI; broader geographic transfer and ecological predictive skill are not established.
