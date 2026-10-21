# ESA WorldCover reference collection

The selected-area workflow can collect an actual WorldCover classification without manual raster imports. It reads at most four official public COG windows and preserves the confirmed geometry, categorical classes, validity mask, source grid metadata, reference year, algorithm version and a hash of the saved sample. The AOI must fit the 25 km² service limit. Output is at most 128 × 128 pixels; it is a nearest-neighbour sample rather than newly created native detail.

The source archive contains 2020/v100 and 2021/v200 maps at approximately 10 m native grid spacing. The latest complete reference year ending at or before the selected observation cutoff is chosen. Full remote tiles are not downloaded or claimed to have a local file hash. [Official data access](https://esa-worldcover.org/en/data-access), [official download implementation](https://github.com/ESA-WorldCover/esa-worldcover-datasets/blob/main/scripts/download.py).

Class 10 (tree cover) and class 95 (mangroves) form the explicitly identified woody-cover proxy used by patch and edge calculations. Other valid classes remain non-tree classes; nodata, unsupported codes and failed tiles remain unknown. This classification is not a universal ecological forest definition, a protected-area boundary or a present-day forest inventory. [Published class legend](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200).

The reference period ends on December 31 of its year. Product publication and retrieval dates are retained separately: 2020/v100 was published October 20, 2021, and 2021/v200 October 28, 2022. Context is retrospective and excluded from neural model inputs and historical forecast/backtest predictors. [2020 publication](https://zenodo.org/records/5571936), [2021 publication](https://zenodo.org/records/7254221).

No clearing estimate is calculated by subtracting these two epochs: the producer changed algorithms as well as years, so apparent differences mix classification and land-cover changes. The public map read supplies no local per-pixel classification probability or independent field validation. Topology also depends on output sampling and the AOI boundary. [ESA's comparison caveat](https://esa-worldcover.org/en/data-access).

The dataset is CC BY 4.0. Exports retain provider attribution, DOI and license. Current integration supports the two released annual epochs; it does not promise an annually updated current forest map. [ESA license and attribution](https://github.com/ESA-WorldCover/esa-worldcover-datasets/blob/main/README.md).

## Verification

### Patch shape diagnostics

Each four-connected patch now exports its area, observed internal edge length and boundary completeness. For patches entirely surrounded by valid classified pixels, perimeter includes observed nonforest holes and compactness is `4 × π × area / perimeter²`. A square on an approximately square grid has compactness π/4; an equal-area elongated rectangle has a lower value. These are grid geometry measurements, not ecological quality or functional connectivity.

Any patch touching the AOI boundary, an excluded polygon pixel or nodata has an unknown full extent. Its perimeter and compactness are null, while the known internal edge remains available. The dashboard shows complete/censored counts and up to 20 patch rows; exports retain all rows. Grid stair-stepping and chosen sampling resolution affect the values. Tests cover analytic shapes, holes, diagonal adjacency, clipped/unknown boundaries and empty or entirely wooded AOIs. The real all-tree Tapajós crop correctly yields one boundary-clipped patch and withholds compactness.

### Earlier source collection check

On October 18, 2026, 17 tests passed, including a real in-memory Rasterio reprojection check, tile boundaries/seams, nodata, polygon masking, temporal cutoffs and source failure. A bounded actual 64 × 64 crop for `[-55.10,-3.50,-55.08,-3.48]` completed in 1.82 seconds from the official `S06W057` 2021/v200 tile. All 4,096 output samples were valid class-10 tree cover. The saved result is `artifacts/worldcover-real-smoke.json` with source, masks, dates, native/source metadata and sample hash. This confirms provider plumbing for that small AOI; it does not independently verify the classification or establish global geographic accuracy.
