# Published forest references and the first external experiment

ForestMind now obtains a small, published reference dataset itself. User-supplied field data is no longer the only route to external evidence. This source supports a particular retrospective point experiment; it does not certify the complete ecological early-warning proposal.

## Source and preserved evidence

The pinned dataset is **A global forest reference set with time series annual change information from 2000 to 2020**, version 1.1, by Guo and collaborators. [Zenodo record 6403863](https://zenodo.org/records/6403863) was published on March 29, 2022 under CC BY 4.0. The [associated paper](https://doi.org/10.1080/01431161.2022.2088256) describes visual interpretation of Landsat, high-resolution imagery and other land-cover/change products at a declared 30 m sample scale. Retain and cite the authors and DOI when using the data.

The actual archive is 229,591 bytes, with provider MD5 `e56c8251ca391693674bf00a9a1bfb77`. It contains 10,339 points: 6,252 in the persisting-forest layer, 2,038 in persisting-nonforest and 2,049 in the change layer. ForestMind preserves the raw Zenodo metadata, original ZIP, provider checksum, local SHA256 hashes and download/decoder provenance in `data/public-references/global-forest-v1.1/`.

```powershell
python scripts/fetch_public_references.py --output data/public-references/global-forest-v1.1
python scripts/fetch_public_references.py --output data/public-references/global-forest-v1.1 --check
```

Downloads are restricted to HTTPS Zenodo, bounded in size, and validated against the pinned version, license, date, file inventory and provider checksum. Existing matching evidence is verified and reused; changed source files are rejected. ZIP files are decoded in memory without extracting paths or executing source contents.

The source provides transition counts and prose descriptions, including undated gains. The decoder reconstructs annual state only when the transition sequence is fully dated and consistent. An explicitly dated loss can remain known even if a later gain date is unknown, but that does not make the state known. Counts, order, original attributes and exclusion reasons are retained. No missing year, confidence score, review date, image identifier or sampling weight is invented.

The real source decoded to 10,286 complete loss-event histories and 9,442 complete state histories. Fifty-three points had unresolved loss descriptions, and a further 844 had undated or same-year transitions preventing complete state reconstruction. These are parsing/coverage counts, not accuracy results.

## Fixed retrospective benchmark

`scripts/benchmark_public_references.py` defines a separate outcome: a source-interpreted loss event in the next calendar year at a point interpreted as forest at the origin. It is **not** the primary 100 ha annual loss-area fraction. It does not manufacture that missing area measurement from a point label.

The reference was released after the historical origins. This experiment therefore measures retrospective feasibility using interpreted history, not historically deployable forecasting. The existing verified-reference harness and its publication-availability guards remain unchanged.

The predeclared design is:

- Hash 5-degree geographic cells into train/validation/calibration/test partitions using fixed seed 17 and 60/10/15/15 percent hash ranges. A point never crosses partitions.
- Exclude all points within 5.1 km of a cell boundary; independently audit at least 10.04 km between cross-partition centres, allowing for the nominal sample footprint. Exact duplicate coordinates are all excluded without choosing one label. This rule was added after the initial identity audit refused duplicates, before any fit or test-performance inspection.
- Use training origins 2005–2009, validation 2012, calibration 2015, and test 2018–2019. Targets are the following calendar years. Temporal gaps separate the partitions.
- Require known forest state at the origin, a complete five-year interpreted history, and a known next-year event. Preserve exclusion counts. Source change strata, total future events, and prose summaries never enter predictors.
- Predictors are latitude/longitude trigonometric terms and past state/event summaries. These are intentionally basic historical/geographic baselines. No satellite feature, climate input or pretrained/multimodal capability is claimed by this experiment.
- Compare no-new-loss, last-year event, five-year event frequency, training event frequency, fixed logistic regression, random forest and gradient boosting. Learned models also have separate sigmoid-calibrated versions fitted only on calibration data.
- Fit normalization only on training data. Validation is retained but does not tune these fixed estimators. All methods use identical test examples. Report Brier, AUROC, average precision, fixed-threshold class metrics, confusion matrices and reliability; exact-day lead times remain unavailable.
- Bootstrap paired Brier improvement over no-new-loss by geographic cell with 1,000 replicates and seed 17. Retain negative results. Intervals are conditional on the fitted models and do not include fitting/calibration uncertainty.

```powershell
python scripts/benchmark_public_references.py freeze --packet data/public-references/global-forest-v1.1 --protocol artifacts/public-reference-study-20261105/protocol.json
python scripts/benchmark_public_references.py run --packet data/public-references/global-forest-v1.1 --protocol artifacts/public-reference-study-20261105/protocol.json --output artifacts/public-reference-study-20261105/results.json
```

Freeze pins source contents, decoded examples, geography, target, settings, code and numerical runtime before performance inspection. Writers refuse existing output files. A changed implementation/source requires a scientifically justified new study, not refreezing inspected test data to tune results. Actual executed results belong in `VALIDATION.md`; commands here alone do not establish that an experiment passed.

The change-enriched source sample and exclusion rules do not represent global event prevalence. Unweighted probability/calibration scores apply only to retained samples. Interpretation may depend on later imagery; annotation errors, selection effects and spatial dependence remain possible. This experiment produces reproducible results and predictions, not an operational checkpoint or calibrated Mudumalai risk map.
