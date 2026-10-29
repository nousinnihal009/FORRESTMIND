# Mudumalai pilot — Tamil Nadu

This is the first named Tamil Nadu study area for the user's zero-budget pilot. The user has no verified reference labels or research/NGO partner yet. The pilot uses the existing computer and public data; no paid service, public hosting, external notification or outreach has been enabled.

## Study area

- Saved name: **Mudumalai forest pilot, Tamil Nadu**; region ID `area-2b3262c5a5e5`.
- WGS84 rectangle: west `76.5238`, south `11.5889`, east `76.5438`, north `11.6089`.
- Approximate area: **4.84 km²**. This is a study rectangle, not an official reserve boundary or a verified research-plot boundary.
- The [Nilgiris district administration](https://nilgiris.nic.in/tourist-place/mudumalai-tiger-reserve/) identifies Mudumalai and its forest habitats. The rectangle is centred near the publicly indexed [ForestGEO Mudumalai site](https://forestgeo.si.edu/sites/asia/mudumalai) coordinate. No ForestGEO field dataset has been obtained, and no partnership or permission to use unpublished data is implied.
- Optical samples use a 64 × 64 grid over that rectangle. Sampling spacing is approximately 35 metres; resampling does not create new native sensor detail.
- The earlier generic **Tamil Nadu, India** record remains separate and preserved.

## Data and interpretation

Sentinel-2 surface reflectance comes from Element 84 Earth Search's public C1 L2A collection. The saved request in [mudumalai.json](mudumalai.json) records the exact initial acquisition settings. The collector retains at most one accepted scene per month, applies scene classification and valid-pixel checks, and leaves failed or cloudy months missing. An observation's date describes satellite acquisition, not the date someone opened the app.

The existing ESA WorldCover 2021 v200 reference covers the study rectangle. It classifies about **97.45%** of valid sampled area as tree/mangrove cover, with **100%** classification sample coverage. This dated woody-cover proxy supplies historical patch/edge context; it is not a current forest inventory, an official boundary or independent field verification. License: CC BY 4.0; attribution: ESA WorldCover / VITO. Source and retrieval information remain in the saved analysis export.

The vegetation condition score is a heuristic index. It is not a trained disease detector, biodiversity assessment or calibrated risk probability. The app withholds analyses when their history requirements are unmet: twelve consecutive usable monthly records for the baseline forecast, longer histories for resilience and recovery diagnostics. These statistical baselines do not establish an ecological tipping point.

## Operation within ₹0

Open the local app, select this area and inspect **Connected analysis**, **Data coverage** and **Monitoring updates**. Successful region selection is remembered across reloads. If no selection is remembered, a real observed area is preferred over a synthetic demonstration when available. Demonstration regions remain explicitly labelled and available for comparison.

Use **Collect missing observations** for a bounded refresh. Use **Resume collection** to continue a stopped job's frozen candidates with another bounded budget. A new request is needed to reconsider candidates already recorded as failed. Existing accepted months are preserved. Local scheduling is off for this pilot; enabling it later requires the computer and ForestMind service to remain running. This is periodic satellite monitoring, not a continuous live feed of all forest activity.

No manual download or payment is needed from the user for the present pilot. Network access, local storage, electricity and provider availability still constrain free operation. Public multi-user hosting and operational supervision remain separate unfinished work.

## Evidence and next steps

The initial job `acq-5cce7597488a421a` accepted a real **30 August 2026** observation. The original historical job `acq-94f09e09a0434c16` exhausted its first budget after repeated raster-read failures. On October 29, a bounded diagnostic successfully read all twelve reflectance bands plus SCL from its previously failing 20 August 2025 candidate, without replacing any cached evidence. The diagnostic is saved in `artifacts/mudumalai-band-diagnostic-20261029.json`. This supports a transient access failure; it does not establish the precise cause of the earlier errors.

Further collection and final verification are recorded in `IMPLEMENTATION_PROGRESS.md` and `VALIDATION.md`. Preserve the frozen plans, progress reports, source files and immutable exports when continuing after an interruption.

The next scientific dependency is a licensed, independently reviewed reference dataset with dated forest-change outcomes. A future collaborator could help verify labels and evaluate warnings. Satellite inputs alone do not provide that independent validation. Until then, the pilot is an evidence inspection and software testing workspace, with missing measurements and unsupported predictions kept explicit.
