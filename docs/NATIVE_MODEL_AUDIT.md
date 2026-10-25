# Native encoder loading audit

Updated October 25, 2026. **Offline native architecture construction, complete local checkpoint loading, and real-input forward execution are now implemented and exercised.** This closes the earlier download-capable-constructor limitation. No pretrained model weights or independently validated ForestMind checkpoint have been supplied. The executed diagnostics used explicitly **random, untrained weights** and do not demonstrate forest predictive accuracy.

## Implemented loading contract

`forestmind/models/foundation/native.py` validates the native contract before inference. `predict_checkpoint` and checkpoint evaluation construct a copy of the stored configuration with `pretrained=False`; they never pass an upstream checkpoint path or ask the native constructor to download pretrained weights. The complete local ForestMind state, including its encoder, projection, fusion, temporal model and task heads, is then loaded. Missing or extra keys, wrong shapes or dtypes, nonfinite values, non-tensor values, sparse, quantized or meta tensors fail explicitly. State validation finishes before model parameters are copied, so a failed load cannot leave a partly updated model.

Native registry dispatch is restricted to the following standard encoder variants:

| Family | Allowed architecture IDs |
| --- | --- |
| Prithvi | `prithvi_eo_v1_100`, `prithvi_eo_v2_300`, `prithvi_eo_v2_600` |
| TerraMind | `terramind_v1_tiny`, `terramind_v1_small`, `terramind_v1_base`, `terramind_v1_large` |

TiM/generative/tokenizer models, coordinate-conditioned Prithvi variants and arbitrary registry names or constructor options are not accepted. The adapter never substitutes a proxy after a native failure. An explicitly requested proxy remains an untrained development architecture and is not eligible as a native model.

The checkpoint must contain `foundation.native_runtime` with exact `terratorch`, `torch`, `torchvision` and `timm` versions matching the execution environment. TerraTorch is currently restricted to the inspected **1.2.13** release. `capture_native_runtime()` supplies this map for training/checkpoint provenance. The full installed dependency set and downloaded package records are retained in the verification artifacts; the four-key model contract is not a claim that the entire operating system is reproducible.

Native inputs require explicit per-band `data.normalization` statistics and `data.optical_units: surface_reflectance_0_1`. Normalization happens once in ForestMind's transform. The checkpoint author must supply statistics appropriate to its actual training units and selected pretrained artifact; the generic synthetic defaults are not accepted for native inference.

Prithvi selects B02/B03/B04/B8A/B11/B12 from the canonical twelve optical bands. Observed SAR and SAR-only timesteps are rejected rather than silently ignored, including during training and checkpoint evaluation. Grid dimensions must be divisible by the architecture's patch size: 16 pixels for the supported v1/300M models, 14 for the 600M model. This channel mapping does not prove transfer from HLS pretraining to a particular Sentinel-2 product or resolution.

TerraMind's `sar_modality` is explicit: `S1RTC` requires `data.sar_units: gamma0_db`; `S1GRD` requires `sigma0_db`. The adapter uses the corresponding modality key and passes only observed modalities. Raw positive linear RTC power must be converted correctly before entering the dB input contract. RTC gamma-naught and GRD sigma-naught products are not interchangeable. Grid dimensions must be divisible by 16. Physical scale, temporal matching, source quality, training geography and missing-modality behavior still need a justified model contract and validation.

The normal monitoring workflow also requires the trusted model registry, an exact checkpoint/evaluation hash match, real training declarations, compatible input/geographic scope and an independent acceptance record. Implementing native loading does not approve any model automatically. No new operational model was registered by these diagnostics.

## Actual installed runtime and execution evidence

An isolated environment exists at `artifacts/native-venv`, created with system site packages to reuse the existing PyTorch installation. The application's default Python environment was not replaced. The installed runtime used in both successful diagnostics was:

| Package | Version |
| --- | --- |
| TerraTorch | 1.2.13 |
| PyTorch | 2.10.0 |
| torchvision | 0.25.0 |
| timm | 1.0.30 |

`artifacts/native-install-result.json` records dependency installation and package archive hashes. Each runtime report also records the full installed package/version inventory.

The following results were produced by `scripts/verify_native_runtime.py` on the preserved **real July 29, 2024 Sentinel-2 observation**, a 64 x 64 twelve-band grid in `artifacts/real-pilot-sequence.npz`. That input contains no SAR observations or ecological labels.

| Actual diagnostic | ForestMind parameter count | Elapsed time | Result artifact |
| --- | ---: | ---: | --- |
| TerraMind v1 tiny | 6,238,153 | 49.493 s | `artifacts/native-terramind-runtime-20261025.json` |
| Prithvi EO v1 100M encoder | 86,471,305 | 45.859 s | `artifacts/native-prithvi-runtime-20261025.json` |

Both runs constructed the actual native encoder, wrote a complete temporary local state, loaded it through the production inference function, and obtained identical predictions on two successive runs. Both recorded finite outputs with five health-class scores, eight driver scores and one configured horizon. Outbound socket connections were denied, an empty Hugging Face weight-cache location was used, and **zero network attempts** occurred. The temporary random checkpoints were removed afterward and were never placed in the operational registry.

These were **random-weight software diagnostics with an explicit identity normalization**, not pretrained-model accuracy tests. Their predictions are intentionally omitted from the diagnostic result because they have no ecological meaning. The successful TerraMind run exercised optical input only; measured RTC forward execution and other allowed model sizes have not been demonstrated by these two artifacts. No GPU inference, licensed pretrained artifact, trained downstream heads or held-out scientific benchmark is established by this evidence.

To reproduce a runtime check after installing the exact compatible environment, choose a new output filename to preserve existing evidence:

```powershell
artifacts/native-venv/Scripts/python.exe scripts/verify_native_runtime.py --foundation terramind --model terramind_v1_tiny --input artifacts/real-pilot-sequence.npz --observed-through 2024-07-29 --output artifacts/native-terramind-new-check.json
artifacts/native-venv/Scripts/python.exe scripts/verify_native_runtime.py --foundation prithvi --model prithvi_eo_v1_100 --input artifacts/real-pilot-sequence.npz --observed-through 2024-07-29 --output artifacts/native-prithvi-new-check.json
```

The script is a standalone diagnostic process. Its network-denial patches are not installed in the running application. It refuses to overwrite an existing result or the input sequence and clearly marks `trained_model`, `predictive_validity` and `registered_operational_model` as false.

## Tests and remaining evidence

`tests/test_native_loading.py` exercises complete-state round trips through small registry doubles, forces `pretrained=False`, checks RTC/GRD modality routing, runtime mismatch and unsupported architecture rejection, checks corrupted/incomplete state refusal without partial loading, and checks native input geometry/modality restrictions in inference, training and evaluation. Registry doubles test software contracts only; the two artifacts above provide separate actual native runtime evidence. The dedicated network-guard test verifies that a requested connection is refused and the original socket API is restored afterward.

The remaining model work is to obtain a licensed pretrained encoder, train the required ForestMind heads on independently labeled observations, preserve its complete state and exact runtime/input contract, and evaluate on protected geographic/temporal examples with appropriate calibration. The model's input units, normalization, resolution, modality sampling and domain must be validated for that study. Only a compatible independently accepted checkpoint should enter normal monitoring. No irreversible tipping-point, causal-intervention or pretrained representation benefit claim is supported yet.

The implementation was checked against the installed [TerraTorch 1.2.13 release](https://pypi.org/project/terratorch/1.2.13/) source. The [official Prithvi builder](https://github.com/torchgeo/terratorch/blob/main/terratorch/models/backbones/prithvi_vit.py) and [TerraMind builder](https://github.com/torchgeo/terratorch/blob/main/terratorch/models/backbones/terramind/model/terramind_register.py) explain upstream construction, while the [TerraMind guide](https://github.com/torchgeo/terratorch/blob/main/docs/guide/terramind.md) documents raw modality keys. These `main` links are background documentation; the installed version, package hashes and executed results above identify the tested runtime.
