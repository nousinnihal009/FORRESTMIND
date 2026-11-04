# Separate annual forest-loss learning contract

The optional annual-loss head estimates the conditional mean **gross tree-cover-loss area fraction** for the [versioned 100 ha study protocol](DATA_CONTRIBUTION.md). Its sigmoid bounds a continuous quantity to 0–1; it is not a calibrated event probability. Health, attribution and condition-trajectory heads retain their existing meanings. Missing/disabled annual configuration adds no parameters to legacy model state dictionaries.

The dataset manifest includes the complete `annual-gross-tree-cover-loss-v1` definition under `target_spec.annual_loss`, with a frozen 365- or 366-day horizon. Each origin is December 31 and the target is the following December 31. These horizons are separate contracts. Every sequence must use the protocol's 100 ha footprint within 5 percent. Labels use `value`, exact target `date`, and independent `source` metadata under `labels.annual_loss`; provenance declarations still require actual source review.

The existing training-data converter produces scalar `annual_loss` and boolean `annual_loss_mask` arrays, explicit units/horizon metadata, protected partitions, train-only normalization and a runnable configuration. Unknown annual labels remain masked; the condition target is never reused as loss. An enabled head requires this explicit configuration:

```yaml
heads:
  annual_loss:
    enabled: true
    horizon_days: 365
    units: fraction_0_1
    protocol_id: annual-gross-tree-cover-loss-v1
training:
  loss_weights:
    annual_loss_mse: 1.0
```

The complete target specification also belongs in `data.target_spec.annual_loss`; the converter supplies it. A partial or changed definition is rejected. Masked mean-squared error trains this head. Checkpoints record actual supervised counts and whether the loss was enabled. MAE/RMSE and raw predictions are retained separately with fraction units. The aggregate checkpoint-selection score includes `1 − annual_loss_mae` when labels exist; it is a heuristic selection score, not scientific accuracy or calibration evidence.

Inference requires an explicitly georeferenced 100 ha input and the exact annual origin/horizon. An annual value is withheld unless the checkpoint records positive supervised counts and enabled support. Nonannual origins retain an unavailable reason. Model/evaluation overrides cannot silently change the annual target; legacy checkpoints remain compatible. The lightweight model convenience prediction does not expose an unchecked annual estimate; use the validated checkpoint inference path.

The new head is software capability. The downloaded public dataset contains point histories, which do not meet this area-fraction contract, and must not be passed to it as invented area labels. No real trained annual-area model is claimed until suitable measured area references are supplied, reviewed, and evaluated on protected data. Controlled one-epoch tests verify conversion, masked training, checkpoints, evaluation, unsupported outputs and calendar/unit rules only.
