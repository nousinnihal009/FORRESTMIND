"""Optional annual gross tree-cover-loss regression, separate from condition."""
from __future__ import annotations

from datetime import date, timedelta
from torch import nn


def annual_loss_contract(config: dict) -> dict | None:
    """Require an explicit versioned annual outcome; legacy models stay unchanged."""
    head = config.get("heads", {}).get("annual_loss")
    if head is None:
        return None
    if not isinstance(head, dict) or type(head.get("enabled")) is not bool:
        raise ValueError("heads.annual_loss requires explicit boolean enabled")
    if not head["enabled"]:
        return None
    if set(head) != {"enabled", "horizon_days", "units", "protocol_id"}:
        raise ValueError("annual_loss head requires enabled, horizon_days, units and protocol_id")
    if type(head["horizon_days"]) is not int or head["horizon_days"] not in (365, 366):
        raise ValueError("annual_loss horizon_days must be exactly 365 or 366")
    if head["units"] != "fraction_0_1" or head["protocol_id"] != "annual-gross-tree-cover-loss-v1":
        raise ValueError("annual_loss requires the versioned gross-loss fraction protocol")
    spec = config.get("data", {}).get("target_spec", {}).get("annual_loss")
    validate_annual_loss_spec(spec)
    if spec["horizon_days"] != head["horizon_days"]:
        raise ValueError("annual_loss head and target_spec horizon differ")
    return dict(spec)


def validate_annual_loss_spec(spec: dict) -> None:
    # Import lazily: constructing a legacy model does not load the reference harness.
    from forestmind.data.reference_review import TARGET
    if not isinstance(spec, dict) or type(spec.get("horizon_days")) is not int or spec["horizon_days"] not in (365, 366):
        raise ValueError("annual_loss target_spec requires an explicit 365/366-day protocol")
    expected = {**TARGET, "horizon_days": spec["horizon_days"]}
    if spec != expected:
        raise ValueError("annual_loss target_spec must preserve the complete annual-gross-tree-cover-loss-v1 protocol")


def annual_loss_target_date(cutoff: str, horizon_days: int) -> str:
    if type(horizon_days) is not int or horizon_days not in (365, 366):
        raise ValueError("annual_loss horizon must be 365 or 366 days")
    origin = date.fromisoformat(cutoff)
    target = origin + timedelta(days=horizon_days)
    if origin.isoformat() != cutoff or (origin.month, origin.day) != (12, 31) or target != date(origin.year + 1, 12, 31):
        raise ValueError("annual_loss requires a December 31 cutoff and exact next December 31 horizon")
    return target.isoformat()


class AnnualLossHead(nn.Module):
    """A bounded conditional mean fraction; sigmoid is not an event probability."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.regression = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1), nn.Sigmoid())

    def forward(self, summary):
        return {"mean": self.regression(summary).squeeze(-1)}
