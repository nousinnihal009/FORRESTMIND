"""Dated alert episodes, configurable screening thresholds and human review state."""
from __future__ import annotations

from datetime import datetime, timezone
import json

DEFAULT_SETTINGS = {"condition_drop_points": 5.0, "moisture_drop": 0.12,
                    "min_valid_fraction": 0.4, "stale_days": 90}


def reconcile_alerts(db, analysis):
    """Run inside the same transaction as the observation/analysis update."""
    configured = db.execute("SELECT payload FROM alert_settings WHERE region_id=?", (analysis["id"],)).fetchone()
    settings = json.loads(configured[0]) if configured else DEFAULT_SETTINGS
    alerts = list(analysis["alerts"])
    rows = analysis["observations"]
    if configured:
        alerts = [a for a in alerts if not a["id"].endswith("-decline")]
        change = analysis.get("health_change")
        if change is not None and change <= -settings["condition_drop_points"]:
            alerts.append({"id": analysis["id"] + "-decline", "severity": "medium", "title": "Recent condition decline",
                "message": f"Observed condition changed {change:+.1f} points; configured decline threshold is {settings['condition_drop_points']}. Seasonality and sensor changes are competing explanations.",
                "date": analysis["latest_date"], "status": "active"})
        valid = [r for r in rows if r.get("ndmi") is not None and r["valid_fraction"] >= settings["min_valid_fraction"]]
        if len(valid) >= 4 and rows[-1] in valid:
            previous = sum(r["ndmi"] for r in valid[-13:-1]) / len(valid[-13:-1])
            delta = valid[-1]["ndmi"] - previous
            if delta <= -settings["moisture_drop"]:
                alerts.append({"id": analysis["id"] + "-moisture", "severity": "medium", "title": "Observed moisture-index decline",
                    "message": f"NDMI change {delta:+.3f}; configured threshold {settings['moisture_drop']}. Spectral moisture change is not an independent drought diagnosis.",
                    "date": analysis["latest_date"], "status": "active"})
    now = datetime.now(timezone.utc).isoformat()
    active_episodes = []
    for alert in alerts:
        episode = alert["id"] + ":" + alert["date"]
        prior = db.execute("SELECT payload FROM alert_history WHERE episode=?", (episode,)).fetchone()
        previous = json.loads(prior[0]) if prior else {}
        alert.update({"episode_id": episode, "kind": "data_quality" if alert["id"].endswith("-coverage") else "ecological_indicator",
                      "status": previous.get("status", "active"), "note": previous.get("note", ""),
                      "reviewed_at": previous.get("reviewed_at"), "first_seen_at": previous.get("first_seen_at", now),
                      "supporting_observations": [{k: r.get(k) for k in ("date", "ndvi", "ndmi", "nbr", "valid_fraction", "health_score")} for r in rows[-15:]],
                      "source": analysis["provenance"]["source"], "thresholds": settings})
        db.execute("INSERT OR REPLACE INTO alert_history VALUES (?,?,?)", (episode, analysis["id"], json.dumps(alert, allow_nan=False)))
        active_episodes.append(episode)
    # Preserve history and resolution. A new observation can open a new episode.
    for old in db.execute("SELECT episode,payload FROM alert_history WHERE region_id=?", (analysis["id"],)).fetchall():
        previous = json.loads(old["payload"])
        if old["episode"] not in active_episodes and previous["status"] in {"active", "acknowledged"}:
            previous.update(status="resolved", resolved_at=now, resolution="Superseded by a newer assessment")
            db.execute("UPDATE alert_history SET payload=? WHERE episode=?", (json.dumps(previous), old["episode"]))
    analysis["alerts"] = alerts
    analysis["alert_count"] = sum(a["status"] == "active" for a in alerts)


def install_alerts(app, store):
    from fastapi import HTTPException
    from pydantic import BaseModel, ConfigDict, Field
    from typing import Literal

    class AlertSettings(BaseModel):
        model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
        condition_drop_points: float = Field(5, ge=0.1, le=100, strict=True)
        moisture_drop: float = Field(0.12, ge=0.01, le=2, strict=True)
        min_valid_fraction: float = Field(0.4, ge=0, le=1, strict=True)
        stale_days: int = Field(90, ge=1, le=1095, strict=True)

    class AlertReview(BaseModel):
        model_config = ConfigDict(extra="forbid")
        status: Literal["open", "acknowledged", "resolved"]
        note: str = Field("", max_length=2000)

    # These local annotations must be resolvable by FastAPI despite future annotations.
    globals()["AlertSettings"] = AlertSettings
    globals()["AlertReview"] = AlertReview

    def require(region_id):
        if store.region(region_id) is None:
            raise HTTPException(404, "Region not found")

    @app.get("/api/v1/regions/{region_id}/alert-settings")
    def settings(region_id: str):
        require(region_id)
        with store.connection() as db:
            row = db.execute("SELECT payload FROM alert_settings WHERE region_id=?", (region_id,)).fetchone()
        return json.loads(row[0]) if row else DEFAULT_SETTINGS

    @app.put("/api/v1/regions/{region_id}/alert-settings")
    def update_settings(region_id: str, payload: AlertSettings):
        require(region_id)
        from forestmind.monitoring.analysis import analyze_region
        with store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR REPLACE INTO alert_settings VALUES (?,?)", (region_id, store._json(payload.model_dump())))
            # Load within write transaction: acquisition cannot change this snapshot mid-update.
            row = db.execute("SELECT metadata,observations FROM regions WHERE id=?", (region_id,)).fetchone()
            store._save_analysis(db, analyze_region(json.loads(row[0]), json.loads(row[1])))
        return payload.model_dump()

    @app.get("/api/v1/regions/{region_id}/alert-history")
    def history(region_id: str):
        require(region_id)
        with store.connection() as db:
            return {"alerts": [json.loads(row[0]) for row in db.execute("SELECT payload FROM alert_history WHERE region_id=? ORDER BY rowid DESC", (region_id,))]}

    @app.patch("/api/v1/alerts/{alert_id}")
    def review(alert_id: str, payload: AlertReview):
        with store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT region_id,payload FROM alerts WHERE id=?", (alert_id,)).fetchone()
            if row is None:
                raise HTTPException(404, "Current alert not found")
            alert = json.loads(row["payload"])
            alert.update(status="active" if payload.status == "open" else payload.status, note=payload.note,
                         reviewed_at=datetime.now(timezone.utc).isoformat())
            db.execute("UPDATE alerts SET payload=? WHERE id=?", (store._json(alert), alert_id))
            db.execute("UPDATE alert_history SET payload=? WHERE episode=?", (store._json(alert), alert["episode_id"]))
            result = json.loads(db.execute("SELECT result FROM analyses WHERE region_id=?", (row["region_id"],)).fetchone()[0])
            result["alerts"] = [alert if a["id"] == alert_id else a for a in result["alerts"]]
            result["alert_count"] = sum(a["status"] == "active" for a in result["alerts"])
            db.execute("UPDATE analyses SET result=? WHERE region_id=?", (store._json(result), row["region_id"]))
        return alert
