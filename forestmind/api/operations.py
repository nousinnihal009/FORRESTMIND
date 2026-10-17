"""Local diagnostics and safe database snapshots. No secrets or arbitrary paths."""
from __future__ import annotations

from datetime import date, datetime, timezone
import json
import shutil
from uuid import uuid4

from fastapi import HTTPException


def install_operations(app, store):
    @app.get("/api/v1/operations")
    def operations():
        with store.connection() as db:
            integrity = db.execute("PRAGMA quick_check").fetchone()[0]
            migrations = [r[0] for r in db.execute("SELECT version FROM schema_migrations ORDER BY version")]
        usage = shutil.disk_usage(store.path.parent)
        return {"status": "ok" if integrity == "ok" else "degraded", "database_check": integrity,
                "schema_migrations": migrations, "database_bytes": store.path.stat().st_size,
                "disk_free_bytes": usage.free, "access": "local-single-user-only",
                "scheduling": "Local scheduled updates run only while this service is running.",
                "notification_delivery": "Disabled unless an operator configures and explicitly authorizes a webhook.",
                "deployment_gate": "Public multi-user deployment blocked: authentication, per-user isolation and hosted acceptance are not implemented."}

    @app.post("/api/v1/operations/backup", status_code=201)
    def backup():
        directory = store.path.parent / "backups"
        if directory.exists() and len(list(directory.glob("*.db"))) >= 20:
            raise HTTPException(409, "Backup retention limit reached (20). Archive reviewed backups before creating another.")
        name = "forestmind-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:6] + ".db"
        store.backup(directory / name)
        return {"filename": name, "verified": True, "kind": "database_snapshot",
                "note": "Includes saved measurements, analyses, schedules and job records. Archive acquisition cache/artifacts separately for full reproducibility. Restore to a new database with the service stopped; see operator guide."}

    @app.get("/api/v1/regions/{region_id}/operational-alerts")
    def operational_alerts(region_id: str):
        result = store.analysis(region_id)
        if result is None:
            raise HTTPException(404, "Region not found")
        with store.connection() as db:
            row = db.execute("SELECT payload FROM alert_settings WHERE region_id=?", (region_id,)).fetchone()
        days = json.loads(row[0])["stale_days"] if row else 90
        age = (datetime.now(timezone.utc).date() - date.fromisoformat(result["latest_date"])).days
        return {"alerts": [{"id": region_id + "-stale", "kind": "monitoring_stale", "severity": "info",
                "date": result["latest_date"], "message": f"Last observation is {age} days old (configured limit {days}). Refresh this area; a historical study may intentionally use an old cutoff."}] if age > days else [],
                "note": "Data freshness is separate from ecological condition. No detected change does not mean enough observations exist."}
