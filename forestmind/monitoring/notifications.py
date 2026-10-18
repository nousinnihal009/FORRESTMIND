"""Opt-in webhook delivery with a durable outbox; disabled by default."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import ipaddress
import json
import os
import socket
import threading
from urllib.parse import urlsplit
from uuid import uuid4


def now():
    return datetime.now(timezone.utc).isoformat()


class Notifications:
    def __init__(self, store, sender=None):
        self.store = store
        self.endpoint = os.environ.get("FORESTMIND_WEBHOOK_URL", "")
        self.secret = os.environ.get("FORESTMIND_WEBHOOK_SECRET", "")
        self.sender = sender or self._send
        self.stop_event = threading.Event()
        self.thread = None
        with store.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS notification_config (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS notification_outbox (
                    id TEXT PRIMARY KEY, event_key TEXT NOT NULL UNIQUE, status TEXT NOT NULL, payload TEXT NOT NULL);
            """)

    def config(self):
        with self.store.connection() as db:
            row = db.execute("SELECT payload FROM notification_config WHERE id=1").fetchone()
        config = json.loads(row[0]) if row else {"enabled": False, "authorized": False}
        return {**config, "endpoint_configured": bool(self.endpoint), "signing_configured": bool(self.secret),
                "destination_host": urlsplit(self.endpoint).hostname if self.endpoint else None,
                "delivery": "At least once; recipients should deduplicate by event_id. No messages are sent until explicit configuration and authorization."}

    def configure(self, enabled, authorized):
        if enabled and (not authorized or not self.endpoint or not self.secret):
            raise ValueError("Set FORESTMIND_WEBHOOK_URL and FORESTMIND_WEBHOOK_SECRET, then explicitly authorize delivery before enabling notifications")
        if enabled:
            self._validate_url()
        value = {"enabled": enabled, "authorized": authorized if enabled else False, "updated_at": now()}
        with self.store.connection() as db:
            db.execute("INSERT OR REPLACE INTO notification_config VALUES (1,?)", (self.store._json(value),))
        return self.config()

    def _validate_url(self):
        p = urlsplit(self.endpoint)
        if p.scheme != "https" or not p.hostname or p.username or p.password or p.port not in (None, 443) or p.fragment:
            raise ValueError("Webhook must be an HTTPS URL on port443 without embedded credentials")
        return p.hostname

    def _send(self, payload):
        import requests
        host = self._validate_url()
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
            raise ValueError("Webhook destination must resolve exclusively to public addresses")
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        signature = hmac.new(self.secret.encode(), body, hashlib.sha256).hexdigest()
        response = requests.post(self.endpoint, data=body,
            headers={"Content-Type": "application/json", "User-Agent": "ForestMind-webhook/0.6",
                     "X-ForestMind-Signature": "sha256=" + signature, "X-ForestMind-Event": payload["event_id"]},
            timeout=(3, 10), allow_redirects=False, stream=True)
        try:
            if not 200 <= response.status_code < 300:
                raise ValueError(f"Webhook returned HTTP {response.status_code}")
        finally:
            response.close()

    def collect(self):
        config = self.config()
        if not config["enabled"] or not config["authorized"]:
            return
        with self.store.connection() as db:
            for row in db.execute("SELECT region_id,payload FROM alerts").fetchall():
                alert = json.loads(row["payload"])
                if alert["status"] != "active":
                    continue
                region_row = db.execute("SELECT metadata FROM regions WHERE id=?", (row["region_id"],)).fetchone()
                if region_row and json.loads(region_row[0]).get("provenance", {}).get("synthetic"):
                    continue
                key = alert.get("episode_id", alert["id"] + ":" + alert["date"])
                payload = {"event_id": "notice-" + hashlib.sha256(key.encode()).hexdigest()[:24],
                           "event_type": "forest_indicator_alert", "region_id": row["region_id"], "created_at": now(),
                           "alert": {k: alert.get(k) for k in ("id", "date", "title", "message", "severity", "kind", "supporting_observations")},
                           "interpretation": "Heuristic screening evidence, not a validated ecological forecast."}
                entry = {"payload": payload, "attempts": 0, "next_attempt_at": now(), "error": None}
                db.execute("INSERT OR IGNORE INTO notification_outbox VALUES (?,?,?,?)", (payload["event_id"], key, "pending", self.store._json(entry)))

    def dispatch_once(self):
        config = self.config()
        if not config["enabled"] or not config["authorized"]:
            return False
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            candidates = db.execute("SELECT id,payload FROM notification_outbox WHERE status IN ('pending','retry') ORDER BY rowid LIMIT 100").fetchall()
            selected = next((r for r in candidates if json.loads(r["payload"])["next_attempt_at"] <= now()), None)
            if selected is None:
                return False
            entry = json.loads(selected["payload"])
            entry["attempts"] += 1
            db.execute("UPDATE notification_outbox SET status='sending',payload=? WHERE id=?", (self.store._json(entry), selected["id"]))
        try:
            self.sender(entry["payload"])
            status = "sent"
            entry.update(sent_at=now(), error=None)
        except Exception as error:
            # Never expose endpoint, tokens, credentials or provider response bodies.
            status = "failed" if entry["attempts"] >= 3 else "retry"
            entry.update(error=f"Delivery failed ({type(error).__name__}); inspect receiver availability.",
                         next_attempt_at=(datetime.now(timezone.utc) + timedelta(seconds=30 * entry["attempts"])).isoformat())
        with self.store.connection() as db:
            db.execute("UPDATE notification_outbox SET status=?,payload=? WHERE id=?", (status, self.store._json(entry), selected["id"]))
        return True

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        with self.store.connection() as db:
            db.execute("UPDATE notification_outbox SET status='retry' WHERE status='sending'")
        def run():
            while not self.stop_event.wait(5):
                try:
                    self.collect()
                    self.dispatch_once()
                except Exception:
                    # Preserve pending entries. Diagnostics are queryable without logging secrets.
                    pass
        self.thread = threading.Thread(target=run, daemon=True, name="forestmind-notifications")
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=15)

    def outbox(self):
        with self.store.connection() as db:
            return [{"id": r["id"], "status": r["status"], "attempts": json.loads(r["payload"])["attempts"],
                     "error": json.loads(r["payload"])["error"]} for r in db.execute("SELECT * FROM notification_outbox ORDER BY rowid DESC LIMIT 100")]


def install_notifications(app, store):
    from fastapi import HTTPException
    from pydantic import BaseModel, ConfigDict, Field
    class NotificationSettings(BaseModel):
        model_config = ConfigDict(extra="forbid")
        enabled: bool = Field(strict=True)
        authorize_delivery: bool = Field(False, strict=True)
    globals()["NotificationSettings"] = NotificationSettings
    service = Notifications(store)
    app.state.notifications = service
    app.add_event_handler("startup", service.start)
    app.add_event_handler("shutdown", service.stop)

    @app.get("/api/v1/notifications")
    def configuration():
        return {"configuration": service.config(), "outbox": service.outbox()}

    @app.put("/api/v1/notifications")
    def configure(payload: NotificationSettings):
        try:
            return service.configure(payload.enabled, payload.authorize_delivery)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
    return service
