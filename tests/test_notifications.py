from datetime import datetime, timedelta, timezone
import json

import pytest

from forestmind.api.store import Store
from forestmind.monitoring.analysis import analyze_region
from forestmind.monitoring.demo import demo_regions
from forestmind.monitoring.notifications import Notifications


def prepared(tmp_path, monkeypatch, sender):
    monkeypatch.setenv("FORESTMIND_WEBHOOK_URL", "https://receiver.example/forest?token=secret")
    monkeypatch.setenv("FORESTMIND_WEBHOOK_SECRET", "test-secret")
    store = Store(tmp_path / "notices.db")
    region, rows = demo_regions()[1]
    # Test fixture only: exercise a nonsynthetic alert path without network delivery.
    region["provenance"]["synthetic"] = False
    store.put_region(region, rows, analyze_region(region, rows))
    return Notifications(store, sender=sender)


def test_disabled_until_authorized_and_deduplicated(tmp_path, monkeypatch):
    sent = []
    service = prepared(tmp_path, monkeypatch, sent.append)
    service.collect()
    assert not service.dispatch_once()
    assert not sent and not service.outbox()
    with pytest.raises(ValueError, match="explicitly authorize"):
        service.configure(True, False)
    service.configure(True, True)
    service.collect()
    count = len(service.outbox())
    assert count > 0
    service.collect()
    assert len(service.outbox()) == count
    assert service.dispatch_once()
    assert sent[0]["alert"]["supporting_observations"]
    assert "secret" not in json.dumps(service.config())
    assert "token" not in json.dumps(service.outbox())
    service.configure(False, False)
    assert not service.dispatch_once()


def test_failed_delivery_bounded_and_redacted(tmp_path, monkeypatch):
    def fail(_):
        raise RuntimeError("https://receiver.example?token=secret")
    service = prepared(tmp_path, monkeypatch, fail)
    service.configure(True, True)
    service.collect()
    for _ in range(3):
        assert service.dispatch_once()
        with service.store.connection() as db:
            row = db.execute("SELECT id,payload FROM notification_outbox ORDER BY rowid LIMIT 1").fetchone()
            value = json.loads(row[1]); value["next_attempt_at"] = "2000-01-01T00:00:00+00:00"
            db.execute("UPDATE notification_outbox SET payload=? WHERE id=?", (json.dumps(value), row[0]))
    failed = [r for r in service.outbox() if r["status"] == "failed"]
    assert failed and failed[0]["attempts"] == 3
    assert "secret" not in json.dumps(failed)


def test_changed_destination_requires_fresh_authorization(tmp_path, monkeypatch):
    service = prepared(tmp_path, monkeypatch, lambda _: None)
    service.configure(True, True)
    service.collect()
    monkeypatch.setenv("FORESTMIND_WEBHOOK_URL", "https://different.example/receiver")
    sent = []
    restarted = Notifications(service.store, sender=sent.append)
    assert restarted.config()["reauthorization_required"]
    assert not restarted.dispatch_once() and not sent
    restarted.configure(True, True)
    assert not restarted.config()["reauthorization_required"]
    assert restarted.dispatch_once() and sent


def test_outbox_capacity_preserves_deduplication_and_records(tmp_path, monkeypatch):
    monkeypatch.setattr("forestmind.monitoring.notifications.MAX_OUTBOX_EVENTS", 1)
    service = prepared(tmp_path, monkeypatch, lambda _: None)
    service.configure(True, True)
    service.collect()
    assert len(service.outbox()) == 1 and service.config()["capacity_reached"]
    assert service.dispatch_once()
    service.collect()
    assert len(service.outbox()) == 1 and service.outbox()[0]["status"] == "sent"
