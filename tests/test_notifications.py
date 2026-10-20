from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import socket

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


@pytest.mark.parametrize("attempts", [1, 2, 3])
def test_restart_preserves_reserved_attempts_and_never_delivers_fourth(tmp_path, monkeypatch, attempts):
    sent = []
    service = prepared(tmp_path, monkeypatch, sent.append)
    service.configure(True, True)
    service.collect()
    with service.store.connection() as db:
        row = db.execute("SELECT id,payload FROM notification_outbox ORDER BY rowid LIMIT 1").fetchone()
        # A process stopped after reserving an attempt, before recording its
        # outcome. It is unknown whether the receiver accepted that attempt.
        value = json.loads(row[1])
        value.update(attempts=attempts, next_attempt_at="2000-01-01T00:00:00+00:00")
        db.execute("DELETE FROM notification_outbox WHERE id<>?", (row[0],))
        db.execute("UPDATE notification_outbox SET status='sending',payload=? WHERE id=?", (json.dumps(value), row[0]))
    restarted = Notifications(service.store, sender=sent.append)
    restarted.start()
    restarted.stop()
    recovered = restarted.outbox()[0]
    assert recovered["attempts"] == attempts
    assert recovered["status"] == ("failed" if attempts == 3 else "retry")
    assert restarted.dispatch_once() is (attempts < 3)
    assert len(sent) == (1 if attempts < 3 else 0)
    settled = restarted.outbox()[0]
    assert settled["attempts"] == min(attempts + 1, 3)
    assert settled["status"] == ("failed" if attempts == 3 else "sent")
    if attempts == 3:
        assert "may have reached" in settled["error"]
    assert not restarted.dispatch_once()


def test_preexisting_exhausted_retry_is_settled_without_delivery(tmp_path, monkeypatch):
    sent = []
    service = prepared(tmp_path, monkeypatch, sent.append)
    service.configure(True, True)
    service.collect()
    with service.store.connection() as db:
        row = db.execute("SELECT id,payload FROM notification_outbox ORDER BY rowid LIMIT 1").fetchone()
        value = json.loads(row[1])
        value.update(attempts=3, next_attempt_at="2000-01-01T00:00:00+00:00")
        db.execute("DELETE FROM notification_outbox WHERE id<>?", (row[0],))
        db.execute("UPDATE notification_outbox SET status='retry',payload=? WHERE id=?", (json.dumps(value), row[0]))
    assert not service.dispatch_once()
    assert not sent
    assert service.outbox()[0]["status"] == "failed"
    assert service.outbox()[0]["attempts"] == 3


@pytest.mark.parametrize("status", [200, 302, 503])
def test_http_transport_signs_exact_body_and_closes_response(tmp_path, monkeypatch, status):
    service = prepared(tmp_path, monkeypatch, None)
    calls, closed = [], []

    class Response:
        status_code = status

        def close(self):
            closed.append(True)

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))])
    monkeypatch.setattr("requests.post", post)
    payload = {"event_id": "notice-test", "area": "Tapajós", "measurement": 0.42}
    if status == 200:
        service._send(payload)
    else:
        with pytest.raises(ValueError, match=f"HTTP {status}"):
            service._send(payload)
    assert len(calls) == 1 and closed == [True]
    _, request = calls[0]
    assert json.loads(request["data"]) == payload
    expected = hmac.new(b"test-secret", request["data"], hashlib.sha256).hexdigest()
    assert request["headers"]["X-ForestMind-Signature"] == "sha256=" + expected
    assert request["headers"]["X-ForestMind-Event"] == payload["event_id"]
    assert request["allow_redirects"] is False
    assert request["timeout"] == (3, 10)
    assert request["stream"] is True


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "::1"])
def test_transport_rejects_nonpublic_dns_before_post(tmp_path, monkeypatch, address):
    service = prepared(tmp_path, monkeypatch, None)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
        (socket.AF_INET6 if ":" in address else socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))])
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: pytest.fail("No request may reach a nonpublic destination"))
    with pytest.raises(ValueError, match="exclusively to public"):
        service._send({"event_id": "notice-test"})


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
