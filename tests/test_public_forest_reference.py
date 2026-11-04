"""Source contracts and conservative annual decoding; fabricated fixtures only."""
import hashlib
import io
import json
import struct
import zipfile

import pytest

from forestmind.data import public_forest_reference as pr


def attrs(prose="loss in 2008", loss=1, gain=0):
    return {"per_f": 0, "per_nonf": 0, "change": 1, "loss": loss,
            "gain": gain, "times": loss + gain, "initial_s": "change", "changeinfo": prose}


def test_dated_loss_reconstruction_and_persisting_reference():
    state, loss, reason = pr.decode_annual_history(attrs())
    assert state["2007"] is True and state["2008"] is False
    assert sum(loss.values()) == 1 and loss["2008"] == 1
    assert reason == "fully_dated_transition_sequence"
    stable = {"per_f": 1, "per_nonf": 0, "change": 0, "loss": 0, "gain": 0, "times": 0, "initial_s": "perf"}
    state, loss, _ = pr.decode_annual_history(stable)
    assert all(state.values()) and set(loss.values()) == {0}


def test_undated_gain_never_becomes_a_known_state():
    state, loss, reason = pr.decode_annual_history(attrs("loss in 2008 than gain", gain=1))
    assert all(v is None for v in state.values())
    assert loss["2008"] == 1 and sum(loss.values()) == 1
    assert reason == "undated_or_same_year_transition_state_withheld"


@pytest.mark.parametrize("prose,loss,gain", [
    ("loss", 1, 0), ("loss in 2008 then loss in 2009", 2, 0),
    ("loss in 2008 then gain in 2007", 1, 1), ("loss in 2099", 1, 0),
    ("unexpected text", 1, 0), ("loss in 2008", 2, 0),
])
def test_ambiguous_or_inconsistent_histories_stay_unknown(prose, loss, gain):
    state, events, _ = pr.decode_annual_history(attrs(prose, loss, gain))
    assert all(v is None for v in state.values())
    assert all(v is None for v in events.values())


def test_same_year_loss_and_gain_has_annual_event_but_no_final_state():
    state, events, _ = pr.decode_annual_history(attrs("loss in 2008 then gain in 2008", gain=1))
    assert events["2008"] == 1
    assert set(state.values()) == {None}


@pytest.mark.parametrize("change", [{"loss": True}, {"times": 4}, {"initial_s": "unknown"}, {"per_f": 1}])
def test_invalid_source_counts_and_flags_are_rejected(change):
    value = attrs()
    value.update(change)
    with pytest.raises(ValueError):
        pr.decode_annual_history(value)


def make_archive():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, layer in enumerate(pr.LAYERS):
            row = {"per_f": int(index == 0), "per_nonf": int(index == 1), "change": int(index == 2),
                   "loss": int(index == 2), "gain": 0, "times": int(index == 2),
                   "initial_s": ["perf", "nonf", "change"][index]}
            if index == 2:
                row["changeinfo"] = "loss in 2008"
            fields = [(k, "N" if isinstance(v, int) else "C", 3 if isinstance(v, int) else 32) for k, v in row.items()]
            header_len, record_len = 33 + len(fields) * 32, 1 + sum(w for _, _, w in fields)
            header = bytearray(header_len)
            header[0] = 3
            struct.pack_into("<IHH", header, 4, 1, header_len, record_len)
            for i, (key, kind, width) in enumerate(fields):
                start = 32 + i * 32
                header[start:start + len(key)] = key.encode()
                header[start + 11] = ord(kind)
                header[start + 16] = width
            header[-1] = 13
            record = b" " + b"".join(str(row[k]).encode().ljust(w) for k, _, w in fields)
            archive.writestr(pr.PREFIX + layer + ".dbf", bytes(header) + record)
            shape = bytearray(128)
            struct.pack_into(">I", shape, 0, 9994)
            struct.pack_into(">I", shape, 24, 64)
            struct.pack_into("<II", shape, 28, 1000, 1)
            struct.pack_into(">II", shape, 100, 1, 10)
            struct.pack_into("<Idd", shape, 108, 1, 12. + index, 22.)
            archive.writestr(pr.PREFIX + layer + ".shp", bytes(shape))
            archive.writestr(pr.PREFIX + layer + ".prj", 'GEOGCS["WGS 84",6378137,0.0174532925199433]')
    return buffer.getvalue()


def test_archive_decode_matches_layers_and_geometry():
    samples, audit = pr.decode_reference_archive(make_archive())
    assert audit["sample_count"] == 3
    assert samples[0]["lon"] == 12 and samples[2]["annual_loss"]["2008"] == 1
    assert not audit["scientifically_validated_forestmind"]


def test_archive_rejects_path_traversal_and_oversize():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../outside", "data")
    with pytest.raises(ValueError, match="Unsafe"):
        pr.decode_reference_archive(buffer.getvalue())
    with pytest.raises(ValueError, match="exceeds bound"):
        pr.decode_reference_archive(b"x" * (1024 * 1024 + 1))


def test_fetch_resume_checks_pinned_metadata_and_rejects_source_mutation(tmp_path, monkeypatch):
    raw = make_archive()
    monkeypatch.setattr(pr, "ARCHIVE_SIZE", len(raw))
    monkeypatch.setattr(pr, "ARCHIVE_MD5", hashlib.md5(raw).hexdigest())
    metadata = {"id": pr.RECORD_ID, "metadata": {"version": "1.1 (from 2000 to 2020)",
                "license": {"id": "cc-by-4.0"}, "publication_date": "2022-03-29", "access_right": "open"},
                "files": [{"key": pr.ARCHIVE_KEY, "size": len(raw), "checksum": "md5:" + pr.ARCHIVE_MD5,
                           "links": {"self": pr.ARCHIVE_URL}}]}
    calls = []
    def download(url, maximum):
        calls.append(url)
        return json.dumps(metadata).encode() if url == pr.METADATA_URL else raw
    monkeypatch.setattr(pr, "_download", download)
    first = pr.fetch_public_references(tmp_path)
    second = pr.fetch_public_references(tmp_path)
    assert first == second and len(calls) == 2
    assert first["metadata"]["published_at"] == "2022-03-29"
    path = tmp_path / "reference.zip"
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="hash or size"):
        pr.load_public_reference(tmp_path)


@pytest.mark.parametrize("url", ["http://zenodo.org/x", "https://example.com/x", "https://user:pass@zenodo.org/x", "https://zenodo.org:8443/x"])
def test_network_boundary_rejects_unapproved_destinations(url):
    with pytest.raises(ValueError):
        pr._url(url)
