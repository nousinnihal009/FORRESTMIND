"""Pinned public forest interpretation evidence; no inferred review or gain dates.

This is a separate 30 m point-reference dataset, not the 100 ha annual-area
protocol. Files are kept immutable and parsed locally without extracting ZIPs.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import urllib.error
import urllib.parse
import urllib.request
import zipfile


RECORD_ID = 6403863
DATASET_ID = "guo-global-forest-reference-v1.1"
METADATA_URL = "https://zenodo.org/api/records/6403863"
ARCHIVE_KEY = "global forest reference set version1.1.zip"
ARCHIVE_URL = METADATA_URL + "/files/" + urllib.parse.quote(ARCHIVE_KEY) + "/content"
ARCHIVE_MD5 = "e56c8251ca391693674bf00a9a1bfb77"
ARCHIVE_SIZE = 229591
YEARS = tuple(range(2000, 2021))
LAYERS = ("persisting_forest", "persisting_nonforest", "forest_change")
PREFIX = "global forest reference set version1.1/"
MAX_METADATA_BYTES = 2 * 1024 * 1024
LIMITATIONS = [
    "Published visual interpretation of Landsat, high-resolution imagery and other land-cover products; not field observations or an independent ForestMind validation campaign.",
    "Annual point-scale forest changes at a declared 30 m reference scale are not 100 ha annual loss fractions, general health, causal drivers or tipping points.",
    "Interpretations were released retrospectively in 2022. Event histories are not proven to have been operationally available at earlier forecast origins.",
    "Source sampling overrepresents forest/change strata and excludes difficult points. Initial stratum codes are retained; exact design/nonresponse weights are not supplied or invented.",
    "Undated gains and ambiguous change descriptions remain masked. Selecting only fully dated or simple histories changes the evaluation population.",
    "The source provides no per-point numerical uncertainty, image identifiers or per-point interpreter review dates. Missing evidence is not synthesized.",
]


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _url(url):
    value = urllib.parse.urlsplit(url)
    if (value.scheme != "https" or value.hostname != "zenodo.org"
            or value.port not in (None, 443) or value.username or value.password or value.fragment):
        raise ValueError("Public reference requests must stay on HTTPS zenodo.org")
    return url


class _ZenodoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return super().redirect_request(request, fp, code, msg, headers, _url(newurl))


def _download(url, maximum):
    request = urllib.request.Request(_url(url), headers={"User-Agent": "ForestMind-public-reference/1"})
    opener = urllib.request.build_opener(_ZenodoRedirect())
    with opener.open(request, timeout=40) as response:
        _url(response.geturl())
        if response.status != 200:
            raise ValueError("Public reference download requires HTTP 200")
        declared = response.headers.get("Content-Length")
        if declared and (not declared.isdigit() or int(declared) > maximum):
            raise ValueError("Public reference exceeds the download bound")
        chunks, size = [], 0
        while True:
            chunk = response.read(min(65536, maximum + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > maximum:
                raise ValueError("Public reference exceeds the download bound")
            chunks.append(chunk)
        if declared and size != int(declared):
            raise ValueError("Incomplete public reference download")
        return b"".join(chunks)


def _validate_metadata(record):
    if not isinstance(record, dict) or record.get("id") != RECORD_ID:
        raise ValueError("Unexpected public reference record")
    metadata = record.get("metadata", {})
    if (metadata.get("version") != "1.1 (from 2000 to 2020)"
            or metadata.get("license", {}).get("id") != "cc-by-4.0"
            or metadata.get("publication_date") != "2022-03-29"
            or metadata.get("access_right") != "open"):
        raise ValueError("Public reference version, publication or license differs from the pinned contract")
    files = record.get("files")
    if not isinstance(files, list) or len(files) != 1:
        raise ValueError("Unexpected public reference file inventory")
    item = files[0]
    if (item.get("key") != ARCHIVE_KEY or item.get("size") != ARCHIVE_SIZE
            or item.get("checksum") != "md5:" + ARCHIVE_MD5
            or item.get("links", {}).get("self") != ARCHIVE_URL):
        raise ValueError("Public reference archive differs from the pinned source")


def _validate_archive(data):
    if len(data) != ARCHIVE_SIZE or hashlib.md5(data).hexdigest() != ARCHIVE_MD5:
        raise ValueError("Public reference archive size or provider checksum mismatch")


def _read_file(root, name, limit):
    path = root / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"Invalid or oversized evidence file: {name}")
    data = path.read_bytes()
    if len(data) > limit:
        raise ValueError(f"Oversized evidence file: {name}")
    return data


def _new_file(root, name, data):
    path = root / name
    if path.exists():
        if _read_file(root, name, len(data)) != data:
            raise ValueError(f"Refusing to replace existing evidence: {name}")
        return
    with path.open("xb") as stream:
        stream.write(data)


def fetch_public_references(output_directory):
    """Fetch the pinned small archive, or verify/resume its immutable packet."""
    root = Path(output_directory)
    if root.is_symlink():
        raise ValueError("Output directory cannot be a symbolic link")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        return load_public_reference(root)
    metadata_data = (_read_file(root, "metadata.json", MAX_METADATA_BYTES)
                     if (root / "metadata.json").exists() else _download(METADATA_URL, MAX_METADATA_BYTES))
    record = _json(metadata_data)
    _validate_metadata(record)
    _new_file(root, "metadata.json", metadata_data)
    archive = (_read_file(root, "reference.zip", ARCHIVE_SIZE)
               if (root / "reference.zip").exists() else _download(ARCHIVE_URL, ARCHIVE_SIZE))
    _validate_archive(archive)
    decode_reference_archive(archive)
    _new_file(root, "reference.zip", archive)
    manifest = {
        "schema_version": 1, "dataset_id": DATASET_ID, "source_record_id": RECORD_ID,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "metadata_url": METADATA_URL, "archive_url": ARCHIVE_URL,
        "files": {"metadata.json": {"sha256": _sha(metadata_data), "bytes": len(metadata_data)},
                  "reference.zip": {"sha256": _sha(archive), "bytes": len(archive), "provider_md5": ARCHIVE_MD5}},
        "download_tool_sha256": _sha(Path(__file__).read_bytes()),
    }
    _new_file(root, "manifest.json", (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode())
    return load_public_reference(root)


def _dbf(data):
    if len(data) < 33 or data[0] != 3:
        raise ValueError("Unsupported reference DBF")
    count = struct.unpack_from("<I", data, 4)[0]
    header, length = struct.unpack_from("<HH", data, 8)
    if not 0 < count <= 12000 or header < 33 or (header - 33) % 32 or header > len(data):
        raise ValueError("Invalid reference DBF dimensions")
    if len(data) not in (header + count * length, header + count * length + 1) or data[header - 1] != 13:
        raise ValueError("Truncated or inconsistent reference DBF")
    fields = []
    for offset in range(32, header - 1, 32):
        field = data[offset:offset + 32]
        name = field[:11].split(b"\0", 1)[0].decode("ascii")
        typ, width, decimal = chr(field[11]), field[16], field[17]
        if not name or name in [f[0] for f in fields] or typ not in ("N", "C") or not width or decimal:
            raise ValueError("Unsupported reference DBF field")
        fields.append((name, typ, width))
    required = {"per_f", "per_nonf", "change", "loss", "gain", "times", "initial_s"}
    if set(f[0] for f in fields) not in (required, required | {"changeinfo"}) or 1 + sum(f[2] for f in fields) != length:
        raise ValueError("Unexpected reference DBF schema")
    rows = []
    for index in range(count):
        row_data = data[header + index * length:header + (index + 1) * length]
        if row_data[0] != 32:
            raise ValueError("Deleted or malformed reference record")
        row, start = {}, 1
        for name, typ, width in fields:
            value = row_data[start:start + width].decode("ascii").strip()
            if typ == "N":
                if not re.fullmatch(r"\d+", value):
                    raise ValueError("Invalid numeric reference attribute")
                value = int(value)
            row[name] = value
            start += width
        rows.append(row)
    return rows


def _points(data):
    if (len(data) < 100 or struct.unpack_from(">I", data)[0] != 9994
            or struct.unpack_from(">I", data, 24)[0] * 2 != len(data)
            or struct.unpack_from("<II", data, 28) != (1000, 1)):
        raise ValueError("Expected a complete point shapefile")
    points, offset = [], 100
    while offset < len(data):
        if offset + 28 > len(data):
            raise ValueError("Truncated reference geometry")
        number, words = struct.unpack_from(">II", data, offset)
        shape, lon, lat = struct.unpack_from("<Idd", data, offset + 8)
        if (number != len(points) + 1 or words != 10 or shape != 1
                or not math.isfinite(lon) or not math.isfinite(lat)
                or not -180 <= lon <= 180 or not -90 <= lat <= 90):
            raise ValueError("Invalid reference point geometry")
        points.append((lon, lat))
        offset += 28
    return points


def decode_annual_history(attributes):
    """Return only unambiguous annual information; unknown transitions stay null."""
    attrs = attributes
    flags = [attrs.get(name) for name in ("per_f", "per_nonf", "change")]
    counts = [attrs.get(name) for name in ("loss", "gain", "times")]
    if (any(type(v) is not int or v not in (0, 1) for v in flags) or sum(flags) != 1
            or any(type(v) is not int or not 0 <= v <= 50 for v in counts)
            or counts[0] + counts[1] != counts[2]
            or attrs.get("initial_s") not in ("perf", "nonf", "change")):
        raise ValueError("Inconsistent reference flags, counts or source stratum")
    empty = {str(year): None for year in YEARS}
    prose = attrs.get("changeinfo", "")
    if not isinstance(prose, str) or len(prose) > 100:
        raise ValueError("Invalid reference change description")
    if not flags[2]:
        if any(counts) or prose.strip():
            raise ValueError("Persisting reference contradicts change evidence")
        return {str(y): bool(flags[0]) for y in YEARS}, {str(y): 0 for y in YEARS}, "persisting_source_class"
    if counts[2] == 0:
        raise ValueError("Changed reference requires at least one transition")
    tokens = re.split(r"\s+(?:then|than)\s+", prose.lower().strip())
    events = []
    for token in tokens:
        match = re.fullmatch(r"(loss|gain)(?: in (\d{4}))?", token.strip())
        if not match:
            return empty.copy(), empty.copy(), "unrecognized_change_description"
        year = int(match[2]) if match[2] else None
        if year is not None and year not in YEARS:
            return empty.copy(), empty.copy(), "change_year_outside_reference_period"
        events.append((match[1], year))
    if (len(events) != counts[2] or sum(kind == "loss" for kind, _ in events) != counts[0]
            or sum(kind == "gain" for kind, _ in events) != counts[1]):
        return empty.copy(), empty.copy(), "change_description_count_mismatch"
    dated = [year for _, year in events if year is not None]
    if dated != sorted(dated) or any(events[i][0] == events[i - 1][0] for i in range(1, len(events))):
        return empty.copy(), empty.copy(), "inconsistent_transition_order"
    losses = [year for kind, year in events if kind == "loss"]
    if any(year is None for year in losses) or len(losses) != len(set(losses)):
        annual_loss = empty.copy()
    else:
        annual_loss = {str(y): int(y in losses) for y in YEARS}
    if any(year is None for _, year in events) or len(dated) != len(set(dated)):
        return empty.copy(), annual_loss, "undated_or_same_year_transition_state_withheld"
    # A dated loss starts from forest; a dated gain starts from nonforest.
    # This reconstructs the interpreted annual state, not measured canopy area.
    state = events[0][0] == "loss"
    annual_state = {}
    for year in YEARS:
        for kind, event_year in events:
            if event_year == year:
                state = kind == "gain"
        annual_state[str(year)] = state
    return annual_state, annual_loss, "fully_dated_transition_sequence"


def decode_reference_archive(data):
    """Decode the source ZIP in memory. Does not write or trust executable files."""
    if len(data) > 1024 * 1024:
        raise ValueError("Reference archive exceeds bound")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        inventory = archive.infolist()
        if len(inventory) > 40 or sum(item.file_size for item in inventory) > 4 * 1024 * 1024:
            raise ValueError("Reference archive expansion exceeds bound")
        names = set()
        for item in inventory:
            path = PurePosixPath(item.filename)
            if (item.filename in names or path.is_absolute() or ".." in path.parts or "\\" in item.filename
                    or item.file_size > 2 * 1024 * 1024 or item.flag_bits & 1
                    or ((item.external_attr >> 16) & 0o170000) == 0o120000):
                raise ValueError("Unsafe or duplicate reference archive entry")
            names.add(item.filename)
        samples = []
        for layer in LAYERS:
            rows = _dbf(archive.read(PREFIX + layer + ".dbf"))
            points = _points(archive.read(PREFIX + layer + ".shp"))
            prj = archive.read(PREFIX + layer + ".prj").decode("ascii")
            if "WGS" not in prj or "6378137" not in prj or "0.0174532925199433" not in prj:
                raise ValueError("Unexpected reference coordinate system")
            if len(rows) != len(points):
                raise ValueError("Reference attributes and geometry counts differ")
            for index, (attrs, (lon, lat)) in enumerate(zip(rows, points), 1):
                expected = (1, 0, 0) if layer == "persisting_forest" else ((0, 1, 0) if layer == "persisting_nonforest" else (0, 0, 1))
                if tuple(attrs.get(name) for name in ("per_f", "per_nonf", "change")) != expected:
                    raise ValueError("Reference flags contradict their source layer")
                state, loss, reason = decode_annual_history(attrs)
                samples.append({"id": f"{layer}:{index:06d}", "lon": lon, "lat": lat,
                    "initial_stratum": attrs["initial_s"], "source_layer": layer,
                    "annual_state": state, "annual_loss": loss, "attributes": attrs,
                    "decoding_reason": reason})
        if len(samples) > 12000:
            raise ValueError("Too many public reference samples")
    except (zipfile.BadZipFile, KeyError, UnicodeError, struct.error) as exc:
        raise ValueError("Invalid or incomplete public reference archive") from exc
    audit = {
        "sample_count": len(samples), "source_layers": dict(Counter(s["source_layer"] for s in samples)),
        "decoding_reasons": dict(Counter(s["decoding_reason"] for s in samples)),
        "eligible_loss_histories": sum(all(v is not None for v in s["annual_loss"].values()) for s in samples),
        "ambiguous_loss_histories": sum(any(v is None for v in s["annual_loss"].values()) for s in samples),
        "complete_state_histories": sum(all(v is not None for v in s["annual_state"].values()) for s in samples),
        "scientifically_validated_forestmind": False,
    }
    return samples, audit


def load_public_reference(packet_dir):
    """Recheck evidence integrity and decode a previously fetched public packet."""
    root = Path(packet_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Public reference packet must be a real directory")
    manifest_data = _read_file(root, "manifest.json", 65536)
    manifest = _json(manifest_data)
    if (manifest.get("schema_version") != 1 or manifest.get("dataset_id") != DATASET_ID
            or manifest.get("source_record_id") != RECORD_ID
            or manifest.get("metadata_url") != METADATA_URL or manifest.get("archive_url") != ARCHIVE_URL
            or set(manifest.get("files", {})) != {"metadata.json", "reference.zip"}):
        raise ValueError("Unexpected public reference manifest")
    content = {}
    for name, limit in (("metadata.json", MAX_METADATA_BYTES), ("reference.zip", ARCHIVE_SIZE)):
        content[name] = _read_file(root, name, limit)
        expected = manifest["files"][name]
        if expected.get("sha256") != _sha(content[name]) or expected.get("bytes") != len(content[name]):
            raise ValueError("Public reference file hash or size mismatch")
    record = _json(content["metadata.json"])
    _validate_metadata(record)
    _validate_archive(content["reference.zip"])
    if manifest["files"]["reference.zip"].get("provider_md5") != ARCHIVE_MD5:
        raise ValueError("Public reference provider digest mismatch")
    samples, audit = decode_reference_archive(content["reference.zip"])
    metadata = {"dataset_id": DATASET_ID, "source_record_id": RECORD_ID,
        "published_at": "2022-03-29", "years": list(YEARS), "declared_resolution_m": 30,
        "retrospective": True, "license": "CC-BY-4.0", "original_record": record,
        "provenance": {"manifest_sha256": _sha(manifest_data), "files": manifest["files"],
                       "metadata_url": METADATA_URL, "archive_url": ARCHIVE_URL,
                       "decoder_sha256": _sha(Path(__file__).read_bytes())},
        "limitations": list(LIMITATIONS)}
    return {"metadata": metadata, "samples": samples, "audit": audit}
