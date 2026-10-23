"""Verify native architecture/state execution offline using RANDOM, UNTRAINED weights.

This diagnostic never downloads pretrained weights, trains a model, or registers
an operational predictor. A successful report verifies runtime plumbing only.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
from importlib import metadata
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from forestmind.inference.predict import predict_checkpoint, report_destination, sha256_file
from forestmind.models.foundation.native import capture_native_runtime, offline_native_config
from forestmind.models.forestmind_model import ForestMindModel


@contextmanager
def deny_network():
    attempts = []

    def refused(*args, **kwargs):
        attempts.append("outbound socket connection requested")
        raise RuntimeError("Network access is disabled during native runtime verification")

    # This runs only in the dedicated diagnostic process, never in the service.
    with patch.object(socket.socket, "connect", refused), patch.object(socket.socket, "connect_ex", refused), patch.object(socket, "create_connection", refused):
        yield attempts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation", choices=("terramind", "prithvi"), required=True)
    parser.add_argument("--model", help="Audited native registry ID; defaults to TerraMind tiny or Prithvi v1 100M")
    parser.add_argument("--input", type=Path, required=True, help="Existing raw NPZ sequence; never altered")
    parser.add_argument("--observed-through", required=True)
    parser.add_argument("--synthetic-input", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = report_destination(args.output, protected_paths=(args.input,))
    if destination.exists():
        parser.error("Output already exists; choose a new evidence filename")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    runtime = capture_native_runtime()
    name = args.foundation
    model_name = args.model or ("terramind_v1_tiny" if name == "terramind" else "prithvi_eo_v1_100")
    options = {"model_name": model_name, "freeze": True, "use_proxy": False, "pretrained": False}
    if name == "terramind":
        options["sar_modality"] = "S1RTC"
    config = {
        "data": {"source": "random-initialization-diagnostic-no-training", "optical_units": "surface_reflectance_0_1",
                 "sar_units": "gamma0_db", "normalization": {"optical_mean": [0.] * 12, "optical_std": [1.] * 12,
                                                             "sar_mean": [0.] * 2, "sar_std": [1.] * 2}},
        "foundation": {"name": name, name: options, "embedding_dim": 16, "native_runtime": runtime},
        "fusion": {"num_heads": 2, "num_layers": 1},
        "temporal": {"depth": 1, "num_heads": 2, "dim_feedforward": 32, "max_seq_len": 48, "dropout": 0.},
        "heads": {"trajectory": {"horizons": [30], "predict_uncertainty": True}},
    }
    torch.set_num_threads(2)
    torch.manual_seed(17)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="native-runtime-check-", dir=destination.parent) as temporary:
        os.environ["HF_HOME"] = str(Path(temporary) / "empty-huggingface-cache")
        with deny_network() as network_attempts:
            model = ForestMindModel(offline_native_config(config))
            parameter_count = sum(p.numel() for p in model.parameters())
            state = Path(temporary) / "random-untrained-state.pt"
            torch.save({"config": config, "model_state_dict": model.state_dict()}, state)
            checkpoint_hash, checkpoint_bytes = sha256_file(state), state.stat().st_size
            del model
            gc.collect()
            first = predict_checkpoint(state, args.input, region_id="native-runtime-diagnostic",
                                       observed_through=args.observed_through, synthetic_input=args.synthetic_input)
            second = predict_checkpoint(state, args.input, region_id="native-runtime-diagnostic",
                                        observed_through=args.observed_through, synthetic_input=args.synthetic_input)
        equal = first["predictions"] == second["predictions"]
        if not equal or network_attempts:
            raise RuntimeError("Native runtime verification did not produce repeatable offline results")
    installed = {dist.metadata["Name"]: dist.version for dist in metadata.distributions() if dist.metadata.get("Name")}
    result = {
        "kind": "forestmind_native_runtime_verification", "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(), "foundation": name, "model_name": model_name,
        "native_runtime": runtime, "installed_packages": dict(sorted(installed.items())),
        "network_denied": True, "network_attempts": len(network_attempts), "empty_weight_cache": True,
        "weights": "Random initialization, not pretrained, not trained; temporary state removed after verification",
        "normalization": "Explicit identity transform used only for runtime verification, not a pretrained normalization claim",
        "trained_model": False, "predictive_validity": False, "registered_operational_model": False,
        "parameter_count": parameter_count, "temporary_checkpoint_bytes": checkpoint_bytes,
        "temporary_checkpoint_sha256": checkpoint_hash, "input": first["input"],
        "strict_state_loaded": True, "repeated_predictions_equal": equal,
        "finite_output_dimensions": {"health_classes": len(first["predictions"]["health"]["class_scores"]),
                                     "driver_scores": len(first["predictions"]["drivers"]),
                                     "forecast_horizons": len(first["predictions"]["trajectory"])},
        "seconds": round(time.perf_counter() - started, 3),
        "limitations": ["Random weights verify execution only. No ecological interpretation is supported.",
                        "Only this exact model/runtime/input combination was exercised.",
                        "A licensed pretrained artifact, trained downstream heads and independent validation are still required."],
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Verified {model_name}: complete local state, repeatable outputs, zero network requests. RANDOM WEIGHTS ONLY.")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
