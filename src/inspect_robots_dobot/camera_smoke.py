"""Camera-only physical smoke test. No robot, serial, model or motion imports."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .camera import Frame, LatestFrameReader
from .camera_v4l2 import V4L2MjpegFrameSource
from .config import (
    CAMERA_NAMES,
    PhysicalCameraConfig,
    load_config,
    validate_camera_hardware_mapping,
)
from .errors import CameraFault, ConfigurationError


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def acquire_after(
    readers: dict[str, LatestFrameReader],
    before: dict[str, tuple[int, int]],
    barrier: float,
) -> dict[str, Frame]:
    frames = {name: reader.latest(after=barrier) for name, reader in readers.items()}
    for name, frame in frames.items():
        if (frame.generation, frame.sequence) <= before[name] or frame.timestamp <= barrier:
            raise CameraFault(f"{name} did not advance after host barrier")
    return frames


def run_camera_smoke(config_path: Path, evidence_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    if {camera.name for camera in config.cameras} != CAMERA_NAMES:
        raise CameraFault("smoke test requires exactly three named cameras")
    validate_camera_hardware_mapping(config.cameras)
    validation = evidence_dir / "validation.json"
    if not validation.is_file():
        raise CameraFault("offline validation.json must be saved before camera smoke")
    specs: tuple[PhysicalCameraConfig, ...] = config.cameras
    evidence_dir.mkdir(parents=True, exist_ok=True)
    samples = evidence_dir / "samples"
    samples.mkdir(exist_ok=True)
    inventory = {spec.name: asdict(spec) for spec in specs}
    _write_json(evidence_dir / "camera_inventory.json", inventory)
    sources = {spec.name: V4L2MjpegFrameSource(spec) for spec in specs}
    readers = {
        spec.name: LatestFrameReader(
            sources[spec.name], spec.frame_config, startup_timeout=spec.startup_timeout
        )
        for spec in specs
    }
    report: dict[str, Any] = {"status": "FAILED", "physical_robot_access": False}
    try:
        for reader in readers.values():
            reader.start()
        first = {name: reader.latest() for name, reader in readers.items()}
        start_counts = {name: source.raw_frame_count for name, source in sources.items()}
        started = time.monotonic()
        end = started + 30.0
        while any(source.raw_frame_count < 30 for source in sources.values()):
            if time.monotonic() > end:
                raise CameraFault("30-frame deadline exceeded")
            time.sleep(0.02)
        duration = time.monotonic() - started
        last = {name: reader.latest() for name, reader in readers.items()}
        camera_results = {}
        for spec in specs:
            name = spec.name
            frame = last[name]
            if frame.rgb.shape != (spec.height, spec.width, 3) or frame.rgb.dtype != np.uint8:
                raise CameraFault(f"{name} has wrong HWC RGB format")
            jpeg = sources[name].latest_jpeg
            if jpeg is None:
                raise CameraFault(f"{name} has no JPEG sample")
            (samples / f"{name}.jpg").write_bytes(jpeg)
            camera_results[name] = {
                "device_path": spec.device_path,
                "shape": list(frame.rgb.shape),
                "dtype": str(frame.rgb.dtype),
                "configured_fps": spec.fps,
                "raw_frame_count": sources[name].raw_frame_count,
                "rgb_frames_decoded": frame.sequence - first[name].sequence + 1,
                "observed_raw_fps": (sources[name].raw_frame_count - start_counts[name]) / duration,
                "generation": frame.generation,
                "latest_host_receive_time_monotonic": frame.timestamp,
                "latest_age_s": time.monotonic() - frame.timestamp,
            }
        report = {
            "status": "PASS",
            "physical_robot_access": False,
            "host_timestamp_semantics": "complete JPEG assembled by host; not sensor exposure",
            "cameras": camera_results,
        }
        _write_json(evidence_dir / "physical_smoke.json", report)
        latencies: list[float] = []
        skews: list[float] = []
        observations = []
        previous = {name: (frame.generation, frame.sequence) for name, frame in last.items()}
        for _ in range(20):
            barrier = time.monotonic()
            fresh = acquire_after(readers, previous, barrier)
            finished = time.monotonic()
            previous = {name: (frame.generation, frame.sequence) for name, frame in fresh.items()}
            times = [frame.timestamp for frame in fresh.values()]
            latencies.append(finished - barrier)
            skews.append(max(times) - min(times))
            observations.append(
                {
                    "barrier": barrier,
                    "latency_s": latencies[-1],
                    "host_receive_skew_s": skews[-1],
                    "frames": {
                        name: {
                            "generation": frame.generation,
                            "sequence": frame.sequence,
                            "host_receive_time_monotonic": frame.timestamp,
                            "age_s": finished - frame.timestamp,
                            "shape": list(frame.rgb.shape),
                            "fresh": True,
                        }
                        for name, frame in fresh.items()
                    },
                }
            )
        _write_json(
            evidence_dir / "freshness_test.json",
            {
                "status": "PASS",
                "checks": len(observations),
                "latency_s_min": min(latencies),
                "latency_s_mean": statistics.mean(latencies),
                "latency_s_max": max(latencies),
                "host_receive_skew_s_max": max(skews),
                "observations": observations,
                "sensor_exposure_freshness_verified": False,
            },
        )
        return report
    except Exception as exc:
        report = {"status": "FAILED", "physical_robot_access": False, "error": str(exc)}
        _write_json(evidence_dir / "physical_smoke.json", report)
        _write_json(evidence_dir / "freshness_test.json", {"status": "NOT_COMPLETED"})
        raise
    finally:
        close_errors = []
        for reader in readers.values():
            try:
                reader.close()
            except Exception as exc:
                close_errors.append(str(exc))
        if close_errors:
            report["status"] = "FAILED"
            report["close_errors"] = close_errors
            _write_json(evidence_dir / "physical_smoke.json", report)
            raise CameraFault(f"camera close failed: {close_errors}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Open only configured V4L2 RGB cameras")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_camera_smoke(args.config, args.evidence_dir)
    except (CameraFault, ConfigurationError, OSError, ValueError) as exc:
        parser.exit(1, f"camera smoke failed: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0
