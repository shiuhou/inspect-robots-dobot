"""Read-only Astra task viewer with an existing local camera preview.

Use --output for a standalone report, or --serve for a loopback-only file watcher.
This module never starts a policy, captures a camera, or dispatches an action.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import math
import os
import stat
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

SCHEMA = "astra_task_view_v1"
ROLES = ("front_rgb", "right_rgb", "wrist_rgb")
MAX_JSON = 2_000_000
MAX_RECORDS = 100
STALE_SECONDS = 10.0
PREVIEW_KEYS = {
    "front_rgb": "icspring-31",
    "right_rgb": "icspring-32",
    "wrist_rgb": "usb-main",
}
STAGES = {
    "observing": "取得觀測",
    "thinking": "Astra 判斷中",
    "proposed": "已提出動作",
    "waiting_approval": "等待操作員確認",
    "executing": "執行中",
    "settling": "等待到位",
    "completed": "流程已完成",
    "stopped": "流程已停止",
    "aborted": "已取消",
    "failed": "執行失敗",
    "rejected": "動作被拒絕",
}
CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'"
)


def preview_csp(port: int | None) -> str:
    if port is None:
        return CSP
    return CSP.replace("img-src 'self' data:", f"img-src 'self' data: http://127.0.0.1:{port}")


class ViewerError(ValueError):
    """Invalid, incomplete, or unsupported file input."""


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any, limit: int = 2000) -> str | None:
    return value[:limit] if isinstance(value, str) and value else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _xyz(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    result = [_number(v) for v in value[:3]]
    return [float(v) for v in result if v is not None] if None not in result else None


def _read(path: Path, limit: int) -> bytes:
    # Do not follow symlinks or block on FIFOs/devices masquerading as JSON/JPEG.
    if path.is_symlink() or not path.is_file():
        raise ViewerError("input must be a regular, non-symlink file")
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ViewerError("device/FIFO inputs are forbidden")
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ViewerError("input exceeds viewer size limit")
    return raw


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(_read(path, MAX_JSON))
    if not isinstance(value, dict):
        raise ViewerError("expected a JSON object")
    return value


def _timestamp(value: Any) -> float:
    if not isinstance(value, str):
        raise ViewerError("updated_at must be an ISO timestamp with timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timezone missing")
        return parsed.timestamp()
    except (ValueError, OverflowError) as exc:
        raise ViewerError("invalid updated_at timestamp") from exc


def _tool_arguments(record: dict[str, Any]) -> dict[str, Any]:
    response = _object(record.get("raw_response"))
    arguments = _object(response.get("tool_call")).get("arguments")
    if isinstance(arguments, str):
        try:
            return _object(json.loads(arguments))
        except ValueError:
            return {}
    return _object(arguments)


def _shadow(record: dict[str, Any], mtime: float) -> dict[str, Any]:
    if not all(
        key in record for key in ("decision_id", "task_instruction", "proposed_high_level_action")
    ):
        raise ViewerError("unsupported record; expected Astra shadow or astra_task_view_v1")
    if record.get("execution_performed") is not False or record.get("execution") is not False:
        raise ViewerError("legacy shadow requires explicit non-execution evidence")
    state = _object(record.get("robot_state_visible"))
    arguments = _tool_arguments(record)
    response = _object(record.get("raw_response"))
    action = _text(record.get("proposed_high_level_action"))
    rejection = _text(record.get("rejection_reason"))
    note = arguments.get("note") or arguments.get("reason") or response.get("rationale_summary")
    proposed_target = _object(record.get("move_to_parameters")).get("target")
    return {
        "mode": "shadow",
        "run_id": _text(record.get("run_id")),
        "decision_id": _text(record.get("decision_id")),
        "task": _text(record.get("task_instruction")),
        "model": _text(record.get("model")),
        "stage": "rejected"
        if rejection
        else "stopped"
        if action in {"done", "give_up"}
        else "proposed",
        "note": _text(note),
        "action_name": action,
        "current_xyz_m": _xyz(state.get("eef_state")),
        "target_xyz_m": _xyz(proposed_target),
        "gripper_intent": _text(record.get("gripper_semantic_target"))
        or _text(record.get("gripper_proposal")),
        "pre_check": "REJECTED" if record.get("pre_check_result") else "NOT_RECORDED",
        "approver": _text(record.get("approver_result")),
        "rejection_reason": rejection,
        "execution_result": "未執行：shadow 決策紀錄",
        "motion_commands_sent": _count(record.get("motion_commands_sent")),
        "gripper_commands_sent": _count(record.get("gripper_commands_sent")),
        "task_success": None,
        "latency_s": _number(record.get("latency_s")),
        "robot_mode": _count(state.get("robot_mode")),
        "timestamp": mtime,
        "time_basis": "檔案修改時間（非量測時間）",
        "cameras": {},
    }


def _compact(record: dict[str, Any]) -> dict[str, Any]:
    mode, stage = record.get("mode"), record.get("stage")
    if (
        not isinstance(mode, str)
        or not isinstance(stage, str)
        or mode not in {"shadow", "live", "replay", "demo"}
        or stage not in STAGES
    ):
        raise ViewerError("invalid mode or stage")
    if not _text(record.get("task")):
        raise ViewerError("task must be a nonempty string")
    success = record.get("task_success")
    if success is not None and type(success) is not bool:
        raise ViewerError("task_success must be true, false or null")
    return {
        "mode": mode,
        "run_id": _text(record.get("run_id")),
        "decision_id": _text(record.get("decision_id")),
        "task": _text(record.get("task")),
        "model": _text(record.get("model")),
        "stage": stage,
        "note": _text(record.get("note")),
        "action_name": _text(record.get("action_name")),
        "current_xyz_m": _xyz(record.get("current_xyz_m")),
        "target_xyz_m": _xyz(record.get("target_xyz_m")),
        "gripper_intent": _text(record.get("gripper_intent")),
        "pre_check": _text(record.get("pre_check")),
        "approver": _text(record.get("approver")),
        "rejection_reason": _text(record.get("rejection_reason")),
        "execution_result": _text(record.get("execution_result")),
        "motion_commands_sent": _count(record.get("motion_commands_sent")),
        "gripper_commands_sent": _count(record.get("gripper_commands_sent")),
        "task_success": success,
        "latency_s": _number(record.get("latency_s")),
        "robot_mode": _count(record.get("robot_mode")),
        "timestamp": _timestamp(record.get("updated_at")),
        "time_basis": "資料產生端 updated_at（非感測器曝光時間）",
    }


class CameraPreview:
    """Read health from camera_live.py; never start it or open a camera device.

    Stream keys/roles are pinned to that script's DEFAULT_CAMERAS. Only the
    explicit loopback port is configurable, not a URL or device path.
    """

    def __init__(self, port: int = 8765) -> None:
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ViewerError("camera preview port must be in [1024,65535]")
        self.port = port

    def descriptor(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "streams": {
                role: f"http://127.0.0.1:{self.port}/stream/{key}.mjpg"
                for role, key in PREVIEW_KEYS.items()
            },
        }

    def health(self) -> dict[str, Any]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=0.75)
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            # Bound length/time; redirects are never followed.
            raw = response.read(65_537)
            if response.status != 200 or len(raw) > 65_536:
                raise ViewerError("preview health unavailable")
            cameras = _object(_object(json.loads(raw)).get("cameras"))
            result = {}
            for role, key in PREVIEW_KEYS.items():
                row = _object(cameras.get(key))
                age = _number(row.get("last_frame_age_seconds"))
                frames = _count(row.get("frames"))
                matching = row.get("label") == role
                result[role] = {
                    "ok": matching
                    and row.get("ok") is True
                    and age is not None
                    and 0 <= age < 2
                    and frames is not None
                    and frames > 0,
                    "mapping_verified": matching,
                    "frames": frames,
                    "age_s": age,
                    "fps": _number(row.get("measured_fps")),
                }
            return {"available": True, "cameras": result}
        except (OSError, ValueError, http.client.HTTPException):
            return {"available": False, "cameras": {}}
        finally:
            connection.close()


class RuntimeCameraPreview:
    """Read JPEG snapshots published by the task's existing camera capture threads."""

    port = None

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def descriptor(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "snapshots": True,
            "streams": {role: f"/frames/{role}.jpg" for role in ROLES},
        }

    def health(self) -> dict[str, Any]:
        try:
            data = _json(self.directory / "cameras.json")
            state = _json(self.directory / "state.json")
            if data.get("run_id") != state.get("run_id"):
                return {"available": False, "cameras": {}}
            result = {}
            for role in ROLES:
                row = _object(_object(data.get("cameras")).get(role))
                captured = _number(row.get("captured_at"))
                age = time.time() - captured if captured is not None else None
                result[role] = {
                    "ok": data.get("active") is True and age is not None and 0 <= age < 2,
                    "mapping_verified": True,
                    "age_s": age,
                    "fps": None,
                    "frame_id": row.get("frame_id"),
                }
            return {"available": data.get("active") is True, "cameras": result}
        except (OSError, ValueError):
            return {"available": False, "cameras": {}}

    def frame(self, role: str) -> bytes:
        if role not in ROLES or not self.health().get("cameras", {}).get(role, {}).get("ok"):
            raise ViewerError("runtime camera frame is unavailable or stale")
        return _read(self.directory / f"{role}.jpg", 8_000_000)


class TaskViewStore:
    """Read a single record or up to 100 newest JSON files in one directory."""

    def __init__(self, source: Path) -> None:
        if source.is_symlink():
            raise ViewerError("source symlinks are not supported")
        self.source = source.absolute()
        if not (self.source.is_file() or self.source.is_dir()):
            raise ViewerError("source must be an existing JSON file or evidence directory")

    def snapshot(self, selection: str | None = None, *, now: float | None = None) -> dict[str, Any]:
        clock = time.time() if now is None else now
        candidates = (
            [self.source]
            if self.source.is_file()
            else sorted(
                (p for p in self.source.glob("*.json") if p.is_file() and not p.is_symlink()),
                key=lambda p: (p.stat().st_mtime_ns, p.name),
                reverse=True,
            )[:MAX_RECORDS]
        )
        records: list[tuple[Path, dict[str, Any]]] = []
        warnings = []
        for path in candidates:
            try:
                raw = _json(path)
                value = (
                    _compact(raw)
                    if raw.get("schema_version") == SCHEMA
                    else _shadow(raw, path.stat().st_mtime)
                )
                value["file"] = path.name
                value["stage_label"] = STAGES[value["stage"]]
                value["age_s"] = max(0.0, clock - value["timestamp"])
                value["stale"] = value["age_s"] > STALE_SECONDS or value["timestamp"] > clock + 5
                value["updated_at"] = datetime.fromtimestamp(
                    value["timestamp"], timezone.utc
                ).isoformat()
                records.append((path, value))
            except (OSError, ValueError, UnicodeError) as exc:
                warnings.append(f"{path.name}: {exc}")
        selected = (
            next(((path, row) for path, row in records if row["file"] == selection), None)
            if selection
            else (records[0] if records else None)
        )
        if selection and selected is None:
            warnings.append("選取的紀錄不存在或目前無法解析")
        current = None
        if selected:
            _, current = selected
            start, target = current["current_xyz_m"], current["target_xyz_m"]
            current["delta_mm"] = (
                [(b - a) * 1000 for a, b in zip(start, target, strict=True)]
                if start and target
                else None
            )
        return {
            "schema_version": SCHEMA,
            "viewer_read_only": True,
            "current": current,
            "history": [
                {key: row[key] for key in ("file", "mode", "action_name", "updated_at")}
                for _, row in records
            ],
            "warnings": warnings,
            "record_limit": MAX_RECORDS,
        }


def render_html(
    snapshot: dict[str, Any],
    *,
    watching: bool = False,
    preview: CameraPreview | RuntimeCameraPreview | None = None,
) -> str:
    template = Path(__file__).with_name("task_viewer.html").read_text(encoding="utf-8")
    # JSON is inert data, never HTML; protect script boundaries even in model text.
    data = json.dumps(
        {
            "snapshot": snapshot,
            "watching": watching,
            "preview": preview.descriptor() if preview else {"enabled": False, "streams": {}},
        },
        ensure_ascii=True,
    )
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return template.replace("__CSP__", preview_csp(preview.port if preview else None)).replace(
        "__INITIAL_STATE__", data
    )


class ViewerApp:
    """Small GET-only application, independently testable without sockets."""

    def __init__(
        self, store: TaskViewStore, preview: CameraPreview | RuntimeCameraPreview | None = None
    ) -> None:
        self.store = store
        self.preview = preview

    def respond(self, method: str, target: str) -> tuple[int, str, bytes]:
        if method != "GET":
            return 405, "text/plain; charset=utf-8", b"Read-only viewer"
        url = urlsplit(target)
        if url.path == "/api/viewer":
            return (
                200,
                "application/json",
                json.dumps(
                    {
                        "source": str(self.store.source),
                        "runtime_frames": isinstance(self.preview, RuntimeCameraPreview),
                    }
                ).encode(),
            )
        if url.path.startswith("/frames/") and isinstance(self.preview, RuntimeCameraPreview):
            role = url.path.removeprefix("/frames/").removesuffix(".jpg")
            try:
                return 200, "image/jpeg", self.preview.frame(role)
            except (OSError, ValueError):
                return 503, "text/plain", b"Camera frame unavailable"
        if url.path == "/api/cameras":
            health = self.preview.health() if self.preview else {"available": False, "cameras": {}}
            return 200, "application/json; charset=utf-8", json.dumps(health).encode()
        if url.path not in {"/", "/api/state"}:
            return 404, "text/plain; charset=utf-8", b"Not found"
        selected = parse_qs(url.query).get("record", [None])[0]
        try:
            snapshot = self.store.snapshot(selected)
        except (OSError, ValueError):
            return 503, "text/plain; charset=utf-8", b"Evidence unavailable; retry next refresh"
        if url.path == "/api/state":
            return 200, "application/json; charset=utf-8", json.dumps(snapshot).encode()
        return (
            200,
            "text/html; charset=utf-8",
            render_html(snapshot, watching=True, preview=self.preview).encode(),
        )


def serve(
    store: TaskViewStore,
    port: int,
    preview: CameraPreview | RuntimeCameraPreview | None = None,
    *,
    ready_file: Path | None = None,
) -> None:
    app = ViewerApp(store, preview)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            status, content_type, body = app.respond("GET", self.path)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                preview_csp(preview.port if preview else None) + "; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            self.send_error(405, "Read-only viewer")

        do_PUT = do_POST
        do_DELETE = do_POST

        def log_message(self, format: str, *args: Any) -> None:
            return

    # Only a local listener and optional existing loopback preview health reader.
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        import errno

        if ready_file is None or exc.errno != errno.EADDRINUSE:
            raise
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    with server:
        actual_port = server.server_port
        if ready_file is not None:
            temporary = ready_file.with_suffix(".tmp")
            temporary.write_text(json.dumps({"port": actual_port, "pid": os.getpid()}))
            temporary.replace(ready_file)
        print(f"Astra 唯讀任務監看：http://127.0.0.1:{actual_port}", flush=True)
        print("只監看本地檔案；Ctrl-C 關閉面板，不會發送機械臂 Stop。", flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--runtime-frames", action="store_true", help="use task-published JPEGs")
    parser.add_argument("--ready-file", type=Path, help=argparse.SUPPRESS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="write a new self-contained offline HTML")
    mode.add_argument("--serve", action="store_true", help="watch files on 127.0.0.1 only")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--camera-preview-port", type=int, default=8765)
    parser.add_argument(
        "--no-camera-preview",
        action="store_true",
        help="disable existing camera_live.py preview (always disabled in --output)",
    )
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be in [1024,65535]")
    try:
        store = TaskViewStore(args.source)
        if args.output is not None:
            html = render_html(store.snapshot())
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(html)
            print(f"Offline viewer: {args.output}")
        else:
            preview: CameraPreview | RuntimeCameraPreview | None = (
                RuntimeCameraPreview(args.source.parent if args.source.is_file() else args.source)
                if args.runtime_frames
                else None
                if args.no_camera_preview
                else CameraPreview(args.camera_preview_port)
            )
            if preview is not None and preview.port == args.port:
                raise ViewerError("viewer and existing camera preview need different ports")
            serve(store, args.port, preview, ready_file=args.ready_file)
        return 0
    except (OSError, ValueError) as exc:
        print(f"Viewer: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
