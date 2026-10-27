"""Task display and shared preview contract; no real sockets/devices/providers."""

from __future__ import annotations

import hashlib
import json
import os

import pytest

from inspect_robots_dobot import task_viewer as viewer

NOW = 1_790_467_200.0


def record(**changes):
    value = {
        "schema_version": viewer.SCHEMA,
        "mode": "demo",
        "task": "把藍色方塊放到盒子中",
        "stage": "proposed",
        "updated_at": "2026-09-27T00:00:00+00:00",
        "current_xyz_m": [0.3, 0.0, 0.2],
        "target_xyz_m": [0.305, 0.0, 0.2],
        "motion_commands_sent": 0,
        "task_success": None,
    }
    value.update(changes)
    return value


def save(tmp_path, value=None, name="state.json"):
    path = tmp_path / name
    path.write_text(json.dumps(record() if value is None else value), encoding="utf-8")
    return path


def test_display_position_units_and_unknowns(tmp_path):
    path = save(tmp_path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    item = viewer.TaskViewStore(path).snapshot(now=NOW)["current"]
    assert item["delta_mm"] == pytest.approx([5, 0, 0])
    assert item["motion_commands_sent"] == 0
    assert item["gripper_commands_sent"] is None
    assert item["task_success"] is None
    assert item["pre_check"] is None
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_shadow_record_without_historical_evidence(tmp_path):
    source = save(
        tmp_path,
        {
            "decision_id": "synthetic-shadow",
            "task_instruction": "Move a block",
            "proposed_high_level_action": "move_to",
            "execution": False,
            "execution_performed": False,
            "robot_state_visible": {"eef_state": [0.3, 0.0, 0.2, 0, 0, 0, 0]},
            "move_to_parameters": {"target": [0.305, 0.0, 0.2, 0, 0, 0, 0]},
            "approver_result": "approved",
            "raw_response": {"rationale_summary": "The blue block is to the right of the gripper."},
        },
    )
    item = viewer.TaskViewStore(source).snapshot()["current"]
    assert item["mode"] == "shadow"
    assert item["delta_mm"] == pytest.approx([5, 0, 0])
    assert item["pre_check"] == "NOT_RECORDED"
    assert item["approver"] == "approved"
    assert item["task_success"] is None
    assert item["note"] == "The blue block is to the right of the gripper."


@pytest.mark.parametrize("action", ["give_up", "done"])
@pytest.mark.parametrize("encoded", [True, False])
def test_official_shadow_tool_and_stop_are_not_physical_success(tmp_path, action, encoded):
    args = {"reason": "Need another view"}
    source = save(
        tmp_path,
        {
            "decision_id": "fixture",
            "task_instruction": "Move a block",
            "proposed_high_level_action": action,
            "execution": False,
            "execution_performed": False,
            "raw_response": {"tool_call": {"arguments": json.dumps(args) if encoded else args}},
            "task_success": True,
        },
    )
    item = viewer.TaskViewStore(source).snapshot()["current"]
    assert item["stage"] == "stopped"
    assert item["task_success"] is None
    assert item["note"] == args["reason"]
    assert item["motion_commands_sent"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"mode": "bad"},
        {"mode": {}},
        {"stage": []},
        {"task": ""},
        {"updated_at": "2026-09-27T00:00:00"},
        {"updated_at": None},
        {"task_success": "yes"},
        {"schema_version": "unsupported"},
    ],
)
def test_bad_record_is_unavailable_not_success(tmp_path, changes):
    state = viewer.TaskViewStore(save(tmp_path, record(**changes))).snapshot()
    assert state["current"] is None
    assert state["warnings"]


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "0.3"])
def test_invalid_numeric_values_are_unknown(tmp_path, value):
    source = save(
        tmp_path, record(current_xyz_m=[value, 0, 0], motion_commands_sent=value, latency_s=value)
    )
    current = viewer.TaskViewStore(source).snapshot()["current"]
    assert current["current_xyz_m"] is None
    assert current["motion_commands_sent"] is None
    assert current["latency_s"] is None
    assert current["delta_mm"] is None


@pytest.mark.parametrize(("offset", "stale"), [(0, False), (11, True), (-6, True)])
def test_record_freshness_and_future_clock(tmp_path, offset, stale):
    store = viewer.TaskViewStore(save(tmp_path))
    timestamp = store.snapshot()["current"]["timestamp"]
    assert store.snapshot(now=timestamp + offset)["current"]["stale"] is stale


def test_empty_directory_then_new_and_partial_records(tmp_path):
    store = viewer.TaskViewStore(tmp_path)
    assert store.snapshot()["current"] is None
    valid = save(tmp_path)
    broken = tmp_path / "broken.json"
    broken.write_text('{"stage":')
    assert store.snapshot()["current"]["file"] == valid.name
    assert store.snapshot()["warnings"]
    broken.write_text(json.dumps(record(task="New decision")))
    os.utime(valid, (100, 100))
    os.utime(broken, (200, 200))
    assert store.snapshot()["current"]["task"] == "New decision"
    assert store.snapshot(valid.name)["current"]["task"] == record()["task"]
    assert store.snapshot("../../other.json")["current"] is None


def test_symlink_and_device_inputs_are_not_read(tmp_path):
    path = save(tmp_path)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(viewer.ViewerError):
        viewer.TaskViewStore(link)
    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo)
    with pytest.raises(viewer.ViewerError):
        viewer.TaskViewStore(fifo)
    assert len(viewer.TaskViewStore(tmp_path).snapshot()["history"]) == 1


def test_bound_input_size_and_history(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, "MAX_RECORDS", 2)
    for index in range(3):
        save(tmp_path, name=f"{index}.json")
    assert len(viewer.TaskViewStore(tmp_path).snapshot()["history"]) == 2
    monkeypatch.setattr(viewer, "MAX_JSON", 2)
    assert viewer.TaskViewStore(tmp_path).snapshot()["current"] is None


def test_html_escapes_model_text_and_excludes_raw_sensitive_fields(tmp_path):
    source = save(
        tmp_path,
        record(
            note='</script><script>alert("x")</script>',
            transcript="private transcript",
            api_key="secret value",
            cameras={"front_rgb": {"url": "http://untrusted.invalid/camera"}},
        ),
    )
    html = viewer.render_html(viewer.TaskViewStore(source).snapshot())
    assert '</script><script>alert("x")' not in html
    assert "\\u003c/script\\u003e" in html
    assert "private transcript" not in html
    assert "secret value" not in html
    assert "untrusted.invalid" not in html
    assert "innerHTML" not in html
    assert "__INITIAL_STATE__" not in html
    assert '"enabled": false' in html


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_no_control_routes(tmp_path, method):
    app = viewer.ViewerApp(viewer.TaskViewStore(save(tmp_path)))
    assert app.respond(method, "/api/state")[0] == 405
    assert app.respond("GET", "/move")[0] == 404
    assert app.respond("GET", "/stop")[0] == 404


def test_get_api_and_no_preview_network_by_default(tmp_path):
    app = viewer.ViewerApp(viewer.TaskViewStore(save(tmp_path)))
    assert app.respond("GET", "/")[0] == 200
    status, _, body = app.respond("GET", "/api/state")
    assert status == 200
    assert json.loads(body)["viewer_read_only"]
    assert json.loads(app.respond("GET", "/api/cameras")[2])["available"] is False
    assert json.loads(app.respond("GET", "/api/state?record=../secret")[2])["current"] is None


def preview_health():
    return {
        "cameras": {
            key: {
                "label": role,
                "ok": True,
                "frames": 12,
                "measured_fps": 29.7,
                "last_frame_age_seconds": 0.025,
            }
            for role, key in viewer.PREVIEW_KEYS.items()
        }
    }


def mock_http(monkeypatch, payload, status=200, failure=None):
    calls = []

    class Response:
        def __init__(self):
            self.status = status

        def read(self, maximum):
            calls.append(("read", maximum))
            return payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    class Connection:
        def __init__(self, host, port, timeout):
            calls.append(("connect", host, port, timeout))

        def request(self, method, path):
            calls.append(("request", method, path))
            if failure:
                raise failure

        def getresponse(self):
            return Response()

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(viewer.http.client, "HTTPConnection", Connection)
    return calls


def test_existing_preview_only_fixed_loopback_read(monkeypatch):
    calls = mock_http(monkeypatch, preview_health())
    preview = viewer.CameraPreview()
    health = preview.health()
    assert all(row["ok"] for row in health["cameras"].values())
    assert calls == [
        ("connect", "127.0.0.1", 8765, 0.75),
        ("request", "GET", "/health"),
        ("read", 65_537),
        ("close",),
    ]
    assert preview.descriptor()["streams"] == {
        "front_rgb": "http://127.0.0.1:8765/stream/icspring-31.mjpg",
        "right_rgb": "http://127.0.0.1:8765/stream/icspring-32.mjpg",
        "wrist_rgb": "http://127.0.0.1:8765/stream/usb-main.mjpg",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"label": "left"},
        {"ok": False},
        {"last_frame_age_seconds": 3},
        {"last_frame_age_seconds": -1},
        {"last_frame_age_seconds": None},
        {"frames": 0},
        {"frames": None},
    ],
)
def test_wrong_mapping_and_stale_preview_are_not_live(monkeypatch, changes):
    payload = preview_health()
    payload["cameras"]["icspring-31"].update(changes)
    mock_http(monkeypatch, payload)
    row = viewer.CameraPreview().health()["cameras"]["front_rgb"]
    assert row["ok"] is False


@pytest.mark.parametrize(
    "payload,status,failure",
    [
        (b"{", 200, None),
        (b"x" * 65_537, 200, None),
        ({}, 302, None),
        ({}, 503, None),
        ({}, 200, TimeoutError()),
    ],
)
def test_preview_error_closes_and_never_launches_hardware(monkeypatch, payload, status, failure):
    calls = mock_http(monkeypatch, payload, status, failure)
    assert viewer.CameraPreview().health() == {"available": False, "cameras": {}}
    assert calls[-1] == ("close",)
    assert sum(call[0] == "request" for call in calls) == 1


def test_camera_status_independent_of_empty_task_directory(tmp_path, monkeypatch):
    mock_http(monkeypatch, preview_health())
    app = viewer.ViewerApp(viewer.TaskViewStore(tmp_path), viewer.CameraPreview())
    assert json.loads(app.respond("GET", "/api/state")[2])["current"] is None
    assert json.loads(app.respond("GET", "/api/cameras")[2])["available"] is True
    html = app.respond("GET", "/")[2].decode()
    assert "http://127.0.0.1:8765/stream/usb-main.mjpg" in html
    assert "img-src 'self' data: http://127.0.0.1:8765" in html


def test_export_never_opens_network_and_refuses_overwrite(tmp_path):
    source = save(tmp_path)
    output = tmp_path / "viewer.html"
    args = ["--source", str(source), "--output", str(output)]
    assert viewer.main(args) == 0  # pytest-socket forbids networking.
    content = output.read_bytes()
    assert b'"enabled": false' in content
    assert viewer.main(args) == 2
    assert output.read_bytes() == content


def test_serve_configuration_uses_existing_preview_without_starting_it(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        viewer,
        "serve",
        lambda store, port, preview, *, ready_file=None: calls.append((port, preview, ready_file)),
    )
    source = save(tmp_path)
    args = ["--source", str(source), "--serve"]
    assert viewer.main(args) == 0
    assert calls[-1][0] == 8766
    assert calls[-1][1].port == 8765
    assert calls[-1][2] is None
    ready = tmp_path / "ready.json"
    assert viewer.main([*args, "--ready-file", str(ready)]) == 0
    assert calls[-1][2] == ready
    assert viewer.main([*args, "--no-camera-preview"]) == 0
    assert calls[-1] == (8766, None, None)
    assert viewer.main([*args, "--camera-preview-port", "8766"]) == 2
    assert viewer.main([*args, "--camera-preview-port", "-1"]) == 2
