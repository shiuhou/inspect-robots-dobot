"""Dream paths remain socket/device-blocked, including a mocked compatible model."""

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fake_sockets import ScriptedFactory, dashboard_socket

from inspect_robots_dobot.clock import FakeClock
from inspect_robots_dobot.config import ConnectionConfig
from inspect_robots_dobot.dashboard import DobotDashboardClient
from inspect_robots_dobot.dream_integration import api
from inspect_robots_dobot.dream_integration.__main__ import main
from inspect_robots_dobot.dream_integration.observation import (
    DashboardMeasurements,
    GripperMeasurement,
    ObservationAdapter,
    fixture,
)
from inspect_robots_dobot.dream_integration.records import load_record, save_record
from inspect_robots_dobot.errors import CameraFault


def contract():
    return api.InferenceContract(
        "test/nova2", "emb_test_nova2", 20.0, 3, 0, 0, "TEST ONLY: fictional training semantics"
    )


def model(**kwargs):
    values = dict(
        id="test/nova2",
        embodiment="emb_test_nova2",
        servable=True,
        cameras=["front", "wrist"],
        embodiment_config=api.nova2_config(),
        parts_withheld=False,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def action():
    return {"per_key": {"action.right_arm": [[0.1] * 6] * 3, "action.right_gripper": [0.5] * 3}}


class Client:
    def __init__(self, models=None, error=None):
        self.models = models if models is not None else [model()]
        self.error = error
        self.calls = []

    def get_models(self):
        if self.error:
            raise self.error
        return self.models

    def policy(self, config):
        self.calls.append(("policy", config))

    def set_observation(self, frames, observation, *, timestamp):
        self.calls.append(("observation", frames, observation, timestamp))

    def get_action(self):
        self.calls.append(("action",))
        return action()


def measured_test_observation():
    # Explicit test double, never a saved physical observation or API input.
    obs, configs = fixture()
    obs = replace(
        obs,
        fixture=False,
        state=replace(obs.state, joints_synthetic=False),
        calibration_evidence="fictional test calibration",
        gripper_state=GripperMeasurement(
            0.5, 10.0, "test encoder", "normalized_0_closed_1_open", "test calibration"
        ),
    )
    return obs, configs


def infer(client, obs, configs, c=None):
    return api.infer_once(
        client,
        obs,
        configs,
        c or contract(),
        "test prompt",
        validation_time=10.02,
        max_age=0.5,
        max_skew=0.1,
    )


def test_missing_credentials_before_sdk_import(monkeypatch):
    monkeypatch.delenv("DREAM_API_KEY", raising=False)
    monkeypatch.setattr(api, "sdk", lambda: pytest.fail("SDK must not load without credentials"))
    with pytest.raises(api.DreamIntegrationError, match="DREAM_API_KEY missing"):
        api.connect_client()


def test_client_uses_explicit_environment_key(monkeypatch):
    monkeypatch.setenv("DREAM_API_KEY", "test-secret")
    monkeypatch.setattr(api, "sdk", lambda: SimpleNamespace(DreamClient=lambda **kw: kw))
    assert api.connect_client() == {"api_key": "test-secret"}


@pytest.mark.parametrize("error", [TimeoutError("private-token"), RuntimeError("private-token")])
def test_discovery_errors_are_sanitized(error):
    with pytest.raises(api.DreamIntegrationError) as exc:
        api.discover(Client(error=error))
    assert "private-token" not in str(exc.value)


def test_discovery():
    assert api.discover(Client())[0].id == "test/nova2"


@pytest.mark.parametrize(
    "changes",
    [
        {"embodiment": "yam"},
        {"embodiment": None},
        {"servable": False},
        {"cameras": ["overhead", "wrist_l", "wrist_r"]},
        {"embodiment_config": None},
        {"parts_withheld": True},
    ],
)
def test_unsupported_model_never_opens_policy(changes):
    client = Client([model(**changes)])
    obs, configs = measured_test_observation()
    with pytest.raises(api.DreamIntegrationError):
        infer(client, obs, configs)
    assert client.calls == []


@pytest.mark.parametrize(
    "change", ["joints", "nan", "pose", "stale", "future", "mode", "synthetic"]
)
def test_bad_robot_observation(change):
    obs, configs = measured_test_observation()
    s = obs.state
    changes = {
        "joints": dict(joints=(0.0,) * 5),
        "nan": dict(joints=(float("nan"),) * 6),
        "pose": dict(pose=replace(s.pose, z=float("inf"))),
        "stale": dict(observed_at=1.0),
        "future": dict(observed_at=11.0),
        "mode": dict(mode=99),
        "synthetic": dict(joints_synthetic=True),
    }
    obs = replace(obs, state=replace(s, **changes[change]))
    with pytest.raises(ValueError):
        infer(Client(), obs, configs)


@pytest.mark.parametrize("change", ["dimensions", "dtype", "stale", "skew", "nan"])
def test_bad_camera_observation(change):
    obs, configs = measured_test_observation()
    f = obs.front_rgb
    changes = {
        "dimensions": dict(rgb=np.zeros((2, 3), dtype=np.uint8)),
        "dtype": dict(rgb=f.rgb.astype(float)),
        "stale": dict(acquisition_started_at=1.0, published_at=1.0),
        "skew": dict(acquisition_started_at=9.8, published_at=9.8),
        "nan": dict(published_at=float("nan")),
    }
    obs = replace(obs, front_rgb=replace(f, **changes[change]))
    with pytest.raises((CameraFault, ValueError)):
        infer(Client(), obs, configs)


def test_missing_camera():
    obs, configs = fixture()
    del configs["wrist_rgb"]
    with pytest.raises(ValueError, match="front and wrist"):
        infer(Client(), obs, configs)


@pytest.mark.parametrize("change", ["missing", "uncalibrated", "native", "nan", "outside"])
def test_gripper_never_fabricated(change):
    obs, configs = measured_test_observation()
    g = obs.gripper_state
    values = {
        "missing": None,
        "uncalibrated": replace(g, calibration_evidence=None),
        "native": replace(g, units="servo_ticks"),
        "nan": replace(g, position=float("nan")),
        "outside": replace(g, position=2.0),
    }
    client = Client()
    with pytest.raises(ValueError):
        infer(client, replace(obs, gripper_state=values[change]), configs)
    assert not client.calls


def test_fixture_cannot_be_sent():
    obs, configs = fixture()
    client = Client()
    with pytest.raises(api.DreamIntegrationError, match="synthetic"):
        infer(client, obs, configs)
    assert not client.calls


def test_calibration_and_coordinate_frames():
    obs, configs = measured_test_observation()
    for invalid in [
        replace(obs, calibration_evidence=None),
        replace(obs, state=replace(obs.state, tool_frame=1)),
    ]:
        with pytest.raises(api.DreamIntegrationError):
            infer(Client(), invalid, configs)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {"per_key": {}},
        {"per_key": {"action.right_arm": [[0] * 5] * 3, "action.right_gripper": [0.5] * 3}},
        {"per_key": {"action.right_arm": [[math.nan] * 6] * 3, "action.right_gripper": [0.5] * 3}},
        {"per_key": {"action.right_arm": [[0] * 6] * 3, "action.right_gripper": [2.0] * 3}},
        {"per_key": {"action.right_arm": [[0] * 6] * 2, "action.right_gripper": [0.5] * 3}},
    ],
)
def test_malformed_action(bad):
    with pytest.raises(api.DreamIntegrationError, match="malformed"):
        api.decode_action(bad, contract())


def test_action_timing_and_units():
    result = api.decode_action(action(), contract())
    assert result["time_from_start_s"] == [0.0, 0.05, 0.1]
    assert result["joint_targets_rad"][0] == [0.1] * 6
    assert result["execution"] == "DISCONNECTED"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(action_hz=0),
        dict(action_hz=math.nan),
        dict(horizon=0),
        dict(joint_units="deg"),
        dict(evidence=""),
        dict(embodiment_id="yam"),
        dict(action_semantics="delta"),
    ],
)
def test_invalid_training_contract(kwargs):
    with pytest.raises(api.DreamIntegrationError):
        replace(contract(), **kwargs).validate()


def test_mocked_inference_zero_motion():
    pytest.importorskip("PIL")
    obs, configs = measured_test_observation()
    client = Client()
    result = infer(client, obs, configs)
    assert [c[0] for c in client.calls] == ["policy", "observation", "action"]
    _, frames, state, timestamp = client.calls[1]
    assert set(frames) == {"video.front", "video.wrist"}
    assert set(state) == {"state.right_arm", "state.right_gripper", "prompt"}
    assert state["state.right_arm"].shape == (6,)
    assert state["state.right_arm"].dtype == np.float32
    assert timestamp == 10.0
    assert result["latency_ms"] >= 0
    assert result["execution"] == "DISCONNECTED"


@pytest.mark.parametrize("error", [TimeoutError("private-key"), RuntimeError("private-key")])
def test_inference_errors_without_retry(error):
    pytest.importorskip("PIL")

    class Failing(Client):
        def get_action(self):
            self.calls.append(("action",))
            raise error

    obs, configs = measured_test_observation()
    client = Failing()
    with pytest.raises(api.DreamIntegrationError) as exc:
        infer(client, obs, configs)
    assert "private-key" not in str(exc.value)
    assert [c[0] for c in client.calls].count("action") == 1


def test_records_preserve_timestamps_and_do_not_overwrite(tmp_path):
    obs, configs = fixture()
    directory = tmp_path / "record"
    save_record(directory, obs, configs, max_age=0.5, max_skew=0.1)
    loaded, loaded_configs, age, skew = load_record(directory)
    assert loaded.fixture and loaded.gripper_state is None
    assert loaded.state == obs.state
    assert loaded.front_rgb.timestamp == obs.front_rgb.timestamp
    np.testing.assert_array_equal(loaded.wrist_rgb.rgb, obs.wrist_rgb.rgb)
    assert loaded_configs == configs and (age, skew) == (0.5, 0.1)
    with pytest.raises(FileExistsError):
        save_record(directory, obs, configs, max_age=0.5, max_skew=0.1)


def test_dashboard_adapter_measured_joints_and_only_queries():
    clock = FakeClock(10.0)
    stream = dashboard_socket()
    client = DobotDashboardClient(
        ConnectionConfig(host="192.0.2.1", user_frame=0, tool_frame=2),
        socket_factory=ScriptedFactory(dashboard=stream),
        clock=clock,
    )
    client.connect()  # injected stream; sockets remain forbidden
    reader = DashboardMeasurements(client, clock)
    s = reader.snapshot()
    assert s.joints == pytest.approx((0, math.pi / 2, -math.pi, math.pi / 4, 0, 0))
    assert s.pose.x == 0.3 and s.tool_frame == 2
    assert not s.joints_synthetic
    assert stream.sent == [
        b"GetAngle()",
        b"GetPose(user=0,tool=2)",
        b"RobotMode()",
        b"GetErrorID()",
    ]
    client.close()


def test_adapter_uses_supplied_running_readers():
    obs, configs = measured_test_observation()
    reader = SimpleNamespace(snapshot=lambda: obs.state)
    camera = SimpleNamespace(latest=lambda: obs.front_rgb)
    adapter = ObservationAdapter(
        reader, camera, camera, configs, FakeClock(10.02), max_age=0.5, max_skew=0.1
    )
    result = adapter.capture()  # no gripper reader => UNKNOWN, never a zero/open guess
    assert result.gripper_state is None
    assert result.joint_positions == obs.state.joints


def test_sdk_local_schema_without_network():
    pytest.importorskip("dream")
    api.validate_local_schema()


def test_cli_missing_credentials_sanitized(monkeypatch, capsys):
    monkeypatch.delenv("DREAM_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["dream", "probe"])
    assert main() == 2
    assert json.loads(capsys.readouterr().out)["execution"] == "DISCONNECTED"


def test_cli_probe_reports_incompatibility_without_policy(monkeypatch, capsys):
    from inspect_robots_dobot.dream_integration import __main__ as cli

    client = Client([model(embodiment="yam", id="dreamlabs/dream-1-beta")])
    client.close = lambda: None
    monkeypatch.setattr(cli, "connect_client", lambda: client)
    monkeypatch.setattr("sys.argv", ["dream", "probe"])
    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert "YAM" in payload["models"][0]["nova2_blocker"]
    assert not client.calls


def test_cli_refuses_public_keyfile(monkeypatch, tmp_path, capsys):
    key = tmp_path / "private"
    key.write_text("test-secret-never-print")
    key.chmod(0o644)
    monkeypatch.setattr("sys.argv", ["dream", "--key-file", str(key), "probe"])
    assert main() == 2
    output = capsys.readouterr().out
    assert "test-secret-never-print" not in output
    assert "0600" in output


def test_record_units_cannot_silently_change(tmp_path):
    obs, configs = fixture()
    path = tmp_path / "record"
    save_record(path, obs, configs, max_age=0.5, max_skew=0.1)
    manifest = path / "observation.json"
    meta = json.loads(manifest.read_text())
    meta["units"]["joints"] = "deg"
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="unsupported"):
        load_record(path)


@pytest.mark.parametrize("value", ["0.1", True])
def test_action_values_must_be_numbers_not_strings_or_bools(value):
    bad = action()
    bad["per_key"]["action.right_arm"] = [[value] * 6] * 3
    with pytest.raises(api.DreamIntegrationError, match="malformed"):
        api.decode_action(bad, contract())
