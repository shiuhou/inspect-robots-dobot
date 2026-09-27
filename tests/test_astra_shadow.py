import json

import numpy as np
import pytest

from inspect_robots_dobot.astra_shadow import (
    AstraDecision,
    AstraPolicyAdapter,
    AstraResponse,
    FakeAstra,
    MissingAstra,
    ShadowExecutor,
    build_fixture_observation,
    fixture_profile,
    generate_candidates,
    move_to_chunk,
    run_fixture_shadow,
)
from inspect_robots_dobot.driver import FakeDobotDriver


def test_fixture_full_shadow_path_is_approved_and_never_dispatches():
    record = run_fixture_shadow()
    assert record.image_roles == ("front_rgb", "right_rgb", "wrist_rgb")
    assert record.approver_result == "approved"
    assert record.pre_check_result is None
    assert record.hypothetical_movl.startswith("WOULD_SEND: MovL(")
    assert record.execution is False
    assert record.execution_performed is False
    assert record.nova_connections == 0
    assert record.gripper_serial_connections == 0
    assert record.number_of_waypoints == 1


def test_shadow_path_does_not_call_fake_move(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("shadow path must not dispatch FakeDobot motion")

    monkeypatch.setattr(FakeDobotDriver, "move_linear", fail)
    record = run_fixture_shadow()
    assert record.execution_performed is False


def test_structured_response_rejects_missing_or_unsupported_action():
    embodiment, observation = build_fixture_observation()
    try:
        candidates = generate_candidates(observation, fixture_profile())
        adapter = AstraPolicyAdapter(FakeAstra({"candidate_id": "+X"}))
        with pytest.raises(ValueError, match="missing action"):
            adapter.decide(observation, candidates)
        adapter = AstraPolicyAdapter(
            FakeAstra(
                {
                    "action": "translate",
                    "candidate_id": "arbitrary-large-motion",
                    "rationale_summary": "bad",
                }
            )
        )
        with pytest.raises(ValueError, match="unsupported candidate"):
            adapter.decide(observation, candidates)
    finally:
        embodiment.close()


def test_candidate_set_is_bounded_orientation_pinned_and_gripper_inactive():
    embodiment, observation = build_fixture_observation()
    try:
        candidates = generate_candidates(observation, fixture_profile())
        translations = [candidate for candidate in candidates if candidate.action == "translate"]
        assert translations
        assert all(
            max(abs(value) for value in candidate.delta_m) <= 0.005 for candidate in translations
        )
        assert all(
            candidate.delta_m == (0.0, 0.0, 0.0)
            for candidate in candidates
            if candidate.action != "translate"
        )
        decision = AstraDecision(
            "translate",
            translations[0].candidate_id,
            "test",
            None,
            False,
            True,
            "grasp",
            translations[0].delta_m,
            {},
        )
        chunk = move_to_chunk(observation, decision, fixture_profile())
        np.testing.assert_array_equal(
            chunk.actions[-1].data[3:], observation.state["eef_state"][3:]
        )
        assert chunk.actions[-1].data[6] == 0
    finally:
        embodiment.close()


def test_response_hash_and_model_request_are_reproducible():
    embodiment, observation = build_fixture_observation()
    try:
        candidates = generate_candidates(observation, fixture_profile())
        provider = FakeAstra(
            {"action": "translate", "candidate_id": "+X", "rationale_summary": "bounded"}
        )
        adapter = AstraPolicyAdapter(provider)
        response, decision, request = adapter.decide(observation, candidates)
        assert response.provider == "fake"
        assert decision.candidate_id == "+X"
        assert request["image_roles"] == ["front_rgb", "right_rgb", "wrist_rgb"]
        assert [image["role"] for image in request["images"]] == [
            "front_rgb",
            "right_rgb",
            "wrist_rgb",
        ]
        assert request["action_semantics"]["interface"] == "move_to"
    finally:
        embodiment.close()


def test_replay_response_can_be_json_string():
    embodiment, observation = build_fixture_observation()
    try:
        candidates = generate_candidates(observation, fixture_profile())
        response = AstraResponse(
            "fake",
            "fixture",
            json.dumps({"action": "translate", "candidate_id": "+X", "rationale_summary": "ok"}),
        )
        decision = AstraPolicyAdapter(FakeAstra(response.raw)).parse(response, candidates)
        assert decision.action == "translate"
    finally:
        embodiment.close()


def test_missing_live_provider_is_explicitly_blocked():
    embodiment, observation = build_fixture_observation()
    try:
        candidates = generate_candidates(observation, fixture_profile())
        with pytest.raises(RuntimeError, match="ASTRA_LIVE_CLIENT_BLOCKED"):
            AstraPolicyAdapter(MissingAstra()).decide(observation, candidates)
    finally:
        embodiment.close()


def test_evidence_writer_refuses_overwrite(tmp_path):
    record = run_fixture_shadow()
    path = ShadowExecutor.write(record, tmp_path)
    assert path.exists()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ShadowExecutor.write(record, tmp_path)
