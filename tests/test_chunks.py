from dataclasses import replace

import numpy as np
import pytest
from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk

from inspect_robots_dobot.chunks import DobotExecutionSession, StagedDobotEmbodiment
from inspect_robots_dobot.config import DobotConfig
from inspect_robots_dobot.errors import ConfigurationError, SafetyRejected
from inspect_robots_dobot.motion import KeepoutBox


class ChunkPolicy:
    def __init__(self, chunk):
        self.chunk, self.calls = chunk, 0

    def act(self, observation):
        self.calls += 1
        return self.chunk


def chunk_to(x=0.31, steps=3):
    return ActionChunk(
        [
            Action(
                np.array([0.3 + (x - 0.3) * i / steps, 0, 0.2, 0, 0, 0, 0]),
                {"chunk_final": i == steps, "note": "original"},
            )
            for i in range(1, steps + 1)
        ],
        control_hz=10,
        inference_latency_s=0.123,
        meta={"source": "test"},
    )


@pytest.fixture
def staged(profile, driver):
    emb = StagedDobotEmbodiment(DobotConfig(profile, 10), driver=driver)
    with DobotExecutionSession(emb) as session:
        obs = emb.reset(Scene(id="staged", instruction="Offline XYZ"))
        yield emb, session, obs, {}


def advance(staged, policy, t):
    emb, session, obs, store = staged
    action = session.controller.next_action(policy, obs, t, store)
    return emb.step(session.approver.review(action, store))


@pytest.mark.parametrize("steps", [1, 2, 10, 100])
def test_one_complete_chunk_yields_one_plan_and_fake_move(staged, driver, pose, steps):
    emb, session, obs, store = staged
    policy = ChunkPolicy(chunk_to(steps=steps))
    for i in range(steps):
        result = advance(staged, policy, i)
        if i < steps - 1:
            np.testing.assert_array_equal(
                result.observation.state["eef_state"], obs.state["eef_state"]
            )
            assert driver.commands == () and not emb.plans
    assert policy.calls == 1
    assert store["_controller_inferences"] == [(0.123, steps)]
    assert session.controller.chunk_metadata == {"source": "test"}
    assert len(emb.plans) == len(emb.execution_results) == len(driver.commands) == 1
    assert driver.commands[0].target == replace(pose, x=0.31)
    assert result.observation.state["eef_state"][0] == 0.31
    assert all(not a.get("physical_sent", False) for a in emb.audit_records)
    assert emb.pending_chunk_id is None


@pytest.mark.parametrize(
    "markers", [[False], [True, True], [True, False], [False, False], [1], ["true"]]
)
def test_missing_duplicate_or_nonboolean_final_rejected(staged, driver, markers):
    emb, session, obs, store = staged
    base = chunk_to(steps=len(markers))
    chunk = replace(
        base,
        actions=[
            replace(a, meta={"chunk_final": m}) for a, m in zip(base.actions, markers, strict=True)
        ],
    )
    with pytest.raises(SafetyRejected, match="chunk_final"):
        session.controller.next_action(ChunkPolicy(chunk), obs, 0, store)
    assert not emb.plans and driver.commands == ()


@pytest.mark.parametrize("replan", [1, 2])
def test_replan_truncation_rejects_before_any_action(profile, driver, replan):
    emb = StagedDobotEmbodiment(DobotConfig(profile, 10), driver=driver)
    with DobotExecutionSession(emb, replan_interval=replan) as s:
        obs = emb.reset(Scene(id="replan", instruction="Offline test"))
        with pytest.raises(ConfigurationError, match="full-chunk playback"):
            s.controller.next_action(ChunkPolicy(chunk_to()), obs, 0, {})
        assert emb.pending_chunk_id is None and not emb.plans and not driver.commands


@pytest.mark.parametrize("replan", [None, 3, 20])
def test_full_replan_permitted(profile, driver, replan):
    emb = StagedDobotEmbodiment(DobotConfig(profile, 10), driver=driver)
    with DobotExecutionSession(emb, replan_interval=replan) as s:
        obs = emb.reset(Scene(id="replan", instruction="Offline test"))
        context = emb, s, obs, {}
        for t in range(3):
            advance(context, ChunkPolicy(chunk_to()), t)
        assert len(emb.plans) == 1


@pytest.mark.parametrize("operation", ["reset", "close", "abort", "trial_end"])
def test_partial_chunk_cleared_and_old_actions_invalid(staged, driver, operation):
    from inspect_robots.rollout import TrialRecord

    emb, s, obs, store = staged
    policy = ChunkPolicy(chunk_to())
    advance(staged, policy, 0)
    old = s.controller.next_action(policy, obs, 1, store)
    if operation == "reset":
        emb.reset(Scene(id="again", instruction="Offline test"))
    elif operation == "close":
        emb.close()
    elif operation == "trial_end":
        s.sink.on_trial_end(TrialRecord(scene_id="trial", epoch=0, seed=None))
    else:
        emb.abort_chunk("cancelled")
    with pytest.raises(SafetyRejected):
        emb.step(old)
    assert not emb.plans and driver.commands == ()
    assert emb.pending_chunk_id is None


def test_external_exception_context_clears_unconsumed_chunk(profile, driver):
    emb = StagedDobotEmbodiment(DobotConfig(profile, 10), driver=driver)
    with pytest.raises(RuntimeError, match="logging failure"), DobotExecutionSession(emb) as s:
        obs = emb.reset(Scene(id="exception", instruction="Offline test"))
        advance((emb, s, obs, {}), ChunkPolicy(chunk_to()), 0)
        raise RuntimeError("logging failure outside adapter")
    assert emb.pending_chunk_id is None and not emb.plans and not driver.commands


def test_active_session_required_and_raw_metadata_not_authority(profile, driver):
    emb = StagedDobotEmbodiment(DobotConfig(profile, 10), driver=driver)
    emb.reset(Scene(id="manual", instruction="Offline test"))
    with pytest.raises(SafetyRejected, match="active"):
        emb.step(chunk_to(steps=1).actions[0])
    emb.close()


def test_approval_required_even_with_issued_labels(staged, driver):
    emb, s, obs, store = staged
    action = s.controller.next_action(ChunkPolicy(chunk_to(steps=1)), obs, 0, store)
    with pytest.raises(SafetyRejected, match="approval"):
        emb.step(action)
    assert not emb.plans and not driver.commands


def test_duplicate_final_cannot_replay(staged, driver):
    emb, s, obs, store = staged
    a = s.approver.review(
        s.controller.next_action(ChunkPolicy(chunk_to(steps=1)), obs, 0, store), store
    )
    emb.step(a)
    with pytest.raises(SafetyRejected, match="stale or duplicate"):
        emb.step(a)
    assert len(emb.plans) == len(driver.commands) == 1


@pytest.mark.parametrize(
    "kind", ["replace", "inplace", "clamp", "identity", "final", "stop", "reject"]
)
def test_framework_rewrite_or_rejection_aborts_entire_chunk(profile, driver, kind):
    class Rewrite:
        calls = 0

        def review(self, action, store):
            self.calls += 1
            if self.calls == 1:
                return action
            if kind == "reject":
                raise SafetyRejected("operator rejected")
            if kind == "inplace":
                action.data[0] -= 0.001
                return action
            if kind == "replace":
                data = action.data.copy()
                data[0] -= 0.001
                return replace(action, data=data)
            key, value = {
                "clamp": ("clamped", True),
                "identity": ("dobot_chunk_id", "bad"),
                "final": ("chunk_final", False),
                "stop": ("request_stop", True),
            }[kind]
            return replace(action, meta={**action.meta, key: value})

    emb = StagedDobotEmbodiment(DobotConfig(profile, 10), driver=driver)
    with DobotExecutionSession(emb, extra_approvers=(Rewrite(),)) as s:
        context = emb, s, emb.reset(Scene(id="rewrite", instruction="Offline test")), {}
        p = ChunkPolicy(chunk_to(steps=2))
        advance(context, p, 0)
        with pytest.raises(SafetyRejected):
            advance(context, p, 1)
        assert not emb.plans and not driver.commands and emb.pending_chunk_id is None


def test_next_chunk_after_abort_does_not_combine_with_stale(staged, driver):
    emb, s, obs, store = staged
    advance(staged, ChunkPolicy(chunk_to()), 0)
    old_id = emb.pending_chunk_id
    emb.abort_chunk("operator cancelled proposal")
    # New trial/store avoids retaining framework delta history from rejected waypoints.
    advance((emb, s, obs, {}), ChunkPolicy(chunk_to(x=0.302, steps=1)), 0)
    assert len(emb.plans) == 1 and emb.plans[0].chunk_id != old_id
    assert len(emb.plans[0].staged_waypoints) == 1 and len(driver.commands) == 1


@pytest.mark.parametrize("mutation", ["pose", "command_id", "alarm"])
def test_measured_start_or_controller_change_prevents_commit(staged, driver, monkeypatch, mutation):
    emb, s, obs, store = staged
    p = ChunkPolicy(chunk_to(steps=2))
    advance(staged, p, 0)
    original = driver.snapshot

    def changed():
        sample = original()
        return replace(
            sample,
            **{
                "pose": {"pose": replace(sample.pose, x=sample.pose.x + 0.0001)},
                "command_id": {"command_id": sample.command_id + 1},
                "alarm": {"errors": (123,)},
            }[mutation],
        )

    monkeypatch.setattr(driver, "snapshot", changed)
    with pytest.raises(SafetyRejected):
        advance(staged, p, 1)
    assert not emb.plans and not driver.commands


def test_policy_stop_never_becomes_a_motion(staged, driver):
    emb, s, obs, store = staged
    stop = ActionChunk(
        [Action(obs.state["eef_state"].copy(), {"request_stop": True, "stop_reason": "done"})]
    )
    result = advance(staged, ChunkPolicy(stop), 0)
    assert result.info["dobot_staging"] == "policy_stop_no_motion"
    assert not emb.plans and not driver.commands


@pytest.mark.parametrize(
    "kind",
    ["nonconsecutive", "skipped_step", "new_store", "duplicate_approval", "mutated_after_approval"],
)
def test_broken_lifecycle_aborts(staged, driver, kind):
    emb, s, obs, store = staged
    p = ChunkPolicy(chunk_to())
    if kind == "nonconsecutive":
        advance(staged, p, 0)
    else:
        action = s.controller.next_action(p, obs, 0, store)
    with pytest.raises(SafetyRejected):
        if kind == "duplicate_approval":
            s.approver.review(action, store)
            s.approver.review(action, store)
        elif kind == "mutated_after_approval":
            reviewed = s.approver.review(action, store)
            reviewed.data[0] += 0.00001
            emb.step(reviewed)
        else:
            s.controller.next_action(
                p, obs, 99 if kind == "nonconsecutive" else 1, {} if kind == "new_store" else store
            )
    assert not emb.plans and not driver.commands


def test_keepout_crossing_between_safe_waypoints_blocks(profile, driver):
    box = KeepoutBox((0.304, -0.001, 0.199), (0.306, 0.001, 0.201))
    emb = StagedDobotEmbodiment(DobotConfig(profile, 10), driver=driver, keepouts=(box,))
    with DobotExecutionSession(emb) as s:
        obs = emb.reset(Scene(id="keepout", instruction="Offline test"))
        assert "keepout" in emb.pre_check(np.stack([a.data for a in chunk_to(steps=1).actions]))
        with pytest.raises(SafetyRejected, match="keepout"):
            s.controller.next_action(ChunkPolicy(chunk_to(steps=1)), obs, 0, {})
    assert not emb.plans and not driver.commands


@pytest.mark.parametrize(
    "key,value",
    [
        ("chunk_final", False),
        ("chunk_index", 9),
        ("chunk_length", 9),
        ("request_stop", True),
        ("chunk_final", 1),
    ],
)
def test_buffered_metadata_mutation_rejected_before_issuance(staged, driver, key, value):
    emb, s, obs, store = staged
    p = ChunkPolicy(chunk_to(steps=2))
    advance(staged, p, 0)
    store["_controller_action_buffer"][0].meta[key] = value
    with pytest.raises(SafetyRejected, match="metadata"):
        s.controller.next_action(p, obs, 1, store)
    assert not emb.plans and not driver.commands


def test_one_guard_cannot_hide_another_guards_rewrite(profile, driver):
    calls = []

    class Rewrite:
        def review(self, action, store):
            calls.append("rewrite")
            return replace(action, data=action.data - np.array([0.001, 0, 0, 0, 0, 0, 0]))

    class Restore:
        def review(self, action, store):
            calls.append("restore")
            return replace(action, data=action.data + np.array([0.001, 0, 0, 0, 0, 0, 0]))

    emb = StagedDobotEmbodiment(DobotConfig(profile, 10), driver=driver)
    with DobotExecutionSession(emb, extra_approvers=(Rewrite(), Restore())) as s:
        obs = emb.reset(Scene(id="rewrite-restore", instruction="Offline"))
        with pytest.raises(SafetyRejected, match="modified"):
            advance((emb, s, obs, {}), ChunkPolicy(chunk_to(steps=1)), 0)
    assert calls == ["rewrite"] and not emb.plans and not driver.commands


def test_unchanged_approver_copy_preserves_labels_and_commits(profile, driver):
    class Copy:
        def review(self, action, store):
            return replace(action, data=action.data.copy(), meta=dict(action.meta))

    emb = StagedDobotEmbodiment(DobotConfig(profile, 10), driver=driver)
    with DobotExecutionSession(emb, extra_approvers=(Copy(),)) as s:
        obs = emb.reset(Scene(id="copy", instruction="Offline"))
        advance((emb, s, obs, {}), ChunkPolicy(chunk_to(steps=1)), 0)
    assert len(emb.plans) == len(driver.commands) == 1


def test_exception_in_policy_clears_trial_controller(staged, driver):
    emb, s, obs, store = staged

    class Broken:
        def act(self, observation):
            raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        s.controller.next_action(Broken(), obs, 0, store)
    assert not emb.plans and not driver.commands and s.controller._store is None


def test_aborted_waypoint_delta_history_rewinds_to_measured_state(staged, driver):
    emb, s, obs, store = staged
    advance(staged, ChunkPolicy(chunk_to(x=0.318, steps=2)), 0)
    emb.abort_chunk("replan from reality")
    # Opposite direction is legal from measured x=.3; old staged x=.309 would clamp it.
    advance(staged, ChunkPolicy(chunk_to(x=0.295, steps=1)), 1)
    assert len(emb.plans) == 1 and driver.get_pose().x == 0.295


def test_missing_buffer_and_new_trial_cannot_reuse_partial(staged, driver):
    emb, s, obs, store = staged
    p = ChunkPolicy(chunk_to())
    advance(staged, p, 0)
    store.pop("_controller_action_buffer")
    with pytest.raises(SafetyRejected, match="incomplete"):
        s.controller.next_action(p, obs, 1, store)
    assert not emb.plans and not driver.commands


def test_fresh_dispatch_read_cannot_change_approved_start(staged, driver, monkeypatch):
    emb, s, obs, store = staged
    original = driver.snapshot

    def drift_after_plan():
        sample = original()
        if emb.plans:
            return replace(sample, pose=replace(sample.pose, y=0.0001))
        return sample

    monkeypatch.setattr(driver, "snapshot", drift_after_plan)
    with pytest.raises(SafetyRejected, match="start changed at fake dispatch"):
        advance(staged, ChunkPolicy(chunk_to(steps=1)), 0)
    assert not any(c.name == "move_linear" for c in driver.commands)
    assert not emb.execution_results and emb.pending_chunk_id is None
