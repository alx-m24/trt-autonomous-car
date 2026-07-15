"""Offline regression test for the parking maneuver state machine.

Runs the REAL ParkingManeuver logic with no ROS context (no rclpy, no clock, no
/cmd_vel) — inputs are passed in directly and it returns an (linear_x, angular_z)
tuple or None. Covers the full happy-path sequence, the gating conditions, aborts,
and the entry-arc curvature math.
"""

import math

import pytest

from auto_drive.parking_maneuver import ParkingManeuver, ParkingConfig


def _cfg(**over):
    base = dict(
        commit_dist_m=6.7, approach_speed=0.10, approach_kp=1.5,
        enter_speed=0.12, enter_distance_m=0.25, enter_use_arc=True,
        min_turn_radius_m=0.35, enter_steer=0.0, enter_timeout_sec=10.0,
        max_yaw_rate=1.0, approach_abort_bearing=math.radians(45.0),
    )
    base.update(over)
    return ParkingConfig(**base)


def _mk(**kw):
    """update() kwargs with sensible defaults for an armed, fresh detection."""
    d = dict(now=0.0, cfg=_cfg(), armed=True, detected=True,
             target=(1.0, 0.0, 0.0), target_fresh=True, dist=7.0,
             robot_xy=(0.0, 0.0), can_start=True)
    d.update(kw)
    return d


def _drive_to_enter(m, cfg, dpsi_deg=0.0, xy0=(0.0, 0.0)):
    target = (1.0, 0.0, math.radians(dpsi_deg))
    m.update(**_mk(cfg=cfg, target=target, dist=7.0, robot_xy=xy0))          # -> APPROACH
    m.update(**_mk(cfg=cfg, target=target, dist=6.0, robot_xy=xy0,
                   can_start=False))                                          # -> COMMIT
    return m.update(**_mk(cfg=cfg, target=target, dist=6.0, robot_xy=xy0,
                          can_start=False))                                   # COMMIT -> ENTER


def _drive_to_parallel_back1(m, cfg, xy0=(0.0, 0.0)):
    """APPROACH -> COMMIT -> PARALLEL_BACK1 for a parallel bay."""
    m.update(**_mk(cfg=cfg, bay_type='parallel', dist=7.0, robot_xy=xy0))
    m.update(**_mk(cfg=cfg, bay_type='parallel', dist=6.0, robot_xy=xy0,
                   can_start=False))
    return m.update(**_mk(cfg=cfg, bay_type='parallel', dist=6.0, robot_xy=xy0,
                          can_start=False))                                   # COMMIT -> BACK1


def test_idle_yields():
    m = ParkingManeuver()
    assert m.update(**_mk(armed=False)) is None
    assert m.state == 'IDLE' and not m.active


def test_can_start_gate_blocks_mid_corner():
    """Parking must not pre-empt another maneuver (can_start False)."""
    m = ParkingManeuver()
    assert m.update(**_mk(can_start=False)) is None
    assert m.state == 'IDLE'


def test_requires_armed_detected_fresh():
    for bad in (dict(armed=False), dict(detected=False), dict(target_fresh=False)):
        m = ParkingManeuver()
        assert m.update(**_mk(**bad)) is None
        assert m.state == 'IDLE'


def test_happy_path_to_parked():
    m = ParkingManeuver()
    cfg = _cfg()
    out = m.update(**_mk(cfg=cfg))                       # IDLE -> APPROACH
    assert m.state == 'APPROACH'
    assert out[0] == pytest.approx(cfg.approach_speed)

    out = m.update(**_mk(cfg=cfg, dist=6.0, can_start=False))   # APPROACH -> COMMIT
    assert m.state == 'COMMIT' and out == (0.0, 0.0)

    out = m.update(**_mk(cfg=cfg, dist=6.0, can_start=False))   # COMMIT -> ENTER
    assert m.state == 'ENTER'
    assert out[0] == pytest.approx(cfg.enter_speed)

    # Not far enough yet.
    m.update(**_mk(cfg=cfg, robot_xy=(0.1, 0.0), can_start=False))
    assert m.state == 'ENTER'
    # Reached the arc length -> PARKED (terminal stop).
    out = m.update(**_mk(cfg=cfg, robot_xy=(0.3, 0.0), can_start=False))
    assert m.state == 'PARKED' and out == (0.0, 0.0)


def test_approach_aborts_off_track_bearing():
    """Off-track guard: a bay bearing beyond the abort limit bails to IDLE."""
    m = ParkingManeuver()
    cfg = _cfg(approach_abort_bearing=math.radians(45.0))
    # Enter APPROACH with a bay dead ahead.
    m.update(**_mk(cfg=cfg, target=(1.0, 0.0, 0.0)))
    assert m.state == 'APPROACH'
    # Now the "bay" is far off to the side (bearing ~63deg > 45) -> bail.
    out = m.update(**_mk(cfg=cfg, target=(1.0, 2.0, 0.0), can_start=False))
    assert m.state == 'IDLE' and out is None


def test_off_nose_detection_never_enters_approach():
    """A persistent off-nose (false) detection must not even START APPROACH, so
    no steer-toward-phantom command is ever emitted (no IDLE<->APPROACH churn)."""
    m = ParkingManeuver()
    cfg = _cfg(approach_abort_bearing=math.radians(45.0))
    for _ in range(5):
        out = m.update(**_mk(cfg=cfg, target=(1.0, 2.0, 0.0)))  # ~63deg off-nose
        assert m.state == 'IDLE' and out is None


def test_approach_abort_disabled_when_zero():
    """abort_bearing <= 0 disables the guard: a side bearing does NOT bail."""
    m = ParkingManeuver()
    cfg = _cfg(approach_abort_bearing=0.0)
    m.update(**_mk(cfg=cfg, target=(1.0, 0.0, 0.0)))
    out = m.update(**_mk(cfg=cfg, target=(1.0, 5.0, 0.0), can_start=False))
    assert m.state == 'APPROACH' and out is not None


def test_enter_terminates_on_timeout():
    m = ParkingManeuver()
    cfg = _cfg(enter_timeout_sec=5.0)
    _drive_to_enter(m, cfg)
    # Car barely moves but the clock passes the timeout -> PARKED.
    out = m.update(**_mk(cfg=cfg, now=6.0, robot_xy=(0.01, 0.0), can_start=False))
    assert m.state == 'PARKED' and out == (0.0, 0.0)


def test_disarm_aborts_from_any_state():
    m = ParkingManeuver()
    cfg = _cfg()
    _drive_to_enter(m, cfg)
    assert m.state == 'ENTER'
    assert m.update(**_mk(cfg=cfg, armed=False, can_start=False)) is None
    assert m.state == 'IDLE' and not m.active


def test_target_lost_in_approach_bails():
    m = ParkingManeuver()
    cfg = _cfg()
    m.update(**_mk(cfg=cfg))                             # -> APPROACH
    assert m.state == 'APPROACH'
    assert m.update(**_mk(cfg=cfg, target_fresh=False, can_start=False)) is None
    assert m.state == 'IDLE'


def test_entry_arc_curvature_matches_heading_error():
    """kappa = dpsi / arc_length, and ENTER commands wz = v * kappa."""
    m = ParkingManeuver()
    cfg = _cfg(enter_distance_m=0.25, min_turn_radius_m=0.35)
    out = _drive_to_enter(m, cfg, dpsi_deg=30.0)
    expected_kappa = math.radians(30.0) / 0.25
    assert (out[1] / cfg.enter_speed) == pytest.approx(expected_kappa, rel=1e-6)


def test_entry_arc_clamped_to_min_radius():
    m = ParkingManeuver()
    cfg = _cfg(enter_distance_m=0.25, min_turn_radius_m=0.35)
    out = _drive_to_enter(m, cfg, dpsi_deg=80.0)         # demands a tighter turn
    kappa_max = 1.0 / 0.35
    assert (out[1] / cfg.enter_speed) == pytest.approx(kappa_max, rel=1e-6)


def test_enter_uses_fixed_steer_when_arc_disabled():
    m = ParkingManeuver()
    cfg = _cfg(enter_use_arc=False, enter_steer=0.07)
    out = _drive_to_enter(m, cfg, dpsi_deg=30.0)
    assert out == (pytest.approx(cfg.enter_speed), pytest.approx(0.07))


# ── PARALLEL parking (reverse-arc entry) ──────────────────────────────────────

def test_parallel_commit_branches_to_back1_and_reverses():
    """A parallel bay at COMMIT hands off to the reverse sequence, not ENTER."""
    m = ParkingManeuver()
    cfg = _cfg()
    out = _drive_to_parallel_back1(m, cfg)
    assert m.state == 'PARALLEL_BACK1'
    assert out[0] == pytest.approx(-cfg.parallel_speed)   # reversing


def test_parallel_full_sequence_to_parked():
    """COMMIT -> BACK1 -> BACK2 -> PARKED, each segment terminated on odom dist."""
    m = ParkingManeuver()
    cfg = _cfg(parallel_back1_dist=0.20, parallel_back2_dist=0.20,
               parallel_adjust_dist=0.0)
    _drive_to_parallel_back1(m, cfg)                                  # BACK1 (seg0=0,0)
    assert m.state == 'PARALLEL_BACK1'
    m.update(**_mk(cfg=cfg, bay_type='parallel', robot_xy=(0.0, -0.10),
                   can_start=False))                                  # not far enough
    assert m.state == 'PARALLEL_BACK1'
    m.update(**_mk(cfg=cfg, bay_type='parallel', robot_xy=(0.0, -0.25),
                   can_start=False))                                  # -> BACK2 (seg reset)
    assert m.state == 'PARALLEL_BACK2'
    out = m.update(**_mk(cfg=cfg, bay_type='parallel', robot_xy=(0.0, -0.50),
                         can_start=False))                            # -> PARKED
    assert m.state == 'PARKED' and out == (0.0, 0.0)


def test_parallel_back_arcs_use_opposite_lock():
    """back1 and back2 steer with opposite sign; magnitude respects R_min."""
    m = ParkingManeuver()
    cfg = _cfg(parallel_back1_dist=0.20, parallel_back2_dist=0.20,
               parallel_first_sign=1.0, max_yaw_rate=0.0)   # 0 = no yaw clamp
    kappa = min(cfg.parallel_kappa, 1.0 / cfg.min_turn_radius_m)
    out1 = _drive_to_parallel_back1(m, cfg)
    assert out1[1] == pytest.approx(cfg.parallel_speed * kappa)
    out2 = m.update(**_mk(cfg=cfg, bay_type='parallel', robot_xy=(0.0, -0.25),
                          can_start=False))
    assert m.state == 'PARALLEL_BACK2'
    assert out2[1] == pytest.approx(-cfg.parallel_speed * kappa)
    assert out1[1] * out2[1] < 0.0                                   # opposite lock


def test_parallel_kappa_clamped_to_min_radius():
    """A huge parallel_kappa is clamped to 1/R_min (holonomic plugin won't)."""
    m = ParkingManeuver()
    cfg = _cfg(parallel_kappa=99.0, min_turn_radius_m=0.35, max_yaw_rate=0.0)
    out = _drive_to_parallel_back1(m, cfg)
    assert out[1] == pytest.approx(cfg.parallel_speed * (1.0 / 0.35))


def test_parallel_adjust_then_parked():
    """With adjust_dist>0, BACK2 -> forward ADJUST -> PARKED."""
    m = ParkingManeuver()
    cfg = _cfg(parallel_back1_dist=0.20, parallel_back2_dist=0.20,
               parallel_adjust_dist=0.10)
    _drive_to_parallel_back1(m, cfg)
    m.update(**_mk(cfg=cfg, bay_type='parallel', robot_xy=(0.0, -0.25),
                   can_start=False))                                  # -> BACK2
    out = m.update(**_mk(cfg=cfg, bay_type='parallel', robot_xy=(0.0, -0.50),
                         can_start=False))                            # -> ADJUST
    assert m.state == 'PARALLEL_ADJUST'
    assert out[0] > 0.0                                               # forward nudge
    out = m.update(**_mk(cfg=cfg, bay_type='parallel', robot_xy=(0.0, -0.62),
                         can_start=False))                            # reach -> PARKED
    assert m.state == 'PARKED' and out == (0.0, 0.0)


def test_parallel_back1_timeout_parks():
    m = ParkingManeuver()
    cfg = _cfg(enter_timeout_sec=5.0, parallel_back1_dist=99.0)       # never reached
    _drive_to_parallel_back1(m, cfg)
    assert m.state == 'PARALLEL_BACK1'
    out = m.update(**_mk(cfg=cfg, now=6.0, bay_type='parallel',
                         robot_xy=(0.0, -0.01), can_start=False))
    assert m.state == 'PARKED' and out == (0.0, 0.0)


def test_disarm_aborts_from_parallel():
    m = ParkingManeuver()
    cfg = _cfg()
    _drive_to_parallel_back1(m, cfg)
    assert m.state == 'PARALLEL_BACK1'
    assert m.update(**_mk(cfg=cfg, armed=False, bay_type='parallel',
                          can_start=False)) is None
    assert m.state == 'IDLE' and not m.active
