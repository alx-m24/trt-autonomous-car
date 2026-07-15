"""parking_maneuver.py — the parking maneuver as a pure, ROS-free state machine.

Extracted from control_node so the behaviour can be unit-tested offline (no
rclpy, no clock, no /cmd_vel) and reasoned about in isolation. control_node
remains the single owner of /cmd_vel: each tick it feeds this class the current
inputs and publishes whatever command it returns.

Strategy (sensors = IMU + wheel encoders + front camera only; Ackermann car
that cannot turn in place): vision closed-loop ONLY during APPROACH while the
bay is in frame; the ENTER nose-in is open-loop dead-reckoned on wheel-encoder
distance, because the front camera is blind to the bay once we drive into it.

    IDLE     : not parking — yields to lane following (update returns None).
    APPROACH : vision closed-loop lineup (steer to the bay-centre bearing).
    COMMIT   : latch odom pose; branch on bay_type (one transient tick).
    ENTER    : PERPENDICULAR entry — open-loop constant-curvature arc (or straight
               nose-in) onto the bay entry heading, terminated on odom distance.
    PARALLEL_BACK1/BACK2 : PARALLEL entry — two opposite-lock REVERSE arcs
               (dead-reckoned on odom arc length): back1 swings the tail into the
               slot, back2 counter-steers to straighten. R_min enforced in
               software (the planar_move plugin is holonomic and won't enforce it).
    PARALLEL_ADJUST : optional forward nudge to centre in the slot (skipped if 0).
    PARKED   : terminal stop (holds until disarmed).

Bay-type branch: perpendicular is the proven forward nose-in; parallel needs
reverse (an Ackermann car cannot enter a parallel bay nose-in). Both entries are
open-loop dead-reckoned because the front camera is blind once the car is in the
slot; segments terminate on real odom distance (wheel encoders) or a timeout.

The class is frame/units agnostic: `dist` is compared to `commit_dist_m` in the
detector's units, while `enter_distance_m` / `parallel_*_dist` / `robot_xy` are
real odom metres. `now` is float seconds. `target` is (x, y, yaw) in
base_footprint, yaw = bay entry heading. `bay_type` is 'parallel' |
'perpendicular' (from /parking/bay_type). Actuation returns (linear_x, angular_z).
"""

import math
from dataclasses import dataclass


@dataclass
class ParkingConfig:
    """Live-tunable maneuver parameters (control_node owns the ROS params)."""
    commit_dist_m: float          # /parking/distance_m at which to commit
    approach_speed: float         # m/s while lining up
    approach_kp: float            # bearing (rad) -> yaw rate
    enter_speed: float            # m/s nosing in
    enter_distance_m: float       # real-odom-metre arc length of the entry
    enter_use_arc: bool           # curve to align (True) vs fixed steer (False)
    min_turn_radius_m: float      # Ackermann R_min; entry curvature clamp
    enter_steer: float            # fixed yaw rate when enter_use_arc is False
    enter_timeout_sec: float      # hard cap on the blind entry
    max_yaw_rate: float           # steering clamp (<=0 disables)
    # Off-track guard: |bearing to bay| (rad) that aborts APPROACH back to
    # lane-follow, so a false detection can't steer the car off the racing
    # line toward a phantom bay. <=0 disables.
    approach_abort_bearing: float
    # ── PARALLEL parking (reverse). Defaulted so existing callers/tests that
    # only do perpendicular keep working unchanged. All *_dist are real odom
    # metres (arc length); tune in sim. ──
    parallel_speed: float = 0.12        # reverse speed magnitude (m/s)
    parallel_kappa: float = 2.857       # arc curvature 1/m (clamped to 1/R_min)
    parallel_first_sign: float = 1.0    # +1/-1: which way to steer back1 (slot side)
    parallel_back1_dist: float = 0.28   # odom arc length of reverse arc 1
    parallel_back2_dist: float = 0.28   # odom arc length of reverse arc 2 (straighten)
    parallel_adjust_dist: float = 0.0   # forward centring nudge (0 = skip)


class ParkingManeuver:
    STATES = ('IDLE', 'APPROACH', 'COMMIT', 'ENTER',
              'PARALLEL_BACK1', 'PARALLEL_BACK2', 'PARALLEL_ADJUST', 'PARKED')

    def __init__(self, log=None):
        self._log = log if log is not None else (lambda _msg: None)
        self.reset()

    def reset(self):
        self.state = 'IDLE'
        self._enter_xy0 = None
        self._enter_start = None
        self._enter_kappa = 0.0
        self._seg_xy0 = None      # start-of-segment odom xy (parallel arcs)
        self._seg_start = None    # start-of-segment time (per-segment timeout)

    def _seg_travelled(self, robot_xy):
        """Odom distance since the current segment began."""
        if robot_xy is None or self._seg_xy0 is None:
            return 0.0
        return math.hypot(robot_xy[0] - self._seg_xy0[0],
                          robot_xy[1] - self._seg_xy0[1])

    def _begin_segment(self, robot_xy, now):
        self._seg_xy0 = robot_xy
        self._seg_start = now

    @property
    def active(self) -> bool:
        """True when the maneuver owns the actuator (any non-IDLE state)."""
        return self.state != 'IDLE'

    def update(self, *, now, cfg, armed, detected, target, target_fresh,
               dist, robot_xy, can_start, bay_type='perpendicular'):
        """Advance one tick and return (linear_x, angular_z), or None if IDLE.

        `can_start` gates the IDLE->APPROACH transition so the maneuver never
        interrupts another controller (e.g. a corner turn) mid-action.
        `bay_type` ('parallel'|'perpendicular') selects the entry at COMMIT.
        """
        # Disarm aborts any phase (PARKED included) back to IDLE.
        if not armed and self.state != 'IDLE':
            self._log('[park] disarmed -> IDLE')
            self.reset()
            return None

        if self.state == 'IDLE':
            # Off-track guard also gates ENTRY: never even start APPROACH toward a
            # bay that is already off-nose. Without this, a persistent off-nose
            # (false) detection oscillates IDLE<->APPROACH and each entry tick
            # still emits one steer-toward-phantom command before the next tick
            # aborts — a ~50% duty-cycle steer off-track. Refusing entry stops any
            # bad command being published at all.
            if (armed and detected and target_fresh and can_start
                    and not self._off_nose(cfg, target)):
                self.state = 'APPROACH'
                self._log(f'[park] APPROACH dist={dist:.2f} '
                          f'entry_heading={math.degrees(target[2]):.0f}deg')
            else:
                return None

        elif self.state == 'APPROACH':
            if not target_fresh:
                self._log('[park] target lost -> IDLE')
                self.reset()
                return None
            # Off-track guard: if the bay bearing swings too far off the nose,
            # this is almost certainly a bad/false detection (a real bay we are
            # lining up on stays roughly ahead). Bail to lane-follow rather than
            # steer the car off the racing line toward a phantom.
            if self._off_nose(cfg, target):
                tx, ty, _ = target
                self._log(f'[park] bearing {math.degrees(math.atan2(ty, max(1e-3, tx))):.0f}deg '
                          f'exceeds abort limit -> IDLE (off-track guard)')
                self.reset()
                return None
            if 0.0 <= dist <= cfg.commit_dist_m:
                self.state = 'COMMIT'

        elif self.state == 'COMMIT':
            if robot_xy is None:
                self._log('[park] no odom at COMMIT — aborting')
                self.reset()
                return None
            # PARALLEL: an Ackermann car can't nose into a parallel bay — hand off
            # to the reverse-arc sequence instead of the forward ENTER.
            if str(bay_type).lower() == 'parallel':
                self._begin_segment(robot_xy, now)
                self.state = 'PARALLEL_BACK1'
                self._log(
                    f'[park] PARALLEL from ({robot_xy[0]:.2f},{robot_xy[1]:.2f}) '
                    f'BACK1 rev-arc dist={cfg.parallel_back1_dist:.2f}m '
                    f'sign={cfg.parallel_first_sign:+.0f}')
                return self._command(cfg, target, target_fresh)
            self._enter_xy0 = robot_xy
            self._enter_start = now
            # Sweep the heading error dpsi (bay entry heading) over the arc
            # length s: kappa = dpsi / s. Driven as a fixed steering angle and
            # stopped on odom distance, the achieved heading change is kappa*s =
            # dpsi regardless of the plugin's speed quirk. Clamp to 1/R_min; if
            # clamped the car cannot fully align in one arc (a two-arc /
            # Reeds-Shepp planner would be needed).
            dpsi = target[2] if target is not None else 0.0
            s = max(1e-3, cfg.enter_distance_m)
            kappa_max = 1.0 / max(1e-3, cfg.min_turn_radius_m)
            kappa = dpsi / s
            clamped = abs(kappa) > kappa_max
            self._enter_kappa = max(-kappa_max, min(kappa_max, kappa))
            self.state = 'ENTER'
            self._log(
                f'[park] ENTER from ({robot_xy[0]:.2f},{robot_xy[1]:.2f}) '
                f'dist_goal={cfg.enter_distance_m:.2f}m dpsi={math.degrees(dpsi):.0f}deg '
                f'kappa={self._enter_kappa:.2f}'
                f'{" (CLAMPED: bay needs a tighter turn than R_min)" if clamped else ""}')

        elif self.state == 'ENTER':
            travelled = 0.0
            if robot_xy is not None and self._enter_xy0 is not None:
                travelled = math.hypot(robot_xy[0] - self._enter_xy0[0],
                                       robot_xy[1] - self._enter_xy0[1])
            elapsed = (now - self._enter_start) if self._enter_start is not None else 0.0
            if travelled >= cfg.enter_distance_m:
                self.state = 'PARKED'
                self._log(f'[park] PARKED (travelled {travelled:.2f}m)')
            elif elapsed >= cfg.enter_timeout_sec:
                self.state = 'PARKED'
                self._log('[park] ENTER timeout -> PARKED (stop)')

        elif self.state == 'PARALLEL_BACK1':
            # Reverse arc 1: tail swings into the slot. Terminate on odom arc length.
            if self._seg_travelled(robot_xy) >= cfg.parallel_back1_dist:
                self._begin_segment(robot_xy, now)
                self.state = 'PARALLEL_BACK2'
                self._log(f'[park] BACK2 counter-arc dist={cfg.parallel_back2_dist:.2f}m')
            elif self._seg_timed_out(now, cfg):
                self.state = 'PARKED'
                self._log('[park] BACK1 timeout -> PARKED (stop)')

        elif self.state == 'PARALLEL_BACK2':
            # Reverse arc 2: counter-steer to straighten parallel in the slot.
            if self._seg_travelled(robot_xy) >= cfg.parallel_back2_dist:
                if cfg.parallel_adjust_dist > 0.0:
                    self._begin_segment(robot_xy, now)
                    self.state = 'PARALLEL_ADJUST'
                    self._log(f'[park] ADJUST fwd centre dist={cfg.parallel_adjust_dist:.2f}m')
                else:
                    self.state = 'PARKED'
                    self._log('[park] PARKED (parallel, no adjust)')
            elif self._seg_timed_out(now, cfg):
                self.state = 'PARKED'
                self._log('[park] BACK2 timeout -> PARKED (stop)')

        elif self.state == 'PARALLEL_ADJUST':
            # Small forward straight nudge to centre longitudinally in the slot.
            if (self._seg_travelled(robot_xy) >= cfg.parallel_adjust_dist
                    or self._seg_timed_out(now, cfg)):
                self.state = 'PARKED'
                self._log('[park] PARKED (parallel)')

        return self._command(cfg, target, target_fresh)

    def _seg_timed_out(self, now, cfg):
        return (self._seg_start is not None
                and (now - self._seg_start) >= cfg.enter_timeout_sec)

    def _parallel_kappa(self, cfg):
        """Curvature magnitude for the parallel arcs, clamped to 1/R_min."""
        kappa_max = 1.0 / max(1e-3, cfg.min_turn_radius_m)
        return min(abs(cfg.parallel_kappa), kappa_max)

    def _off_nose(self, cfg, target):
        """True if the bay bearing exceeds the off-track abort limit.

        Shared by the IDLE->APPROACH entry gate and the in-APPROACH guard so a
        bad detection can neither start nor sustain a steer toward a phantom bay.
        Disabled (always False) when approach_abort_bearing <= 0.
        """
        if cfg.approach_abort_bearing <= 0.0 or target is None:
            return False
        tx, ty, _ = target
        return abs(math.atan2(ty, max(1e-3, tx))) > cfg.approach_abort_bearing

    def _command(self, cfg, target, target_fresh):
        """Actuation for the current (post-transition) state."""
        if self.state == 'APPROACH':
            wz = 0.0
            if target_fresh and target is not None:
                tx, ty, _ = target
                bearing = math.atan2(ty, max(1e-3, tx))     # +y is left
                wz = cfg.approach_kp * bearing
                if cfg.max_yaw_rate > 0.0:
                    wz = max(-cfg.max_yaw_rate, min(cfg.max_yaw_rate, wz))
            return (cfg.approach_speed, wz)

        if self.state == 'ENTER':
            # angular.z = v * kappa => the Ackermann plugin picks steering angle
            # atan(L*kappa), independent of the actual (halved) speed.
            wz = (cfg.enter_speed * self._enter_kappa if cfg.enter_use_arc
                  else cfg.enter_steer)
            return (cfg.enter_speed, wz)

        if self.state in ('PARALLEL_BACK1', 'PARALLEL_BACK2'):
            # Reverse (linear.x < 0). angular.z = |v| * kappa with kappa clamped to
            # 1/R_min so the (holonomic) plugin still respects the Ackermann turn
            # limit. back2 uses the opposite lock to straighten.
            kappa = self._parallel_kappa(cfg)
            sign = cfg.parallel_first_sign if self.state == 'PARALLEL_BACK1' \
                else -cfg.parallel_first_sign
            wz = sign * cfg.parallel_speed * kappa
            if cfg.max_yaw_rate > 0.0:
                wz = max(-cfg.max_yaw_rate, min(cfg.max_yaw_rate, wz))
            return (-cfg.parallel_speed, wz)

        if self.state == 'PARALLEL_ADJUST':
            return (cfg.parallel_speed, 0.0)

        # COMMIT (transient) or PARKED — stop.
        return (0.0, 0.0)
