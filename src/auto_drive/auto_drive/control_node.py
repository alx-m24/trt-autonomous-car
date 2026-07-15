import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float32, String, Bool
from nav_msgs.msg import Odometry, Path
from gazebo_msgs.srv import SetEntityState, SetModelState
from gazebo_msgs.msg import EntityState, ModelState, ModelStates
from auto_drive.parking_maneuver import ParkingManeuver, ParkingConfig
import math
import numpy as np
import sys


class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')

        self.create_subscription(Float32, '/lane_error', self.error_cb, 10)
        self.create_subscription(Path, '/lane_path', self.path_cb, 10)
        self.create_subscription(Bool, '/lane_detected', self.lane_detected_cb, 10)

        # ── UPGRADED: Added the missing proactive braking scene subscriber ──
        self.create_subscription(String, '/scene_state', self.scene_cb, 10)

        # ── NEW: anticipatory corner cues from perception ──
        self.create_subscription(String,  '/corner/direction',  self.corner_dir_cb,  10)
        self.create_subscription(Float32, '/corner/distance_m', self.corner_dist_cb, 10)
        self.create_subscription(Float32, '/corner/angle_deg',  self.corner_ang_cb,  10)
        # ── NEW: EKF-fused odom for precise dead-reckoned turn termination ──
        self.create_subscription(Odometry, '/odometry/filtered', self.filtered_cb, 10)

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # DIAG (2026-07-07, turning-radius investigation): last commanded twist,
        # for comparing against what Gazebo's planar_move actually realizes.
        self.last_cmd_linear_x = 0.0
        self.last_cmd_angular_z = 0.0

        # ── NEW: parking auto-arm + detection awareness ──
        # control_node arms the GATED parking detector by zone and listens to its
        # result. Detection-only for now — the PARKING maneuver hook is marked in
        # _parking_arm_tick but not yet wired (control still owns /cmd_vel).
        self.parking_arm_pub = self.create_publisher(Bool, '/mission/parking_active', 10)
        # The maneuver branches on bay_type at COMMIT: perpendicular = forward
        # nose-in, parallel = reverse-arc sequence (an Ackermann car can't nose
        # into a parallel bay). So /parking/bay_type IS a control input again.
        self.create_subscription(Bool,        '/parking/detected',   self.parking_detected_cb, 10)
        self.create_subscription(PoseStamped, '/parking/target',     self.parking_target_cb,   10)
        self.create_subscription(Float32,     '/parking/distance_m', self.parking_dist_cb,     10)
        self.create_subscription(String,      '/parking/bay_type',   self.parking_bay_type_cb, 10)

        # ── Parameters (tune without rebuilding) ──────────────────────────────
        self.declare_parameter('max_steer', 1.0)               
        self.declare_parameter('max_error_deg', 45.0)          
        self.declare_parameter('wheelbase', 0.24)              

        # ── UPGRADED: PID values adjusted to kill the "wonky" oscillations ──
        self.declare_parameter('kp', 0.018)                    
        self.declare_parameter('kd_multiplier', 0.40)          
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('integral_max', 20.0)

        self.declare_parameter('alpha', 0.80)                  
        self.declare_parameter('alpha_min', 1.0)              
        self.declare_parameter('alpha_max', 1.0)               
        self.declare_parameter('beta', 0.85)                    
        self.declare_parameter('deadband_deg', 1.5)

        self.declare_parameter('error_sign', -1.0)
        self.declare_parameter('max_yaw_rate', 5.0)
        # Yaw-rate actuation correction (2026-07-07, MEASURED not guessed): see
        # the full derivation in _pub()'s docstring. Gazebo's planar_move here
        # realizes only ~65.8% of commanded angular.z (linear.x ~100%), flat
        # across magnitude and motion type -- a scale bug, not friction. This
        # gain (1/0.658) is applied once, at the final publish point, to every
        # cmd_vel regardless of which code path built it.
        self.declare_parameter('angular_cmd_gain', 1.520)

        # Pure Pursuit 
        self.declare_parameter('use_pure_pursuit', True)
        
        # ── UPGRADED: Lookahead pushed deeper to see the curve ──
        self.declare_parameter('pp_lookahead_distance', 0.45)   # 2026-07-02: lowered from 0.65 for tighter/centred tracking
        self.declare_parameter('pp_lookahead_turn_m', 0.80)  # metric re-base: real path tops out ~1.2m
        # 0.80 -> 0.50 (2026-07-03): the long curve-lookahead reached past the
        # on-road near path into the fit's extrapolated far end on a straight
        # perception briefly mislabelled 'curve' -> ~32deg point at the 16deg gate
        # boundary -> pass/reject flicker = the wag that drove the car off a road
        # the magenta line was still on. See control_params.yaml for the full note.
        # 2026-07-07: 0.60 -> 0.50. A 0.60m reach lands the lookahead deep around
        # the sweeping bend (R ~0.4-0.5m) -> car turns in ~0.6m early and clips
        # the inside. See control_params.yaml for the full history of this knob.
        self.declare_parameter('pp_lookahead_curve_m', 0.48)
        
        self.declare_parameter('pp_min_speed', 0.10)
        self.declare_parameter('pp_max_speed', 0.25)
        self.declare_parameter('pp_curvature_slowdown', 1.40)
        # Curvature FEEDFORWARD blend [0..1]. Single-point pure pursuit commands
        # kappa = 2y/ld^2 from ONE lookahead point; on a constant-curvature arc
        # that chord always UNDERSHOOTS the arc's real curvature, so the car runs
        # wide / under-turns (odom-measured realized R~2.8m on a much tighter road
        # arc, 2026-07-06). This blends in the LIVE curvature fitted from the
        # /lane_path centreline so the car turns at the road's actual rate.
        # Dynamic (from the live fit), NOT a preset arc; 0 = plain pure pursuit,
        # 1 = follow the fitted curvature only (loses pure-pursuit centering).
        #   car still runs wide on arcs -> RAISE toward 0.8.
        #   arcs feel twitchy / over-turn or wobble -> LOWER toward 0.4.
        self.declare_parameter('pp_curvature_ff_gain', 0.3)
        # Forward reach (m) of the feedforward's near-band curvature fit. The
        # metric-BEV path only STARTS at x~0.24m, so the original 0.45 cap left
        # <4 poses in-band and the fit returned None — kff read 'na' on every
        # frame and the feedforward silently never engaged (2026-07-07 log).
        # 0.60 gives ~0.36m of extent while staying short of the 0.75 that fed
        # an upcoming corner's curvature forward too early (2026-07-06 lesson).
        #   kff still 'na' on a clean curve -> RAISE a notch (0.65)
        #   car turns in early before bends  -> LOWER toward 0.50
        self.declare_parameter('pp_ff_band_max_x_m', 0.60)
        # Near-field centering (2026-07-07, the "keep the near road centered"
        # term): pure pursuit aims at ONE far point, so mid-bend it tolerates a
        # standing lateral offset (the inside drift). This adds a curvature
        # correction proportional to the lateral offset of the NEAR path (the
        # bottom of the BEV, x <= pp_center_band_max_x_m) so the car is pushed
        # back over the centreline continuously. Correction is clamped to
        # +/- pp_center_max_kappa. 0 gain disables.
        #   still drifts inside on bends -> RAISE gain toward 2.5
        #   weaves/oscillates on straights -> LOWER gain toward 0.8
        self.declare_parameter('pp_center_gain', 1.5)
        self.declare_parameter('pp_center_band_max_x_m', 0.50)
        self.declare_parameter('pp_center_max_kappa', 0.60)
        # Scene-flap debounce (2026-07-07): mid-bend, perception can still flap
        # curve<->roundabout; each flap instantly swaps the lookahead
        # 0.50<->0.80m and near-doubles commanded curvature (log-proven kappa
        # -0.27 -> -0.53 on consecutive frames). Only escalate to
        # pp_lookahead_turn_m (and the turn hard-brake) after this many
        # CONSECUTIVE turn/roundabout scene messages; unconfirmed flap frames
        # drive as 'curve'. 1 = old instant behaviour.
        self.declare_parameter('pp_turn_scene_confirm_frames', 5)
        self.declare_parameter('pp_path_timeout_sec', 0.15)
        # Lookahead-point plausibility gate: a followable road centreline point
        # cannot be wildly off to the side. When perception's fit rails to the BEV
        # edge (|y| -> 2.5m = clamped image edge), the point's heading exceeds any
        # sane steering direction; reject it and steer to the nearest plausible
        # (near, both-edge-supported) point instead. Data 2026-07-03: healthy
        # |y|<=0.35 at x~1.2; railed |y|>=1.09 -> gate cuts cleanly between them.
        #   car still drifts off on a railing curve -> LOWER heading_deg / floor_m.
        #   clips a genuine tight-but-followable curve -> RAISE heading_deg.
        self.declare_parameter('pp_point_max_heading_deg', 16.0)
        self.declare_parameter('pp_point_lateral_floor_m', 0.30)

        # Degenerate-path guard: perception sometimes publishes a RAIL path — a
        # handful of far-only poses all leaning to one side, with NO near-field
        # support (confirmed 2026-07-03 via /lane_path echo on a straight: 2 poses
        # at x=1.21/1.51, both y~0.4 left, nothing near). Control has no reliable
        # near point to lock onto and steers hard toward the far-left rail = off a
        # straight road. When a path has fewer than pp_min_path_points valid poses,
        # OR no pose within pp_near_point_max_x_m ahead, don't chase it: decay the
        # previous steer toward straight and coast at reduced speed until a real
        # (near-supported) path returns.
        #   coasts straight THROUGH a real tight bend it should take -> RAISE
        #       pp_near_point_max_x_m (accept a slightly-farther near point) or
        #       LOWER pp_min_path_points.
        #   still gets yanked by a sparse far path -> LOWER pp_near_point_max_x_m.
        self.declare_parameter('pp_min_path_points', 3)
        self.declare_parameter('pp_near_point_max_x_m', 0.70)

        # Straight-road steering hold (zero steer when path is vertically aligned)
        self.declare_parameter('straight_y_tol_m', 0.03)
        self.declare_parameter('straight_min_points', 6)
        self.declare_parameter('straight_line_tol_m', 0.1)
        self.declare_parameter('straight_min_ratio', 0.25)
        self.declare_parameter('straight_full_ratio', 0.80)
        self.declare_parameter('straight_hold_frames', 6)
        self.declare_parameter('straight_bias_gain', 0.6)
        self.declare_parameter('straight_bias_max', 1.5)
        self.declare_parameter('straight_angle_tol_deg', 6.0)

        # ── Boundary repulsion (APF) ───────────────────────────────────────
        # Pushes the pure-pursuit target away from the track edge. The center
        # band must stay APF-free or the repulsion fires in pulses as the car
        # crosses the deadband edge -> relay limit cycle (the residual straight
        # swerve). Keep apf_safe_margin comfortably above normal lateral wander.
        # apf_track_half_width : lateral distance (m) from centre to track edge.
        # apf_safe_margin      : |y| below this gets NO repulsion (the free band).
        # apf_repulsion_gain   : strength of the exponential edge push.
        self.declare_parameter('apf_track_half_width', 0.15)
        self.declare_parameter('apf_safe_margin', 0.06)
        # APF DISABLED by default (2026-07-02): its metre thresholds (track_half_width
        # 0.15 / safe_margin 0.06) don't match the ~8x-inflated bird_m_per_px_x y-scale,
        # so |y| crosses the edge easily -> dist_to_edge clamps to 0.01 -> repulsion
        # detonates (~8.0) and flings the target the other way = offroad thrash even on
        # a straight. Pure pursuit on the smoothed centreline drives straights, curves
        # AND corners cleanly on its own. Re-enable only after recalibrating to the
        # real y-scale.
        self.declare_parameter('apf_repulsion_gain', 0.0)
        self.declare_parameter('apf_repulsion_gain_straight', 0.0)

        # ── NEW: dead-reckoned 90deg corner maneuver ──────────────────────────
        # use_corner_maneuver  : master switch for the whole behavior layer.
        # corner_trigger_dist_m: start braking (APPROACH) when a corner is this
        #                        close ahead.
        # corner_commit_dist_m : commit to the dead-reckoned turn at this range
        #                        (or earlier if the path ahead is lost).
        # corner_confirm_frames: stable corner detections required before acting.
        # turn_speed           : forward speed held through the arc (m/s).
        # turn_radius_m        : arc radius. Default 0.30m ~= the car's min turn
        #                        radius (L/tan(steer_max)=0.24/tan(0.6)~0.35m) and
        #                        ~the lane width, so a 90deg corner fits the track.
        # turn_angle_deg       : fallback turn angle if perception doesn't supply.
        # turn_angle_tol_deg   : stop the turn this many deg before target.
        # turn_timeout_sec     : hard safety cap on a single maneuver.
        # recover_stable_frames: stable lane frames before handing back to PP.
        # ENABLED by default (2026-07-02): now gated by the multi-band SHARP-corner
        # discriminator in perception (fires only on a real corner cliff, not curves),
        # so it no longer false-fires the way the plain reach cue did. Pure pursuit
        # still drives straights/curves; the maneuver only takes the sharp corners.
        self.declare_parameter('use_corner_maneuver', True)
        # Close the dead-reckoned corner turn on raw /odom yaw (fresh & reliable in
        # sim) instead of the EKF /odometry/filtered (which needs use_sim_time and
        # goes stale, stalling the turn mid-arc). Set False to use the EKF.
        self.declare_parameter('corner_yaw_use_odom', True)
        # corner_trigger_dist_m: UNUSED with the multi-band cue (APPROACH triggers on
        # the sharp detection itself). corner_commit_dist_m: commit the turn when the
        # camera's centre-MID proximity ramps below this (0..0.35; lower = commit later
        # / closer to the corner).
        self.declare_parameter('corner_trigger_dist_m', 0.40)
        self.declare_parameter('corner_commit_dist_m', 0.05)   # (legacy vision-commit; unused now that commit is odom-distance)
        # Commit the turn after the car has DRIVEN this far (m, odom) past the point of
        # detection — consistent timing, unlike the vision collapse. turns too EARLY
        # (before the corner) -> RAISE; too LATE (overshoots) -> LOWER.
        # Geometry-derived (my_car.urdf): cam is 0.13m up, 0.13m ahead of the pivot
        # (base_footprint; planar_move rotates about it), 20deg pitch, ~64deg VFOV. The
        # mid-band detection lands ~0.5m ahead of the cam = ~0.65m ahead of the pivot;
        # minus ~0.1m for the leading front axle -> ~0.55m of travel to reach the apex.
        self.declare_parameter('corner_commit_travel_m', 0.48)
        self.declare_parameter('corner_confirm_frames', 3)
        # Straight dead-reckoned leg driven between CHAINED back-to-back turns
        # (the S-jog gap between corner 1's exit and corner 2's pivot).
        self.declare_parameter('corner_chain_gap_m', 0.30)
        self.declare_parameter('turn_speed', 0.10)
        self.declare_parameter('turn_radius_m', 0.18)
        self.declare_parameter('turn_angle_deg', 90.0)
        self.declare_parameter('turn_angle_tol_deg', 6.0)
        # Angle-agnostic turn termination: stop EXECUTE_TURN the instant
        # perception re-acquires road straight ahead (corner_dir=='none'),
        # provided we've rotated at least turn_min_angle_deg. turn_angle_deg
        # (90) becomes a FALLBACK for when the road never re-acquires cleanly.
        # turn_max_angle_deg is a hard cap so a bad cue can't spin us forever.
        # This self-corrects for skewed entries / non-90 bends (was overshoot).
        self.declare_parameter('turn_min_angle_deg', 80.0)   # 2026-07-02: was 35 -> turn stopped short (under-turn)
        self.declare_parameter('turn_max_angle_deg', 120.0)
        self.declare_parameter('turn_timeout_sec', 6.0)
        self.declare_parameter('recover_stable_frames', 5)
        # Tracked-bend abort (2026-07-07): the reach cue's cliff classifier fires
        # on a large-radius SWEEPING bend too (centre-mid drains laterally, just
        # slower), and the fixed turn_radius_m pivot then cuts to the inside of
        # the arc. Discriminator: on a sweeping bend pure pursuit is ALREADY
        # steering through it (|kcmd| ramps to ~1.0 over seconds, healthy path),
        # while a true 90 is approached STRAIGHT (kcmd ~0 until the road ends).
        # If the smoothed commanded curvature already exceeds this (1/m) in the
        # SAME direction as the pending turn, refuse to arm/commit and let pure
        # pursuit keep the bend. <=0 disables the gate.
        #   still pivots inside a sweeping bend -> LOWER (gate sooner)
        #   real corner suppressed (check CORNER_BEND log at that moment) -> RAISE
        self.declare_parameter('corner_bend_abort_kappa', 0.50)

        # ── NEW: parking auto-arm (zone-based) ─────────────────────────────────
        # parking_auto_arm        : master switch for auto-arming the detector.
        # parking_zone_{x,y}      : zone centre in the /odom frame (metres). MUST
        #                           be set to the real parking-area location; the
        #                           (0,0) default + the travel guard below means
        #                           it will not self-arm out of the box.
        # parking_zone_radius_m   : arm when the robot is within this radius.
        # parking_arm_min_travel_m: don't arm until the car has driven this far
        #                           from spawn (stops self-arming at the origin).
        # parking_arm_latch       : once armed, stay armed (keep scanning) even
        #                           after leaving the zone.
        # park_search_speed       : lane-follow HAND-OFF creep. While armed and
        #                           inside the zone but the maneuver hasn't taken
        #                           over yet, cap the lane-follow speed to this so
        #                           the car doesn't blow past the pocket before
        #                           detection locks (close-range + ~1.8 s vote).
        #                           <=0 disables the creep (full-speed search).
        self.declare_parameter('parking_auto_arm', True)
        self.declare_parameter('parking_zone_x', 0.0)
        self.declare_parameter('parking_zone_y', 0.0)
        self.declare_parameter('parking_zone_radius_m', 1.0)
        self.declare_parameter('parking_arm_min_travel_m', 2.0)
        self.declare_parameter('parking_arm_latch', True)
        self.declare_parameter('park_search_speed', 0.12)

        # ── NEW: parking maneuver skeleton (APPROACH -> COMMIT -> ENTER) ───────
        # Strategy for IMU + wheel-encoders + front-cam-only Ackermann car: vision
        # closed-loop ONLY during APPROACH (bay still in frame); the ENTER nose-in
        # is open-loop dead-reckoned on odom distance, because the front camera
        # can't see the bay once we start driving into it. Values are starting
        # stubs — tune on the real maneuver.
        #   park_enable          : master switch for the maneuver layer.
        #   park_commit_dist_m   : /parking/distance_m at which to stop steering
        #                          on vision and commit to the blind entry.
        #   park_approach_speed  : forward speed while lining up (m/s).
        #   park_approach_kp     : P-gain, bearing-to-bay (rad) -> yaw rate.
        #   park_enter_speed     : forward speed nosing into the bay (m/s).
        #   park_enter_distance_m: open-loop ARC LENGTH to drive into the bay.
        #   park_enter_use_arc   : if true, curve during entry so the car finishes
        #                          aligned with the bay entry heading (single
        #                          constant-curvature arc); if false, drive the
        #                          fixed park_enter_steer instead. DEFAULT FALSE:
        #                          the straight nose-in (steer=0) is the only
        #                          entry CONFIRMED to park the car on the live sim;
        #                          the arc is unproven opt-in (see PARKING_PLAN.md).
        #   park_min_turn_radius_m: kinematic min turn radius (Ackermann). The
        #                          entry curvature is clamped to 1/this; if the
        #                          bay demands a tighter turn the car can't fully
        #                          align in one arc (a two-arc/Reeds-Shepp planner
        #                          would be needed — see TODO in _update_...).
        #   park_enter_steer     : fixed yaw rate during entry when use_arc=false.
        #   park_enter_timeout_sec: hard cap on the blind entry.
        #   park_target_timeout_sec: treat /parking/target older than this as lost.
        #                          1.5 (was 0.5): the detector loses the bay for a
        #                          frame or two mid-APPROACH (close-range, sparse
        #                          detection); 0.5 aborted APPROACH on the gap.
        #                          1.5 rides through it — CONFIRMED to park the car
        #                          on the live teleport-to-approach run (2026-07-01).
        #   park_approach_abort_deg: OFF-TRACK GUARD. If the bearing to the bay
        #                          swings beyond this (deg) during APPROACH, treat
        #                          it as a bad/false detection and bail back to
        #                          lane-follow rather than steer off the racing
        #                          line toward a phantom bay. <=0 disables.
        # NOTE on units: park_commit_dist_m is compared against /parking/distance_m,
        # which is in the detector's FICTITIOUS BEV metre (bird_m_per_px=0.01 over a
        # ~10 m-wide BEV), NOT real metres. Measured on the live approach: the bay
        # is first reliably detected at dist≈7.1 and falls to ≈5.7 as the car noses
        # in, so commit at 6.8. By contrast park_enter_distance_m IS real odom metres
        # (wheel encoders): the car is inside the pocket ≈0.25–0.30 m past commit.
        # Defaults below were CONFIRMED on the live sim (2026-07-01): the car
        # detects the pocket, commits, dead-reckons 0.25 m in, and PARKS inside
        # (BEV ~78% dark at the final pose). Note the Ackermann plugin delivers
        # ~half the commanded speed at low speed, hence the higher enter speed +
        # 10 s timeout so the maneuver terminates on the 0.25 m goal, not timeout.
        self.declare_parameter('park_enable', True)
        self.declare_parameter('park_commit_dist_m', 6.7)     # fictitious BEV metre
        self.declare_parameter('park_approach_speed', 0.10)
        self.declare_parameter('park_approach_kp', 1.5)
        self.declare_parameter('park_enter_speed', 0.12)
        self.declare_parameter('park_enter_distance_m', 0.25)  # real odom metres
        self.declare_parameter('park_enter_use_arc', False)  # proven straight nose-in
        self.declare_parameter('park_min_turn_radius_m', 0.35)
        self.declare_parameter('park_enter_steer', 0.0)
        self.declare_parameter('park_enter_timeout_sec', 10.0)
        self.declare_parameter('park_target_timeout_sec', 1.5)
        self.declare_parameter('park_approach_abort_deg', 45.0)  # off-track guard
        # PARALLEL parking (reverse-arc entry). All *_dist are real odom metres;
        # tune in sim. first_sign (+1/-1) sets which way back1 steers (slot side).
        self.declare_parameter('park_parallel_speed', 0.12)
        self.declare_parameter('park_parallel_kappa', 2.857)     # ~1/R_min
        self.declare_parameter('park_parallel_first_sign', 1.0)
        self.declare_parameter('park_parallel_back1_dist', 0.28)
        self.declare_parameter('park_parallel_back2_dist', 0.28)
        self.declare_parameter('park_parallel_adjust_dist', 0.0)  # 0 = skip nudge

        self.declare_parameter('base_speed', 0.10)
        self.declare_parameter('max_speed_reduction', 0.30)    
        self.declare_parameter('speed_reduction_gamma', 1.60)
        self.declare_parameter('yaw_ref_speed', 0.20)          

        self.max_steer = float(self.get_parameter('max_steer').value)
        self.max_error_deg = float(self.get_parameter('max_error_deg').value)
        self.wheelbase = float(self.get_parameter('wheelbase').value)

        self.Kp = float(self.get_parameter('kp').value)
        self.kd_multiplier = float(self.get_parameter('kd_multiplier').value)
        self.Kd = self.Kp * self.kd_multiplier
        self.Ki = float(self.get_parameter('ki').value)
        self.integral_max = float(self.get_parameter('integral_max').value)

        self.alpha = float(self.get_parameter('alpha').value)
        self.alpha_min = float(self.get_parameter('alpha_min').value)
        self.alpha_max = float(self.get_parameter('alpha_max').value)
        self.beta = float(self.get_parameter('beta').value)
        self.deadband_deg = float(self.get_parameter('deadband_deg').value)
        self.error_sign = float(self.get_parameter('error_sign').value)
        self.max_yaw_rate = float(self.get_parameter('max_yaw_rate').value)
        self.angular_cmd_gain = float(self.get_parameter('angular_cmd_gain').value)

        self.use_pure_pursuit = bool(self.get_parameter('use_pure_pursuit').value)
        self.pp_lookahead_distance = float(self.get_parameter('pp_lookahead_distance').value)
        self.pp_lookahead_turn_m = float(self.get_parameter('pp_lookahead_turn_m').value)
        self.pp_lookahead_curve_m = float(self.get_parameter('pp_lookahead_curve_m').value)
        self.pp_min_speed = float(self.get_parameter('pp_min_speed').value)
        self.pp_max_speed = float(self.get_parameter('pp_max_speed').value)
        self.pp_curvature_slowdown = float(self.get_parameter('pp_curvature_slowdown').value)
        self.pp_curvature_ff_gain = float(self.get_parameter('pp_curvature_ff_gain').value)
        self.pp_ff_band_max_x_m = float(self.get_parameter('pp_ff_band_max_x_m').value)
        self.pp_center_gain = float(self.get_parameter('pp_center_gain').value)
        self.pp_center_band_max_x_m = float(self.get_parameter('pp_center_band_max_x_m').value)
        self.pp_center_max_kappa = float(self.get_parameter('pp_center_max_kappa').value)
        self.pp_turn_scene_confirm_frames = int(self.get_parameter('pp_turn_scene_confirm_frames').value)
        self._kff_ema = 0.0              # EMA-smoothed feedforward curvature
        self._kff_alpha = 0.4            # EMA weight on the new frame (tames spikes)
        self.pp_point_max_heading_deg = float(self.get_parameter('pp_point_max_heading_deg').value)
        self.pp_point_lateral_floor_m = float(self.get_parameter('pp_point_lateral_floor_m').value)
        self.pp_min_path_points = int(self.get_parameter('pp_min_path_points').value)
        self.pp_near_point_max_x_m = float(self.get_parameter('pp_near_point_max_x_m').value)
        self.pp_path_timeout_sec = float(self.get_parameter('pp_path_timeout_sec').value)

        self.straight_y_tol_m = float(self.get_parameter('straight_y_tol_m').value)
        self.straight_min_points = int(self.get_parameter('straight_min_points').value)
        self.straight_line_tol_m = float(self.get_parameter('straight_line_tol_m').value)
        self.straight_min_ratio = float(self.get_parameter('straight_min_ratio').value)
        self.straight_full_ratio = float(self.get_parameter('straight_full_ratio').value)
        self.straight_hold_frames = int(self.get_parameter('straight_hold_frames').value)
        self.straight_bias_gain = float(self.get_parameter('straight_bias_gain').value)
        self.straight_bias_max = float(self.get_parameter('straight_bias_max').value)
        self.apf_track_half_width = float(self.get_parameter('apf_track_half_width').value)
        self.apf_safe_margin = float(self.get_parameter('apf_safe_margin').value)
        self.apf_repulsion_gain = float(self.get_parameter('apf_repulsion_gain').value)
        self.apf_repulsion_gain_straight = float(self.get_parameter('apf_repulsion_gain_straight').value)
        self.straight_angle_tol_deg = float(self.get_parameter('straight_angle_tol_deg').value)

        self.base_speed = float(self.get_parameter('base_speed').value)
        self.max_speed_reduction = float(self.get_parameter('max_speed_reduction').value)
        self.speed_reduction_gamma = float(self.get_parameter('speed_reduction_gamma').value)
        self.yaw_ref_speed = float(self.get_parameter('yaw_ref_speed').value)

        self.use_corner_maneuver = bool(self.get_parameter('use_corner_maneuver').value)
        self.corner_yaw_use_odom = bool(self.get_parameter('corner_yaw_use_odom').value)
        self.corner_trigger_dist_m = float(self.get_parameter('corner_trigger_dist_m').value)
        self.corner_commit_dist_m = float(self.get_parameter('corner_commit_dist_m').value)
        self.corner_confirm_frames = int(self.get_parameter('corner_confirm_frames').value)
        self.corner_commit_travel_m = float(self.get_parameter('corner_commit_travel_m').value)
        self.corner_chain_gap_m = float(self.get_parameter('corner_chain_gap_m').value)
        self.turn_speed = float(self.get_parameter('turn_speed').value)
        self.turn_radius_m = float(self.get_parameter('turn_radius_m').value)
        self.turn_angle_deg = float(self.get_parameter('turn_angle_deg').value)
        self.turn_angle_tol_deg = float(self.get_parameter('turn_angle_tol_deg').value)
        self.turn_min_angle_deg = float(self.get_parameter('turn_min_angle_deg').value)
        self.turn_max_angle_deg = float(self.get_parameter('turn_max_angle_deg').value)
        self.turn_timeout_sec = float(self.get_parameter('turn_timeout_sec').value)
        self.recover_stable_frames = int(self.get_parameter('recover_stable_frames').value)
        self.corner_bend_abort_kappa = float(self.get_parameter('corner_bend_abort_kappa').value)

        self.parking_auto_arm = bool(self.get_parameter('parking_auto_arm').value)
        self.parking_zone_x = float(self.get_parameter('parking_zone_x').value)
        self.parking_zone_y = float(self.get_parameter('parking_zone_y').value)
        self.parking_zone_radius_m = float(self.get_parameter('parking_zone_radius_m').value)
        self.parking_arm_min_travel_m = float(self.get_parameter('parking_arm_min_travel_m').value)
        self.parking_arm_latch = bool(self.get_parameter('parking_arm_latch').value)
        self.park_search_speed = float(self.get_parameter('park_search_speed').value)

        self.park_enable = bool(self.get_parameter('park_enable').value)
        self.park_commit_dist_m = float(self.get_parameter('park_commit_dist_m').value)
        self.park_approach_speed = float(self.get_parameter('park_approach_speed').value)
        self.park_approach_kp = float(self.get_parameter('park_approach_kp').value)
        self.park_enter_speed = float(self.get_parameter('park_enter_speed').value)
        self.park_enter_distance_m = float(self.get_parameter('park_enter_distance_m').value)
        self.park_enter_use_arc = bool(self.get_parameter('park_enter_use_arc').value)
        self.park_min_turn_radius_m = float(self.get_parameter('park_min_turn_radius_m').value)
        self.park_enter_steer = float(self.get_parameter('park_enter_steer').value)
        self.park_enter_timeout_sec = float(self.get_parameter('park_enter_timeout_sec').value)
        self.park_target_timeout_sec = float(self.get_parameter('park_target_timeout_sec').value)
        self.park_approach_abort_deg = float(self.get_parameter('park_approach_abort_deg').value)
        self.park_parallel_speed = float(self.get_parameter('park_parallel_speed').value)
        self.park_parallel_kappa = float(self.get_parameter('park_parallel_kappa').value)
        self.park_parallel_first_sign = float(self.get_parameter('park_parallel_first_sign').value)
        self.park_parallel_back1_dist = float(self.get_parameter('park_parallel_back1_dist').value)
        self.park_parallel_back2_dist = float(self.get_parameter('park_parallel_back2_dist').value)
        self.park_parallel_adjust_dist = float(self.get_parameter('park_parallel_adjust_dist').value)

        self.add_on_set_parameters_callback(self._param_cb)

        self.prev_error = 0.0
        self.integral = 0.0
        self.smoothed_error = 0.0
        self.prev_angular_z = 0.0

        # ── NEW: parking auto-arm state ──
        self.parking_armed = False
        self.parking_detected = False
        self.parking_in_zone = False   # currently within the parking zone radius
        self._spawn_xy = None     # first odom pose, for the travel-distance guard

        # ── NEW: parking maneuver ──
        # The maneuver state machine lives in ParkingManeuver (pure logic, unit-
        # tested offline); control_node stays the sole /cmd_vel owner and just
        # feeds it inputs each tick. We only hold the latest detector inputs here.
        self.park_target = None          # (x, y, yaw) in base_footprint, latest
        self.park_target_time = None
        self.park_dist = -1.0            # /parking/distance_m
        self.parking_bay_type = 'perpendicular'   # /parking/bay_type (COMMIT branch)
        self.parking = ParkingManeuver(log=self.get_logger().info)

        # Track the scene for proactive braking
        self.current_scene = 'straight'
        self.turn_scene_frames = 0        # consecutive turn/roundabout scene frames (debounce)

        # ── NEW: corner maneuver state machine ──
        # mode: LANE_FOLLOW -> APPROACH -> EXECUTE_TURN -> RECOVER -> LANE_FOLLOW
        self.mode = 'LANE_FOLLOW'
        self.corner_dir = 'none'          # latest cue from perception
        self.corner_dist = -1.0
        self.corner_angle = 0.0
        self.corner_confirm_count = 0
        self.recover_count = 0
        self.fused_yaw = None             # rad, from /odometry/filtered
        self.fused_yaw_time = None
        self.raw_yaw = None               # rad, from raw /odom (sim ground truth)
        self.raw_yaw_time = None
        self.turn_yaw0 = None             # latched heading at turn start
        self.turn_dir = 0                 # +1 = left (CCW), -1 = right (CW)
        self.turn_target = 0.0            # rad, magnitude to rotate
        self.approach_start_xy = None     # odom (x,y) captured at APPROACH entry, for travel-based commit
        self.turn_start_xy = None         # odom (x,y) captured at EXECUTE_TURN entry, for realized-radius diagnostic
        self.turn_start_time = None
        self.turn_expected_time = 0.0     # open-loop fallback when no yaw
        self.pp_kappa_ema = 0.0           # smoothed commanded curvature (tracked-bend gate)

        # Back-to-back corner handoff: on this track corner 2's clean detection
        # window can open WHILE corner 1's EXECUTE_TURN is still running (proven
        # live 2026-07-04 -- a right cue read mid_prox=0.06 at turned=54deg of a
        # LEFT turn, then vanished before RECOVER completed). EXECUTE_TURN never
        # consumes corner_dir for anything but the road-reacquired early-stop, so
        # that reading was silently discarded and the 2nd corner was missed
        # entirely. Fix: latch an OPPOSITE-direction, persistent corner cue seen
        # during EXECUTE_TURN, then act on it immediately at RECOVER instead of
        # falling back to plain LANE_FOLLOW and waiting for a fresh confirm.
        self.pending_corner_dir = None
        self.pending_corner_confirm = 0
        # Deepest (lowest) mid_prox the pending corner reached during the turn.
        # If it collapsed to commit level (<= corner_commit_dist_m) the car is
        # ALREADY at the next corner's decision point when this turn ends --
        # proven live 2026-07-04: corner 2 read right/0.06 at ~60deg into
        # corner 1's turn, and after the turn the camera stared at open
        # surround (no_signal), so a post-RECOVER APPROACH could never see a
        # prox cue. In that case chain DIRECTLY into the next EXECUTE_TURN.
        self.pending_corner_prox = -1.0
        # CHAIN_GAP: the S-jog's two corners are separated by a short straight
        # (roughly one road width). A mid-turn prox reading is a TRANSIENT taken
        # while rotating -- it justifies chaining, but NOT pivoting on the spot:
        # turn 1's exit is ~a gap-length short of corner 2's true pivot. So a
        # chained maneuver first drives a short dead-reckoned straight leg
        # (corner_chain_gap_m, yaw-held on odom), then starts rotation 2.
        self.chain_dir = 'none'           # direction to turn after the gap
        self.chain_gap_start_xy = None
        self.chain_gap_yaw = None         # heading to hold during the gap leg
        self.chain_gap_start_time = None

        self.frame_count          = 0
        self.first_error_received = False
        self.last_path: Path | None = None
        self.last_path_time = None

        self._pp_tick_count = 0
        
        self.control_timer = self.create_timer(0.033, self.control_tick)  # 30 Hz

        # 2 Hz auto-arm heartbeat for the gated parking detector (kept off the
        # 30 Hz steering path — arming is a slow, position-driven decision).
        self.parking_arm_timer = self.create_timer(0.5, self._parking_arm_tick)

        self.lane_detected = True
        self.lane_lost_count = 0
        self.lane_lost_stop_frames = 5

        self.straight_frames = 0
        self.straight_active = False
        
        self.robot_name = 'my_car'
        # Startup "reset to start line" teleport target. Configurable so a test
        # can place the car on a specific approach (e.g. the parking pocket)
        # instead of the world origin. Default is an on-road point for the
        # track_race mesh (the origin is off the re-centred road, so the car
        # would fall through). NOTE: this teleport only fires when /model_states
        # is published (model_seen); this world's gazebo_ros_state plugin does
        # publish it, so the target IS honoured.
        self.declare_parameter('start_x', 0.63)
        self.declare_parameter('start_y', 1.06)
        # Spawn origin == wheel-contact plane; road top is z~=0.003, so 0.02
        # drops the car only ~1.7 cm (gentle settle, not a 10 cm fall).
        self.declare_parameter('start_z', 0.02)
        self.declare_parameter('start_yaw_deg', 0.0)
        self.target_x = float(self.get_parameter('start_x').value)
        self.target_y = float(self.get_parameter('start_y').value)
        self.target_z = float(self.get_parameter('start_z').value)
        _tyaw = math.radians(float(self.get_parameter('start_yaw_deg').value))
        self._target_qz = math.sin(_tyaw / 2.0)
        self._target_qw = math.cos(_tyaw / 2.0)

        self.set_entity_clients = [
            (self.create_client(SetEntityState, '/set_entity_state'), '/set_entity_state'),
            (self.create_client(SetEntityState, '/gazebo/set_entity_state'), '/gazebo/set_entity_state'),
        ]
        self.set_model_clients = [
            (self.create_client(SetModelState, '/set_model_state'), '/set_model_state'),
            (self.create_client(SetModelState, '/gazebo/set_model_state'), '/gazebo/set_model_state'),
        ]

        self.model_seen = False
        self.last_model_pose = None
        self.model_states_source = None
        self._model_states_subs = []
        for topic in ('/model_states', '/gazebo/model_states'):
            self._model_states_subs.append(
                self.create_subscription(ModelStates, topic, lambda msg, t=topic: self._model_states_cb(msg, t), 10)
            )

        self.last_odom_time = None
        self.last_odom_pose = None
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        
        self.get_logger().info('ControlNode started.')
        
        self.teleport_attempts = 0
        self.max_teleport_attempts = 30
        # Startup grace: until the reset-to-start teleport has settled, the
        # corner state machine must stay idle. At node start /odom still
        # reports the PREVIOUS run's end pose (often mid-track or off-road);
        # arming on that stale pose caused a spurious EXECUTE_TURN right at
        # the start line (live log 2026-07-06: via=travel traveled=2.38m at
        # t+0.5s, car pirouetted off the spawn straight before teleport).
        self.start_pose_settled = False
        self.teleport_timer = self.create_timer(0.5, self.teleport_tick)
        self.teleport_future = None
        self.teleport_mode = None  
        self.teleport_service = ''
        
        self.get_logger().info('Braking car...')
        for i in range(20):  
            self._pub(Twist())
        self.get_logger().info('Car should be stopped. Waiting for lane error signal...')

    def _param_cb(self, params):
        for p in params:
            if p.name == 'kp':
                self.Kp = float(p.value)
                self.Kd = self.Kp * self.kd_multiplier
            elif p.name == 'kd_multiplier':
                self.kd_multiplier = float(p.value)
                self.Kd = self.Kp * self.kd_multiplier
            elif p.name == 'ki': self.Ki = float(p.value)
            elif p.name == 'max_steer': self.max_steer = float(p.value)
            elif p.name == 'alpha_min': self.alpha_min = float(p.value)
            elif p.name == 'alpha_max': self.alpha_max = float(p.value)
            elif p.name == 'base_speed': self.base_speed = float(p.value)
            elif p.name == 'max_speed_reduction': self.max_speed_reduction = float(p.value)
            elif p.name == 'speed_reduction_gamma': self.speed_reduction_gamma = float(p.value)
            elif p.name == 'use_pure_pursuit': self.use_pure_pursuit = bool(p.value)
            elif p.name == 'pp_lookahead_distance': self.pp_lookahead_distance = float(p.value)
            elif p.name == 'pp_point_max_heading_deg': self.pp_point_max_heading_deg = float(p.value)
            elif p.name == 'pp_point_lateral_floor_m': self.pp_point_lateral_floor_m = float(p.value)
            elif p.name == 'pp_min_path_points': self.pp_min_path_points = int(p.value)
            elif p.name == 'pp_near_point_max_x_m': self.pp_near_point_max_x_m = float(p.value)
            elif p.name == 'pp_lookahead_turn_m': self.pp_lookahead_turn_m = float(p.value)
            elif p.name == 'pp_lookahead_curve_m': self.pp_lookahead_curve_m = float(p.value)
            elif p.name == 'pp_min_speed': self.pp_min_speed = float(p.value)
            elif p.name == 'pp_max_speed': self.pp_max_speed = float(p.value)
            elif p.name == 'pp_curvature_slowdown': self.pp_curvature_slowdown = float(p.value)
            elif p.name == 'pp_curvature_ff_gain': self.pp_curvature_ff_gain = float(p.value)
            elif p.name == 'pp_ff_band_max_x_m': self.pp_ff_band_max_x_m = float(p.value)
            elif p.name == 'pp_center_gain': self.pp_center_gain = float(p.value)
            elif p.name == 'pp_center_band_max_x_m': self.pp_center_band_max_x_m = float(p.value)
            elif p.name == 'pp_center_max_kappa': self.pp_center_max_kappa = float(p.value)
            elif p.name == 'pp_turn_scene_confirm_frames': self.pp_turn_scene_confirm_frames = int(p.value)
            elif p.name == 'pp_path_timeout_sec': self.pp_path_timeout_sec = float(p.value)
            elif p.name == 'straight_y_tol_m': self.straight_y_tol_m = float(p.value)
            elif p.name == 'straight_min_points': self.straight_min_points = int(p.value)
            elif p.name == 'straight_line_tol_m': self.straight_line_tol_m = float(p.value)
            elif p.name == 'straight_min_ratio': self.straight_min_ratio = float(p.value)
            elif p.name == 'straight_full_ratio': self.straight_full_ratio = float(p.value)
            elif p.name == 'straight_hold_frames': self.straight_hold_frames = int(p.value)
            elif p.name == 'straight_bias_gain': self.straight_bias_gain = float(p.value)
            elif p.name == 'straight_bias_max': self.straight_bias_max = float(p.value)
            elif p.name == 'apf_track_half_width': self.apf_track_half_width = float(p.value)
            elif p.name == 'apf_safe_margin': self.apf_safe_margin = float(p.value)
            elif p.name == 'apf_repulsion_gain': self.apf_repulsion_gain = float(p.value)
            elif p.name == 'apf_repulsion_gain_straight': self.apf_repulsion_gain_straight = float(p.value)
            elif p.name == 'straight_angle_tol_deg': self.straight_angle_tol_deg = float(p.value)
            elif p.name == 'use_corner_maneuver': self.use_corner_maneuver = bool(p.value)
            elif p.name == 'corner_trigger_dist_m': self.corner_trigger_dist_m = float(p.value)
            elif p.name == 'corner_commit_dist_m': self.corner_commit_dist_m = float(p.value)
            elif p.name == 'corner_confirm_frames': self.corner_confirm_frames = int(p.value)
            elif p.name == 'corner_commit_travel_m': self.corner_commit_travel_m = float(p.value)
            elif p.name == 'corner_chain_gap_m': self.corner_chain_gap_m = float(p.value)
            elif p.name == 'turn_speed': self.turn_speed = float(p.value)
            elif p.name == 'turn_radius_m': self.turn_radius_m = float(p.value)
            elif p.name == 'turn_angle_deg': self.turn_angle_deg = float(p.value)
            elif p.name == 'turn_angle_tol_deg': self.turn_angle_tol_deg = float(p.value)
            elif p.name == 'turn_min_angle_deg': self.turn_min_angle_deg = float(p.value)
            elif p.name == 'turn_max_angle_deg': self.turn_max_angle_deg = float(p.value)
            elif p.name == 'turn_timeout_sec': self.turn_timeout_sec = float(p.value)
            elif p.name == 'recover_stable_frames': self.recover_stable_frames = int(p.value)
            elif p.name == 'corner_bend_abort_kappa': self.corner_bend_abort_kappa = float(p.value)
            elif p.name == 'angular_cmd_gain': self.angular_cmd_gain = float(p.value)
            elif p.name == 'parking_auto_arm': self.parking_auto_arm = bool(p.value)
            elif p.name == 'parking_zone_x': self.parking_zone_x = float(p.value)
            elif p.name == 'parking_zone_y': self.parking_zone_y = float(p.value)
            elif p.name == 'parking_zone_radius_m': self.parking_zone_radius_m = float(p.value)
            elif p.name == 'parking_arm_min_travel_m': self.parking_arm_min_travel_m = float(p.value)
            elif p.name == 'parking_arm_latch': self.parking_arm_latch = bool(p.value)
            elif p.name == 'park_search_speed': self.park_search_speed = float(p.value)
            elif p.name == 'park_enable': self.park_enable = bool(p.value)
            elif p.name == 'park_commit_dist_m': self.park_commit_dist_m = float(p.value)
            elif p.name == 'park_approach_speed': self.park_approach_speed = float(p.value)
            elif p.name == 'park_approach_kp': self.park_approach_kp = float(p.value)
            elif p.name == 'park_enter_speed': self.park_enter_speed = float(p.value)
            elif p.name == 'park_enter_distance_m': self.park_enter_distance_m = float(p.value)
            elif p.name == 'park_enter_use_arc': self.park_enter_use_arc = bool(p.value)
            elif p.name == 'park_min_turn_radius_m': self.park_min_turn_radius_m = float(p.value)
            elif p.name == 'park_enter_steer': self.park_enter_steer = float(p.value)
            elif p.name == 'park_enter_timeout_sec': self.park_enter_timeout_sec = float(p.value)
            elif p.name == 'park_target_timeout_sec': self.park_target_timeout_sec = float(p.value)
            elif p.name == 'park_approach_abort_deg': self.park_approach_abort_deg = float(p.value)
            elif p.name == 'park_parallel_speed': self.park_parallel_speed = float(p.value)
            elif p.name == 'park_parallel_kappa': self.park_parallel_kappa = float(p.value)
            elif p.name == 'park_parallel_first_sign': self.park_parallel_first_sign = float(p.value)
            elif p.name == 'park_parallel_back1_dist': self.park_parallel_back1_dist = float(p.value)
            elif p.name == 'park_parallel_back2_dist': self.park_parallel_back2_dist = float(p.value)
            elif p.name == 'park_parallel_adjust_dist': self.park_parallel_adjust_dist = float(p.value)
            self.get_logger().info(f'Parameter {p.name} updated to {p.value}')
        return SetParametersResult(successful=True)

    def _pub(self, cmd: Twist):
        """Publish wrapper: applies the measured yaw-rate actuation correction,
        then records the (post-correction) commanded twist so odom_cb can log
        commanded-vs-realized velocity side by side.

        2026-07-07 CALIBRATED FIX: isolated rclpy probes against this exact
        Gazebo/planar_move setup (pure rotation, pure forward, combined, at
        magnitudes 0.10 and 0.556 rad/s) measured a flat, magnitude-independent
        realized/commanded yaw-rate ratio of ~0.658 -- linear.x is realized at
        ~1.00. Confirmed live during an actual off-road drift: fd_wz/cmd_wz =
        0.153/0.232 = 0.659, matching to 3 sig figs. angular_cmd_gain =
        1/0.658 = 1.520 compensates it at the single point every cmd_vel
        publish already funnels through (EXECUTE_TURN, pure pursuit, parking),
        so internal curvature math (kappa, kctr, kff, pp_kappa_ema, the corner
        bend-abort gate) is untouched -- only what's actually sent to Gazebo
        changes. Re-clamps to max_yaw_rate afterward since scaling can push a
        pre-clamped value over it.
          car still runs wide / drifts off-road on curves -> RAISE toward 1.6
          car now oscillates / overshoots corrections      -> LOWER toward 1.3
          if the physics/URDF ever changes, re-measure with the isolated
          probe script (see yaw-rate-actuation-deficit memory) before assuming
          this value still holds.
        """
        cmd.angular.z = float(cmd.angular.z) * float(self.angular_cmd_gain)
        if self.max_yaw_rate > 0.0:
            cmd.angular.z = float(max(-self.max_yaw_rate, min(self.max_yaw_rate, cmd.angular.z)))
        self.last_cmd_linear_x = float(cmd.linear.x)
        self.last_cmd_angular_z = float(cmd.angular.z)
        self.pub.publish(cmd)

    # ── UPGRADED: Receive the scene update ──
    def scene_cb(self, msg):
        self.current_scene = msg.data
        # Consecutive turn-class frames, for the lookahead-escalation debounce.
        if msg.data in ('turn', 'roundabout', 'sharp_turn'):
            self.turn_scene_frames += 1
        else:
            self.turn_scene_frames = 0

    def _turn_scene_confirmed(self):
        return self.turn_scene_frames >= max(1, int(self.pp_turn_scene_confirm_frames))

    # ── NEW: corner cue + fused-odom callbacks ──
    def corner_dir_cb(self, msg: String):
        self.corner_dir = msg.data

    def corner_dist_cb(self, msg: Float32):
        self.corner_dist = float(msg.data)

    def corner_ang_cb(self, msg: Float32):
        self.corner_angle = float(msg.data)

    def filtered_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        # yaw from quaternion (planar, so roll/pitch ~ 0)
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.fused_yaw = math.atan2(siny, cosy)
        self.fused_yaw_time = self.get_clock().now()

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Signed smallest difference a-b wrapped to [-pi, pi]."""
        d = a - b
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        return d

    def _fused_yaw_fresh(self) -> bool:
        if self.fused_yaw is None or self.fused_yaw_time is None:
            return False
        age = (self.get_clock().now() - self.fused_yaw_time).nanoseconds * 1e-9
        return age <= 0.5

    def _raw_yaw_fresh(self) -> bool:
        if self.raw_yaw is None or self.raw_yaw_time is None:
            return False
        age = (self.get_clock().now() - self.raw_yaw_time).nanoseconds * 1e-9
        return age <= 0.5

    def _turn_yaw(self):
        """Yaw source for the corner maneuver: raw /odom by default (reliable in
        sim), or the EKF if corner_yaw_use_odom is False."""
        return self.raw_yaw if self.corner_yaw_use_odom else self.fused_yaw

    def _turn_yaw_fresh(self) -> bool:
        return self._raw_yaw_fresh() if self.corner_yaw_use_odom else self._fused_yaw_fresh()

    def lane_detected_cb(self, msg: Bool):
        self.lane_detected = bool(msg.data)

    def path_cb(self, msg: Path):
        self.last_path = msg
        self.last_path_time = self.get_clock().now()

    def _odom_xy(self):
        """Current robot (x, y) from raw /odom, or None if no odom yet."""
        p = getattr(self, 'last_odom_pose', None)
        if p is None:
            return None
        return (float(p.position.x), float(p.position.y))

    def _path_is_fresh(self) -> bool:
        if self.last_path is None or self.last_path_time is None:
            return False
        age = (self.get_clock().now() - self.last_path_time).nanoseconds * 1e-9
        return age <= float(self.pp_path_timeout_sec)

    def _pick_lookahead_point(self, path: Path, lookahead_override: float = None):
        if not path.poses:
            return None
        ld = max(0.05, float(lookahead_override if lookahead_override is not None else self.pp_lookahead_distance))
        # Plausibility gate: reject lookahead points that are too far off to the
        # side to be a followable road centreline. Perception's fit rails to the
        # BEV edge (|y|->2.5m) when the inner lane edge leaves frame on a curve;
        # steering to that edge point is what drags the car off-road. A real
        # centreline point stays within floor + x*tan(max_heading) of dead ahead.
        floor = float(self.pp_point_lateral_floor_m)
        tan_gate = math.tan(math.radians(float(self.pp_point_max_heading_deg)))
        best = None
        for ps in path.poses:
            p = ps.pose.position
            x = float(p.x)
            y = float(p.y)
            if x <= 0.0:
                continue
            if abs(y) > floor + x * tan_gate:
                continue                      # railed / off-corridor point -> ignore
            d = math.hypot(x, y)
            if d >= ld:
                return (x, y)
            best = (x, y)                     # nearest plausible so far
        return best

    def _path_curvature(self, path: Path):
        """Signed local curvature (1/m) of the /lane_path centreline near the car,
        from a quadratic fit y(x) over the near band. +ve = curving LEFT (matches
        the kappa=2y/ld^2 sign convention). Returns None when the path is too
        short, too straight, or the fit is unreliable — caller falls back to plain
        pure pursuit. This is the DYNAMIC arc measurement (fitted live each frame),
        not a preset arc."""
        if path is None or not path.poses:
            return None
        xs, ys = [], []
        for ps in path.poses:
            px = float(ps.pose.position.x)
            py = float(ps.pose.position.y)
            # NEAR band only: the curvature the car is ON right now, NOT the
            # sharp corner still ahead. A long window (0.75) reached into the
            # upcoming corner and fed its curvature forward early, so the car
            # turned hard while still centred on the lead-in and ran off
            # (live 2026-07-06: y~0, kff spiked -1.5). But the metric-BEV path
            # STARTS at x~0.24, so the old 0.45 cap left <4 poses in-band and
            # kff read 'na' every frame (2026-07-07) — the band max is now the
            # pp_ff_band_max_x_m param (default 0.60).
            if 0.05 <= px <= float(self.pp_ff_band_max_x_m):
                xs.append(px)
                ys.append(py)
        if len(xs) < 4:
            return None
        xs = np.asarray(xs); ys = np.asarray(ys)
        # Need real forward extent, else the fit is ill-conditioned.
        if float(xs.max() - xs.min()) < 0.15:
            return None
        try:
            c2, c1, _c0 = np.polyfit(xs, ys, 2)
        except (np.linalg.LinAlgError, ValueError):
            return None
        kappa = 2.0 * float(c2) / (1.0 + float(c1) ** 2) ** 1.5
        if not math.isfinite(kappa):
            return None
        # Clamp to a physically sane band (tightest realizable ~0.15 m radius).
        return max(-6.7, min(6.7, kappa))

    def _latch_corner(self):
        """Capture the turn direction + target angle when committing to a corner."""
        self.turn_dir = 1 if self.corner_dir == 'left' else -1   # left = +yaw (CCW)
        ang = self.corner_angle if self.corner_angle > 1.0 else self.turn_angle_deg
        self.turn_target = math.radians(ang)
        # Open-loop time fallback (used only if fused yaw is unavailable):
        # arc time = angle / omega, omega = v / R.
        omega = max(1e-3, self.turn_speed / max(1e-3, self.turn_radius_m))
        self.turn_expected_time = self.turn_target / omega

    def _bend_already_tracked(self, corner_dir):
        """True when pure pursuit is already steering a sustained bend in the
        cue's direction — i.e. the 'corner' is a sweeping curve the follower is
        handling, so a dead-reckoned pivot would cut inside the arc.
        Sign convention: kappa > 0 = left (matches turn_dir +1 = left)."""
        thresh = float(self.corner_bend_abort_kappa)
        if thresh <= 0.0:
            return False
        want = 1.0 if corner_dir == 'left' else -1.0
        ema = self.pp_kappa_ema
        return abs(ema) >= thresh and (ema > 0) == (want > 0)

    def _update_corner_state(self):
        """Behavior state machine that wraps pure pursuit for sharp 90deg turns.

        Only EXECUTE_TURN fully overrides control; APPROACH/RECOVER let pure
        pursuit drive (APPROACH just caps speed). Transitions are intentionally
        conservative — a flapping/false corner falls straight back to LANE_FOLLOW.
        """
        now = self.get_clock().now()
        # STARTUP GRACE: /odom is stale (previous run's end pose) until the
        # reset teleport settles — any arming/committing on it is garbage.
        # Hold LANE_FOLLOW and keep every corner latch clean until then.
        if not self.start_pose_settled:
            self.mode = 'LANE_FOLLOW'
            self.corner_confirm_count = 0
            self.pending_corner_dir = None
            self.pending_corner_confirm = 0
            self.pending_corner_prox = -1.0
            self.chain_dir = 'none'
            return
        corner = self.corner_dir in ('left', 'right')
        # Multi-band camera flow: perception fires ONLY on a real sharp corner (curves
        # excluded), and EARLY. corner_dist is the centre-MID proximity, which ramps
        # 0.35 -> 0 as the car reaches the corner. APPROACH as soon as a sharp corner
        # is seen (no distance gate); commit when centre-mid collapses below commit.
        if self.frame_count % 15 == 0 and (corner or self.mode != 'LANE_FOLLOW'):
            self.get_logger().info(
                f'CORNER_DISC dir={self.corner_dir} mid_prox={self.corner_dist:.2f} '
                f'commit<={self.corner_commit_dist_m:.2f} mode={self.mode}')

        if self.mode == 'LANE_FOLLOW':
            if corner:
                if self._bend_already_tracked(self.corner_dir):
                    self.corner_confirm_count = 0
                    if self.frame_count % 15 == 0:
                        self.get_logger().info(
                            f'CORNER_BEND arm suppressed: dir={self.corner_dir} '
                            f'kappa_ema={self.pp_kappa_ema:+.2f} '
                            f'>= {self.corner_bend_abort_kappa:.2f} (pursuit has the bend)')
                    return
                self.corner_confirm_count += 1
                if self.corner_confirm_count >= max(1, self.corner_confirm_frames):
                    self.mode = 'APPROACH'
                    self._latch_corner()
                    self.approach_start_xy = self._odom_xy()   # commit-distance origin
                    self.get_logger().info(
                        f'[corner] APPROACH dir={self.corner_dir} '
                        f'travel_commit={self.corner_commit_travel_m:.2f}m angle={math.degrees(self.turn_target):.0f}')
            else:
                self.corner_confirm_count = 0

        elif self.mode == 'APPROACH':
            # LATCH, then commit when the car actually REACHES the corner.
            #   The reach cue publishes corner_dist (aka mid_prox): ~0.84 while
            #   the corner is still far, collapsing monotonically to ~0 as the
            #   nose arrives. That is a distance-TO-CORNER signal, and it is the
            #   right commit trigger — it is invariant to WHERE detection first
            #   armed. The far drift cue arms APPROACH ~1m early at mid_prox~0.84,
            #   so committing on odom-travel-from-first-detection would pivot far
            #   too early (the 1st-corner "cut"). We commit on PROXIMITY instead.
            #   A brief cue dropout must NOT abort (that stranded the 2nd corner);
            #   only an OPPOSITE-direction cue aborts. Travel is a raised safety
            #   net for a real corner whose proximity never collapses cleanly
            #   (e.g. the 2nd corner, whose cue the roundabout's open geometry
            #   keeps suppressing) — set large enough not to undercut proximity.
            cur = self._odom_xy()
            traveled = 0.0
            if self.approach_start_xy is not None and cur is not None:
                traveled = math.hypot(cur[0] - self.approach_start_xy[0],
                                      cur[1] - self.approach_start_xy[1])
            if corner:
                opposite = ((self.corner_dir == 'right' and self.turn_dir > 0) or
                            (self.corner_dir == 'left' and self.turn_dir < 0))
                if opposite:
                    self.mode = 'LANE_FOLLOW'
                    self.corner_confirm_count = 0
                    self.get_logger().info(
                        f'[corner] APPROACH cue reversed to {self.corner_dir} '
                        f'@ {traveled:.2f}m -> LANE_FOLLOW')
                else:
                    self._latch_corner()
            # Primary: proximity collapse (car is at the corner). Fallback:
            # travel safety-net (raised so it never fires before proximity on a
            # clean corner). corner_commit_dist_m = mid_prox threshold to commit.
            reached_prox = (corner and 0.0 <= self.corner_dist <= self.corner_commit_dist_m)
            reached_travel = (traveled >= self.corner_commit_travel_m)
            if self.mode == 'APPROACH' and (reached_prox or reached_travel):
                trig = 'prox' if reached_prox else 'travel'
                # Tracked-bend abort: by commit time the follower's curvature is
                # the ground truth for "is this a bend pure pursuit is already
                # taking". Arming may happen back on the straight (EMA ~0), so
                # this commit-time check is the one that catches sweeping bends.
                if self._bend_already_tracked('left' if self.turn_dir > 0 else 'right'):
                    self.mode = 'LANE_FOLLOW'
                    self.corner_confirm_count = 0
                    self.get_logger().info(
                        f'[corner] CORNER_BEND commit ABORT via={trig} '
                        f'kappa_ema={self.pp_kappa_ema:+.2f} '
                        f'>= {self.corner_bend_abort_kappa:.2f} traveled={traveled:.2f}m '
                        f'-> LANE_FOLLOW (pursuit has the bend)')
                    return
                if self._turn_yaw_fresh():
                    self.turn_yaw0 = self._turn_yaw()
                else:
                    self.turn_yaw0 = None   # fall back to open-loop timing
                self.turn_start_time = now
                self.turn_start_xy = cur   # exact pose for the realized-radius diagnostic at RECOVER
                self.mode = 'EXECUTE_TURN'
                self.pending_corner_dir = None
                self.pending_corner_confirm = 0
                self.pending_corner_prox = -1.0
                self.get_logger().info(
                    f'[corner] EXECUTE_TURN dir={"L" if self.turn_dir > 0 else "R"} '
                    f'via={trig} mid_prox={self.corner_dist:.2f} traveled={traveled:.2f}m '
                    f'yaw0={"n/a" if self.turn_yaw0 is None else f"{self.turn_yaw0:.2f}"}')

        elif self.mode == 'EXECUTE_TURN':
            # Watch for the NEXT corner's cue while still executing this one —
            # see the back-to-back handoff note above self.pending_corner_dir.
            # Only an OPPOSITE-direction reading counts as "the next corner":
            # a same-direction reading during the turn is far more likely an
            # echo/residual of the corner already being executed.
            opposite_mid_turn = (corner and
                                  ((self.corner_dir == 'right' and self.turn_dir > 0) or
                                   (self.corner_dir == 'left' and self.turn_dir < 0)))
            if opposite_mid_turn:
                self.pending_corner_confirm += 1
                if self.pending_corner_confirm >= max(1, self.corner_confirm_frames):
                    if self.pending_corner_dir != self.corner_dir:
                        self.get_logger().info(
                            f'[corner] pending {self.corner_dir} latched mid-turn '
                            f'(seen during EXECUTE_TURN dir={"L" if self.turn_dir > 0 else "R"})')
                    self.pending_corner_dir = self.corner_dir
                # Track how CLOSE the pending corner got (its mid_prox ramps
                # 0.84 -> 0 as its decision point is reached). Only once the
                # latch is confirmed, so a single stray frame can't record a
                # bogus commit-level proximity.
                if (self.pending_corner_dir == self.corner_dir
                        and self.corner_dist >= 0.0
                        and (self.pending_corner_prox < 0.0
                             or self.corner_dist < self.pending_corner_prox)):
                    self.pending_corner_prox = self.corner_dist
            else:
                self.pending_corner_confirm = 0

            done = False
            turned = None
            elapsed = (now - self.turn_start_time).nanoseconds * 1e-9
            if self.turn_yaw0 is not None and self._turn_yaw_fresh():
                turned = abs(self._angle_diff(self._turn_yaw(), self.turn_yaw0))
                # Angle-agnostic stop: once we've rotated past the minimum and
                # perception sees open road straight ahead again, the corner is
                # cleared — stop here rather than driving a blind fixed 90 that
                # overshoots on skewed entries / shallow bends.
                road_reacquired = (self.corner_dir == 'none')
                if turned >= math.radians(self.turn_min_angle_deg) and road_reacquired:
                    done = True
                    self.get_logger().info(
                        f'[corner] road re-acquired @ {math.degrees(turned):.0f}deg -> RECOVER')
                # Fallback: road never re-acquired cleanly -> the fixed target.
                elif turned >= self.turn_target - math.radians(self.turn_angle_tol_deg):
                    done = True
                # Hard cap: a bad/flapping cue must not let us spin forever.
                elif turned >= math.radians(self.turn_max_angle_deg):
                    done = True
                    self.get_logger().warn(
                        f'[corner] max turn cap {self.turn_max_angle_deg:.0f}deg hit -> RECOVER')
            elif elapsed >= self.turn_expected_time:
                # No trustworthy yaw — terminate open-loop on expected arc time.
                done = True
            if elapsed >= self.turn_timeout_sec:
                done = True
                self.get_logger().warn('[corner] turn timeout — forcing RECOVER')
            if done:
                # Realized-radius diagnostic: chord = 2*R*sin(turned/2) for a
                # constant-(v,omega) arc, so R_realized = chord / (2*sin(turned/2)).
                # Commanded R is turn_speed/turn_radius_m's turn_radius_m by
                # construction (_run_execute_turn sets omega = v/turn_radius_m).
                # If realized >> commanded, the car is physically sweeping a
                # WIDER arc than configured -> reads as "cutting" through
                # track features the tight nominal radius should have missed.
                if turned is not None and turned > math.radians(5.0) and self.turn_start_xy is not None:
                    cur = self._odom_xy()
                    if cur is not None:
                        chord = math.hypot(cur[0] - self.turn_start_xy[0],
                                            cur[1] - self.turn_start_xy[1])
                        realized_r = chord / (2.0 * math.sin(turned * 0.5))
                        self.get_logger().info(
                            f'[corner] TURN_RADIUS_DBG commanded={self.turn_radius_m:.2f}m '
                            f'realized={realized_r:.2f}m chord={chord:.2f}m turned={math.degrees(turned):.0f}deg')
                # BACK-TO-BACK CHAIN: if the pending corner's proximity already
                # collapsed to commit level DURING this turn, the next corner is
                # (roughly) a gap-length ahead -- going to RECOVER would stare at
                # open surround (no lane to reacquire) and the maneuver dies.
                # But do NOT pivot on the spot either: the mid-turn prox was a
                # transient taken while rotating; turn 1's exit sits ~one road
                # width short of corner 2's pivot (proven live: an immediate
                # chained pivot ground along the surround, yaw bouncing 10->1).
                # Drive a dead-reckoned straight CHAIN_GAP leg first.
                chain = (self.pending_corner_dir in ('left', 'right')
                         and 0.0 <= self.pending_corner_prox <= self.corner_commit_dist_m
                         and elapsed < self.turn_timeout_sec)
                if chain:
                    self.chain_dir = self.pending_corner_dir
                    self.chain_gap_start_xy = self._odom_xy()
                    self.chain_gap_yaw = self._turn_yaw() if self._turn_yaw_fresh() else None
                    self.chain_gap_start_time = now
                    self.mode = 'CHAIN_GAP'
                    self.get_logger().info(
                        f'[corner] CHAIN_GAP -> {self.chain_dir} after '
                        f'{self.corner_chain_gap_m:.2f}m straight '
                        f'(pending mid_prox={self.pending_corner_prox:.2f}, no RECOVER between)')
                    self.pending_corner_dir = None
                    self.pending_corner_confirm = 0
                    self.pending_corner_prox = -1.0
                else:
                    self.mode = 'RECOVER'
                    self.recover_count = 0
                    self.get_logger().info('[corner] RECOVER')

        elif self.mode == 'CHAIN_GAP':
            # Dead-reckoned straight leg between chained turns.
            # PRIMARY commit: the live reach cue. Once turn 1's rotation stops,
            # the camera faces straight down the jog leg at corner 2 — the cue
            # re-fires in the leg (road-end ~0.7m ahead at leg start) and its
            # mid_prox collapse marks the true pivot point, self-correcting for
            # wherever turn 1 actually exited (which moves with turn radius).
            # FALLBACK: odom gap distance (then time if odom is unavailable);
            # hard timeout aborts to RECOVER so a stuck car can't creep forever.
            cur = self._odom_xy()
            chain_prox = (self.corner_dir == self.chain_dir
                          and 0.0 <= self.corner_dist <= self.corner_commit_dist_m)
            gap_done = False
            if cur is not None and self.chain_gap_start_xy is not None:
                gap_traveled = math.hypot(cur[0] - self.chain_gap_start_xy[0],
                                          cur[1] - self.chain_gap_start_xy[1])
                gap_done = gap_traveled >= self.corner_chain_gap_m
            gap_elapsed = (now - self.chain_gap_start_time).nanoseconds * 1e-9
            expected_t = self.corner_chain_gap_m / max(0.02, self.turn_speed)
            if cur is None or self.chain_gap_start_xy is None:
                gap_done = gap_elapsed >= expected_t     # open-loop fallback
            if gap_elapsed >= 3.0 * expected_t:
                self.get_logger().warn('[corner] CHAIN_GAP timeout -> RECOVER')
                self.chain_dir = 'none'
                self.mode = 'RECOVER'
                self.recover_count = 0
            elif gap_done or chain_prox:
                trig = 'prox' if chain_prox else 'gap leg done'
                self.corner_dir = self.chain_dir
                self._latch_corner()
                if self._turn_yaw_fresh():
                    self.turn_yaw0 = self._turn_yaw()
                else:
                    self.turn_yaw0 = None
                self.turn_start_time = now
                self.turn_start_xy = cur
                self.chain_dir = 'none'
                self.mode = 'EXECUTE_TURN'
                self.get_logger().info(
                    f'[corner] EXECUTE_TURN dir={"L" if self.turn_dir > 0 else "R"} '
                    f'via=chained ({trig}, mid_prox={self.corner_dist:.2f})')

        elif self.mode == 'RECOVER':
            if self.lane_detected and self._path_is_fresh():
                self.recover_count += 1
                if self.recover_count >= max(1, self.recover_stable_frames):
                    if self.pending_corner_dir in ('left', 'right'):
                        # A corner was already confirmed while we were still
                        # executing the PREVIOUS turn — its detection window
                        # may already be closing (that's exactly why it needed
                        # latching), so skip LANE_FOLLOW's fresh confirm-frames
                        # wait and go straight into APPROACH for it.
                        self.corner_dir = self.pending_corner_dir
                        self.mode = 'APPROACH'
                        self.corner_confirm_count = 0
                        self._latch_corner()
                        self.approach_start_xy = self._odom_xy()
                        self.get_logger().info(
                            f'[corner] APPROACH dir={self.corner_dir} (from pending mid-turn '
                            f'latch) travel_commit={self.corner_commit_travel_m:.2f}m '
                            f'angle={math.degrees(self.turn_target):.0f}')
                    else:
                        self.mode = 'LANE_FOLLOW'
                        self.corner_confirm_count = 0
                        self.get_logger().info('[corner] LANE_FOLLOW (recovered)')
                    self.pending_corner_dir = None
                    self.pending_corner_confirm = 0
                    self.pending_corner_prox = -1.0
            else:
                self.recover_count = 0

    def _run_execute_turn(self):
        """Publish the dead-reckoned arc command. Returns True (handled)."""
        turned_deg = None
        if self.turn_yaw0 is not None and self._turn_yaw() is not None:
            turned_deg = math.degrees(abs(self._angle_diff(self._turn_yaw(), self.turn_yaw0)))
        # Safety: if we were closing the loop on yaw but it went stale, stop
        # rather than spin blind.
        if self.turn_yaw0 is not None and not self._turn_yaw_fresh():
            self.get_logger().warn(
                f'[turn] yaw STALE -> stopping. yaw0={self.turn_yaw0:.2f} '
                f'yaw={self._turn_yaw()} turned={turned_deg}')
            self._pub(Twist())
            return True
        omega = self.turn_speed / max(1e-3, self.turn_radius_m)   # v / R
        if self.max_yaw_rate > 0.0:
            omega = min(omega, self.max_yaw_rate)
        cmd = Twist()
        cmd.linear.x = float(self.turn_speed)
        cmd.angular.z = float(self.turn_dir * omega)
        self._pub(cmd)
        self.prev_angular_z = cmd.angular.z
        self._turn_log = getattr(self, '_turn_log', 0) + 1
        if self._turn_log % 5 == 0:
            self.get_logger().info(
                f'[turn] dir={"L" if self.turn_dir > 0 else "R"} wz={cmd.angular.z:+.2f} '
                f'turned={turned_deg if turned_deg is None else f"{turned_deg:.0f}"}deg '
                f'target={math.degrees(self.turn_target):.0f}')
        return True

    # ── NEW: parking maneuver bridge (logic lives in ParkingManeuver) ─────────
    def _parking_cfg(self):
        """Snapshot the live-tunable ROS params into the maneuver's config."""
        return ParkingConfig(
            commit_dist_m=self.park_commit_dist_m,
            approach_speed=self.park_approach_speed,
            approach_kp=self.park_approach_kp,
            enter_speed=self.park_enter_speed,
            enter_distance_m=self.park_enter_distance_m,
            enter_use_arc=self.park_enter_use_arc,
            min_turn_radius_m=self.park_min_turn_radius_m,
            enter_steer=self.park_enter_steer,
            enter_timeout_sec=self.park_enter_timeout_sec,
            max_yaw_rate=self.max_yaw_rate,
            approach_abort_bearing=math.radians(self.park_approach_abort_deg),
            parallel_speed=self.park_parallel_speed,
            parallel_kappa=self.park_parallel_kappa,
            parallel_first_sign=self.park_parallel_first_sign,
            parallel_back1_dist=self.park_parallel_back1_dist,
            parallel_back2_dist=self.park_parallel_back2_dist,
            parallel_adjust_dist=self.park_parallel_adjust_dist,
        )

    def _run_parking_maneuver(self):
        """Tick the maneuver. Returns True if it owns the actuator this tick.

        control_node stays the single /cmd_vel owner: we publish whatever the
        maneuver returns and, while it is active, mirror its phase into
        self.mode so the corner machine yields and the STATE log reflects it.
        can_start is gated on LANE_FOLLOW so parking never pre-empts a corner turn.
        """
        now_s = self.get_clock().now().nanoseconds * 1e-9
        out = self.parking.update(
            now=now_s, cfg=self._parking_cfg(),
            armed=self.parking_armed, detected=self.parking_detected,
            target=self.park_target, target_fresh=self._park_target_fresh(),
            dist=self.park_dist, robot_xy=self._robot_xy(),
            can_start=(self.mode == 'LANE_FOLLOW'),
            bay_type=self.parking_bay_type)
        if self.parking.active:
            self.mode = 'PARK_' + self.parking.state
            cmd = Twist()
            cmd.linear.x, cmd.angular.z = float(out[0]), float(out[1])
            self._pub(cmd)
            self.prev_angular_z = cmd.angular.z
            return True
        if self.mode.startswith('PARK_'):   # maneuver just ended -> hand back
            self.mode = 'LANE_FOLLOW'
        return False

    def control_tick(self):
        if not self.use_pure_pursuit:
            return

        # Diagnostic: tie car position to mode + corner cue so a failing corner
        # can be located from the trajectory (run is headless).
        self._state_tick = getattr(self, '_state_tick', 0) + 1
        if self._state_tick % 15 == 0 and self.last_odom_pose is not None:
            p = self.last_odom_pose.position
            self.get_logger().info(
                f'STATE mode={self.mode} corner={self.corner_dir}/{self.corner_dist:.2f} '
                f'pos=({p.x:.2f},{p.y:.2f},{p.z:.2f})'
            )

        # ── Parking maneuver (highest priority; fully overrides lane following) ──
        if self.park_enable and self._run_parking_maneuver():
            return

        # ── Corner maneuver state machine (overrides pure pursuit for sharp turns) ──
        if self.use_corner_maneuver:
            self._update_corner_state()
            if self.mode == 'EXECUTE_TURN':
                self._run_execute_turn()
                return
            if self.mode == 'CHAIN_GAP':
                # Dead-reckoned straight leg between chained turns — perception
                # is typically blind here (facing the jog), so drive on odom:
                # constant turn_speed forward with a light yaw-hold on the
                # heading captured at the gap's start.
                cmd = Twist()
                cmd.linear.x = float(self.turn_speed)
                wz = 0.0
                if self.chain_gap_yaw is not None and self._turn_yaw_fresh():
                    err = self._angle_diff(self.chain_gap_yaw, self._turn_yaw())
                    wz = max(-0.3, min(0.3, 1.5 * err))
                cmd.angular.z = float(wz)
                self._pub(cmd)
                return

        if not self.lane_detected:
            cmd = Twist()
            cmd.linear.x = float(self.base_speed) * 0.25
            cmd.angular.z = float(self.prev_angular_z) * 0.5
            self._pub(cmd)
            return

        if not self._path_is_fresh() or self.last_path is None:
            if self.prev_angular_z != 0.0:
                cmd = Twist()
                cmd.linear.x  = float(self.base_speed) * 0.40
                
                # ── UPGRADED: 1.0 multiplier holds the steering angle during stutter ──
                cmd.angular.z = float(self.prev_angular_z) * 1.0  
                
                self._pub(cmd)
            return

        # Scene-adaptive lookahead: look FURTHER ahead on turns so the lookahead
        # point lands inside the corner geometry before the car reaches it.
        # Turn-class escalation is DEBOUNCED: a single-frame curve->roundabout
        # flap must not swap the lookahead (it near-doubles commanded curvature).
        if self.current_scene in ('turn', 'roundabout', 'sharp_turn'):
            _ld = (float(self.pp_lookahead_turn_m) if self._turn_scene_confirmed()
                   else float(self.pp_lookahead_curve_m))
        elif self.current_scene == 'curve':
            _ld = float(self.pp_lookahead_curve_m)
        else:
            _ld = float(self.pp_lookahead_distance)

        # Straight-road steering hold: if the path is vertically aligned for
        # a few frames, zero the steering until the pattern changes.
        line_ratio = 0.0
        ratio_vertical = 0.0
        line_angle_deg = None
        if self.last_path is not None and self.current_scene in ('straight', 'curve'):
            aligned = 0
            checked = 0
            sum_x = 0.0
            sum_y = 0.0
            sum_x2 = 0.0
            sum_xy = 0.0
            for ps in self.last_path.poses:
                p = ps.pose.position
                x = float(p.x)
                y = float(p.y)
                if x <= 0.05:
                    continue
                checked += 1
                sum_x += x
                sum_y += y
                sum_x2 += x * x
                sum_xy += x * y
                if abs(y) <= float(self.straight_y_tol_m):
                    aligned += 1
            checked_min = max(1, int(self.straight_min_points))
            if checked >= checked_min:
                ratio_vertical = float(aligned) / float(checked)
            else:
                ratio_vertical = 0.0

            line_ratio = 0.0
            if checked >= max(2, checked_min):
                denom = (float(checked) * sum_x2) - (sum_x * sum_x)
                if abs(denom) > 1e-9:
                    m = (float(checked) * sum_xy - (sum_x * sum_y)) / denom
                    b = (sum_y - m * sum_x) / float(checked)
                    # Angle of fitted line in the controller frame (x forward, y lateral).
                    # A perfectly straight-ahead path is y = 0 => slope m≈0 => angle≈0°.
                    line_angle_deg = abs(math.degrees(math.atan(m)))
                    in_line = 0
                    for ps in self.last_path.poses:
                        p = ps.pose.position
                        x = float(p.x)
                        y = float(p.y)
                        if x <= 0.05:
                            continue
                        if abs(y - (m * x + b)) <= float(self.straight_line_tol_m):
                            in_line += 1
                    line_ratio = float(in_line) / float(checked)

            is_straight = (checked >= checked_min) and (line_ratio >= float(self.straight_min_ratio))
            is_vertical_angle = (line_angle_deg is not None) and (line_angle_deg <= float(self.straight_angle_tol_deg))
            is_fully_aligned = (checked >= checked_min) and (ratio_vertical >= float(self.straight_full_ratio)) and is_vertical_angle

            if is_straight and is_fully_aligned:
                self.straight_frames += 1
            else:
                self.straight_frames = 0
                self.straight_active = False

            if self.frame_count % 10 == 0:
                self.get_logger().info(
                    f"ALIGN_DBG "
                    f"line_ratio={line_ratio:.2f} "
                    f"vert_ratio={ratio_vertical:.2f} "
                    f"angle={(line_angle_deg if line_angle_deg is not None else -1):.1f} "
                    f"frames={self.straight_frames}"
                )

        if self.current_scene not in ('straight', 'curve'):
            self.straight_frames = 0
            self.straight_active = False
        else:
            if self.straight_frames >= max(1, int(self.straight_hold_frames)):
                self.straight_active = True

        self._pp_tick_count += 1
        if self._pp_tick_count % 30 == 0 and self.current_scene in ('straight', 'curve'):
            ang_str = 'None' if line_angle_deg is None else f'{line_angle_deg:.2f}'
            self.get_logger().info(
                f'straight_dbg: ratio_line={line_ratio:.2f}  ratio_vert={ratio_vertical:.2f}  '
                f'angle_deg={ang_str}  hold={self.straight_active}'
            )

        # ── Degenerate-path guard ──────────────────────────────────────────
        # A rail path (few far-only poses, no near support) has no point worth
        # steering hard toward. Count valid poses (x>0.05) and whether any lands
        # in the near field (x <= pp_near_point_max_x_m). If the path is too
        # sparse OR has no near point, don't chase the far rail: decay the last
        # steer toward straight and coast at reduced speed until a real path
        # returns. Only guards the following scenes — the corner maneuver and
        # parking own their own logic and can legitimately run on sparse paths.
        if self.current_scene in ('straight', 'curve'):
            n_valid = 0
            has_near = False
            for ps in self.last_path.poses:
                px = float(ps.pose.position.x)
                if px <= 0.05:
                    continue
                n_valid += 1
                if px <= float(self.pp_near_point_max_x_m):
                    has_near = True
            if n_valid < int(self.pp_min_path_points) or not has_near:
                decayed = float(self.prev_angular_z) * 0.30
                cmd = Twist()
                cmd.linear.x = float(self.base_speed) * 0.50
                cmd.angular.z = decayed
                self._pub(cmd)
                self.prev_angular_z = decayed
                if self.frame_count % 10 == 0:
                    self.get_logger().warn(
                        f"PATH_DEGEN n_valid={n_valid} has_near={has_near} "
                        f"(min_pts={self.pp_min_path_points} "
                        f"near_x<={self.pp_near_point_max_x_m:.2f}) -> coast straight "
                        f"z={decayed:.3f}"
                    )
                return

        pt = self._pick_lookahead_point(self.last_path, lookahead_override=_ld)
        if pt is None:
            # No plausible road point (whole fit railed / empty path). Steer
            # straight ahead rather than freezing the last command — a held
            # wrong-way steer is exactly what drives the car off-road.
            pt = (max(0.1, float(_ld)), 0.0)
        x, y = pt

        ld2 = max(1e-4, x * x + y * y)
        kappa = 2.0 * y / ld2

        # ── Curvature feedforward: blend the LIVE fitted arc curvature toward the
        # single-point pure-pursuit command so the car turns at the road's actual
        # rate on an arc instead of running wide (see pp_curvature_ff_gain). At
        # gain 0, or when the path is too short/straight to fit, this is a no-op.
        kappa_ff = self._path_curvature(self.last_path)
        if kappa_ff is not None:
            # EMA-smooth to kill the frame-to-frame spikes the raw fit produced.
            self._kff_ema = self._kff_alpha * kappa_ff + (1.0 - self._kff_alpha) * self._kff_ema
            kappa_ff = self._kff_ema
        g_ff = float(self.pp_curvature_ff_gain)
        if kappa_ff is not None and g_ff > 0.0:
            kappa_cmd = (1.0 - g_ff) * kappa + g_ff * kappa_ff
        else:
            kappa_cmd = kappa

        # ── Near-field centering ("keep the bottom of the BEV road centred") ──
        # Pure pursuit aims at ONE far point, so mid-bend it tolerates a standing
        # lateral offset — the inside drift. This continuously steers the offset
        # of the NEAR path (x <= pp_center_band_max_x_m, i.e. the road at the
        # car's nose) back to zero. No-op at gain 0 or when converged (y_near~0),
        # so straights and centred driving are untouched.
        kappa_center = 0.0
        g_c = float(self.pp_center_gain)
        if g_c > 0.0:
            near_ys = [float(ps.pose.position.y) for ps in self.last_path.poses
                       if 0.05 <= float(ps.pose.position.x) <= float(self.pp_center_band_max_x_m)]
            if near_ys:
                y_near = sum(near_ys) / len(near_ys)
                k_max = float(self.pp_center_max_kappa)
                kappa_center = max(-k_max, min(k_max, g_c * y_near))
                kappa_cmd += kappa_center

        # Smoothed commanded curvature for the tracked-bend corner gate (~0.5s
        # window at 30Hz). Only updated while pure pursuit actually publishes,
        # so a flood (PATH_DEGEN returns above) freezes the pre-flood value —
        # a straight-then-flooded real corner keeps a small EMA.
        self.pp_kappa_ema = 0.12 * kappa_cmd + 0.88 * self.pp_kappa_ema

        # ── UPGRADED: Proactive braking based on what the camera sees ──
        if self.current_scene in ['turn', 'roundabout'] and self._turn_scene_confirmed():
            scene_speed_limit = float(self.pp_max_speed) * 0.40  # Hard brake for sharp turns
        elif self.current_scene == 'curve' or self.current_scene in ['turn', 'roundabout']:
            scene_speed_limit = float(self.pp_max_speed) * 0.75  # Mild brake for curves
        else:
            scene_speed_limit = float(self.pp_max_speed)         # Full speed for straights

        # APPROACH: brake to turn speed so the dead-reckoned commit is smooth.
        if self.mode == 'APPROACH':
            scene_speed_limit = min(scene_speed_limit, float(self.turn_speed))

        speed = scene_speed_limit / (1.0 + float(self.pp_curvature_slowdown) * abs(kappa_cmd))
        speed = max(float(self.pp_min_speed), min(scene_speed_limit, speed))

        # ══════════════════════════════════════════════════════════════════
        # THE FIX: True Kinematics + Artificial Potential Field (APF)
        # ══════════════════════════════════════════════════════════════════
        # 1. Boundary Repulsion (APF)
        # Repulsion only acts OUTSIDE the safe margin; the center band is left
        # to pure pursuit. apf_safe_margin must exceed normal lateral wander or
        # the car oscillates on the deadband edge (relay limit cycle).
        track_half_width = float(self.apf_track_half_width)
        safe_margin = float(self.apf_safe_margin)

        apf_offset = 0.0
        if abs(y) > safe_margin:
            # Calculate distance to the physical edge (prevent divide-by-zero)
            dist_to_edge = max(0.01, track_half_width - abs(y))

            # Scene-conditional gain: soften only on straights (kills the swerve);
            # keep the full push on curves/turns where it's needed to stay on road.
            eff_gain = (self.apf_repulsion_gain_straight
                        if self.current_scene == 'straight'
                        else self.apf_repulsion_gain)

            # Exponential push-back: gets violently stronger the closer you get to the edge
            repulsion_force = float(eff_gain) / (dist_to_edge ** 2)

            # Apply force in the opposite direction of the drift
            apf_offset = -math.copysign(repulsion_force, y)
            
        # APF adds a curvature offset (equivalent to shifting target_y); apply it
        # to the feedforward-blended command so both the arc-following and the
        # boundary push are preserved. APF is a no-op while its gains are 0.
        apf_delta_kappa = 2.0 * apf_offset / ld2
        adjusted_kappa = kappa_cmd + apf_delta_kappa
        
        # 2. Kinematic Steering Angle (Ackermann Bicycle Model)
        # delta = atan(kappa * L)
        steering_angle = math.atan(adjusted_kappa * self.wheelbase)
        
        # Clip to physical steering limits (e.g. servo limits)
        steering_angle = max(-self.max_steer, min(self.max_steer, steering_angle))
        
        # 3. Convert back to Twist yaw_rate for the simulator
        # yaw_rate = v * tan(delta) / L
        yaw_rate = speed * math.tan(steering_angle) / self.wheelbase
        
        if self.max_yaw_rate > 0.0:
            yaw_rate = float(max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate)))

        # ── Straight-road steering hold ────────────────────────────────────
        # When perception has confirmed a vertically-aligned, straight path for
        # `straight_hold_frames` frames, attenuate the steering command. This is
        # the damper that was computed (`straight_active`) but never applied:
        # without it a small steady lateral bias drives a back-and-forth limit
        # cycle on straights. Conditional on `straight_active`, so it never
        # touches turn/curve behaviour. `straight_bias_gain` is the retained
        # steering fraction (0 = hard hold, 1 = no damping), live-tunable.
        if self.straight_active:
            hold_gain = float(min(1.0, max(0.0, self.straight_bias_gain)))
            yaw_rate *= hold_gain

        if self.frame_count % 10 == 0:
            self.get_logger().info(
                f"CTRL_DBG "
                f"scene={self.current_scene} "
                f"y={y:.3f} "
                f"kappa={kappa:.3f} "
                f"kff={('%.3f' % kappa_ff) if kappa_ff is not None else 'na'} "
                f"kctr={kappa_center:+.3f} "
                f"kcmd={kappa_cmd:.3f} "
                f"speed={speed:.3f} "
                f"yaw={yaw_rate:.3f} "
                f"hold={self.straight_active}"
            )

        # ── Parking hand-off: creep while searching for the bay ──────────────
        # Reaching here means the maneuver has NOT taken over (it returns early
        # above when active). If we're armed and inside the parking zone, cap the
        # lane-follow speed so the car doesn't overshoot the pocket before the
        # close-range detection (+ ~1.8 s vote) confirms and the maneuver engages.
        if (self.park_enable and self.parking_armed and self.parking_in_zone
                and self.park_search_speed > 0.0):
            speed = min(speed, float(self.park_search_speed))

        cmd = Twist()
        cmd.linear.x = float(speed)
        cmd.angular.z = float(yaw_rate)
        self._pub(cmd)

        self.prev_angular_z = yaw_rate

    def _model_states_cb(self, msg: ModelStates, topic: str):
        try:
            idx = msg.name.index(self.robot_name)
        except ValueError:
            return
        self.model_seen = True
        self.last_model_pose = msg.pose[idx]
        self.model_states_source = topic

    def odom_cb(self, msg: Odometry):
        self.last_odom_time = self.get_clock().now()
        self.last_odom_pose = msg.pose.pose
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.raw_yaw = math.atan2(siny, cosy)
        self.raw_yaw_time = self.get_clock().now()

        # ── DIAG_YAW (2026-07-07, turning-radius investigation) ──────────────
        # Answers: is the physically-realized yaw rate/speed equal to what we
        # commanded? Uses SIM time from the odom header (NOT node wall-clock,
        # which is unrelated to RTF) for the finite-difference cross-check, and
        # cross-validates against the twist the planar_move plugin itself
        # reports for that same tick. Active only during EXECUTE_TURN/APPROACH
        # or a curve/roundabout scene, throttled to ~10Hz. Remove after the
        # investigation closes.
        if self.mode in ('APPROACH', 'EXECUTE_TURN') or self.current_scene in ('curve', 'roundabout', 'turn'):
            self._diag_tick = getattr(self, '_diag_tick', 0) + 1
            if self._diag_tick % 3 == 0:
                t_sim = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                cur_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)
                prev_t = getattr(self, '_diag_prev_t', None)
                prev_yaw = getattr(self, '_diag_prev_yaw', None)
                prev_xy = getattr(self, '_diag_prev_xy', None)
                fd_v = fd_wz = None
                if prev_t is not None and t_sim > prev_t:
                    dt = t_sim - prev_t
                    fd_wz = self._angle_diff(self.raw_yaw, prev_yaw) / dt
                    fd_v = math.hypot(cur_xy[0] - prev_xy[0], cur_xy[1] - prev_xy[1]) / dt
                self.get_logger().info(
                    f'DIAG_YAW mode={self.mode} scene={self.current_scene} '
                    f'cmd_v={self.last_cmd_linear_x:+.3f} cmd_wz={self.last_cmd_angular_z:+.3f} '
                    f'odomtw_v={msg.twist.twist.linear.x:+.3f} odomtw_wz={msg.twist.twist.angular.z:+.3f} '
                    f'fd_v={"na" if fd_v is None else f"{fd_v:+.3f}"} '
                    f'fd_wz={"na" if fd_wz is None else f"{fd_wz:+.3f}"}'
                )
                self._diag_prev_t = t_sim
                self._diag_prev_yaw = self.raw_yaw
                self._diag_prev_xy = cur_xy

    # ── NEW: parking detector callbacks + zone-based auto-arm ──
    def parking_detected_cb(self, msg: Bool):
        self.parking_detected = bool(msg.data)

    def parking_target_cb(self, msg: PoseStamped):
        q = msg.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z), 1.0 - 2.0 * (q.z * q.z))
        self.park_target = (float(msg.pose.position.x), float(msg.pose.position.y), yaw)
        self.park_target_time = self.get_clock().now()

    def parking_dist_cb(self, msg: Float32):
        self.park_dist = float(msg.data)

    def parking_bay_type_cb(self, msg: String):
        # 'parallel' | 'perpendicular' | 'none'. Latch the last real label; the
        # maneuver reads it at COMMIT to pick forward nose-in vs reverse arcs.
        val = str(msg.data).lower()
        if val in ('parallel', 'perpendicular'):
            self.parking_bay_type = val

    def _park_target_fresh(self) -> bool:
        if self.park_target is None or self.park_target_time is None:
            return False
        age = (self.get_clock().now() - self.park_target_time).nanoseconds * 1e-9
        return age <= float(self.park_target_timeout_sec)

    def _robot_xy(self):
        """Best-available robot position: /odom, else Gazebo model state."""
        if self.last_odom_pose is not None:
            p = self.last_odom_pose.position
            return p.x, p.y
        if self.last_model_pose is not None:
            p = self.last_model_pose.position
            return p.x, p.y
        return None

    def _parking_arm_tick(self):
        """Arm the gated parking detector when the car reaches the parking zone.

        Publishes /mission/parking_active at the timer rate so a late-joining
        detector stays in sync. Arming is latched once tripped (detection-only
        scanning); leaving the zone won't disarm unless parking_arm_latch=False.
        """
        if not self.parking_auto_arm:
            return
        xy = self._robot_xy()
        if xy is None:
            return
        if self._spawn_xy is None:
            self._spawn_xy = xy

        travelled = math.hypot(xy[0] - self._spawn_xy[0], xy[1] - self._spawn_xy[1])
        in_zone = (math.hypot(xy[0] - self.parking_zone_x,
                              xy[1] - self.parking_zone_y) <= self.parking_zone_radius_m)
        self.parking_in_zone = in_zone   # drives the lane-follow search creep
        should_arm = (travelled >= self.parking_arm_min_travel_m) and in_zone
        if self.parking_arm_latch and self.parking_armed:
            should_arm = True   # stay armed once tripped

        if should_arm != self.parking_armed:
            self.parking_armed = should_arm
            self.get_logger().info(
                f'[parking] detector {"ARMED" if should_arm else "disarmed"} '
                f'(pos=({xy[0]:.2f},{xy[1]:.2f}) travelled={travelled:.2f}m)'
            )

        # The maneuver itself (APPROACH -> ... -> PARKED) is driven from
        # control_tick via ParkingManeuver; here we only arm the detector and
        # publish the arm state. Log when a bay is confirmed in view.
        if self.parking_armed and self.parking_detected:
            self.get_logger().info(
                f'[parking] bay confirmed in view (dist={self.park_dist:.2f})',
                throttle_duration_sec=2.0)

        self.parking_arm_pub.publish(Bool(data=bool(self.parking_armed)))

    def teleport_tick(self):
        if not self.model_seen:
            self.teleport_attempts += 1
            if self.teleport_attempts >= self.max_teleport_attempts:
                self._settle_start_pose('no /model_states — giving up on teleport')
                self.teleport_timer.cancel()
            return
        if self.last_model_pose is not None:
            dx = self.last_model_pose.position.x - self.target_x
            dy = self.last_model_pose.position.y - self.target_y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < 0.2:
                self._settle_start_pose(f'at start pose (d={dist:.2f}m)')
                self.teleport_timer.cancel()
                return
        if self.teleport_future is not None:
            if not self.teleport_future.done():
                return
            ok = self._handle_teleport_response()
            if ok:
                self._settle_start_pose(f'teleport ok via {self.teleport_service}')
                self.teleport_timer.cancel()
                return
            self.teleport_future = None
            self.teleport_mode = None
        self.teleport_attempts += 1
        self._start_teleport_request()
        if self.teleport_attempts >= self.max_teleport_attempts:
            self._settle_start_pose('teleport attempts exhausted')
            self.teleport_timer.cancel()

    def _settle_start_pose(self, why: str):
        """Lift the startup grace: the corner state machine may now arm."""
        if not self.start_pose_settled:
            self.start_pose_settled = True
            self.get_logger().info(f'[corner] start pose settled ({why}) — corner maneuvers armed')

    def _start_teleport_request(self):
        for client, srv_name in self.set_entity_clients:
            if not (client.service_is_ready() or client.wait_for_service(timeout_sec=0.0)):
                continue
            entity_state = EntityState()
            entity_state.name = self.robot_name
            entity_state.pose.position.x = self.target_x
            entity_state.pose.position.y = self.target_y
            entity_state.pose.position.z = self.target_z
            entity_state.pose.orientation.z = self._target_qz
            entity_state.pose.orientation.w = self._target_qw
            request = SetEntityState.Request()
            if hasattr(request, 'state'): request.state = entity_state
            else: request.entity_state = entity_state
            self.teleport_future = client.call_async(request)
            self.teleport_mode = 'entity'
            self.teleport_service = srv_name
            return
        for client, srv_name in self.set_model_clients:
            if not (client.service_is_ready() or client.wait_for_service(timeout_sec=0.0)):
                continue
            model_state = ModelState()
            model_state.model_name = self.robot_name
            model_state.pose.position.x = self.target_x
            model_state.pose.position.y = self.target_y
            model_state.pose.position.z = self.target_z
            model_state.pose.orientation.z = self._target_qz
            model_state.pose.orientation.w = self._target_qw
            request = SetModelState.Request()
            request.model_state = model_state
            self.teleport_future = client.call_async(request)
            self.teleport_mode = 'model'
            self.teleport_service = srv_name
            return

    def _handle_teleport_response(self) -> bool:
        try: resp = self.teleport_future.result()
        except Exception: return False
        if resp is None: return False
        if getattr(resp, 'success', False): return True
        return False

    def error_cb(self, msg):
        if self.use_pure_pursuit and self._path_is_fresh():
            return
        if not self.first_error_received:
            for _ in range(5):
                self._pub(Twist())
            self.first_error_received = True

        raw_error = float(msg.data) * self.error_sign

        if not self.lane_detected:
            self.lane_lost_count += 1
            if self.lane_lost_count >= self.lane_lost_stop_frames:
                self.integral = 0.0
                self.prev_error = 0.0
                self.smoothed_error = 0.0
                self.prev_angular_z = 0.0
                cmd = Twist()
                cmd.linear.x = float(self.base_speed)
                cmd.angular.z = 0.0
                self._pub(cmd)
                self.frame_count += 1
                return
        else:
            self.lane_lost_count = 0

        alpha_min = float(max(0.0, min(1.0, self.alpha_min)))
        alpha_max = float(max(alpha_min, min(1.0, self.alpha_max)))
        turn_strength_alpha = min(1.0, abs(self.smoothed_error) / 15.0)
        effective_alpha = alpha_min + (alpha_max - alpha_min) * turn_strength_alpha

        self.smoothed_error = (effective_alpha * raw_error + (1.0 - effective_alpha) * self.smoothed_error)
        error = self.smoothed_error

        if abs(error) < self.deadband_deg:
            error = 0.0

        derivative = error - self.prev_error
        if abs(self.Ki) > 1e-12:
            self.integral = float(max(-self.integral_max, min(self.integral_max, self.integral + error)))
        else:
            self.integral = 0.0

        steering_angle = -(self.Kp * error + self.Kd * derivative + self.Ki * self.integral)
        steering_angle = float(max(-self.max_steer, min(self.max_steer, steering_angle)))

        if abs(self.wheelbase) < 1e-6:
            curvature = 0.0
        else:
            curvature = math.tan(steering_angle) / self.wheelbase
            
        yaw_rate = self.yaw_ref_speed * curvature

        if self.max_yaw_rate > 0.0:
            yaw_rate = float(max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate)))

        yaw_rate = (self.beta * yaw_rate + (1.0 - self.beta) * self.prev_angular_z)
        self.prev_angular_z = yaw_rate

        turn_strength = min(1.0, abs(error) / self.max_error_deg)
        gamma = max(1e-6, float(self.speed_reduction_gamma))
        reduction = self.max_speed_reduction * (turn_strength ** gamma)
        speed = self.base_speed * (1.0 - reduction)

        cmd = Twist()
        cmd.linear.x = float(speed)
        cmd.angular.z = float(yaw_rate)
        self._pub(cmd)

        self.prev_error = error
        self.frame_count += 1

def main():
    rclpy.init()
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()