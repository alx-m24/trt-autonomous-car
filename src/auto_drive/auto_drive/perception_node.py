import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String, Bool
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import math

class LaneKalmanFilter:
    def __init__(self, dt=0.033):
        # State: [position (cx), velocity (dx)]
        self.x = np.array([[640.0], [0.0]]) 
        
        # State Transition Matrix (Prediction physics: x = x + v*dt)
        self.F = np.array([[1.0, dt], 
                           [0.0, 1.0]])
        
        # Measurement Matrix (We only measure position, not velocity)
        self.H = np.array([[1.0, 0.0]])
        
        # Covariance (Uncertainty in our state)
        self.P = np.array([[1000.0, 0.0], 
                           [0.0, 1000.0]])
        
        # Process Noise (How much we trust our physics model)
        # Increase if the lane center changes directions violently
        self.Q = np.array([[1.0, 0.0], 
                           [0.0, 3.0]])
        
        # Measurement Noise (How much we trust the camera)
        # Increase if the mask is very noisy/flickery
        self.R = np.array([[50.0]])
        
        self.I = np.eye(2)

        self.innovation = 0.0

    def predict(self):
        """Predict where the lane center will be next frame."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[0, 0]

    def update(self, z):
        """Correct the prediction with the actual camera measurement."""
        Z = np.array([[z]])
        
        # Calculate error between measurement and prediction (Innovation)
        y = Z - np.dot(self.H, self.x)

        self.innovation = float(y[0, 0])
        
        # Calculate Kalman Gain (Who do we trust more: prediction or measurement?)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update State and Covariance
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((self.I - np.dot(K, self.H)), self.P)
        
        return self.x[0, 0]

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.bridge = CvBridge()

        self.image_topics = ['/front_camera/image_raw', '/camera/image_raw']
        self.active_image_topic = None

        self.show_windows = bool(os.environ.get('DISPLAY'))

        self._image_subs = []
        for topic in self.image_topics:
            self._image_subs.append(
                self.create_subscription(
                    Image,
                    topic,
                    lambda msg, t=topic: self.image_cb(msg, t),
                    10,
                )
            )

        self.lane_error_pub    = self.create_publisher(Float32, '/lane_error',          10)
        self.lane_detected_pub = self.create_publisher(Bool,    '/lane_detected',       10)
        self.lane_path_pub     = self.create_publisher(Path,    '/lane_path',           10)
        
        # ── UPGRADED: Tell the control node what scene we are in ──
        self.scene_pub         = self.create_publisher(String,  '/scene_state',         10)

        # ── NEW: Anticipatory 90deg corner detector ──
        # Direction the upcoming sharp turn goes ('left'/'right'/'none'), how far
        # ahead the straight road runs out (m), and the turn angle (deg, 90 v1).
        # The control node uses these to trigger a dead-reckoned turn maneuver.
        self.corner_dir_pub    = self.create_publisher(String,  '/corner/direction',    10)
        self.corner_dist_pub   = self.create_publisher(Float32, '/corner/distance_m',   10)
        self.corner_ang_pub    = self.create_publisher(Float32, '/corner/angle_deg',    10)

        self.debug_pub         = self.create_publisher(Image,   '/debug/lane_mask',     10)

        # ── NEW: Diagnostic Telemetry Publishers ──
        self.kf_cx_pub    = self.create_publisher(Float32, '/debug/kf_cx',         10)
        self.la_cx_pub    = self.create_publisher(Float32, '/debug/raw_la_cx',     10)
        self.kf_vel_pub   = self.create_publisher(Float32, '/debug/kf_velocity',   10)
        self.kf_innov_pub = self.create_publisher(Float32, '/debug/kf_innovation', 10)

        # Perspective points — centered trapezoid for balanced left/right capture
        # Input frame: 640x480. Camera is mounted symmetrically.
        # Format: (x, y) — top-left, bottom-left, top-right, bottom-right
        self.tl = (140, 260)
        self.bl = (0,   720)
        self.tr = (1140, 260)
        self.br = (1280, 720)

        self.bird_w = 1000
        self.bird_h = 720

        # Perspective warp matrix only changes when the trapezoid changes, so
        # compute it once instead of every frame in get_bird_eye().
        self._warp_matrix = None
        self._recompute_warp_matrix()

        # ── Tunable parameters ──────────────────────────────────────────────
        self.declare_parameter('max_error_deg',             45.0)
        self.declare_parameter('lookahead_y_frac',           0.18)
        self.declare_parameter('bird_m_per_px_x',            0.005)  # 2026-07-02 winning default (was 0.01)
        self.declare_parameter('bird_m_per_px_y',            0.01)
        self.declare_parameter('path_points',               25)
        self.declare_parameter('path_min_x_m',               0.2)
        self.declare_parameter('path_max_x_m',               1.8)
        self.declare_parameter('path_y_offset_m',            0.0)
        # ── Vertical connectivity gate (stops the path bridging a GAP) ──
        # The row scan builds the centreline near->far. Once it has locked onto
        # the car's road, it must NOT jump across a non-road gap to a DIFFERENT
        # road blob ahead (a roundabout, or a second perpendicular road with
        # bright surface between). Two cuts, both only after the near road is
        # found (valid>0), so leading empties near the car are still skipped:
        #   (a) a near-empty strip = a non-road gap band  -> stop the centreline.
        #   (b) the chosen road cluster jumps laterally from the running centre
        #       by more than path_gap_jump_frac * width -> a different road, stop.
        # Set path_break_on_gap False to restore the old bridge-the-gap behaviour.
        self.declare_parameter('path_break_on_gap',         True)
        self.declare_parameter('path_gap_jump_frac',        0.30)
        # Row-fill-profile cut (strip-count-independent backstop for the gap):
        # scan the road mask bottom->top; a band of path_gap_min_rows consecutive
        # rows whose road-fill fraction drops below path_gap_row_min_fill is a
        # non-road gap -> zero everything above it so only the car's contiguous
        # road survives into the centroid scan. Catches gaps too thin to land a
        # whole empty strip, and far roads that are column-aligned (no lateral
        # jump). Only active when path_break_on_gap is True.
        self.declare_parameter('path_gap_row_min_fill',     0.02)
        self.declare_parameter('path_gap_min_rows',         3)
        # Near-field connectivity gate (the LEADING-gap fix). The two gates above
        # only fire AFTER the car's road has started (valid > 0). If the near
        # field (bottom strips, directly in front) is empty or one-sided-off-
        # image, those strips are skipped and the centreline STARTS on whatever
        # road it first finds higher up — a DIFFERENT road across the bright
        # surround border. Confirmed 2026-07-03: path had 2 poses at x=1.21/1.51
        # (BEV rows ~569/599) and NOTHING below row 600 (near field empty) — it
        # had bridged to a far road, poisoning scene classification into
        # roundabout/turn and firing a spurious dead-reckon turn. This gate
        # refuses to start the centreline when more than this many leading near
        # strips were skipped (road doesn't connect to the car). 4 strips of 40
        # over the 7.2m BEV ~ 0.72m, matching control's pp_near_point_max_x_m.
        #   real road dropped when the near bumper strip is briefly empty ->
        #       RAISE toward 6. still bridges to a far road -> LOWER toward 2.
        #   0 disables (restores start-on-first-road behaviour).
        self.declare_parameter('near_connect_max_empty_strips', 4)
        self.declare_parameter('ransac_n_iter',             50)
        self.declare_parameter('ransac_residual_threshold', 15.0)
        # ── Stage 1: confidence-weighted centreline fit (perception-only) ──
        # When enabled, the published /lane_path comes from ONE weighted
        # polynomial fit over the row centroids instead of the raw per-strip
        # points. Kills the two-section split (continuous by construction) and
        # gives pure pursuit a stable lookahead point. 2026-07-02: ON by default
        # (winning config) — it's what kills the two-section wiggle; degree pinned to
        # 1 (a curve-following straight-line fit; degree 2 flickered "drunk-driver").
        self.declare_parameter('centerline_fit_enable',     True)
        self.declare_parameter('centerline_fit_degree',     1)      # 1=stable heading line; 2 flickers
        self.declare_parameter('centerline_w_both',         1.0)    # both edges in-frame: real measurement
        self.declare_parameter('centerline_w_oneside',      0.20)   # one edge: extrapolated (corner lever)
        self.declare_parameter('centerline_w_flood',        0.05)   # flooded/carried last_cx: no edge info
        self.declare_parameter('centerline_min_total_w',    2.0)    # below this total weight -> fall back to raw
        # Temporal smoothing of the fitted centreline: each frame's fit is
        # independent, so the far end wags even at deg-1 ("drunk driver"). EMA
        # the sampled cx per row across frames on a FIXED grid (so points align).
        # alpha in (0,1]: 1.0 = no smoothing (raw fit), lower = steadier + laggier.
        self.declare_parameter('centerline_smooth_enable',  True)
        self.declare_parameter('centerline_smooth_alpha',   0.35)
        # Consecutive voted frames required to switch OUT of a smooth
        # curve/straight scene INTO a disruptive roundabout/turn one (asymmetric
        # scene hysteresis — see _committed_scene in __init__). Debounces the
        # tight-curve->roundabout flicker that made the fit cut across the bend.
        self.declare_parameter('scene_switch_frames', 4)
        # Fit-dropout hold: when _fit_centerline momentarily returns None, keep
        # publishing the LAST good fit for this many frames instead of snapping
        # /lane_path back to the raw one-sided centroids (that snap is the offroad
        # jerk). 0 disables the hold (immediate raw fallback).
        self.declare_parameter('fit_hold_frames',           5)
        # Forward distance (m) at which the BEV yellow dot is drawn ON the published
        # fit path — mirrors control's pp_lookahead_distance so the dot you watch is
        # the point the car actually steers to (not the unused /lane_error centre).
        self.declare_parameter('overlay_lookahead_m',       0.45)   # keep matched to control pp_lookahead_distance
        # ── track_race recalibration: BEV field-of-view + white threshold, LIVE ──
        # Processed camera frame is 1280x720. The bird's-eye source trapezoid is
        # derived from 3 intuitive knobs so it can be tuned on the running sim:
        #   warp_top_y      : camera row of the FAR edge (smaller = see further
        #                     ahead). THE key corner knob — raise the horizon so a
        #                     flooded corner's road-end / white surround comes into
        #                     view instead of filling the frame with road.
        #   warp_top_inset  : x inset of the top corners (tl.x, tr.x=1280-inset).
        #   warp_bot_inset  : x inset of the bottom corners (bl.x, br.x=1280-inset).
        self.declare_parameter('warp_top_y',        260)
        self.declare_parameter('warp_top_inset',    140)
        self.declare_parameter('warp_bot_inset',      0)
        # ── METRIC BEV (2026-07-03, the calibration root-fix) ─────────────────
        # The legacy trapezoid BEV had NO metric meaning: audit showed it covers
        # only ~0.23–0.86 m of ground while bird_m_per_px declared 7.2 m, and the
        # real lateral scale varies 4x bottom-to-top (0.00023→0.00096 m/px vs the
        # constant 0.005 used). Every downstream "metre" was fiction, and the
        # row-varying road width caused the two-section centreline + rail paths.
        # metric_bev=true derives the image→ground homography from the CAMERA
        # MODEL (URDF: 640x480 @ 80° HFOV, mounted cam_forward_m ahead of
        # base_footprint, cam_height_m up, pitched cam_pitch_deg down) and warps
        # the NATIVE frame (no 1280x720 stretch). The output BEV window is
        # x∈[bev_x_min_m, bev_x_max_m] ahead of base_footprint, y∈±bev_y_half_m,
        # at a CONSTANT bev_m_per_px. bird_m_per_px_x/y are overridden to
        # bev_m_per_px, so all /lane_path coords become REAL metres.
        # Ground the camera can actually see starts ~0.23 m ahead; resolution
        # fades beyond ~1.2 m (one image pixel spans many BEV rows), hence the
        # default window. Areas of the BEV outside the camera view are masked
        # invalid (per-row bounds) so black unknowns can't read as "road".
        self.declare_parameter('metric_bev',        True)
        self.declare_parameter('bev_m_per_px',      0.002)   # 2 mm/px, constant
        self.declare_parameter('bev_x_min_m',       0.24)    # visible ground starts ~0.23
        self.declare_parameter('bev_x_max_m',       1.24)    # beyond ~1.2 m resolution is mush
        self.declare_parameter('bev_y_half_m',      0.50)    # road ±0.151 + surround margin
        self.declare_parameter('cam_img_w',         640)     # URDF camera sensor
        self.declare_parameter('cam_img_h',         480)
        self.declare_parameter('cam_hfov_rad',      1.3962634)  # URDF horizontal_fov
        self.declare_parameter('cam_height_m',      0.13)    # base z 0.08 + cam z 0.05
        self.declare_parameter('cam_forward_m',     0.13)    # cam x in base_link
        self.declare_parameter('cam_pitch_deg',     20.0)    # URDF rpy pitch 0.349
        # Real road width (m): seeds the lane-width estimator and its sanity
        # window in metric mode. track_race road measured 0.302 m from the STL;
        # old_track is 0.20 m (set this if driving old_track).
        self.declare_parameter('road_width_m',      0.302)
        # White (surround) LAB threshold — retune for the new mesh material.
        self.declare_parameter('white_l_lo', 150)
        self.declare_parameter('white_a_lo',  90)
        self.declare_parameter('white_b_lo',  90)
        self.declare_parameter('white_l_hi', 255)
        self.declare_parameter('white_a_hi', 130)
        self.declare_parameter('white_b_hi', 130)
        self.declare_parameter('lane_width_min_frac',        0.20)
        self.declare_parameter('lane_width_max_frac',        0.95)

        # ── Dark-road validation ────────────────────────────────────────────
        self.declare_parameter('road_dark_threshold',  80)
        self.declare_parameter('road_patch_radius',    12)
        self.declare_parameter('road_dark_min_fill',    0.25)

        # ── logging ────────────────────────────────────────────
        self.declare_parameter('verbose_logging', False)
        self.verbose_logging = bool(self.get_parameter('verbose_logging').value)


        # ── NEW: centroid pipeline parameters ──────────────────────────────
        # ema_alpha       : EMA weight on the NEW frame (0=fully smooth, 1=raw).
        #                   Lower = more temporal smoothing = stable on straight roads.
        # ema_alpha_turn  : faster EMA used during turns/roundabouts to track
        #                   rapid heading changes without lag.
        # min_lane_conf   : fraction of centroid strips that must find white pixels
        #                   before the system trusts lane markings.  Below this it
        #                   switches to drivable-region (road_mask) centroid mode.
        self.declare_parameter('ema_alpha',       0.25)
        self.declare_parameter('ema_alpha_turn',  0.30)
        self.declare_parameter('min_lane_conf',   0.20)

        # ── NEW: corner-detector parameters ────────────────────────────────
        # corner_drift_px : how far (px) the road centreline must break from the
        #                   image centre to count as a sharp turn onset. Clamped
        #                   centroids span [0, bird_w], so centre=500 and a hard
        #                   90deg pushes cx toward 0/1000 (dev ~500). 250 = half.
        # corner_min_pts  : min centreline points before trusting the signal.
        # corner_vote_win : majority-vote window (frames) on direction to suppress
        #                   single-frame flicker.
        self.declare_parameter('corner_drift_px',          250.0)
        self.declare_parameter('corner_min_pts',           6)
        self.declare_parameter('corner_vote_win',          5)
        # EMA weight on the new frame's centreline deviation. Lower = steadier
        # direction (more temporal smoothing), at the cost of a little lag.
        self.declare_parameter('corner_dev_alpha',         0.25)

        # ── NEW: floor-asymmetry corner cue ────────────────────────────────
        # At a 90deg corner the dark-road centreline degenerates: the far band
        # of the bird's-eye is replaced by the bright off-road floor, so the
        # centreline pins to image-centre and reads "straight" (deterministic
        # ~9deg under-read). The off-road floor itself is the signal: when the
        # road bends left, white floods the RIGHT of the far band, and vice
        # versa. This cue fires when the centreline drift can't.
        # corner_floor_band_frac : top fraction of the bird's-eye (far field)
        #                          scanned for floor asymmetry.
        # corner_floor_ratio     : min L/R imbalance of off-road white fill in
        #                          that band, (R-L)/(R+L+eps), to call a corner.
        # corner_floor_min_fill  : min total off-road white fill fraction in the
        #                          band before the cue is trusted (ignores noise
        #                          when the far field is still mostly road).
        self.declare_parameter('corner_floor_band_frac',   0.40)
        self.declare_parameter('corner_floor_ratio',       0.55)
        self.declare_parameter('corner_floor_min_fill',    0.15)
        # ── Road forward-reach corner cue (distance-free, autonomous) ──────────
        # Reads the CLEAN road mask geometry instead of the noisy far-band white
        # count. Split the BEV into left/centre/right thirds; measure how far
        # FORWARD each band's road reaches (contiguous solid rows up from the
        # bottom, 0..1). Centre reach collapsing = straight ended (corner here);
        # the side reaching furthest = which way the road goes. The published
        # /corner distance becomes the centre reach (0..1, robust & monotonic),
        # so control's trigger/commit gate on "fraction of road ahead remaining"
        # — reset corner_trigger_dist_m/commit to ~0.60/0.20 when using this.
        # Validated offline on captured raw frames (scratchpad/reach_test.py):
        # the corner is only visible in the RAW image far field, not the BEV.
        # Road = grayscale <= road_max; in a far-field ROI band, take road-fill
        # fraction per L/C/R third. Centre fill collapsing = corner; higher side
        # = direction. Centre fill (0..1, high=open) is the published distance.
        self.declare_parameter('corner_use_reach',       True)
        self.declare_parameter('corner_reach_road_max',  60)     # gray <= this = road (near-black)
        self.declare_parameter('corner_reach_roi_top',   0.24)   # ROI top frac (below horizon/OMar object)
        self.declare_parameter('corner_reach_roi_bot',   0.55)   # ROI bottom frac (skip near foreground)
        self.declare_parameter('corner_reach_center_max',0.40)   # centre fill below this => corner
        self.declare_parameter('corner_reach_side_min',  0.25)   # winning side must exceed this
        self.declare_parameter('corner_reach_ratio',     0.08)   # min |L-R| fill asymmetry for a direction
        # Reach cue authority gate. DISPROVED on this track (2026-07-02): real
        # corners here keep a HEALTHY centreline (n_pts~34, not a flood), so gating
        # the reach cue on `len < corner_min_pts` suppressed a correctly-detected
        # real corner and the car ran off. Default False = reach cue always wins
        # (its original behaviour). Kept as a param only for A/B; do not enable
        # without a real discriminator (n_pts is NOT one).
        self.declare_parameter('corner_reach_require_flood', False)
        # Multi-band SHARP-corner classifier (validated on captured curve+corner
        # REACH_MB 2026-07-02): a sharp corner keeps centre-NEAR full while centre-MID
        # collapses (road ends abruptly ahead = cliff); a curve drops near+mid together
        # (ramp) and never satisfies both -> pursuit keeps it. Fire only when
        # C_near >= near_min AND C_mid <= mid_max.
        self.declare_parameter('corner_sharp_near_min', 0.75)   # centre-near still ~full
        self.declare_parameter('corner_sharp_mid_max',  0.35)   # centre-mid has collapsed
        # Open-road corroboration gate (defense-in-depth for the reach cue).
        # The reach cue may only fire the maneuver when the BEV centreline ALSO
        # says the road ENDS within view: the farthest strip backed by a REAL
        # edge (weight > w_flood; flood-carried strips don't count) must be at
        # most this many BEV metres ahead. Offline-validated on scratchpad/cam
        # 2026-07-03 (gate_test.py): every straight/curve frame reads the FULL
        # BEV height (7.11m — road runs off the top of the frame), while the
        # real corner reads <= 6.75m from the first correct cue fire. So the
        # gate is really "does the road reach the far edge of the BEV or not";
        # 6.9 sits between the two with one-strip (~0.18m) quantization margin.
        # NOT the disproved n_pts flood gate: keys on edge-informed EXTENT, not
        # point count (flood emits many degenerate points; counts stay high).
        #   real corner fires LATE / never -> RAISE toward 7.2 (watch edge_ext
        #       in REACH_DBG on the corner approach).
        #   still false-fires on a straight -> check edge_ext in the log; if it
        #       reads < 6.9 on that straight the road mask has a false end
        #       (gap/shadow) — fix the mask, don't lower the gate blindly.
        #   <= 0 disables the gate (previous behaviour).
        self.declare_parameter('corner_path_end_max_m', 6.9)

        # 2026-07-04: near the roundabout, edge_ext SATURATES to the same BEV-depth
        # ceiling (~1.23m) whether the far edge belongs to the car's own real
        # corner or to a different road segment/the island visible far off — so
        # corner_path_end_max_m can't tell them apart there (confirmed live: a
        # genuine right corner and an earlier false alarm both read edge_ext=1.23).
        # Corroborate with the independently-computed drift EMA instead: if the
        # BEV centreline is ALSO already bending the reach cue's claimed direction
        # (even below its own standalone corner_drift_px threshold), the reach fire
        # is very likely real, so let it override the open-road gate.
        #   2nd corner still doesn't fire -> raise this (less corroboration needed).
        #   false corner near the roundabout reappears -> lower this or add a
        #       persistence requirement (consult REACH_DBG + CORNER_DBG ema jointly).
        self.declare_parameter('corner_reach_drift_corroborate_px', 30.0)

        # ── NEW: perf — number of horizontal strips scanned for row centroids.
        # Fewer strips = less Python per-frame overhead, coarser path. 40 keeps
        # the original behavior; ~24 noticeably faster with little quality loss.
        self.declare_parameter('centroid_strips', 40)

        self.max_error_deg             = float(self.get_parameter('max_error_deg').value)
        self.lookahead_y_frac          = float(self.get_parameter('lookahead_y_frac').value)
        self.bird_m_per_px_x           = float(self.get_parameter('bird_m_per_px_x').value)
        self.bird_m_per_px_y           = float(self.get_parameter('bird_m_per_px_y').value)
        self.path_points               = int(self.get_parameter('path_points').value)
        self.path_min_x_m              = float(self.get_parameter('path_min_x_m').value)
        self.path_max_x_m              = float(self.get_parameter('path_max_x_m').value)
        self.path_break_on_gap         = bool(self.get_parameter('path_break_on_gap').value)
        self.path_gap_jump_frac        = float(self.get_parameter('path_gap_jump_frac').value)
        self.path_gap_row_min_fill     = float(self.get_parameter('path_gap_row_min_fill').value)
        self.path_gap_min_rows         = int(self.get_parameter('path_gap_min_rows').value)
        self.near_connect_max_empty_strips = int(self.get_parameter('near_connect_max_empty_strips').value)
        self.path_y_offset_m           = float(self.get_parameter('path_y_offset_m').value)
        self.ransac_n_iter             = int(self.get_parameter('ransac_n_iter').value)
        self.ransac_residual_threshold = float(self.get_parameter('ransac_residual_threshold').value)
        self.cl_fit_enable             = bool(self.get_parameter('centerline_fit_enable').value)
        self.cl_fit_degree             = int(self.get_parameter('centerline_fit_degree').value)
        self.cl_w_both                 = float(self.get_parameter('centerline_w_both').value)
        self.cl_w_oneside              = float(self.get_parameter('centerline_w_oneside').value)
        self.cl_w_flood                = float(self.get_parameter('centerline_w_flood').value)
        self.cl_min_total_w            = float(self.get_parameter('centerline_min_total_w').value)
        self.cl_smooth_enable          = bool(self.get_parameter('centerline_smooth_enable').value)
        self.cl_smooth_alpha           = float(self.get_parameter('centerline_smooth_alpha').value)
        self._scene_switch_frames      = max(1, int(self.get_parameter('scene_switch_frames').value))
        self.fit_hold_frames           = int(self.get_parameter('fit_hold_frames').value)
        self.overlay_lookahead_m       = float(self.get_parameter('overlay_lookahead_m').value)
        self.warp_top_y                = int(self.get_parameter('warp_top_y').value)
        self.warp_top_inset            = int(self.get_parameter('warp_top_inset').value)
        self.warp_bot_inset            = int(self.get_parameter('warp_bot_inset').value)
        self.metric_bev                = bool(self.get_parameter('metric_bev').value)
        self.bev_m_per_px              = float(self.get_parameter('bev_m_per_px').value)
        self.bev_x_min_m               = float(self.get_parameter('bev_x_min_m').value)
        self.bev_x_max_m               = float(self.get_parameter('bev_x_max_m').value)
        self.bev_y_half_m              = float(self.get_parameter('bev_y_half_m').value)
        self.cam_img_w                 = int(self.get_parameter('cam_img_w').value)
        self.cam_img_h                 = int(self.get_parameter('cam_img_h').value)
        self.cam_hfov_rad              = float(self.get_parameter('cam_hfov_rad').value)
        self.cam_height_m              = float(self.get_parameter('cam_height_m').value)
        self.cam_forward_m             = float(self.get_parameter('cam_forward_m').value)
        self.cam_pitch_deg             = float(self.get_parameter('cam_pitch_deg').value)
        self.road_width_m              = float(self.get_parameter('road_width_m').value)
        self.white_l_lo                = int(self.get_parameter('white_l_lo').value)
        self.white_a_lo                = int(self.get_parameter('white_a_lo').value)
        self.white_b_lo                = int(self.get_parameter('white_b_lo').value)
        self.white_l_hi                = int(self.get_parameter('white_l_hi').value)
        self.white_a_hi                = int(self.get_parameter('white_a_hi').value)
        self.white_b_hi                = int(self.get_parameter('white_b_hi').value)
        self.lane_width_min_frac       = float(self.get_parameter('lane_width_min_frac').value)
        self.lane_width_max_frac       = float(self.get_parameter('lane_width_max_frac').value)
        self.road_dark_threshold       = int(self.get_parameter('road_dark_threshold').value)
        self.road_patch_radius         = int(self.get_parameter('road_patch_radius').value)
        self.road_dark_min_fill        = float(self.get_parameter('road_dark_min_fill').value)
        self._ema_alpha                = float(self.get_parameter('ema_alpha').value)
        self._ema_alpha_turn           = float(self.get_parameter('ema_alpha_turn').value)
        self._min_lane_conf            = float(self.get_parameter('min_lane_conf').value)
        self._corner_drift_px          = float(self.get_parameter('corner_drift_px').value)
        self._corner_min_pts           = int(self.get_parameter('corner_min_pts').value)
        self._corner_vote_win          = int(self.get_parameter('corner_vote_win').value)
        self._corner_dev_alpha         = float(self.get_parameter('corner_dev_alpha').value)
        self._corner_dev_ema           = 0.0
        self._corner_history           = []
        self._corner_floor_band_frac   = float(self.get_parameter('corner_floor_band_frac').value)
        self._corner_floor_ratio       = float(self.get_parameter('corner_floor_ratio').value)
        self._corner_floor_min_fill    = float(self.get_parameter('corner_floor_min_fill').value)
        self._corner_use_reach         = bool(self.get_parameter('corner_use_reach').value)
        self._corner_reach_road_max    = int(self.get_parameter('corner_reach_road_max').value)
        self._corner_reach_roi_top     = float(self.get_parameter('corner_reach_roi_top').value)
        self._corner_reach_roi_bot     = float(self.get_parameter('corner_reach_roi_bot').value)
        self._corner_reach_center_max  = float(self.get_parameter('corner_reach_center_max').value)
        self._corner_reach_side_min    = float(self.get_parameter('corner_reach_side_min').value)
        self._corner_reach_ratio       = float(self.get_parameter('corner_reach_ratio').value)
        self._corner_reach_require_flood = bool(self.get_parameter('corner_reach_require_flood').value)
        self._corner_sharp_near_min    = float(self.get_parameter('corner_sharp_near_min').value)
        self._corner_sharp_mid_max     = float(self.get_parameter('corner_sharp_mid_max').value)
        self._corner_path_end_max_m    = float(self.get_parameter('corner_path_end_max_m').value)
        self._corner_reach_drift_corroborate_px = float(self.get_parameter('corner_reach_drift_corroborate_px').value)
        self._reach_dbg                = (0.0, 0.0, 0.0)   # last (L,C,R) forward reach
        self._reach_mb                 = (0.0,)*9          # last near/mid/far x L/C/R (curve-vs-corner R&D)
        self._corner_camdir_ema        = 0.0               # EMA of the far-field road-mass offset (turn direction)
        # #2: confidence that reflects only strips where a REAL road edge was
        # found (not the flooded "cx=centre" fallback). Set each frame by the
        # is_road pass of _row_centroids; used so a flooded-straight frame can't
        # masquerade as a high-confidence straight.
        self._road_edge_conf           = 0.0
        self._cl_ema_xs                = None   # per-grid-row EMA of the fitted centreline
        self._last_good_fit_pts        = None   # last successful fit sample, for dropout hold
        self._fit_dropout_count        = 0      # consecutive frames the fit has been None
        self._floor_cue_str            = 0.0   # last floor-asymmetry imbalance
        self._centroid_strips          = max(8, int(self.get_parameter('centroid_strips').value))

        self.add_on_set_parameters_callback(self._param_cb)

        # LAB thresholds — white lane markings / surround: high L, neutral a/b.
        # Built from live params (see _apply_white_params) so the new mesh can be
        # retuned without a rebuild. Also re-derive the BEV trapezoid from params.
        self._apply_white_params()
        self._apply_warp_params()

        # ── NEW: Dynamic Lane Width Estimator ──
        # Metric mode: seed from the REAL road width (constant across BEV rows
        # now that the scale is true) with a ±40% sanity window. Legacy mode
        # keeps the old 65%-of-width guess and 45–75% window.
        if self.metric_bev:
            self.estimated_lane_w = float(self.road_width_m) / float(self.bev_m_per_px)
            self._lane_w_lo_px = 0.6 * self.estimated_lane_w
            self._lane_w_hi_px = 1.4 * self.estimated_lane_w
            self._two_lines_min_px = 0.5 * self.estimated_lane_w
        else:
            self.estimated_lane_w = float(self.bird_w) * 0.65
            self._lane_w_lo_px = float(self.bird_w) * 0.45
            self._lane_w_hi_px = float(self.bird_w) * 0.75
            self._two_lines_min_px = float(self.bird_w) * 0.30
        # A slow EMA alpha (e.g., 0.05) prevents noise from violently changing the width
        self.lane_w_alpha = 0.05

        # Legacy sliding-window history (kept for _sliding_windows compatibility)
        self.prevLx = []
        self.prevRx = []

        self.frame_count = 0

        # ── Centroid pipeline state ─────────────────────────────────────────
        # _ema_cx           : exponentially-smoothed lookahead centre-x (pixels).
        #                     Initialised to image centre. Persists across frames.
        # _ema_valid_frames : how many frames have updated the EMA so far.
        #                     Used to gate lane_detected before we have real data.
        # _scene_history    : short rolling buffer of per-frame scene classifications
        #                     for majority-vote stabilisation.
        # _SCENE_VOTE_WIN   : window length (frames) for the scene majority vote.
        self.kf = LaneKalmanFilter(dt=0.033) # 30Hz frame rate
        self.filter_initialized = False
        self.current_cx = float(self.bird_w) / 2.0
        self._scene_history     = []
        self._SCENE_VOTE_WIN    = 5
        # Asymmetric scene hysteresis. The 5-frame vote above is symmetric, so a
        # TIGHT CURVE (whose centroids spread past TURN_VAR_THRESH for ~half the
        # frames and read 'roundabout') still flip-flops the committed scene ->
        # 'roundabout' switches the path to centroid_flow, which publishes a
        # STRAIGHT cut-across fit (screenshot-proven 2026-07-06), jerking the
        # lookahead ~0.15m at ~3Hz. Fix: it's HARD to leave a smooth
        # lane-following scene (curve/straight) for a DISRUPTIVE one
        # (roundabout/turn) — that needs scene_switch_frames consecutive voted
        # frames — but EASY to return to smooth tracking (switches immediately).
        # A genuine, sustained roundabout still crosses the threshold; a 1-2
        # frame variance spike on a curve can't. Tunable (scene_switch_frames):
        #   real roundabout entered LATE / cuts in -> LOWER toward 2.
        #   tight curve still flips to the straight cut -> RAISE toward 6.
        self._committed_scene     = 'straight'
        self._scene_switch_count  = 0

        self.get_logger().info('PerceptionNode started — centroid lookahead pipeline active.')
        self.get_logger().info(f'Subscribed to: {self.image_topics}')
        self.get_logger().info(f'OpenCV windows: {self.show_windows}')
        self.get_logger().info(
            f'Perspective: TL{self.tl} BL{self.bl} TR{self.tr} BR{self.br}'
        )
        self.get_logger().info(
            f'Dark-road validation: L<{self.road_dark_threshold}  '
            f'patch_r={self.road_patch_radius}px  '
            f'min_fill={self.road_dark_min_fill}'
        )
        self.get_logger().info(
            f'EMA alpha: straight/curve={self._ema_alpha}  turn/roundabout={self._ema_alpha_turn}  '
            f'min_lane_conf={self._min_lane_conf}'
        )

    # ── Live parameter callback ─────────────────────────────────────────────

    def _param_cb(self, params):
        for p in params:
            if p.name == 'lookahead_y_frac':
                self.lookahead_y_frac = float(p.value)
                self.get_logger().info(f'lookahead_y_frac -> {self.lookahead_y_frac}')
            elif p.name == 'max_error_deg':
                self.max_error_deg = float(p.value)
            elif p.name == 'bird_m_per_px_x':
                if self.metric_bev:
                    self.get_logger().warn(
                        'bird_m_per_px_x ignored: metric_bev forces the TRUE scale bev_m_per_px')
                else:
                    self.bird_m_per_px_x = float(p.value)
            elif p.name == 'bird_m_per_px_y':
                if self.metric_bev:
                    self.get_logger().warn(
                        'bird_m_per_px_y ignored: metric_bev forces the TRUE scale bev_m_per_px')
                else:
                    self.bird_m_per_px_y = float(p.value)
            elif p.name == 'metric_bev':
                self.metric_bev = bool(p.value)
                self._apply_warp_params()
                self.get_logger().info(f'metric_bev -> {self.metric_bev}')
            elif p.name in ('bev_m_per_px', 'bev_x_min_m', 'bev_x_max_m', 'bev_y_half_m',
                            'cam_hfov_rad', 'cam_height_m', 'cam_forward_m', 'cam_pitch_deg'):
                setattr(self, p.name, float(p.value))
                if self.metric_bev:
                    self._apply_warp_params()
                    self._cl_ema_xs = None      # BEV geometry changed: drop EMA history
            elif p.name in ('cam_img_w', 'cam_img_h'):
                setattr(self, p.name, int(p.value))
                if self.metric_bev:
                    self._apply_warp_params()
            elif p.name == 'road_width_m':
                self.road_width_m = float(p.value)
                if self.metric_bev:
                    self.estimated_lane_w = self.road_width_m / self.bev_m_per_px
                    self._lane_w_lo_px = 0.6 * self.estimated_lane_w
                    self._lane_w_hi_px = 1.4 * self.estimated_lane_w
                    self._two_lines_min_px = 0.5 * self.estimated_lane_w
                    self.get_logger().info(
                        f'road_width_m -> {self.road_width_m} (lane_w={self.estimated_lane_w:.0f}px)')
            elif p.name == 'path_points':
                self.path_points = int(p.value)
            elif p.name == 'path_min_x_m':
                self.path_min_x_m = float(p.value)
            elif p.name == 'path_max_x_m':
                self.path_max_x_m = float(p.value)
            elif p.name == 'path_break_on_gap':
                self.path_break_on_gap = bool(p.value)
            elif p.name == 'path_gap_jump_frac':
                self.path_gap_jump_frac = float(p.value)
            elif p.name == 'path_gap_row_min_fill':
                self.path_gap_row_min_fill = float(p.value)
            elif p.name == 'path_gap_min_rows':
                self.path_gap_min_rows = int(p.value)
            elif p.name == 'near_connect_max_empty_strips':
                self.near_connect_max_empty_strips = int(p.value)
            elif p.name == 'path_y_offset_m':
                self.path_y_offset_m = float(p.value)
            elif p.name == 'ransac_n_iter':
                self.ransac_n_iter = int(p.value)
            elif p.name == 'ransac_residual_threshold':
                self.ransac_residual_threshold = float(p.value)
            elif p.name == 'centerline_fit_enable':
                self.cl_fit_enable = bool(p.value)
                self.get_logger().info(f'centerline_fit_enable -> {self.cl_fit_enable}')
            elif p.name == 'centerline_fit_degree':
                self.cl_fit_degree = int(p.value)
            elif p.name == 'centerline_w_both':
                self.cl_w_both = float(p.value)
            elif p.name == 'centerline_w_oneside':
                self.cl_w_oneside = float(p.value)
            elif p.name == 'centerline_w_flood':
                self.cl_w_flood = float(p.value)
            elif p.name == 'centerline_min_total_w':
                self.cl_min_total_w = float(p.value)
            elif p.name == 'centerline_smooth_enable':
                self.cl_smooth_enable = bool(p.value)
                self._cl_ema_xs = None
            elif p.name == 'centerline_smooth_alpha':
                self.cl_smooth_alpha = float(p.value)
            elif p.name == 'scene_switch_frames':
                self._scene_switch_frames = max(1, int(p.value))
            elif p.name == 'fit_hold_frames':
                self.fit_hold_frames = int(p.value)
            elif p.name == 'overlay_lookahead_m':
                self.overlay_lookahead_m = float(p.value)
            elif p.name in ('warp_top_y', 'warp_top_inset', 'warp_bot_inset'):
                setattr(self, p.name, int(p.value))
                self._apply_warp_params()
                self.get_logger().info(
                    f'{p.name} -> {int(p.value)}  (BEV TL{self.tl} TR{self.tr})')
            elif p.name in ('white_l_lo', 'white_a_lo', 'white_b_lo',
                            'white_l_hi', 'white_a_hi', 'white_b_hi'):
                setattr(self, p.name, int(p.value))
                self._apply_white_params()
                self.get_logger().info(f'{p.name} -> {int(p.value)}  '
                                       f'(white {self.lower_white}..{self.upper_white})')
            elif p.name == 'lane_width_min_frac':
                self.lane_width_min_frac = float(p.value)
            elif p.name == 'lane_width_max_frac':
                self.lane_width_max_frac = float(p.value)
            elif p.name == 'road_dark_threshold':
                self.road_dark_threshold = int(p.value)
                self.get_logger().info(f'road_dark_threshold -> {self.road_dark_threshold}')
            elif p.name == 'road_patch_radius':
                self.road_patch_radius = int(p.value)
            elif p.name == 'road_dark_min_fill':
                self.road_dark_min_fill = float(p.value)
                self.get_logger().info(f'road_dark_min_fill -> {self.road_dark_min_fill}')
            elif p.name == 'ema_alpha':
                self._ema_alpha = float(p.value)
                self.get_logger().info(f'ema_alpha -> {self._ema_alpha}')
            elif p.name == 'ema_alpha_turn':
                self._ema_alpha_turn = float(p.value)
                self.get_logger().info(f'ema_alpha_turn -> {self._ema_alpha_turn}')
            elif p.name == 'corner_drift_px':
                self._corner_drift_px = float(p.value)
                self.get_logger().info(f'corner_drift_px -> {self._corner_drift_px}')
            elif p.name == 'corner_min_pts':
                self._corner_min_pts = int(p.value)
                self.get_logger().info(f'corner_min_pts -> {self._corner_min_pts}')
            elif p.name == 'corner_dev_alpha':
                self._corner_dev_alpha = float(p.value)
                self.get_logger().info(f'corner_dev_alpha -> {self._corner_dev_alpha}')
            elif p.name == 'corner_vote_win':
                self._corner_vote_win = int(p.value)
                self.get_logger().info(f'corner_vote_win -> {self._corner_vote_win}')
            elif p.name == 'corner_floor_band_frac':
                self._corner_floor_band_frac = float(p.value)
                self.get_logger().info(f'corner_floor_band_frac -> {self._corner_floor_band_frac}')
            elif p.name == 'corner_floor_ratio':
                self._corner_floor_ratio = float(p.value)
                self.get_logger().info(f'corner_floor_ratio -> {self._corner_floor_ratio}')
            elif p.name == 'corner_floor_min_fill':
                self._corner_floor_min_fill = float(p.value)
                self.get_logger().info(f'corner_floor_min_fill -> {self._corner_floor_min_fill}')
            elif p.name == 'corner_use_reach':
                self._corner_use_reach = bool(p.value)
                self.get_logger().info(f'corner_use_reach -> {self._corner_use_reach}')
            elif p.name == 'corner_reach_road_max':
                self._corner_reach_road_max = int(p.value)
            elif p.name == 'corner_reach_roi_top':
                self._corner_reach_roi_top = float(p.value)
            elif p.name == 'corner_reach_roi_bot':
                self._corner_reach_roi_bot = float(p.value)
            elif p.name == 'corner_reach_center_max':
                self._corner_reach_center_max = float(p.value)
            elif p.name == 'corner_reach_side_min':
                self._corner_reach_side_min = float(p.value)
            elif p.name == 'corner_reach_ratio':
                self._corner_reach_ratio = float(p.value)
            elif p.name == 'corner_reach_require_flood':
                self._corner_reach_require_flood = bool(p.value)
                self.get_logger().info(f'corner_reach_require_flood -> {self._corner_reach_require_flood}')
            elif p.name == 'corner_sharp_near_min':
                self._corner_sharp_near_min = float(p.value)
            elif p.name == 'corner_sharp_mid_max':
                self._corner_sharp_mid_max = float(p.value)
            elif p.name == 'corner_path_end_max_m':
                self._corner_path_end_max_m = float(p.value)
                self.get_logger().info(f'corner_path_end_max_m -> {self._corner_path_end_max_m}')
            elif p.name == 'corner_reach_drift_corroborate_px':
                self._corner_reach_drift_corroborate_px = float(p.value)
                self.get_logger().info(f'corner_reach_drift_corroborate_px -> {self._corner_reach_drift_corroborate_px}')
            elif p.name == 'centroid_strips':
                self._centroid_strips = max(8, int(p.value))
                self.get_logger().info(f'centroid_strips -> {self._centroid_strips}')
            elif p.name == 'verbose_logging':
                self.verbose_logging = bool(p.value)
                self.get_logger().info(f'verbose_logging -> {self.verbose_logging}')
            elif p.name == 'min_lane_conf':
                self._min_lane_conf = float(p.value)
                self.get_logger().info(f'min_lane_conf -> {self._min_lane_conf}')
        return SetParametersResult(successful=True)

    # ── RANSAC polynomial fit ───────────────────────────────────────────────
    def fit_ransac_poly(
        self,
        y_vals,
        x_vals,
        degree: int = 2,
        n_iter: int = 50,
        residual_threshold: float = 15.0,
    ):
        """Fit a polynomial with numpy-only RANSAC."""
        y = np.asarray(y_vals, dtype=np.float32)
        x = np.asarray(x_vals, dtype=np.float32)
        n = int(min(len(y), len(x)))
        if n < (degree + 1):
            return None

        y = y[:n].reshape(-1)
        x = x[:n].reshape(-1)

        rng               = np.random.default_rng()
        best_inlier_mask  = None
        best_inlier_count = 0

        for _ in range(int(n_iter)):
            try:
                idx           = rng.choice(n, size=degree + 1, replace=False)
                coeffs        = np.polyfit(y[idx], x[idx], degree)
                residuals     = np.abs(np.polyval(coeffs, y) - x)
                inlier_mask   = residuals < float(residual_threshold)
                inlier_count  = int(np.count_nonzero(inlier_mask))
            except Exception:
                continue

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inlier_mask  = inlier_mask

        if best_inlier_mask is not None and best_inlier_count > (degree + 1):
            try:
                return np.polyfit(y[best_inlier_mask], x[best_inlier_mask], degree)
            except Exception:
                pass

        # Fallback: use all points
        try:
            return np.polyfit(y, x, degree)
        except Exception:
            return None

    # ── Dark-road centre validation ─────────────────────────────────────────

    def _centre_is_dark(self, road_mask: np.ndarray, cx: int, cy: int) -> bool:
        """Return True if the patch around (cx, cy) is predominantly dark road."""
        r     = max(1, int(self.road_patch_radius))
        h, w  = road_mask.shape[:2]
        x0    = max(0, cx - r);  x1 = min(w, cx + r + 1)
        y0    = max(0, cy - r);  y1 = min(h, cy + r + 1)
        patch = road_mask[y0:y1, x0:x1]
        if patch.size == 0:
            return False
        fill = float(np.count_nonzero(patch)) / float(patch.size)
        return fill >= float(self.road_dark_min_fill)

    # ── Lane path publisher (legacy — kept for fit-based callers) ───────────

    def _publish_lane_path_from_fits(self, left_fit, right_fit,
                                     y_min: float, y_max: float):
        """Publish centreline Path from polynomial fits (REP-103)."""
        n         = max(5, int(self.path_points))
        y_samples = np.linspace(y_max, y_min, n)

        path                  = Path()
        path.header.stamp     = self.get_clock().now().to_msg()
        path.header.frame_id  = 'base_footprint'

        for y_px in y_samples:
            left_x      = float(np.polyval(left_fit,  y_px))
            right_x     = float(np.polyval(right_fit, y_px))
            centre_x_px = 0.5 * (left_x + right_x)

            x_m = self._bev_x0() + (float(self.bird_h) - float(y_px)) * float(self.bird_m_per_px_y)
            if x_m < float(self.path_min_x_m):
                continue
            y_m = (float(self.bird_w) * 0.5 - centre_x_px) * float(self.bird_m_per_px_x)

            ps                    = PoseStamped()
            ps.header             = path.header
            ps.pose.position.x    = float(x_m)
            ps.pose.position.y    = float(y_m)
            ps.pose.position.z    = 0.0
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        if len(path.poses) >= 2:
            first = path.poses[0].pose.position
            last = path.poses[-1].pose.position

            angle_deg = math.degrees(
                math.atan2(
                    last.y - first.y,
                    last.x - first.x,
                )
            )

            self.get_logger().info(
                f"PATH_DBG "
                f"angle={angle_deg:.1f} "
                f"first=({first.x:.2f},{first.y:.2f}) "
                f"last=({last.x:.2f},{last.y:.2f})"
            )

        self.lane_path_pub.publish(path)

    # ── Perspective warp ────────────────────────────────────────────────────

    def _apply_white_params(self):
        """Rebuild the white LAB threshold arrays from live params."""
        self.lower_white = np.array([self.white_l_lo, self.white_a_lo, self.white_b_lo])
        self.upper_white = np.array([self.white_l_hi, self.white_a_hi, self.white_b_hi])

    def _bev_x0(self) -> float:
        """Forward offset (m) of the BEV's BOTTOM row from base_footprint.
        Metric BEV starts at bev_x_min_m (the camera can't see closer);
        the legacy warp pretended the bottom row was x=0."""
        return float(self.bev_x_min_m) if self.metric_bev else 0.0

    def _project_ground_to_image(self, X, Y):
        """Project a ground point (X ahead, Y left of base_footprint, z=0) to
        NATIVE image pixels using the URDF camera model. Returns (u, v) floats
        (may be outside the frame — that's fine for homography anchors)."""
        p  = math.radians(self.cam_pitch_deg)
        fx = (self.cam_img_w * 0.5) / math.tan(self.cam_hfov_rad * 0.5)
        fy = fx                                   # square pixels
        cx = self.cam_img_w * 0.5
        cy = self.cam_img_h * 0.5
        # Vector from the camera to the ground point, in body axes (x fwd, y left, z up)
        dx = X - self.cam_forward_m
        dy = Y
        dz = -self.cam_height_m
        # Camera axes in body frame (pitched down by p):
        #   optical z_c=(cos p, 0, -sin p), x_c=right=(0,-1,0), y_c=down=(-sin p, 0, -cos p)
        z_c = dx * math.cos(p) - dz * math.sin(p)
        x_c = -dy
        y_c = -dx * math.sin(p) - dz * math.cos(p)
        return (cx + fx * x_c / z_c, cy + fy * y_c / z_c)

    def _apply_metric_bev(self):
        """Build the metric ground-plane homography and validity mask.

        BEV pixel convention (kept from the legacy warp so downstream is
        unchanged): row 0 = FAR (x = bev_x_max_m), bottom row = NEAR
        (x = bev_x_min_m); col 0 = LEFT (+y), so
            x_m = bev_x_min_m + (bird_h - row) * bev_m_per_px
            y_m = (bird_w/2 - col) * bev_m_per_px
        Regions of the BEV the camera cannot see (near-field corners) are
        marked invalid; _row_centroids uses the per-row valid bounds so the
        view boundary is never mistaken for a road edge."""
        s = float(self.bev_m_per_px)
        self.bird_w = max(8, int(round(2.0 * self.bev_y_half_m / s)))
        self.bird_h = max(8, int(round((self.bev_x_max_m - self.bev_x_min_m) / s)))
        # Force the metre-per-pixel constants to the TRUE scale so every
        # existing conversion site publishes real metres.
        self.bird_m_per_px_x = s
        self.bird_m_per_px_y = s
        # 4 ground anchors (BEV corners) -> native image pixels -> homography.
        corners_ground = [
            (self.bev_x_max_m,  self.bev_y_half_m),   # BEV (0, 0)          top-left
            (self.bev_x_max_m, -self.bev_y_half_m),   # BEV (bird_w, 0)     top-right
            (self.bev_x_min_m,  self.bev_y_half_m),   # BEV (0, bird_h)     bottom-left
            (self.bev_x_min_m, -self.bev_y_half_m),   # BEV (bird_w, bird_h) bottom-right
        ]
        src = np.float32([self._project_ground_to_image(X, Y) for X, Y in corners_ground])
        dst = np.float32([[0, 0], [self.bird_w, 0], [0, self.bird_h], [self.bird_w, self.bird_h]])
        self._warp_matrix = cv2.getPerspectiveTransform(src, dst)
        # Validity: warp an all-white native frame; anything not fully sampled
        # from inside the camera image is unknown. Also mask above-horizon rows
        # (can't happen inside this window, but cheap to be exact).
        ones = np.full((self.cam_img_h, self.cam_img_w), 255, np.uint8)
        valid = cv2.warpPerspective(ones, self._warp_matrix, (self.bird_w, self.bird_h))
        self._bev_valid = (valid >= 250).astype(np.uint8) * 255
        lo = np.full(self.bird_h, -1, np.int32)
        hi = np.full(self.bird_h, -1, np.int32)
        for r in range(self.bird_h):
            cols = np.flatnonzero(self._bev_valid[r])
            if cols.size:
                lo[r], hi[r] = int(cols[0]), int(cols[-1])
        self._bev_row_lo, self._bev_row_hi = lo, hi
        self.get_logger().info(
            f'METRIC BEV: {self.bird_w}x{self.bird_h}px @ {s*1000:.1f}mm/px, '
            f'x=[{self.bev_x_min_m:.2f},{self.bev_x_max_m:.2f}]m y=±{self.bev_y_half_m:.2f}m, '
            f'cam h={self.cam_height_m} fwd={self.cam_forward_m} pitch={self.cam_pitch_deg}°')

    def _apply_warp_params(self):
        """Recompute the BEV warp. Metric mode uses the camera-model homography;
        legacy mode derives the hand trapezoid from the 3 warp knobs
        (frame stretched to 1280x720; top corners inset by warp_top_inset)."""
        if self.metric_bev:
            self._apply_metric_bev()
            return
        self._bev_valid = None
        self._bev_row_lo = self._bev_row_hi = None
        # Restore the legacy BEV size + declared (fictitious) scales in case
        # metric mode was toggled off at runtime.
        self.bird_w = 1000
        self.bird_h = 720
        self.bird_m_per_px_x = float(self.get_parameter('bird_m_per_px_x').value)
        self.bird_m_per_px_y = float(self.get_parameter('bird_m_per_px_y').value)
        W = 1280
        ty = int(self.warp_top_y)
        ti = int(self.warp_top_inset)
        bi = int(self.warp_bot_inset)
        self.tl = (ti,      ty)
        self.tr = (W - ti,  ty)
        self.bl = (bi,      720)
        self.br = (W - bi,  720)
        self._recompute_warp_matrix()

    def _recompute_warp_matrix(self):
        pts1 = np.float32([self.tl, self.bl, self.tr, self.br])
        pts2 = np.float32([[0, 0], [0, self.bird_h], [self.bird_w, 0], [self.bird_w, self.bird_h]])
        self._warp_matrix = cv2.getPerspectiveTransform(pts1, pts2)

    def get_bird_eye(self, frame):
        return cv2.warpPerspective(frame, self._warp_matrix, (self.bird_w, self.bird_h))

    # ── Sliding windows ─────────────────────────────────────────────────────

    def sliding_windows(self, mask, bird_h=None, bird_w=None):
        if bird_h is None: bird_h = self.bird_h
        if bird_w is None: bird_w = self.bird_w

        h, w      = mask.shape[:2]
        histogram = np.sum(mask[h//2:, :], axis=0)
        midpoint  = w // 2

        def find_lane_base(hist, max_val):
            if np.sum(hist) == 0:
                return max_val // 2
            indices      = np.arange(len(hist))
            weighted_sum = np.sum(indices * hist.astype(np.float32))
            total        = np.sum(hist.astype(np.float32))
            return int(weighted_sum / total) if total > 0 else max_val // 2

        left_base  = find_lane_base(histogram[:midpoint], midpoint)
        right_base = find_lane_base(histogram[midpoint:], midpoint) + midpoint

        y             = h
        lx, rx        = [], []
        msk           = mask.copy()
        window_height = max(18, h // 20)

        while y > 0:
            for base, lst in [(left_base, lx), (right_base, rx)]:
                search_width = max(40, w // 8)
                img_s = mask[
                    max(0, y - window_height):y,
                    max(0, base - search_width):min(w, base + search_width)
                ]
                contours, _ = cv2.findContours(img_s, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    M = cv2.moments(c)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        lane_x = max(0, base - search_width) + cx
                        lst.extend([lane_x, lane_x, lane_x])
                        if lst is lx: left_base  = max(0, base - search_width) + cx
                        else:         right_base = max(0, base - search_width) + cx
            cv2.rectangle(
                msk,
                (max(0, left_base  - search_width), y),
                (min(w, left_base  + search_width), max(0, y - window_height)),
                (255, 255, 255), 2
            )
            cv2.rectangle(
                msk,
                (max(0, right_base - search_width), y),
                (min(w, right_base + search_width), max(0, y - window_height)),
                (255, 255, 255), 2
            )
            y -= window_height

        if len(lx) == 0: lx = self.prevLx
        else:            self.prevLx = lx
        if len(rx) == 0: rx = self.prevRx
        else:            self.prevRx = rx

        return lx, rx, left_base, right_base, msk

    # ── Helpers ─────────────────────────────────────────────────────────────

    def update_perspective(self, tl=None, bl=None, tr=None, br=None):
        if tl is not None: self.tl = tl
        if bl is not None: self.bl = bl
        if tr is not None: self.tr = tr
        if br is not None: self.br = br
        self._recompute_warp_matrix()
        self.get_logger().info(
            f'Perspective updated: TL{self.tl} BL{self.bl} TR{self.tr} BR{self.br}'
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ── NEW: Centroid pipeline helpers ──────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    def _cut_road_at_gap(self, road_mask: np.ndarray) -> np.ndarray:
        """Keep only the road that is vertically contiguous from the car.

        Scans the road mask bottom->top (bottom row = directly in front of the
        car). Once road has started, a run of `path_gap_min_rows` rows whose
        fill fraction is below `path_gap_row_min_fill` is a non-road gap band —
        everything above it (a different road: roundabout / 2nd perpendicular
        road across the bright gap) is zeroed so it can't feed the centreline.
        Returns the mask unchanged when no gap is found (normal road, flooded
        corner, or a genuinely connected road ahead — that's path_max_x_m's job).
        """
        h, w = road_mask.shape[:2]
        if w == 0:
            return road_mask
        row_fill = np.count_nonzero(road_mask, axis=1).astype(np.float32) / float(w)
        started = False
        gap_run = 0
        for y in range(h - 1, -1, -1):
            if row_fill[y] >= self.path_gap_row_min_fill:
                started = True
                gap_run = 0
            elif started:
                gap_run += 1
                if gap_run >= max(1, self.path_gap_min_rows):
                    # First (bottom-most) low row of this run is the gap edge;
                    # keep the near band below it, drop everything at/above.
                    cut_y = y + gap_run          # exclusive upper bound to keep
                    out = road_mask.copy()
                    out[:cut_y, :] = 0
                    return out
        return road_mask

    def _row_centroids(self, mask: np.ndarray, is_road: bool, n_strips: int = None):
        if n_strips is None:
            n_strips = self._centroid_strips
        h, w   = mask.shape[:2]
        sh     = max(1, h // n_strips)
        valid  = 0
        edge_valid = 0   # #2: strips where a REAL road edge was seen (is_road)
        centroids = []
        weights   = []   # Stage 1: per-strip reliability, aligned with centroids

        # Lane width tracking: estimated_lane_w is seeded from road_width_m in
        # metric mode (constant across rows now the scale is true) or the legacy
        # 65%-of-width guess; the sanity window lives in _lane_w_lo/hi_px.
        half_lane = self.estimated_lane_w * 0.5

        last_cx = float(w) * 0.5
        lead_skip = 0   # near strips skipped (empty / off-image) before first valid

        # A one-sided EXTRAPOLATED centre (edge ± half_lane) that lands outside
        # this corridor is off-image = the inner lane edge is fully out of frame
        # (unfollowable / off-track). Such strips are DROPPED below rather than
        # pinned to the image edge — that pin was the 2.5m lateral target that
        # steered the car off-road.
        corridor_lo = w * 0.05
        corridor_hi = w * 0.95

        # Iterate bottom → top so index 0 is nearest to the car.
        for i in range(n_strips - 1, -1, -1):
            y_bot = min(h,  (i + 1) * sh)
            y_top = max(0,   i      * sh)
            strip = mask[y_top:y_bot, :]

            # View bounds for THIS strip: in metric mode the camera can't see
            # the whole BEV width near the bottom, so "touching the frame edge"
            # must be tested against the per-row VALID bounds, or the view
            # boundary is mistaken for a road edge.
            if self.metric_bev and self._bev_row_lo is not None:
                mid_r = min(h - 1, (y_top + y_bot) // 2)
                vlo = int(self._bev_row_lo[mid_r])
                vhi = int(self._bev_row_hi[mid_r])
                if vhi <= vlo:            # row entirely outside the camera view
                    if valid == 0:
                        lead_skip += 1
                    continue
            else:
                vlo, vhi = 0, w

            if np.count_nonzero(strip) < 6:   # near-empty strip
                # Vertical connectivity gate (a): once the car's road is locked,
                # a non-road gap band means the road AHEAD is a different one
                # (bright surface between). Stop rather than bridge across it.
                if self.path_break_on_gap and valid > 0:
                    break
                lead_skip += 1                 # leading near-field empty
                continue                       # else: leading empties -> skip

            cols = np.where(strip > 0)[1]

            # ── THE FIX: The "Sea Parting" Road Splitter ──
            # If there is a white gap larger than 40 pixels, treat them as separate physical roads.
            split_indices = np.where(np.diff(cols) > 40)[0] + 1
            clusters = np.split(cols, split_indices)

            # Pick the cluster closest to the previous row's center
            best_cluster = min(clusters, key=lambda c: abs(((c[0] + c[-1]) * 0.5) - last_cx))

            # Vertical connectivity gate (b): if the chosen road cluster jumps
            # laterally from the running centre by more than a lane-ish width,
            # it is a DIFFERENT road blob (roundabout / offset road across a
            # gap), not the continuation of ours. Stop the centreline here.
            cluster_cx = (float(best_cluster[0]) + float(best_cluster[-1])) * 0.5
            if self.path_break_on_gap and valid > 0 \
                    and abs(cluster_cx - last_cx) > self.path_gap_jump_frac * w:
                break

            left_edge = best_cluster[0]
            right_edge = best_cluster[-1]

            # ── NEW: Dynamically grab the live estimated width ──
            half_lane = self.estimated_lane_w * 0.5

            if is_road:
                # ── SCENARIO A: Tracking the Dark Drivable Area ──
                # Check if the edges are "real" or just clipping the VIEW bounds
                # (image edge in legacy mode; per-row camera-visibility bounds in
                # metric mode — the near-field BEV corners are outside the view).
                left_visible = left_edge > vlo + 5
                right_visible = right_edge < (vhi - 5)

                if left_visible or right_visible:
                    # At least one true road edge is inside the frame -> this
                    # strip's centre is a real measurement, not a flood guess.
                    edge_valid += 1

                if left_visible and right_visible:
                    # OPPORTUNISTIC UPDATE: Both walls are visible!
                    measured_w = float(right_edge - left_edge)
                    # Sanity Check: only accept widths near the physical road
                    # width (metric: road_width_m ±40%; legacy: 45–75% of BEV).
                    if self._lane_w_lo_px < measured_w < self._lane_w_hi_px:
                        self.estimated_lane_w = (self.lane_w_alpha * measured_w) + ((1.0 - self.lane_w_alpha) * self.estimated_lane_w)
                        # Clamp so a corner-distorted width can't inflate half_lane
                        # and throw the one-sided extrapolation off the image (the
                        # rail-to-edge positive feedback).
                        self.estimated_lane_w = float(min(max(self.estimated_lane_w, self._lane_w_lo_px), self._lane_w_hi_px))
                        # Re-calculate half_lane with the new highly-accurate data
                        half_lane = self.estimated_lane_w * 0.5
                    # Both walls visible: true center is exactly in the middle
                    cx = (float(left_edge) + float(right_edge)) * 0.5
                    w_strip = self.cl_w_both
                elif left_visible and not right_visible:
                    # Only left wall visible: push center to the right
                    cx = float(left_edge) + half_lane
                    w_strip = self.cl_w_oneside
                elif right_visible and not left_visible:
                    # Only right wall visible: push center to the left
                    cx = float(right_edge) - half_lane
                    w_strip = self.cl_w_oneside
                else:
                    # Road fills the entire screen (straight tunnel), default to middle
                    if valid > 0: cx = last_cx
                    else: cx = float(w) * 0.5
                    w_strip = self.cl_w_flood

            else:
                # ── SCENARIO B: Tracking White Lane Lines (Fallback) ──
                spread = right_edge - left_edge
                raw_mean = float(np.mean(best_cluster))

                if spread > self._two_lines_min_px:
                    # OPPORTUNISTIC UPDATE: Both lines are in frame!
                    if self._lane_w_lo_px < float(spread) < self._lane_w_hi_px:
                        self.estimated_lane_w = (self.lane_w_alpha * float(spread)) + ((1.0 - self.lane_w_alpha) * self.estimated_lane_w)
                        self.estimated_lane_w = float(min(max(self.estimated_lane_w, self._lane_w_lo_px), self._lane_w_hi_px))
                        half_lane = self.estimated_lane_w * 0.5
                    # Both left and right lines are in frame; mean is perfect
                    cx = raw_mean
                    w_strip = self.cl_w_both
                else:
                    # Only ONE line is in frame
                    if raw_mean < (w * 0.5):
                        # Line is on the left; push center right
                        cx = raw_mean + half_lane
                    else:
                        # Line is on the right; push center left
                        cx = raw_mean - half_lane
                    w_strip = self.cl_w_oneside

            # Source rail guard: a one-sided EXTRAPOLATED centre that lands
            # off-image is unfollowable — drop the strip instead of letting the
            # publish-time clamp pin it to the edge (that pin = 2.5m lateral
            # target = the sustained wrong-way steer that drove the car off-road).
            # Both-edge and flood centres are already on-image, so they pass.
            if w_strip == self.cl_w_oneside and not (corridor_lo <= cx <= corridor_hi):
                if valid == 0:
                    lead_skip += 1             # leading near-field off-image drop
                continue

            # Near-field connectivity gate: if the road doesn't begin until the
            # near field has been skipped past (empty / off-image), the road we
            # just found is a DIFFERENT one across the bright surround border, not
            # the car's own road. Don't start the centreline on it — return what
            # we have (empty) so control coasts straight instead of chasing a far
            # road, and scene classification isn't poisoned into a false corner.
            if valid == 0 and self.path_break_on_gap \
                    and self.near_connect_max_empty_strips > 0 \
                    and lead_skip > self.near_connect_max_empty_strips:
                break

            last_cx = cx
            centroids.append(((y_top + y_bot) * 0.5, cx))
            weights.append(w_strip)
            valid += 1

        confidence = float(valid) / float(n_strips)
        if is_road:
            # #2: fraction of strips backed by a real edge. Equals `confidence`
            # on a normal road; collapses toward 0 when the dark road floods the
            # frame edge-to-edge (the corner case that pins cx to centre).
            self._road_edge_conf = float(edge_valid) / float(n_strips)
        return centroids, confidence, weights

    def _classify_scene(self, centroids, confidence: float) -> str:
        TURN_CONF_THRESH   = 0.20
        # 2026-07-04 metric re-base: TURN_VAR (was 15_000 px^2) and CURVE_SLOPE
        # (was 2.5 px/strip) were tuned on the old 1000px-wide BEV. The metric
        # BEV is 500px wide, so every lateral px signal halved and a real bend
        # classified 'straight' (live log: slope 2.1 < 2.5 on a screenshot-
        # confirmed corner). Scale both by bird_w so sensitivity is
        # width-independent — old values are reproduced exactly at bird_w=1000.
        TURN_VAR_THRESH    = (float(self.bird_w) * 0.1225) ** 2
        CURVE_SLOPE_THRESH = float(self.bird_w) * 0.0025
        ROUNDABOUT_DRIFT   = self.bird_w * 0.12

        if confidence < TURN_CONF_THRESH or len(centroids) < 3:
            return 'turn'

        cx_arr = np.array([c[1] for c in centroids], dtype=np.float32)
        var    = float(np.var(cx_arr))

        if var > TURN_VAR_THRESH:
            ema_delta = abs(self.current_cx - float(np.mean(cx_arr)))
            return 'roundabout' if ema_delta > ROUNDABOUT_DRIFT else 'turn'

        ys = np.arange(len(cx_arr), dtype=np.float32)
        slope = abs(float(np.polyfit(ys, cx_arr, 1)[0])) if len(ys) >= 4 else 0.0

        if slope > CURVE_SLOPE_THRESH:
            return 'curve'

        return 'straight'

    def _floor_corner_cue(self, white_mask):
        """Corner cue from off-road (white floor) asymmetry in the far band.

        When the dark-road centreline degenerates at a 90deg corner (the far
        field becomes bright floor, so cx pins to image-centre), the floor
        itself still says which way the road went: road bends LEFT  -> floor
        floods the RIGHT of the far band; road bends RIGHT -> floor on the LEFT.

        Returns (direction, distance_m, angle_deg, strength). strength is the
        signed imbalance (R-L)/(R+L) in [-1, 1]; >0 => floor-right => bend left.
        """
        if white_mask is None:
            return 'none', -1.0, 0.0, 0.0
        h, w = white_mask.shape[:2]
        band_h = max(1, int(h * self._corner_floor_band_frac))
        band   = white_mask[0:band_h, :]            # row 0 = far ahead
        half   = w // 2
        area   = float(band_h * half) + 1e-6
        left_fill  = float(np.count_nonzero(band[:, :half])) / area
        right_fill = float(np.count_nonzero(band[:, half:])) / area

        total = left_fill + right_fill
        # Need enough floor in the far band; an empty/road-filled band is not a
        # corner, it's open road ahead.
        if total < self._corner_floor_min_fill:
            return 'none', -1.0, 0.0, 0.0

        imbalance = (right_fill - left_fill) / (total + 1e-6)
        if abs(imbalance) < self._corner_floor_ratio:
            return 'none', -1.0, 0.0, imbalance

        # Floor on the right -> road bends left, and vice versa.
        direction = 'left' if imbalance > 0 else 'right'
        # Distance = forward range of the near edge of the far band.
        distance_m = max(0.0, self._bev_x0() + (float(self.bird_h) - float(band_h)) * float(self.bird_m_per_px_y))
        return direction, distance_m, 90.0, imbalance

    def _raw_corner_cue(self, frame):
        """Distance-free corner cue from the RAW camera image (validated offline
        on scratchpad/cam/ — see reach_test.py).

        The corner is only visible in the raw far field; the BEV crops it away.
        Threshold road = grayscale <= road_max; in a far-field ROI band take the
        road-fill FRACTION per L/C/R third. Centre fill collapsing = the straight
        ended (corner here); the higher side = which way the road goes. Returns
        (direction, center_fill, angle_deg) — center_fill (0..1, high=open ahead,
        low=at the corner) doubles as a robust pseudo-distance. 'none' otherwise.
        """
        if frame is None:
            return 'none', -1.0, 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        road = (gray <= self._corner_reach_road_max)
        y0 = int(h * self._corner_reach_roi_top)
        y1 = int(h * self._corner_reach_roi_bot)
        roi = road[y0:y1, :]
        t = max(1, w // 3)
        left_f   = float(roi[:, :t].mean())
        center_f = float(roi[:, t:2 * t].mean())
        right_f  = float(roi[:, 2 * t:].mean())
        self._reach_dbg = (left_f, center_f, right_f)

        # ── Multi-band instrumentation: curve-vs-sharp-corner discriminator R&D ──
        # Split the ROI into near/mid/far vertical thirds (forward camera: the TOP of
        # this band, y0, is FARTHER; the BOTTOM, y1, is NEARER). Hypothesis: a CURVE
        # collapses centre fill GRADUALLY with distance (near>=mid>=far, smooth ramp)
        # while a SHARP corner keeps centre full near then drops it ABRUPTLY (a cliff:
        # near high, mid+far ~0). Logged for BOTH cases (before the decision returns)
        # so a curve and a corner can be compared at matched approach stages. No
        # behaviour change yet — instrumentation only.
        bh = max(1, (y1 - y0) // 3)
        def _band(c0, c1):
            seg = road[y0:y1, c0:c1]
            far  = float(seg[0:bh].mean())          # top of band = farther ahead
            mid  = float(seg[bh:2 * bh].mean())
            near = float(seg[2 * bh:].mean())        # bottom of band = nearer the car
            return near, mid, far
        cN, cM, cF = _band(t, 2 * t)                 # centre third
        lN, lM, lF = _band(0, t)                     # left third
        rN, rM, rF = _band(2 * t, w)                 # right third
        self._reach_mb = (cN, cM, cF, lN, lM, lF, rN, rM, rF)
        if self.frame_count % 10 == 0:
            self.get_logger().info(
                f'REACH_MB C[n/m/f]={cN:.2f}/{cM:.2f}/{cF:.2f} '
                f'L[n/m/f]={lN:.2f}/{lM:.2f}/{lF:.2f} '
                f'R[n/m/f]={rN:.2f}/{rM:.2f}/{rF:.2f}')

        # SHARP-corner classifier (multi-band cliff, validated on REACH_MB 2026-07-02):
        # a sharp corner keeps centre-NEAR full while centre-MID has collapsed (road
        # ends abruptly ahead = cliff); a curve drops near+mid together (ramp) so it
        # never qualifies -> pure pursuit keeps it (no false-fire).
        is_sharp = (cN >= self._corner_sharp_near_min) and (cM <= self._corner_sharp_mid_max)
        if not is_sharp:
            # Relax the direction EMA toward 0 between corners, so the NEXT corner
            # (especially an opposite-direction back-to-back) starts UNBIASED instead of
            # carrying this corner's turn direction for a few frames.
            self._corner_camdir_ema *= 0.80
            return 'none', -1.0, 0.0                       # curve / straight -> pursuit keeps it
        # ROBUST DIRECTION = centre-of-mass of the road pixels in the MID+FAR rows (the
        # turn region, where the road bends away). This is *where the road goes*, as a
        # mass-weighted average of every road pixel — it can't flip on a close L/R call
        # or a stray dark blob the way the old fill-threshold comparison did. Near rows
        # (the symmetric straight approach) are excluded. EMA-smoothed for stability.
        turn_rows = road[y0:y0 + 2 * bh, :]                # far + mid bands
        col_mass  = turn_rows.sum(axis=0).astype(np.float64)   # road-pixel count per column
        total     = float(col_mass.sum())
        if total < 1.0:
            return 'none', -1.0, 0.0                       # no road in the turn region
        road_cx = float((col_mass * np.arange(w, dtype=np.float64)).sum() / total)
        offset  = (road_cx - w * 0.5) / (w * 0.5)          # [-1,1]; <0 = road mass LEFT
        a = 0.40                                           # EMA weight on the new frame
        self._corner_camdir_ema = a * offset + (1.0 - a) * self._corner_camdir_ema
        ema = self._corner_camdir_ema
        if abs(ema) < 0.06:                                # dead-zone: road ahead but no clear side
            return 'none', -1.0, 0.0
        direction = 'left' if ema < 0.0 else 'right'
        if self.frame_count % 5 == 0:
            self.get_logger().info(
                f'[corner-cam] SHARP {direction} offset={ema:+.2f} cN={cN:.2f} cM={cM:.2f}')
        # Proximity = centre-MID: it RAMPS 0.35 -> 0 through the approach (control uses
        # odom-distance for commit timing now; this is kept for the log/legacy).
        return direction, float(cM), 90.0

    def _edge_extent_m(self, centroids, weights):
        """Forward BEV range (m) of the farthest strip backed by a REAL edge
        (weight > w_flood). Flood-carried strips are excluded, so a flooded
        corner reads SHORT while a straight with edges in view reads LONG.
        Returns -1.0 when it can't be computed (no weights / mismatch)."""
        if not centroids or not weights or len(weights) != len(centroids):
            return -1.0
        far_row = None
        for (row_y, _), w_strip in zip(centroids, weights):
            if w_strip > self.cl_w_flood + 1e-6:
                far_row = row_y if far_row is None else min(far_row, row_y)
        if far_row is None:
            return 0.0        # full flood: no real edge anywhere -> road "ends" at the car
        return self._bev_x0() + (float(self.bird_h) - float(far_row)) * float(self.bird_m_per_px_y)

    def _detect_corner(self, centroids, white_mask=None, road_mask=None,
                       raw_frame=None, weights=None):
        """Anticipatory sharp-turn detector based on the road CENTERLINE drift.

        Rationale: on this track the dark-road mask is a thin curve (~5% fill), so
        occupancy-of-a-band is a near-zero, useless signal. The centroid pipeline,
        however, tracks the lane centre robustly. Walking the centreline from near
        the car outward, the first row whose centre-x deviates hard from the image
        centre marks where a sharp turn begins; the side of that deviation is the
        turn direction, and its forward distance is how far ahead the turn is.

        centroids: list of (row_y, cx), index 0 = nearest the car (bottom of view).
        Returns (direction, distance_m, angle_deg).
        Bird's-eye convention: row 0 = far ahead, row bird_h = at the car.
        """
        # Primary: raw-image road-fill cue (distance-free, sees the corner the BEV
        # crops away). When it fires it wins; on straights/gentle curves (centre
        # open) it returns 'none' and we fall through to the drift/floor logic.
        if self._corner_use_reach and raw_frame is not None:
            rdir, rdist, rang = self._raw_corner_cue(raw_frame)   # also updates _reach_dbg
            n_pts = len(centroids) if centroids else 0
            path_healthy = n_pts >= self._corner_min_pts
            # Gate: only trust the reach cue as sole authority when the path has
            # degenerated (flood). A healthy path on a straight must not be steered
            # by a far-field ROI that's looking at a different road/edge.
            reach_gated = self._corner_reach_require_flood and path_healthy
            # Open-road corroboration gate: the BEV centreline must AGREE the road
            # ends ahead (edge-informed extent short) before the reach cue may fire
            # the maneuver. Keys on edge-informed EXTENT, not point count — the
            # flooded mesh corner keeps a long DEGENERATE centroid list (that's the
            # disproved n_pts gate) but its real-edge extent collapses.
            edge_ext = self._edge_extent_m(centroids, weights)
            open_gated = (self._corner_path_end_max_m > 0.0 and edge_ext >= 0.0
                          and edge_ext > self._corner_path_end_max_m)
            # Near the roundabout, edge_ext SATURATES to the same BEV-depth
            # ceiling whether the far edge belongs to the car's own corner or to
            # a different road segment/the island visible far off (confirmed
            # live 2026-07-04: a genuine right corner and an earlier false alarm
            # both read edge_ext=1.23) -> edge_ext alone can't discriminate there.
            # Corroborate with the drift EMA instead: if the centreline is
            # ALREADY bending the reach cue's claimed direction (even below its
            # own standalone corner_drift_px threshold), the fire is real.
            drift_corroborates = False
            if open_gated and rdir != 'none':
                ema = self._corner_dev_ema
                reach_sign = 1.0 if rdir == 'right' else -1.0
                drift_corroborates = (abs(ema) >= self._corner_reach_drift_corroborate_px
                                       and (ema > 0) == (reach_sign > 0))
                if drift_corroborates:
                    open_gated = False
            if rdir != 'none' and open_gated and self.frame_count % 5 == 0:
                self.get_logger().info(
                    f'[corner-gate] reach {rdir} SUPPRESSED: edge road extends '
                    f'{edge_ext:.2f}m > {self._corner_path_end_max_m:.2f}m (open ahead), '
                    f'drift_ema={self._corner_dev_ema:+.0f}px not corroborating')
            elif rdir != 'none' and drift_corroborates and self.frame_count % 5 == 0:
                self.get_logger().info(
                    f'[corner-gate] reach {rdir} OVERRIDE: edge_ext={edge_ext:.2f}m '
                    f'gated but drift_ema={self._corner_dev_ema:+.0f}px corroborates')
            if self.frame_count % 15 == 0 and (rdir != 'none' or reach_gated):
                lf, cf, rf = self._reach_dbg
                self.get_logger().info(
                    f'REACH_DBG L={lf:.2f} C={cf:.2f} R={rf:.2f} raw={rdir} '
                    f'n_pts={n_pts} healthy={path_healthy} gated={reach_gated} '
                    f'edge_ext={edge_ext:.2f} open_gated={open_gated}')
            if rdir != 'none' and not reach_gated and not open_gated:
                self._floor_cue_str = 0.0
                return rdir, rdist, rang

        if not centroids or len(centroids) < self._corner_min_pts:
            self._corner_dev_ema *= (1.0 - self._corner_dev_alpha)  # decay toward 0
            # Centreline gone entirely (e.g. road flooded the frame) -> this is
            # exactly when the floor cue is most reliable.
            fdir, fdist, fang, fstr = self._floor_corner_cue(white_mask)
            self._floor_cue_str = fstr
            return fdir, fdist, fang

        center = float(self.bird_w) * 0.5
        w = float(self.bird_w)
        # Clamp each centroid into the image: sharp turns extrapolate cx to
        # ~1300/-340, and a single raw point flips the direction frame-to-frame.
        devs = [(float(row_y), min(max(float(cx), 0.0), w) - center) for row_y, cx in centroids]

        # Robust per-frame deviation = MEDIAN of the far ~40% of the centreline
        # (centroids are ordered near->far, so the tail is "ahead").
        far = devs[int(len(devs) * 0.6):] or devs
        far_sorted = sorted(d for _, d in far)
        far_dev = far_sorted[len(far_sorted) // 2]

        # Temporal EMA so the DIRECTION can't flip on one noisy frame.
        self._corner_dev_ema = (self._corner_dev_alpha * far_dev
                                + (1.0 - self._corner_dev_alpha) * self._corner_dev_ema)
        ema = self._corner_dev_ema

        if self.frame_count % 15 == 0:
            self.get_logger().info(
                f'CORNER_DBG far_dev={far_dev:+.0f} ema={ema:+.0f}px  (drift>{self._corner_drift_px:.0f}px)'
            )

        if abs(ema) < float(self._corner_drift_px):
            # Centreline reads "straight" — but at a flooded corner that's the
            # deterministic under-read. Defer to the floor-asymmetry cue.
            fdir, fdist, fang, fstr = self._floor_corner_cue(white_mask)
            self._floor_cue_str = fstr
            if fdir != 'none':
                return fdir, fdist, fang
            return 'none', -1.0, 0.0

        # 2026-07-04: open-road corroboration gate, mirrored from the reach cue
        # above. This drift/EMA branch fires on the far centreline BENDING —
        # which is exactly what a normal curve or an S-jog lead-in also does,
        # not just a sharp corner. Un-gated, it was firing APPROACH as soon as
        # "the blue dots align and form the correct path" through any curve
        # (the 'fires too early' report), while a corner masked behind the
        # roundabout's geometry (dots pulled toward open road) doesn't bend the
        # far centreline until very late (the 'fires too late' report) — same
        # weak signal, both symptoms. Requiring the edge-informed extent to
        # ALSO read short (like the reach cue already must) stops this branch
        # from mistaking an ordinary bend for the real corner.
        edge_ext = self._edge_extent_m(centroids, weights)
        if (self._corner_path_end_max_m > 0.0 and edge_ext >= 0.0
                and edge_ext > self._corner_path_end_max_m):
            if self.frame_count % 5 == 0:
                self.get_logger().info(
                    f'[corner-gate] drift {"right" if ema > 0 else "left"} SUPPRESSED: '
                    f'edge road extends {edge_ext:.2f}m > {self._corner_path_end_max_m:.2f}m (open ahead)')
            return 'none', -1.0, 0.0

        self._floor_cue_str = 0.0
        direction = 'right' if ema > 0 else 'left'
        sign = 1.0 if ema > 0 else -1.0
        # Distance = forward range of the NEAREST point already bent to the
        # smoothed side; shrinks as the bend approaches (drives APPROACH->commit).
        onset_row = far[0][0]
        for row_y, d in devs:                      # near -> far
            if sign * d >= float(self._corner_drift_px):
                onset_row = row_y
                break
        distance_m = max(0.0, self._bev_x0() + (float(self.bird_h) - onset_row) * float(self.bird_m_per_px_y))
        return direction, distance_m, 90.0

    def _pick_lookahead(self, centroids):
        if not centroids:
            return None, None
        target_y = float(self.bird_h) * float(self.lookahead_y_frac)
        best     = min(centroids, key=lambda c: abs(c[0] - target_y))
        return best[1], best[0]   # (cx, row_y)

    def _overlay_lookahead_px(self, fit_pts):
        """Pixel (cx, row_y) of the point on the PUBLISHED fit path that control's
        pure pursuit would steer to — mirrors control._pick_lookahead_point on the
        same (x_m, y_m) geometry so the drawn yellow dot IS the real steering target,
        not the unused /lane_error centre. Returns None if the path is empty."""
        if not fit_pts:
            return None
        ld   = max(0.05, float(self.overlay_lookahead_m))
        best = None
        for row_y, cx in fit_pts:                       # near -> far, matching control
            x_m = self._bev_x0() + (float(self.bird_h) - float(row_y)) * float(self.bird_m_per_px_y)
            if x_m < 0.05 or x_m > float(self.path_max_x_m):
                continue
            cx_clamped = min(max(float(cx), 0.0), float(self.bird_w))
            y_m = (float(self.bird_w) * 0.5 - cx_clamped) * float(self.bird_m_per_px_x) \
                + float(self.path_y_offset_m)
            best = (int(cx_clamped), int(row_y))
            if math.hypot(x_m, y_m) >= ld:
                return best
        return best

    def _fit_centerline(self, centroids, weights):
        """Stage 1: one confidence-weighted polynomial fit over the row centroids.

        Returns (coeffs, mu, sd, y_lo, y_hi) for x = polyval(coeffs, (y-mu)/sd),
        or None if there isn't enough real signal (caller falls back to the raw
        per-strip path). The fit is deterministic weighted least-squares — NOT
        RANSAC — so it can't jitter frame-to-frame from a reseeded RNG, and it
        uses the reliability tiers computed in _row_centroids.
        """
        n = len(centroids)
        if n < 2 or len(weights) != n:
            return None
        ws = np.asarray(weights, dtype=np.float64)
        if float(ws.sum()) < self.cl_min_total_w:
            return None

        rows = np.asarray([c[0] for c in centroids], dtype=np.float64)
        cxs  = np.asarray([c[1] for c in centroids], dtype=np.float64)

        # Degree adapts to the number of *measured* (both-edge) strips, capped by
        # the param: sharp corners (few/no both-strips) collapse to a deg-1 tilt
        # line; straights/curves with support get curvature. Uses the both-count
        # discriminator, never the scene classifier.
        both_count = int(np.count_nonzero(ws >= self.cl_w_both - 1e-6))
        degree = min(int(self.cl_fit_degree), max(1, both_count - 1), n - 1)

        # Condition the fit: normalise the independent var (raw pixel rows make
        # a deg-2 Vandermonde ill-conditioned).
        mu = float(rows.mean())
        sd = float(rows.std()) or 1.0
        t  = (rows - mu) / sd
        try:
            coeffs = np.polyfit(t, cxs, degree, w=ws)
        except Exception:
            return None
        return coeffs, mu, sd, float(rows.min()), float(rows.max())

    def _fit_to_centroids(self, fit, n):
        """Sample a centreline fit into (row_y, cx) tuples, near->far (largest
        row_y first, matching the raw path order the controller walks). Sampling
        stays within the observed row span — no extrapolation past the data."""
        coeffs, mu, sd, y_lo, y_hi = fit
        ys = np.linspace(y_hi, y_lo, max(2, int(n)))
        xs = np.polyval(coeffs, (ys - mu) / sd)
        return [(float(y), float(x)) for y, x in zip(ys, xs)]

    def _fit_to_centroids_smoothed(self, fit, n):
        """Like _fit_to_centroids but EMA-smoothed across frames to stop the far
        end wagging. Samples on a FIXED full-height row grid so each grid row is
        the same physical range every frame and can be blended with its own
        history; only rows inside the observed span [y_lo, y_hi] are published
        (no extrapolation past where the road was actually seen). near->far."""
        if not self.cl_smooth_enable:
            return self._fit_to_centroids(fit, n)
        coeffs, mu, sd, y_lo, y_hi = fit
        n = max(2, int(n))
        grid = np.linspace(float(self.bird_h) - 1.0, 0.0, n)   # near (bottom) -> far (top)
        xs   = np.polyval(coeffs, (grid - mu) / sd)
        within = (grid >= y_lo) & (grid <= y_hi)
        if self._cl_ema_xs is None or len(self._cl_ema_xs) != n:
            self._cl_ema_xs = np.full(n, np.nan, dtype=np.float64)
        a = float(np.clip(self.cl_smooth_alpha, 1e-3, 1.0))
        out = []
        for i in range(n):
            if not within[i]:
                self._cl_ema_xs[i] = np.nan          # road doesn't reach here now
                continue
            if np.isnan(self._cl_ema_xs[i]):
                self._cl_ema_xs[i] = float(xs[i])    # first sample at this row -> init
            else:
                self._cl_ema_xs[i] = a * float(xs[i]) + (1.0 - a) * self._cl_ema_xs[i]
            out.append((float(grid[i]), float(self._cl_ema_xs[i])))
        return out

    def _publish_path_from_centroids(self, centroids):
        """Publish the ACTUAL curve geometry by converting the blue dots directly to real-world points."""
        path                  = Path()
        path.header.stamp     = self.get_clock().now().to_msg()
        path.header.frame_id  = 'base_footprint'

        for row_y, cx in centroids:
            x_m = self._bev_x0() + (float(self.bird_h) - float(row_y)) * float(self.bird_m_per_px_y)
            # Lowered the minimum distance so the controller has points right at the bumper
            if x_m < 0.05:
                continue
            if x_m > float(self.path_max_x_m):
                continue
            # Clamp the centroid to the image. In sharp turns the lane-width
            # extrapolation can push cx far outside [0, bird_w] (seen as la_cx≈1300
            # / -340), which becomes a metres-off-track lateral target and steers
            # the car off the course. Bounding cx caps the lateral error.
            cx_clamped = min(max(float(cx), 0.0), float(self.bird_w))
            y_m = (float(self.bird_w) * 0.5 - cx_clamped) * float(self.bird_m_per_px_x) + float(self.path_y_offset_m)

            ps                    = PoseStamped()
            ps.header             = path.header
            ps.pose.position.x    = float(x_m)
            ps.pose.position.y    = float(y_m)
            ps.pose.position.z    = 0.0
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        if path.poses:
            self.lane_path_pub.publish(path)

    # ── Main image callback ─────────────────────────────────────────────────

    def image_cb(self, msg, topic: str = ''):
        if topic and self.active_image_topic is None:
            self.active_image_topic = topic
            self.get_logger().info(f'Receiving images on: {self.active_image_topic}')

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        if self.metric_bev:
            # Metric mode works at the NATIVE camera resolution — the legacy
            # 1280x720 resize was a non-uniform stretch (x2.0 / x1.5) that
            # distorted all BEV geometry. Only resize if the camera ever
            # differs from the calibrated cam_img_w/h.
            if frame.shape[1] != self.cam_img_w or frame.shape[0] != self.cam_img_h:
                frame = cv2.resize(frame, (self.cam_img_w, self.cam_img_h))
        else:
            frame = cv2.resize(frame, (1280, 720))

        # ── Bird's-eye + colour masks ──────────────────────────────────────
        bird       = self.get_bird_eye(frame)
        bird_debug = bird.copy()
        lab        = cv2.cvtColor(bird, cv2.COLOR_BGR2LAB)

        mask = cv2.inRange(lab, self.lower_white, self.upper_white)

        road_mask = cv2.inRange(
            lab,
            np.array([0,   100, 100]),
            np.array([self.road_dark_threshold, 145, 145])
        )

        # Metric mode: pixels outside the camera's view warp to BLACK, which the
        # dark-road threshold would happily call "road". Mask both to the
        # actually-observed region.
        if self.metric_bev and self._bev_valid is not None:
            road_mask = cv2.bitwise_and(road_mask, self._bev_valid)
            mask      = cv2.bitwise_and(mask,      self._bev_valid)

        # Drop any road beyond a non-road gap so the centreline can't bridge onto
        # a different road ahead (roundabout / 2nd road across the bright gap).
        if self.path_break_on_gap:
            road_mask = self._cut_road_at_gap(road_mask)

        # Only pay the upscale + serialize cost when something is subscribed.
        if self.debug_pub.get_subscription_count() > 0:
            debug_display = cv2.resize(mask, (1280, 720))
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_display, 'mono8'))

        self.frame_count += 1

        # ══════════════════════════════════════════════════════════════════
        # STAGE 1 — Row-centroid extraction
        # ══════════════════════════════════════════════════════════════════
        lane_centroids, lane_conf, lane_w = self._row_centroids(mask, is_road=False)
        road_centroids, road_conf, road_w = self._row_centroids(road_mask, is_road=True)

        # ══════════════════════════════════════════════════════════════════
        # STAGE 2 — Scene classification
        # ══════════════════════════════════════════════════════════════════
        pri_centroids = road_centroids if road_conf > 0.05 else lane_centroids
        pri_conf      = road_conf if road_conf > 0.05 else lane_conf
        pri_weights   = road_w if road_conf > 0.05 else lane_w

        raw_scene = self._classify_scene(pri_centroids, pri_conf)
        self._scene_history.append(raw_scene)
        if len(self._scene_history) > self._SCENE_VOTE_WIN:
            self._scene_history.pop(0)
        voted_scene = max(set(self._scene_history), key=self._scene_history.count)

        # Asymmetric hysteresis (see _committed_scene in __init__): leaving a
        # smooth curve/straight for a disruptive roundabout/turn needs sustained
        # evidence; returning to smooth tracking is immediate. Kills the
        # tight-curve->roundabout flicker that swapped in the straight cut-across
        # fit. curve<->straight and any ->stable transition switch at once.
        _STABLE  = ('curve', 'straight')
        _DISRUPT = ('roundabout', 'turn')
        if self._committed_scene in _STABLE and voted_scene in _DISRUPT:
            self._scene_switch_count += 1
            if self._scene_switch_count >= self._scene_switch_frames:
                self._committed_scene = voted_scene
                self._scene_switch_count = 0
            # else: hold the stable scene for this frame
        else:
            self._scene_switch_count = 0
            self._committed_scene = voted_scene
        scene = self._committed_scene

        # ── UPGRADED: Publish the scene string so the control node can brake ──
        scene_msg = String()
        scene_msg.data = scene
        self.scene_pub.publish(scene_msg)

        # ── NEW: Anticipatory corner detection (direction + distance + angle) ──
        # Uses the road centreline drift (pri_centroids), the robust signal here.
        # Direction is majority-voted to kill single-frame flicker.
        raw_dir, corner_dist, corner_ang = self._detect_corner(
            pri_centroids, white_mask=mask, road_mask=road_mask, raw_frame=frame,
            weights=pri_weights)
        self._corner_history.append(raw_dir)
        if len(self._corner_history) > max(1, self._corner_vote_win):
            self._corner_history.pop(0)
        corner_dir = max(set(self._corner_history), key=self._corner_history.count)
        if corner_dir == 'none':
            corner_dist, corner_ang = -1.0, 0.0

        self.corner_dir_pub.publish(String(data=corner_dir))
        self.corner_dist_pub.publish(Float32(data=float(corner_dist)))
        self.corner_ang_pub.publish(Float32(data=float(corner_ang)))

        # ══════════════════════════════════════════════════════════════════
        # STAGE 3 — Select active centroid source (Dark Road is Primary)
        # ══════════════════════════════════════════════════════════════════
        if road_conf > 0.05:
            active_centroids = road_centroids
            if scene == 'roundabout':
                mode_str = 'centroid_flow'
            elif scene == 'turn':
                mode_str = 'centroid_turn'
            elif scene == 'curve':
                mode_str = 'road_curve'
            else:
                mode_str = 'road_straight'
        elif lane_centroids and lane_conf >= self._min_lane_conf:
            active_centroids = lane_centroids
            mode_str = 'lane_straight' if scene == 'straight' else 'lane_curve'
        else:
            active_centroids = []
            mode_str = 'no_signal'

        # ══════════════════════════════════════════════════════════════════
        # STAGE 4 — Pick the single lookahead point
        # ══════════════════════════════════════════════════════════════════
        la_cx, la_y = self._pick_lookahead(active_centroids)

        # ══════════════════════════════════════════════════════════════════
        # STAGE 5 — Kalman Predictive Smoothing
        # ══════════════════════════════════════════════════════════════════
        if la_cx is not None:
            if not self.filter_initialized:
                # First frame initialization to prevent massive snap
                self.kf.x[0, 0] = la_cx
                self.filter_initialized = True
                self.current_cx = la_cx
            else:
                # 2. Update with the new camera measurement
                self.current_cx = self.kf.update(la_cx)
        else:
            # Camera lost the lane!
            if self.filter_initialized:
                # Coast on the Kalman prediction so the car keeps steering
                # naturally through the corner for a few frames.
                self.current_cx = self.kf.predict()
            else:
                # No measurement yet and no filter state: hold image centre.
                self.current_cx = float(self.bird_w) * 0.5
            
        # Use self.current_cx for your error calculation instead of self._ema_cx
        lane_detected = self.filter_initialized

        # ══════════════════════════════════════════════════════════════════
        # STAGE 6 — Compute lane error from the single smoothed centre
        # ══════════════════════════════════════════════════════════════════
        lane_detected = self.filter_initialized

        if lane_detected:
            # FIX: Use self.current_cx instead of self._ema_cx
            offset = float(self.bird_w) * 0.5 - self.current_cx
            error  = float(np.clip(
                (offset / (float(self.bird_w) * 0.5)) * float(self.max_error_deg),
                -self.max_error_deg, self.max_error_deg
            ))
        else:
            error = 0.0
        # ══════════════════════════════════════════════════════════════════
        # STAGE 7 — Publish actual path for pure pursuit
        # ══════════════════════════════════════════════════════════════════
        fit = None
        fit_pts = None
        if lane_detected and active_centroids:
            active_w = road_w if road_conf > 0.05 else lane_w
            fit = self._fit_centerline(active_centroids, active_w) if self.cl_fit_enable else None
            if fit is not None:
                # Sample ONCE (this advances the EMA); reuse for both publish + overlay.
                fit_pts = self._fit_to_centroids_smoothed(fit, self.path_points)
                self._publish_path_from_centroids(fit_pts)
                self._last_good_fit_pts = fit_pts     # remember for dropout hold
                self._fit_dropout_count = 0
            elif self.cl_fit_enable and self._last_good_fit_pts is not None \
                    and self._fit_dropout_count < max(0, self.fit_hold_frames):
                # Momentary fit dropout: HOLD the last good centreline instead of
                # snapping /lane_path back to the raw one-sided centroids (the snap
                # is what jerks the car offroad). EMA history is preserved so the
                # fit resumes smoothly when it recovers.
                self._fit_dropout_count += 1
                fit_pts = self._last_good_fit_pts
                self._publish_path_from_centroids(fit_pts)
            else:
                # Fit disabled, or held too long -> fall back to the raw baseline.
                self._cl_ema_xs = None            # drop stale EMA history
                self._last_good_fit_pts = None
                self._publish_path_from_centroids(active_centroids)

        # ── Debug overlay ──────────────────────────────────────────────────
        for row_y, cx in active_centroids:
            cv2.circle(bird_debug, (int(cx), int(row_y)), 3, (200, 200, 0), -1)

        # Stage 1: draw the ACTUAL published (smoothed) centreline in magenta so
        # what you see is what the car follows — a steady line here = fixed wiggle.
        if fit_pts and len(fit_pts) >= 2:
            pts = np.array([[int(x), int(y)] for y, x in fit_pts], dtype=np.int32)
            cv2.polylines(bird_debug, [pts], False, (255, 0, 255), 2, cv2.LINE_AA)

        # Yellow dot = the point control's pure pursuit actually steers to, drawn ON
        # the published fit path (NOT the unused /lane_error centre). If the fit path
        # is unavailable, fall back to the raw smoothed centre so the dot never vanishes.
        la_pt = self._overlay_lookahead_px(fit_pts) if fit_pts else None
        if la_pt is not None:
            cv2.circle(bird_debug, la_pt, 8, (0, 255, 255), -1)
        elif la_y is not None:
            cv2.circle(bird_debug, (int(self.current_cx), int(la_y)), 8, (0, 200, 255), -1)

        road_vis          = np.zeros_like(bird_debug)
        road_vis[:, :, 1] = road_mask
        bird_debug        = cv2.addWeighted(bird_debug, 1.0, road_vis, 0.15, 0)

        colour_map = {
            'road_straight':  (255, 255,   0), # Cyan
            'road_curve':     (200, 200,   0), 
            'lane_straight':  (0,   255,   0),
            'lane_curve':     (0,   200,  60),
            'centroid_turn':  (0,   165, 255),
            'centroid_flow':  (180,   0, 255),
            'no_signal':      (0,     0, 255),
        }
        label_colour = colour_map.get(mode_str, (200, 200, 200))
        _lr, _cr, _rr = self._reach_dbg
        label        = (f'{mode_str} | {scene} | conf={pri_conf:.2f} '
                        f'edge={self._road_edge_conf:.2f} | {corner_dir} '
                        f'reach L{_lr:.2f} C{_cr:.2f} R{_rr:.2f} | e={error:.1f}deg')
        cv2.putText(bird_debug, label, (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, label_colour, 2, cv2.LINE_AA)

        # ── Publish lane topics ──────────────────────────────────────────────
        lane_msg      = Float32(); lane_msg.data = float(error)
        det_msg       = Bool();    det_msg.data  = bool(lane_detected)
        self.lane_error_pub.publish(lane_msg)
        self.lane_detected_pub.publish(det_msg)

        # ── NEW: Publish Telemetry for Graphing ──
        if lane_detected:
            msg_kf = Float32(); msg_kf.data = float(self.current_cx)
            self.kf_cx_pub.publish(msg_kf)
            
            msg_vel = Float32(); msg_vel.data = float(self.kf.x[1, 0])
            self.kf_vel_pub.publish(msg_vel)
            
            msg_innov = Float32(); msg_innov.data = float(self.kf.innovation)
            self.kf_innov_pub.publish(msg_innov)
            
            if la_cx is not None:
                msg_raw = Float32(); msg_raw.data = float(la_cx)
                self.la_cx_pub.publish(msg_raw)

        # ── UPGRADED: Respect the verbose override ──
        if self.verbose_logging or self.frame_count % 30 == 0:
            self.get_logger().info(
                f'scene={scene}  mode={mode_str}  '
                f'lane_conf={lane_conf:.2f}  road_conf={road_conf:.2f}  '
                f'edge_conf={self._road_edge_conf:.2f}  '
                f'corner={corner_dir}/{corner_dist:.2f}  '
                f'reach L{self._reach_dbg[0]:.2f} C{self._reach_dbg[1]:.2f} R{self._reach_dbg[2]:.2f}  '
                f'kf_cx={self.current_cx:.1f}  la_cx={la_cx}  '
                f'error={error:.2f}°'
            )

        if self.show_windows:
            try:
                cv2.imshow('Camera',    frame)
                cv2.imshow('Bird Eye',  bird_debug)
                cv2.imshow('Lane Mask', mask)
                cv2.waitKey(1)
            except cv2.error as e:
                self.get_logger().warn(f'imshow failed; disabling windows: {e}')
                self.show_windows = False


def main():
    rclpy.init()
    node = PerceptionNode()
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
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()