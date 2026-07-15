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
        self.declare_parameter('bird_m_per_px_x',            0.01)
        self.declare_parameter('bird_m_per_px_y',            0.01)
        self.declare_parameter('path_points',               25)
        self.declare_parameter('path_min_x_m',               0.2)
        self.declare_parameter('path_max_x_m',               1.8)
        self.declare_parameter('path_y_offset_m',            0.0)
        self.declare_parameter('ransac_n_iter',             50)
        self.declare_parameter('ransac_residual_threshold', 15.0)
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
        self.path_y_offset_m           = float(self.get_parameter('path_y_offset_m').value)
        self.ransac_n_iter             = int(self.get_parameter('ransac_n_iter').value)
        self.ransac_residual_threshold = float(self.get_parameter('ransac_residual_threshold').value)
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
        # #2: confidence that reflects only strips where a REAL road edge was
        # found (not the flooded "cx=centre" fallback). Set each frame by the
        # is_road pass of _row_centroids; used so a flooded-straight frame can't
        # masquerade as a high-confidence straight.
        self._road_edge_conf           = 0.0
        self._floor_cue_str            = 0.0   # last floor-asymmetry imbalance
        self._centroid_strips          = max(8, int(self.get_parameter('centroid_strips').value))

        self.add_on_set_parameters_callback(self._param_cb)

        # LAB thresholds — white lane markings: high L, neutral a/b
        self.lower_white = np.array([150, 90,  90])
        self.upper_white = np.array([255, 130, 130])

        # ── NEW: Dynamic Lane Width Estimator ──
        # We start with a 65% guess, but it will rapidly correct itself.
        self.estimated_lane_w = float(self.bird_w) * 0.65 
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
                self.bird_m_per_px_x = float(p.value)
            elif p.name == 'bird_m_per_px_y':
                self.bird_m_per_px_y = float(p.value)
            elif p.name == 'path_points':
                self.path_points = int(p.value)
            elif p.name == 'path_min_x_m':
                self.path_min_x_m = float(p.value)
            elif p.name == 'path_max_x_m':
                self.path_max_x_m = float(p.value)
            elif p.name == 'path_y_offset_m':
                self.path_y_offset_m = float(p.value)
            elif p.name == 'ransac_n_iter':
                self.ransac_n_iter = int(p.value)
            elif p.name == 'ransac_residual_threshold':
                self.ransac_residual_threshold = float(p.value)
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

            x_m = (float(self.bird_h) - float(y_px)) * float(self.bird_m_per_px_y)
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

    def _row_centroids(self, mask: np.ndarray, is_road: bool, n_strips: int = None):
        if n_strips is None:
            n_strips = self._centroid_strips
        h, w   = mask.shape[:2]
        sh     = max(1, h // n_strips)
        valid  = 0
        edge_valid = 0   # #2: strips where a REAL road edge was seen (is_road)
        centroids = []

        # ── TUNE THIS: Estimated physical lane width in pixels ──
        # 0.65 assumes the lane takes up 65% of the bird's eye view width (1000px).
        assumed_lane_w = w * 0.65 
        half_lane = assumed_lane_w * 0.5
        
        last_cx = float(w) * 0.5

        # Iterate bottom → top so index 0 is nearest to the car.
        for i in range(n_strips - 1, -1, -1):
            y_bot = min(h,  (i + 1) * sh)
            y_top = max(0,   i      * sh)
            strip = mask[y_top:y_bot, :]

            if np.count_nonzero(strip) < 6:   # skip near-empty strips
                continue

            cols = np.where(strip > 0)[1]
            
            # ── THE FIX: The "Sea Parting" Road Splitter ──
            # If there is a white gap larger than 40 pixels, treat them as separate physical roads.
            split_indices = np.where(np.diff(cols) > 40)[0] + 1
            clusters = np.split(cols, split_indices)
            
            # Pick the cluster closest to the previous row's center
            best_cluster = min(clusters, key=lambda c: abs(((c[0] + c[-1]) * 0.5) - last_cx))
            
            left_edge = best_cluster[0]
            right_edge = best_cluster[-1]

            # ── NEW: Dynamically grab the live estimated width ──
            half_lane = self.estimated_lane_w * 0.5

            if is_road:
                # ── SCENARIO A: Tracking the Dark Drivable Area ──
                # Check if the edges are "real" or just clipping the camera frame
                left_visible = left_edge > 5
                right_visible = right_edge < (w - 5)

                if left_visible or right_visible:
                    # At least one true road edge is inside the frame -> this
                    # strip's centre is a real measurement, not a flood guess.
                    edge_valid += 1

                if left_visible and right_visible:
                    # OPPORTUNISTIC UPDATE: Both walls are visible!
                    measured_w = float(right_edge - left_edge)
                    # Sanity Check: Only accept widths that physically make sense 
                    # (e.g. between 30% and 95% of the camera width) to ignore corner distortions
                    if (w * 0.3) < measured_w < (w * 0.95):
                        self.estimated_lane_w = (self.lane_w_alpha * measured_w) + ((1.0 - self.lane_w_alpha) * self.estimated_lane_w)
                        # Re-calculate half_lane with the new highly-accurate data
                        half_lane = self.estimated_lane_w * 0.5
                    # Both walls visible: true center is exactly in the middle
                    cx = (float(left_edge) + float(right_edge)) * 0.5
                elif left_visible and not right_visible:
                    # Only left wall visible: push center to the right
                    cx = float(left_edge) + half_lane
                elif right_visible and not left_visible:
                    # Only right wall visible: push center to the left
                    cx = float(right_edge) - half_lane
                else:
                    # Road fills the entire screen (straight tunnel), default to middle
                    if valid > 0: cx = last_cx
                    else: cx = float(w) * 0.5

            else:
                # ── SCENARIO B: Tracking White Lane Lines (Fallback) ──
                spread = right_edge - left_edge
                raw_mean = float(np.mean(best_cluster))
                
                if spread > w * 0.3:
                    # OPPORTUNISTIC UPDATE: Both lines are in frame!
                    if (w * 0.3) < float(spread) < (w * 0.95):
                        self.estimated_lane_w = (self.lane_w_alpha * float(spread)) + ((1.0 - self.lane_w_alpha) * self.estimated_lane_w)
                        half_lane = self.estimated_lane_w * 0.5
                    # Both left and right lines are in frame; mean is perfect
                    cx = raw_mean
                else:
                    # Only ONE line is in frame
                    if raw_mean < (w * 0.5):
                        # Line is on the left; push center right
                        cx = raw_mean + half_lane
                    else:
                        # Line is on the right; push center left
                        cx = raw_mean - half_lane

            last_cx = cx
            centroids.append(((y_top + y_bot) * 0.5, cx))
            valid += 1

        confidence = float(valid) / float(n_strips)
        if is_road:
            # #2: fraction of strips backed by a real edge. Equals `confidence`
            # on a normal road; collapses toward 0 when the dark road floods the
            # frame edge-to-edge (the corner case that pins cx to centre).
            self._road_edge_conf = float(edge_valid) / float(n_strips)
        return centroids, confidence

    def _classify_scene(self, centroids, confidence: float) -> str:
        TURN_CONF_THRESH   = 0.20          
        TURN_VAR_THRESH    = 15_000.0      
        CURVE_SLOPE_THRESH = 2.5           
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
        distance_m = max(0.0, (float(self.bird_h) - float(band_h)) * float(self.bird_m_per_px_y))
        return direction, distance_m, 90.0, imbalance

    def _detect_corner(self, centroids, white_mask=None):
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
        distance_m = max(0.0, (float(self.bird_h) - onset_row) * float(self.bird_m_per_px_y))
        return direction, distance_m, 90.0

    def _pick_lookahead(self, centroids):
        if not centroids:
            return None, None
        target_y = float(self.bird_h) * float(self.lookahead_y_frac)
        best     = min(centroids, key=lambda c: abs(c[0] - target_y))
        return best[1], best[0]   # (cx, row_y)

    def _publish_path_from_centroids(self, centroids):
        """Publish the ACTUAL curve geometry by converting the blue dots directly to real-world points."""
        path                  = Path()
        path.header.stamp     = self.get_clock().now().to_msg()
        path.header.frame_id  = 'base_footprint'

        for row_y, cx in centroids:
            x_m = (float(self.bird_h) - float(row_y)) * float(self.bird_m_per_px_y)
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

        # Only pay the upscale + serialize cost when something is subscribed.
        if self.debug_pub.get_subscription_count() > 0:
            debug_display = cv2.resize(mask, (1280, 720))
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_display, 'mono8'))

        self.frame_count += 1

        # ══════════════════════════════════════════════════════════════════
        # STAGE 1 — Row-centroid extraction
        # ══════════════════════════════════════════════════════════════════
        lane_centroids, lane_conf = self._row_centroids(mask, is_road=False)
        road_centroids, road_conf = self._row_centroids(road_mask, is_road=True)

        # ══════════════════════════════════════════════════════════════════
        # STAGE 2 — Scene classification
        # ══════════════════════════════════════════════════════════════════
        pri_centroids = road_centroids if road_conf > 0.05 else lane_centroids
        pri_conf      = road_conf if road_conf > 0.05 else lane_conf

        raw_scene = self._classify_scene(pri_centroids, pri_conf)
        self._scene_history.append(raw_scene)
        if len(self._scene_history) > self._SCENE_VOTE_WIN:
            self._scene_history.pop(0)
        scene = max(set(self._scene_history), key=self._scene_history.count)

        # ── UPGRADED: Publish the scene string so the control node can brake ──
        scene_msg = String()
        scene_msg.data = scene
        self.scene_pub.publish(scene_msg)

        # ── NEW: Anticipatory corner detection (direction + distance + angle) ──
        # Uses the road centreline drift (pri_centroids), the robust signal here.
        # Direction is majority-voted to kill single-frame flicker.
        raw_dir, corner_dist, corner_ang = self._detect_corner(pri_centroids, white_mask=mask)
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
        if lane_detected and active_centroids:
            self._publish_path_from_centroids(active_centroids)

        # ── Debug overlay ──────────────────────────────────────────────────
        for row_y, cx in active_centroids:
            cv2.circle(bird_debug, (int(cx), int(row_y)), 3, (200, 200, 0), -1)

        if la_y is not None:
            cv2.circle(bird_debug, (int(self.current_cx), int(la_y)), 8, (0, 255, 255), -1)

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
        label        = (f'{mode_str} | {scene} | conf={pri_conf:.2f} '
                        f'edge={self._road_edge_conf:.2f} | {corner_dir} '
                        f'floor={self._floor_cue_str:+.2f} | e={error:.1f}deg')
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
                f'corner={corner_dir}  floor={self._floor_cue_str:+.2f}  '
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