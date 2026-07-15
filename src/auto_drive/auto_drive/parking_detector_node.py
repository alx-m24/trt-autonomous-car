"""
parking_detector_node.py — vision-based parking-bay detector (pure OpenCV).

Role in the stack
------------------
This node *only detects and localises* a parking bay; it never touches
/cmd_vel. control_node owns the actuator. The detector publishes a versatile,
strategy-agnostic target so any future maneuver planner (geometric two-arc,
Reeds–Shepp, MPC, …) can consume it unchanged:

    /parking/detected     std_msgs/Bool             — a stable bay is in view
    /parking/bay_type      std_msgs/String           — 'parallel'|'perpendicular'|'none'
    /parking/target        geometry_msgs/PoseStamped — bay centre + approach heading,
                                                        base_footprint frame
    /parking/bay_corners   geometry_msgs/PolygonStamped — 4 bay corners (base_footprint)
    /parking/distance_m    std_msgs/Float32          — range to bay centre
    /parking/occupied      std_msgs/Bool             — bay interior looks blocked
    /debug/parking         sensor_msgs/Image         — annotated bird's-eye (gated)

Track-grounded detection method (tuned 2026-07 against tracknxgv.png)
---------------------------------------------------------------------
The track surface is DARK road/bays on a BRIGHT off-track surround, and the
bays are separated by WHITE dotted divider lines. So:
  * A LAB white mask captures the bright surround AND the dotted dashes.
  * The dashes are the only reliable bay cue. We isolate them as connected
    components that are small + elongated (the huge off-track blob and noise
    are filtered out by area/elongation), NOT with Canny+Hough — which on this
    polarity just fragments every dark/white boundary into noise (and is the
    expensive op this hardware struggled with).
  * Bay type = orientation of the dotted dividers relative to travel:
        dividers ALONG travel  (tall dashes)  -> perpendicular bays
        dividers ACROSS travel (wide dashes)  -> parallel bays
    Validated on the real texture: both parking clusters -> 'perpendicular';
    plain road / roundabout / top connector -> 'none'.
  * Empty bay interior is dark road; a parked car/obstacle is brighter ->
    occupancy from the dark-fill fraction inside the bay box.

The bird's-eye is warped so travel = up. If you ever approach the bays
head-on instead of up the aisle, flip `travel_is_vertical`.
"""

import math
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import PoseStamped, PolygonStamped, Point32
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


class ParkingDetectorNode(Node):
    def __init__(self):
        super().__init__('parking_detector_node')
        self.bridge = CvBridge()

        self.show_windows = bool(os.environ.get('DISPLAY'))

        # ── Camera in ───────────────────────────────────────────────────────
        # Prefer the COMPRESSED stream: a JPEG frame is ~50-100 kB vs ~2.7 MB
        # for raw 1280x720 BGR, so subscribing compressed keeps this detector
        # (which runs alongside perception's own warp) off the transport hot
        # path. If no compressed frames arrive within a grace period we fall
        # back to the raw topics, so this still works on setups without the
        # image_transport republisher.
        self.image_topics = ['/front_camera/image_raw', '/camera/image_raw']
        self.compressed_topics = [t + '/compressed' for t in self.image_topics]
        self.active_image_topic = None
        self._got_compressed = False
        self._raw_fallback_done = False
        self._image_subs = []
        for topic in self.compressed_topics:
            self._image_subs.append(
                self.create_subscription(
                    CompressedImage, topic,
                    lambda msg, t=topic: self.compressed_cb(msg, t), 10
                )
            )
        # One-shot-style check: add raw subs only if compressed never showed up.
        self.create_timer(3.0, self._maybe_raw_fallback)

        # ── Arming gate ─────────────────────────────────────────────────────
        # Idle (and ~free) until control_node / an operator flips this true.
        self.armed = False
        self.create_subscription(Bool, '/mission/parking_active', self._arm_cb, 10)

        # ── Robot heading (EKF-fused) ───────────────────────────────────────
        # Used to rotate the fitted divider direction into the world frame so
        # the bay-type label is invariant to the car's approach heading (it
        # otherwise flickers as the car steers, because the BEV is car-fixed).
        self._yaw = None       # radians, world frame; None until first sample
        self.create_subscription(
            Odometry, '/odometry/filtered', self._odom_cb, 10)

        # ── Outputs (signals only — control_node owns /cmd_vel) ─────────────
        self.detected_pub = self.create_publisher(Bool,          '/parking/detected',    10)
        self.type_pub     = self.create_publisher(String,        '/parking/bay_type',    10)
        self.target_pub   = self.create_publisher(PoseStamped,   '/parking/target',      10)
        self.corners_pub  = self.create_publisher(PolygonStamped,'/parking/bay_corners', 10)
        self.dist_pub     = self.create_publisher(Float32,       '/parking/distance_m',  10)
        self.occ_pub      = self.create_publisher(Bool,          '/parking/occupied',    10)
        self.debug_pub    = self.create_publisher(Image,         '/debug/parking',       10)

        # ── Bird's-eye calibration — seeded from perception_node so the two
        #    share one source of truth for this camera. ─────────────────────
        self.declare_parameter('tl', [140.0, 260.0])
        self.declare_parameter('bl', [0.0,   720.0])
        self.declare_parameter('tr', [1140.0, 260.0])
        self.declare_parameter('br', [1280.0, 720.0])
        self.declare_parameter('bird_w', 1000)
        self.declare_parameter('bird_h', 720)
        self.declare_parameter('bird_m_per_px_x', 0.01)
        self.declare_parameter('bird_m_per_px_y', 0.01)

        # White (painted line / bright off-track) LAB mask — matches perception.
        self.declare_parameter('white_lo', [150.0, 90.0,  90.0])
        self.declare_parameter('white_hi', [255.0, 130.0, 130.0])
        # Dark = drivable road / empty bay interior (L below this is "dark").
        # Measured from a live BEV capture (2026-07-01): road L≈87, so 80 was
        # too tight and read the road as "not dark". 100 gives margin.
        self.declare_parameter('road_dark_threshold', 100)

        # ── Detector tunables (defaults tuned against tracknxgv.png) ─────────
        self.declare_parameter('detector_decimation', 3)      # process 1 of N frames
        self.declare_parameter('travel_is_vertical', True)    # see module docstring
        # Dotted-dash connected-component filter, as fractions of bird area /
        # aspect, so they survive a re-calibrated BEV without re-tuning:
        #   dash_area_min_frac : reject specks below this fraction of the BEV.
        #   dash_area_max_frac : reject blobs above this (the off-track surround).
        #   dash_min_elong     : min long/short side ratio (a dash is elongated).
        self.declare_parameter('dash_area_min_frac', 0.0006)
        self.declare_parameter('dash_area_max_frac', 0.05)
        self.declare_parameter('dash_min_elong',     1.8)
        # Reject blobs longer than this fraction of the BEV's long side. On the
        # live approach the road/off-track boundary slips through as a near
        # full-width elongated sliver (maxdim≈1.0); real dashes measured
        # ≈0.06–0.18, so 0.20 keeps every dash and drops the sliver + oversized
        # pocket-corner blobs (≈0.28). Tuned on a real driven approach 2026-07.
        self.declare_parameter('dash_max_len_frac',  0.20)
        self.declare_parameter('min_dashes',         3)       # collinear, one divider
        # Collinear clustering: dashes whose perpendicular offset (across the
        # dominant divider direction) falls within this fraction of the BEV
        # short side belong to the SAME dotted divider line.
        self.declare_parameter('divider_bin_frac',   0.05)
        # Reject a bay whose classification confidence (collinearity × angular
        # concentration) is below this — kills scattered noise that happens to
        # have the right dash count but no coherent divider.
        self.declare_parameter('min_confidence',     0.35)
        # Approach-invariant classification: rotate the fitted divider direction
        # into the world frame using EKF yaw and classify it against a FIXED
        # aisle heading, instead of the car's momentary heading. This keeps the
        # bay-type label stable while the car steers during the approach.
        #   use_world_frame          : off => legacy car-relative (BEV-vertical).
        #   parking_aisle_heading_deg : world-frame heading (deg, CCW from +x)
        #     the car drives ALONG to reach the bays. A divider PARALLEL to this
        #     aisle => perpendicular bays; PERPENDICULAR to it => parallel bays.
        #     Default 90 = +y (matches the documented south->north approach).
        self.declare_parameter('use_world_frame',    True)
        self.declare_parameter('parking_aisle_heading_deg', 90.0)
        # Tier 1 dimension-aware classification: when >=2 dividers give a pitch,
        # classify by bay aspect (depth/width) instead of the divider angle —
        # deep-narrow => perpendicular, long-shallow => parallel. Only trusted
        # when the shape is decisive: max(aspect, 1/aspect) >= this ratio;
        # otherwise fall back to the angle/world-frame method.
        self.declare_parameter('aspect_min_ratio', 1.2)
        # Distance band (FICTITIOUS BEV metre, like /parking/distance_m). Measured
        # live: the bay is detected at dist≈7.1 and falls to ≈5.7 as the car noses
        # in, so 3.0–9.0 brackets the whole usable window. (The old 0.20–1.50 was
        # in assumed real metres and never matched, so detection never confirmed.)
        self.declare_parameter('opt_min_m', 3.0)
        self.declare_parameter('opt_max_m', 9.0)
        self.declare_parameter('stable_hits', 6)              # of last `vote_win`
        self.declare_parameter('vote_win', 10)
        # Occupancy: empty bay interior is mostly dark road.
        self.declare_parameter('occupied_dark_max', 0.45)     # <this dark frac => occupied

        self._load_params()
        self.add_on_set_parameters_callback(self._param_cb)
        self._recompute_warp()

        # ── State ───────────────────────────────────────────────────────────
        self.frame_count = 0
        self._type_hist = []   # rolling bay-type votes for temporal stability

        self.get_logger().info(
            'ParkingDetectorNode started — IDLE until /mission/parking_active=true.'
        )
        self.get_logger().info(f'Subscribed to: {self.image_topics}')

    # ────────────────────────────────────────────────────────────────────────
    # Parameters
    # ────────────────────────────────────────────────────────────────────────
    def _load_params(self):
        g = self.get_parameter
        self.tl = tuple(g('tl').value)
        self.bl = tuple(g('bl').value)
        self.tr = tuple(g('tr').value)
        self.br = tuple(g('br').value)
        self.bird_w = int(g('bird_w').value)
        self.bird_h = int(g('bird_h').value)
        self.mpp_x = float(g('bird_m_per_px_x').value)
        self.mpp_y = float(g('bird_m_per_px_y').value)
        self.white_lo = np.array(g('white_lo').value, dtype=np.uint8)
        self.white_hi = np.array(g('white_hi').value, dtype=np.uint8)
        self.road_dark = int(g('road_dark_threshold').value)
        self.decim = max(1, int(g('detector_decimation').value))
        self.travel_vertical = bool(g('travel_is_vertical').value)
        self.dash_area_min_frac = float(g('dash_area_min_frac').value)
        self.dash_area_max_frac = float(g('dash_area_max_frac').value)
        self.dash_min_elong = float(g('dash_min_elong').value)
        self.dash_max_len_frac = float(g('dash_max_len_frac').value)
        self.min_dashes = int(g('min_dashes').value)
        self.divider_bin_frac = float(g('divider_bin_frac').value)
        self.min_confidence = float(g('min_confidence').value)
        self.use_world_frame = bool(g('use_world_frame').value)
        self.aisle_heading_deg = float(g('parking_aisle_heading_deg').value)
        self.aspect_min_ratio = float(g('aspect_min_ratio').value)
        self.opt_min = float(g('opt_min_m').value)
        self.opt_max = float(g('opt_max_m').value)
        self.stable_hits = int(g('stable_hits').value)
        self.vote_win = max(1, int(g('vote_win').value))
        self.occupied_dark_max = float(g('occupied_dark_max').value)

    def _param_cb(self, params):
        self._load_params()
        self._recompute_warp()
        return SetParametersResult(successful=True)

    def _arm_cb(self, msg: Bool):
        if msg.data != self.armed:
            self.get_logger().info(f'parking detector {"ARMED" if msg.data else "disarmed"}')
        self.armed = bool(msg.data)
        if not self.armed:
            self._type_hist.clear()

    def _odom_cb(self, msg: Odometry):
        """Cache the EKF-fused planar yaw (world frame) for world-frame classify."""
        q = msg.pose.pose.orientation
        self._yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                               1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    # ────────────────────────────────────────────────────────────────────────
    # Bird's-eye + geometry helpers
    # ────────────────────────────────────────────────────────────────────────
    def _recompute_warp(self):
        pts1 = np.float32([self.tl, self.bl, self.tr, self.br])
        pts2 = np.float32([[0, 0], [0, self.bird_h],
                           [self.bird_w, 0], [self.bird_w, self.bird_h]])
        self._warp = cv2.getPerspectiveTransform(pts1, pts2)

    def _bev_to_base(self, px, py):
        """Bird's-eye pixel -> base_footprint metres (REP-103: x fwd, y left)."""
        x_m = (float(self.bird_h) - float(py)) * self.mpp_y
        y_m = (float(self.bird_w) * 0.5 - float(px)) * self.mpp_x
        return x_m, y_m

    @staticmethod
    def _yaw_to_quat(yaw):
        return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))

    # ────────────────────────────────────────────────────────────────────────
    # Detection — dotted dividers via connected components (no Canny/Hough)
    # ────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _component_pca(labels, stats, i):
        """Rotation-aware orientation + elongation of connected component i.

        Fits the component's pixel cloud with PCA. Returns
        (angle_deg in [0,180), elongation = sqrt(lam_max/lam_min)). This
        replaces the axis-aligned bounding-box aspect, which mis-reads a
        perspective-tilted dash (a 45deg thin dash has a near-square bbox but
        is genuinely elongated). Angle 90deg == vertical (image y is down).
        """
        x0 = int(stats[i, cv2.CC_STAT_LEFT]); y0 = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH]);  h = int(stats[i, cv2.CC_STAT_HEIGHT])
        sub = labels[y0:y0 + h, x0:x0 + w] == i
        ys, xs = np.nonzero(sub)
        if xs.size < 5:                       # too few px for a stable fit -> bbox
            return (90.0 if h >= w else 0.0), (max(w, h) / max(1, min(w, h)))
        xs = xs.astype(np.float64) - xs.mean()
        ys = ys.astype(np.float64) - ys.mean()
        sxx = float(np.mean(xs * xs)); syy = float(np.mean(ys * ys))
        sxy = float(np.mean(xs * ys))
        evals, evecs = np.linalg.eigh(np.array([[sxx, sxy], [sxy, syy]]))  # asc
        vx, vy = evecs[0, 1], evecs[1, 1]     # eigenvector of the larger eigenvalue
        ang = math.degrees(math.atan2(vy, vx)) % 180.0
        elong = math.sqrt(max(evals[1], 1e-6) / max(evals[0], 1e-6))
        return ang, elong

    def _find_dashes(self, white_mask):
        """Return dotted-line dashes as [(cx, cy, w, h, orient, angle_deg)].

        A dash is a small, elongated white connected component. The huge
        off-track surround (area too big) and specks (area too small) are
        rejected on bounding-box area; road-edge slivers on bbox length.
        Orientation and elongation come from PCA on the pixels (rotation-aware),
        so a divider seen at an angle through the perspective warp still reads
        as one elongated dash. `orient` ('along'/'across' travel) is kept for
        the debug overlay; the classifier uses the fitted angle directly.
        """
        n, labels, stats, cent = cv2.connectedComponentsWithStats(white_mask, 8)
        area = float(self.bird_w * self.bird_h)
        a_min = self.dash_area_min_frac * area
        a_max = self.dash_area_max_frac * area
        len_max = self.dash_max_len_frac * max(self.bird_w, self.bird_h)
        dashes = []
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            if not (a_min <= a <= a_max):
                continue
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            if max(w, h) > len_max:          # reject road-edge slivers / oversized blobs
                continue
            angle, elong = self._component_pca(labels, stats, i)
            if elong < self.dash_min_elong:
                continue
            dv = abs(angle - 90.0)            # 0 => vertical
            long_is_vertical = dv <= 45.0
            along = long_is_vertical if self.travel_vertical else not long_is_vertical
            dashes.append((float(cent[i][0]), float(cent[i][1]),
                           int(w), int(h), 'along' if along else 'across',
                           float(angle)))
        return dashes

    @staticmethod
    def _circular_mean_deg(angles):
        """Orientation mean on the [0,180) circle (via the double angle)."""
        a = np.deg2rad(np.asarray(angles, dtype=np.float64)) * 2.0
        s, c = float(np.mean(np.sin(a))), float(np.mean(np.cos(a)))
        theta = 0.5 * math.degrees(math.atan2(s, c)) % 180.0
        concentration = math.hypot(s, c)     # 0 (spread) .. 1 (aligned)
        return theta, concentration

    def _divider_lines(self, dashes):
        """Cluster dashes into collinear dotted dividers.

        The dashes of one divider are collinear, so once we know the dominant
        dash direction theta we project every centroid onto the axis
        PERPENDICULAR to theta: dashes on the same divider share ~one offset.
        Binning those offsets recovers the individual divider lines — this is
        the real signal (a divider is a line, not a bag of blobs) and lets us
        require genuine collinear evidence and measure inter-divider pitch.

        Returns (theta_deg, concentration, [line_dashes...], [line_offset...])
        with lines sorted largest-first, or None if no divider has min_dashes.
        """
        if len(dashes) < self.min_dashes:
            return None
        theta, concentration = self._circular_mean_deg([d[5] for d in dashes])
        th = math.radians(theta)
        vx, vy = -math.sin(th), math.cos(th)     # perpendicular to divider dir
        cents = np.array([(d[0], d[1]) for d in dashes])
        offs = cents[:, 0] * vx + cents[:, 1] * vy
        order = list(np.argsort(offs))
        tol = self.divider_bin_frac * min(self.bird_w, self.bird_h)
        clusters = [[order[0]]]
        for k in order[1:]:
            if offs[k] - offs[clusters[-1][-1]] <= tol:
                clusters[-1].append(k)
            else:
                clusters.append([k])
        lines, line_offs = [], []
        for idx in clusters:
            if len(idx) >= self.min_dashes:
                lines.append([dashes[j] for j in idx])
                line_offs.append(float(np.mean([offs[j] for j in idx])))
        if not lines:
            return None
        keyed = sorted(zip(lines, line_offs), key=lambda t: -len(t[0]))
        lines, line_offs = [t[0] for t in keyed], [t[1] for t in keyed]
        return theta, concentration, lines, line_offs

    @staticmethod
    def _line_angle_diff(a_deg, b_deg):
        """Smallest angle between two undirected lines, folded to [0, 90]."""
        d = (a_deg - b_deg + 90.0) % 180.0 - 90.0
        return abs(d)

    def _classify(self, dashes):
        """Bay type + strongest divider's dashes + geometry info.

        Robust discriminator: the fitted divider-LINE direction, compared to a
        reference "aisle" direction. A divider PARALLEL to the aisle separates
        perpendicular bays; a divider PERPENDICULAR to the aisle separates
        parallel bays (validated on the track texture + the live approach).

        The reference is the aisle expressed in the BEV. In legacy car-relative
        mode it is simply BEV-vertical (travel = up). In world-frame mode we
        rotate a FIXED world-frame aisle heading into the current BEV using EKF
        yaw, so the label no longer flips as the car steers on approach:

            theta_world = (90 - theta_bev + yaw_deg) mod 180   [divider, world]
            divider ∥ aisle  <=>  theta_world ≈ aisle_heading

        which, solved for the BEV reference angle, gives
            ref_bev = (90 - aisle_heading + yaw_deg) mod 180.

        `info` reports the fitted BEV angle, the world-frame angle, whether the
        world frame was actually used, a confidence (collinearity share ×
        angular concentration), the divider count, and pitch.
        """
        empty = {'angle_deg': -1.0, 'angle_world_deg': -1.0, 'world_frame': False,
                 'confidence': 0.0, 'n_dividers': 0, 'pitch_m': -1.0,
                 'bay_w_m': -1.0, 'bay_d_m': -1.0, 'aspect': -1.0,
                 'classified_by': 'none', 'bay_w_px': -1.0}
        res = self._divider_lines(dashes)
        if res is None:
            return 'none', [], empty
        theta, concentration, lines, line_offs = res

        use_world = self.use_world_frame and self._yaw is not None
        if use_world:
            yaw_deg = math.degrees(self._yaw)
            ref = (90.0 - self.aisle_heading_deg + yaw_deg) % 180.0
            theta_world = (90.0 - theta + yaw_deg) % 180.0
        else:
            ref = 90.0 if self.travel_vertical else 0.0   # aisle = BEV-vertical
            theta_world = -1.0
        # Divider PARALLEL to the aisle (small diff) => perpendicular bays.
        bay_type = ('perpendicular'
                    if self._line_angle_diff(theta, ref) <= 45.0 else 'parallel')

        on_lines = sum(len(l) for l in lines)
        confidence = min(1.0, (on_lines / max(1, len(dashes))) * concentration)

        pitch_px = -1.0
        if len(line_offs) >= 2:
            so = sorted(line_offs)
            gaps = [so[i + 1] - so[i] for i in range(len(so) - 1)]
            pitch_px = float(np.median(gaps))
        pitch_m = pitch_px * self.mpp_x if pitch_px > 0 else -1.0

        # ── Tier 1: measure the bay from geometry we already have ────────────
        # width ~ divider pitch (gap to the next divider; needs >=2 dividers);
        # depth ~ how far the strongest divider's dashes extend ALONG the line.
        th = math.radians(theta)
        ux, uy = math.cos(th), math.sin(th)
        proj = [d[0] * ux + d[1] * uy for d in lines[0]]
        depth_px = (max(proj) - min(proj)) if len(proj) >= 2 else -1.0
        bay_d_m = depth_px * self.mpp_x if depth_px > 0 else -1.0
        aspect = (depth_px / pitch_px) if (pitch_px > 0 and depth_px > 0) else -1.0

        # Classify by aspect when a pitch exists and the shape is decisive
        # (deep-narrow => perpendicular, long-shallow => parallel); this needs
        # no aisle assumption. Else keep the angle/world-frame result above.
        classified_by = 'angle'
        if aspect > 0.0 and max(aspect, 1.0 / aspect) >= self.aspect_min_ratio:
            classified_by = 'aspect'
            bay_type = 'perpendicular' if aspect >= 1.0 else 'parallel'

        info = {'angle_deg': float(theta), 'angle_world_deg': float(theta_world),
                'world_frame': bool(use_world), 'confidence': float(confidence),
                'n_dividers': len(lines), 'pitch_m': pitch_m,
                'bay_w_m': float(pitch_m), 'bay_d_m': float(bay_d_m),
                'aspect': float(aspect), 'classified_by': classified_by,
                'bay_w_px': float(pitch_px)}
        return bay_type, lines[0], info

    def _bay_box(self, group, width_px=None):
        """Axis-aligned BEV box around the divider dashes, padded to the bay.

        The dashes mark the divider between bays; the bay sits beside it, so we
        pad the dash cluster's bounding box laterally. When a measured divider
        pitch is available (Tier 1) we pad by half of it, so the box spans about
        one real bay width; otherwise we fall back to a fixed fraction of the
        BEV. Returns (corners_px[4x2], centre_px) or None.
        """
        if len(group) < 2:
            return None
        xs = np.array([d[0] for d in group])
        ys = np.array([d[1] for d in group])
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        # Pad laterally toward the adjacent bay (~half a bay width).
        pad = 0.5 * float(width_px) if (width_px and width_px > 0) else 0.10 * self.bird_w
        if self.travel_vertical:      # vertical divider -> bay is to the side
            x0 -= pad; x1 += pad
        else:                         # horizontal divider -> bay is fore/aft
            y0 -= pad; y1 += pad
        x0 = max(0.0, x0); y0 = max(0.0, y0)
        x1 = min(float(self.bird_w), x1); y1 = min(float(self.bird_h), y1)
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
        centre = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], np.float32)
        return corners, centre

    def _occupied(self, lab, corners):
        """Empty bay interior is mostly dark road; bright/textured => occupied."""
        x0 = int(max(0, corners[:, 0].min()))
        x1 = int(min(self.bird_w, corners[:, 0].max()))
        y0 = int(max(0, corners[:, 1].min()))
        y1 = int(min(self.bird_h, corners[:, 1].max()))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return False
        L = lab[y0:y1, x0:x1, 0]
        dark_frac = float(np.mean(L < self.road_dark))
        return dark_frac < self.occupied_dark_max

    # ────────────────────────────────────────────────────────────────────────
    # Camera callbacks
    # ────────────────────────────────────────────────────────────────────────
    def _maybe_raw_fallback(self):
        """Subscribe to the raw topics iff no compressed frame ever arrived."""
        if self._raw_fallback_done:
            return
        self._raw_fallback_done = True
        if self._got_compressed:
            return
        for topic in self.image_topics:
            self._image_subs.append(
                self.create_subscription(
                    Image, topic, lambda msg, t=topic: self.image_cb(msg, t), 10
                )
            )
        self.get_logger().warn(
            'no compressed camera frames — falling back to raw image topics')

    def _should_process(self, topic):
        """Arm gate + single-source lock + frame decimation."""
        if not self.armed:
            return False
        if self.active_image_topic is None:
            self.active_image_topic = topic
            self.get_logger().info(f'Receiving images on: {self.active_image_topic}')
        elif topic != self.active_image_topic:
            return False       # locked to one source; ignore the other stream
        self.frame_count += 1
        return self.frame_count % self.decim == 0

    def compressed_cb(self, msg, topic=''):
        self._got_compressed = True
        if not self._should_process(topic):
            return
        self._process_frame(self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8'))

    def image_cb(self, msg, topic=''):
        if not self._should_process(topic):
            return
        self._process_frame(self.bridge.imgmsg_to_cv2(msg, 'bgr8'))

    # ────────────────────────────────────────────────────────────────────────
    # Main processing
    # ────────────────────────────────────────────────────────────────────────
    def _process_frame(self, frame):
        frame = cv2.resize(frame, (1280, 720))
        bird = cv2.warpPerspective(frame, self._warp, (self.bird_w, self.bird_h))
        lab = cv2.cvtColor(bird, cv2.COLOR_BGR2LAB)
        white = cv2.inRange(lab, self.white_lo, self.white_hi)

        dashes = self._find_dashes(white)
        bay_type, group, cinfo = self._classify(dashes)

        box = (self._bay_box(group, width_px=cinfo['bay_w_px'])
               if bay_type != 'none' else None)
        in_band = False
        dist_m = -1.0
        occupied = False
        corners_m = None
        goal = None

        if box is not None:
            corners_px, centre_px = box
            cx_m, cy_m = self._bev_to_base(centre_px[0], centre_px[1])
            dist_m = float(math.hypot(cx_m, cy_m))
            in_band = self.opt_min <= dist_m <= self.opt_max
            occupied = self._occupied(lab, corners_px)
            corners_m = [self._bev_to_base(px, py) for px, py in corners_px]
            # Target orientation = the bay ENTRY HEADING in base_footprint, i.e.
            # the direction the car should point to nose straight in. The bay
            # centerline runs parallel to the dotted divider, so the entry
            # heading is the divider direction expressed in the base frame:
            #   theta_base = 90deg - theta_bev   (BEV up = +x forward)
            # folded into (-90, 90] so it points forward (into the bay). The
            # maneuver turns onto this heading; a straight-ahead bay gives ~0.
            # (This replaces the old bearing-to-centre, which carried no info
            # about how the bay itself is oriented.)
            yaw = math.radians(90.0 - cinfo['angle_deg'])
            goal = (cx_m, cy_m, yaw)

        # ── Temporal stability vote ─────────────────────────────────────────
        confident = cinfo['confidence'] >= self.min_confidence
        vote = bay_type if (box is not None and in_band and confident) else 'none'
        self._type_hist.append(vote)
        if len(self._type_hist) > self.vote_win:
            self._type_hist.pop(0)
        stable = max(set(self._type_hist), key=self._type_hist.count)
        confirmed = (stable != 'none'
                     and self._type_hist.count(stable) >= self.stable_hits)

        # ── Publish ─────────────────────────────────────────────────────────
        self.detected_pub.publish(Bool(data=bool(confirmed)))
        self.type_pub.publish(String(data=stable if confirmed else 'none'))
        self.dist_pub.publish(Float32(data=float(dist_m)))
        self.occ_pub.publish(Bool(data=bool(occupied)))

        if confirmed and goal is not None:
            now = self.get_clock().now().to_msg()

            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = 'base_footprint'
            ps.pose.position.x = float(goal[0])
            ps.pose.position.y = float(goal[1])
            qx, qy, qz, qw = self._yaw_to_quat(goal[2])
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            self.target_pub.publish(ps)

            poly = PolygonStamped()
            poly.header = ps.header
            for xm, ym in corners_m:
                poly.polygon.points.append(Point32(x=float(xm), y=float(ym), z=0.0))
            self.corners_pub.publish(poly)

        # ── Debug overlay (gated) ───────────────────────────────────────────
        if self.debug_pub.get_subscription_count() > 0 or self.show_windows:
            dbg = bird.copy()
            for cx, cy, w, h, orient, ang in dashes:
                col = (0, 255, 0) if orient == 'along' else (255, 120, 0)
                cv2.rectangle(dbg, (int(cx - w / 2), int(cy - h / 2)),
                              (int(cx + w / 2), int(cy + h / 2)), col, 2)
            # Draw the fitted divider direction through the strongest divider's
            # centroid — this is what the classifier actually decides on.
            if group and cinfo['angle_deg'] >= 0.0:
                gx = float(np.mean([d[0] for d in group]))
                gy = float(np.mean([d[1] for d in group]))
                th = math.radians(cinfo['angle_deg'])
                dxp, dyp = math.cos(th) * 120.0, math.sin(th) * 120.0
                cv2.line(dbg, (int(gx - dxp), int(gy - dyp)),
                         (int(gx + dxp), int(gy + dyp)), (0, 255, 255), 2)
            if box is not None:
                cv2.polylines(dbg, [box[0].astype(int)], True,
                              (0, 0, 255) if occupied else (0, 255, 255), 2)
            cv2.circle(dbg, (self.bird_w // 2, self.bird_h - 1), 6, (0, 0, 255), -1)
            label = (f'{stable if confirmed else "scan"} '
                     f'd={dist_m:.2f}m occ={occupied} dashes={len(dashes)}')
            frm = (f'W{cinfo["angle_world_deg"]:.0f}' if cinfo['world_frame']
                   else 'car')
            geom = (f'ang={cinfo["angle_deg"]:.0f}/{frm} conf={cinfo["confidence"]:.2f} '
                    f'ndiv={cinfo["n_dividers"]} pitch={cinfo["pitch_m"]:.2f}')
            cv2.putText(dbg, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if confirmed else (0, 200, 255), 2)
            cv2.putText(dbg, geom, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if confirmed else (0, 200, 255), 2)
            dims = (f'by={cinfo["classified_by"]} w={cinfo["bay_w_m"]:.2f} '
                    f'd={cinfo["bay_d_m"]:.2f} aspect={cinfo["aspect"]:.2f}')
            cv2.putText(dbg, dims, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if confirmed else (0, 200, 255), 2)
            if self.debug_pub.get_subscription_count() > 0:
                self.debug_pub.publish(self.bridge.cv2_to_imgmsg(dbg, 'bgr8'))
            if self.show_windows:
                try:
                    cv2.imshow('parking', dbg)
                    cv2.waitKey(1)
                except cv2.error:
                    self.show_windows = False


def main():
    rclpy.init()
    node = ParkingDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
