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
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import PoseStamped, PolygonStamped, Point32
from cv_bridge import CvBridge


class ParkingDetectorNode(Node):
    def __init__(self):
        super().__init__('parking_detector_node')
        self.bridge = CvBridge()

        self.show_windows = bool(os.environ.get('DISPLAY'))

        # ── Camera in (same topics perception uses) ─────────────────────────
        self.image_topics = ['/front_camera/image_raw', '/camera/image_raw']
        self.active_image_topic = None
        self._image_subs = []
        for topic in self.image_topics:
            self._image_subs.append(
                self.create_subscription(
                    Image, topic, lambda msg, t=topic: self.image_cb(msg, t), 10
                )
            )

        # ── Arming gate ─────────────────────────────────────────────────────
        # Idle (and ~free) until control_node / an operator flips this true.
        self.armed = False
        self.create_subscription(Bool, '/mission/parking_active', self._arm_cb, 10)

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
        self.declare_parameter('min_dashes',         3)       # of one orientation
        self.declare_parameter('opt_min_m', 0.20)             # fire only inside band
        self.declare_parameter('opt_max_m', 1.50)
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
    def _find_dashes(self, white_mask):
        """Return dotted-line dashes as [(cx, cy, w, h, 'along'/'across')].

        A dash is a small, elongated white connected component. The huge
        off-track surround (area too big) and specks (area too small) are
        rejected. Orientation is taken from the bounding box: with travel
        vertical, a tall dash (h>=w) runs ALONG travel.
        """
        n, _, stats, cent = cv2.connectedComponentsWithStats(white_mask, 8)
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
            if max(w, h) / max(1, min(w, h)) < self.dash_min_elong:
                continue
            long_is_vertical = h >= w
            along = long_is_vertical if self.travel_vertical else not long_is_vertical
            dashes.append((float(cent[i][0]), float(cent[i][1]),
                           int(w), int(h), 'along' if along else 'across'))
        return dashes

    def _classify(self, dashes):
        """Bay type + the dashes of the dominant divider orientation.

        Requires >= min_dashes of one orientation — there is no length
        fallback because both bay types have prominent along-travel edges, so
        only the dotted divider's orientation disambiguates them.
        """
        along = [d for d in dashes if d[4] == 'along']
        across = [d for d in dashes if d[4] == 'across']
        if len(along) >= self.min_dashes and len(along) >= len(across):
            return 'perpendicular', along    # dividers along travel
        if len(across) >= self.min_dashes and len(across) > len(along):
            return 'parallel', across         # dividers across travel
        return 'none', []

    def _bay_box(self, group):
        """Axis-aligned BEV box around the divider dashes, padded to the bay.

        The dashes mark the divider between bays; the bay sits beside it, so we
        pad the dash cluster's bounding box by half the inter-dash spacing on
        the short axis. Returns (corners_px[4x2], centre_px) or None.
        """
        if len(group) < 2:
            return None
        xs = np.array([d[0] for d in group])
        ys = np.array([d[1] for d in group])
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        # Pad laterally toward the adjacent bay (half a typical bay width).
        pad = 0.10 * self.bird_w
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
    # Main callback
    # ────────────────────────────────────────────────────────────────────────
    def image_cb(self, msg, topic=''):
        if not self.armed:
            return
        if topic and self.active_image_topic is None:
            self.active_image_topic = topic
            self.get_logger().info(f'Receiving images on: {self.active_image_topic}')

        self.frame_count += 1
        if self.frame_count % self.decim != 0:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        frame = cv2.resize(frame, (1280, 720))
        bird = cv2.warpPerspective(frame, self._warp, (self.bird_w, self.bird_h))
        lab = cv2.cvtColor(bird, cv2.COLOR_BGR2LAB)
        white = cv2.inRange(lab, self.white_lo, self.white_hi)

        dashes = self._find_dashes(white)
        bay_type, group = self._classify(dashes)

        box = self._bay_box(group) if bay_type != 'none' else None
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
            # Approach heading: point from the car toward the bay centre. The
            # final in-bay orientation is the maneuver planner's job; this gives
            # it a sane, geometry-derived seed instead of a hard-coded angle.
            yaw = math.atan2(cy_m, cx_m)
            goal = (cx_m, cy_m, yaw)

        # ── Temporal stability vote ─────────────────────────────────────────
        vote = bay_type if (box is not None and in_band) else 'none'
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
            for cx, cy, w, h, orient in dashes:
                col = (0, 255, 0) if orient == 'along' else (255, 120, 0)
                cv2.rectangle(dbg, (int(cx - w / 2), int(cy - h / 2)),
                              (int(cx + w / 2), int(cy + h / 2)), col, 2)
            if box is not None:
                cv2.polylines(dbg, [box[0].astype(int)], True,
                              (0, 0, 255) if occupied else (0, 255, 255), 2)
            cv2.circle(dbg, (self.bird_w // 2, self.bird_h - 1), 6, (0, 0, 255), -1)
            label = (f'{stable if confirmed else "scan"} '
                     f'd={dist_m:.2f}m occ={occupied} dashes={len(dashes)}')
            cv2.putText(dbg, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
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
