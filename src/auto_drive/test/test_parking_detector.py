"""Offline regression test for the parking-bay classifier.

Runs the REAL detector methods (`_find_dashes` + `_classify`) against fixed
fixtures so a future change that breaks bay detection on this track fails the
build. Two fixture sets, testing two different things:

1. *_perpendicular / *_none — crops of the flat track texture (tracknxgv.png),
   treated as clean top-down BEV. These unit-test the orientation logic: the
   texture's vertical dividers -> 'perpendicular'.

2. live_* — REAL bird's-eye frames captured from the running sim while the car
   drove a south->north approach to the pocket (warped through perception's
   trapezoid). These integration-test real detection.

IMPORTANT — the bay TYPE label is approach-dependent (documented in the
parking-detector memo): the SAME physical pocket reads 'perpendicular' from the
flat texture but 'parallel' on the live south->north approach, because the
divider line projects ACROSS the path. So the live test asserts *detection*
(a bay is found at close range, nothing at distance) plus the observed live
orientation — not that the two fixture sets agree on the label.

Polarity: dark road/bays, bright off-track surround, white dotted dividers. The
node methods need only a few attributes, so we build a bare instance with
object.__new__ and avoid spinning rclpy.
"""

import os

import cv2
import numpy as np
import pytest

from auto_drive.parking_detector_node import ParkingDetectorNode

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures', 'parking')

# Flat-texture crops (clean top-down) -> orientation-logic check.
TEXTURE_CASES = [
    ('right_perpendicular.png', 'perpendicular'),
    ('left_perpendicular.png',  'perpendicular'),
    ('road_none.png',           'none'),
    ('topbar_none.png',         'none'),
    ('roundabout_none.png',     'none'),
]

# Live sim BEV frames from a real driven approach -> detection check.
# (filename, expected_detected, expected_live_type)
LIVE_CASES = [
    ('live_close_bay.png',     True,  'parallel'),  # at the pocket: divider across path
    ('live_approach_none.png', False, 'none'),       # too far: only 2 dashes, below min
    ('live_far_none.png',      False, 'none'),       # far: nothing in usable BEV
]


def _stub_node():
    """A detector instance with default params but no ROS context."""
    n = object.__new__(ParkingDetectorNode)
    # Mirror the node's declared defaults (see parking_detector_node.py).
    n.bird_w = 1000
    n.bird_h = 720
    n.white_lo = np.array([150, 90, 90], dtype=np.uint8)
    n.white_hi = np.array([255, 130, 130], dtype=np.uint8)
    n.dash_area_min_frac = 0.0006
    n.dash_area_max_frac = 0.05
    n.dash_min_elong = 1.8
    n.dash_max_len_frac = 0.20
    n.min_dashes = 3
    n.travel_vertical = True
    n.divider_bin_frac = 0.05
    n.min_confidence = 0.35
    n.mpp_x = 0.01
    # World-frame classification off by default (no EKF yaw in offline tests) ->
    # car-relative behaviour, so the fixture expectations are unchanged.
    n.use_world_frame = True
    n.aisle_heading_deg = 90.0
    n._yaw = None
    n.aspect_min_ratio = 1.2
    return n


def _classify_fixture(node, filename):
    img = cv2.imread(os.path.join(FIXTURES, filename))
    assert img is not None, f'missing fixture {filename}'
    bev = cv2.resize(img, (node.bird_w, node.bird_h), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(bev, cv2.COLOR_BGR2LAB)
    white = cv2.inRange(lab, node.white_lo, node.white_hi)
    dashes = node._find_dashes(white)
    bay_type, group, info = node._classify(dashes)
    return bay_type, dashes, info


@pytest.mark.parametrize('filename,expected', TEXTURE_CASES)
def test_texture_orientation(filename, expected):
    node = _stub_node()
    bay_type, _, _ = _classify_fixture(node, filename)
    assert bay_type == expected, (
        f'{filename}: expected {expected}, got {bay_type}')


def test_perpendicular_dashes_run_along_travel():
    """Texture dividers are vertical -> tall dashes -> 'along'."""
    node = _stub_node()
    _, dashes, _ = _classify_fixture(node, 'right_perpendicular.png')
    along = [d for d in dashes if d[4] == 'along']
    assert len(along) >= node.min_dashes, f'too few along-travel dashes: {len(along)}'


@pytest.mark.parametrize('filename,detected,live_type', LIVE_CASES)
def test_live_approach_detection(filename, detected, live_type):
    node = _stub_node()
    bay_type, _, _ = _classify_fixture(node, filename)
    assert (bay_type != 'none') == detected, (
        f'{filename}: detected={bay_type != "none"} expected {detected} (got {bay_type})')
    assert bay_type == live_type, (
        f'{filename}: expected live type {live_type}, got {bay_type}')


def test_live_close_rejects_road_edge_sliver():
    """The full-width road/off-track boundary must NOT survive as a dash."""
    node = _stub_node()
    _, dashes, _ = _classify_fixture(node, 'live_close_bay.png')
    longest = max((max(d[2], d[3]) for d in dashes), default=0)
    cap = node.dash_max_len_frac * max(node.bird_w, node.bird_h)
    assert longest <= cap, f'a blob of {longest}px slipped past the {cap:.0f}px cap'


def test_detection_is_confident():
    """A real divider should classify with confidence above the gate."""
    node = _stub_node()
    _, _, info = _classify_fixture(node, 'live_close_bay.png')
    assert info['confidence'] >= node.min_confidence, (
        f'confidence {info["confidence"]:.2f} below gate {node.min_confidence}')
    assert info['n_dividers'] >= 1


def _synthetic_dashes(angle_deg, n=5, spacing=70, cx=500, cy=360, length=40, thick=14):
    """A dark BEV with n collinear white dashes along `angle_deg` (from +x)."""
    img = np.zeros((720, 1000, 3), np.uint8)
    th = np.deg2rad(angle_deg)
    ux, uy = np.cos(th), np.sin(th)          # divider (line) direction
    # dashes are laid along the line direction, each dash elongated along it
    for k in range(n):
        off = (k - (n - 1) / 2) * spacing
        px, py = cx + ux * off, cy + uy * off
        a = (int(px - ux * length / 2), int(py - uy * length / 2))
        b = (int(px + ux * length / 2), int(py + uy * length / 2))
        cv2.line(img, a, b, (255, 255, 255), thick)
    return img


def _classify_synthetic(node, angle_deg):
    img = _synthetic_dashes(angle_deg)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    white = cv2.inRange(lab, node.white_lo, node.white_hi)
    dashes = node._find_dashes(white)
    return node._classify(dashes)


@pytest.mark.parametrize('angle,expected', [
    (90, 'perpendicular'),   # vertical divider -> along travel
    (75, 'perpendicular'),   # tilted, still along-travel band
    (0,  'parallel'),        # horizontal divider -> across travel
    (20, 'parallel'),        # tilted, still across-travel band
])
def test_tilted_divider_is_rotation_robust(angle, expected):
    """PCA orientation must classify a tilted dotted divider correctly where
    the old axis-aligned bbox aspect would have failed (near-square bbox)."""
    node = _stub_node()          # _yaw None -> car-relative (BEV vertical = aisle)
    bay_type, _, info = _classify_synthetic(node, angle)
    assert bay_type == expected, (
        f'{angle}deg divider: expected {expected}, got {bay_type} '
        f'(fitted angle {info["angle_deg"]:.0f})')


@pytest.mark.parametrize('yaw_deg', [60, 90, 120])
def test_world_frame_label_is_approach_invariant(yaw_deg):
    """The SAME physical bay must classify the same from any approach yaw once
    EKF yaw rotates the divider into the world frame. A divider parallel to the
    aisle (heading 90) is 'perpendicular' bays; perpendicular to it is
    'parallel' — regardless of how the car is currently steered."""
    node = _stub_node()          # aisle_heading=90, use_world_frame=True
    node._yaw = np.deg2rad(yaw_deg)
    # Divider PARALLEL to the aisle: theta_world = 90 -> theta_bev = yaw.
    par_type, _, info = _classify_synthetic(node, yaw_deg % 180)
    assert par_type == 'perpendicular', (
        f'yaw={yaw_deg}: parallel-to-aisle divider should be perpendicular bays, '
        f'got {par_type} (world angle {info["angle_world_deg"]:.0f})')
    # Divider PERPENDICULAR to the aisle: theta_world = 0 -> theta_bev = 90+yaw.
    perp_type, _, _ = _classify_synthetic(node, (90 + yaw_deg) % 180)
    assert perp_type == 'parallel', (
        f'yaw={yaw_deg}: perpendicular-to-aisle divider should be parallel bays, '
        f'got {perp_type}')


def test_world_frame_fixes_the_car_relative_flip():
    """Demonstrates the bug the world frame fixes: a fixed physical divider seen
    from two different yaws flips the CAR-RELATIVE label but not the world one."""
    # Same physical bay (parallel to a 90deg aisle) seen at yaw 90 vs yaw 30.
    bev_at_90 = 90 % 180          # vertical in BEV when driving along the aisle
    bev_at_30 = 30 % 180          # skewed in BEV when the car is yawed 60deg off

    car = _stub_node(); car.use_world_frame = False
    assert _classify_synthetic(car, bev_at_90)[0] == 'perpendicular'
    assert _classify_synthetic(car, bev_at_30)[0] == 'parallel'   # <-- flips!

    world = _stub_node()          # use_world_frame True
    world._yaw = np.deg2rad(90)
    assert _classify_synthetic(world, bev_at_90)[0] == 'perpendicular'
    world._yaw = np.deg2rad(30)
    assert _classify_synthetic(world, bev_at_30)[0] == 'perpendicular'  # stable


def _synthetic_dividers(angle_deg, n_lines, line_len, pitch, cx=500, cy=360, thick=14):
    """A dark BEV with n_lines parallel dotted dividers.

    Each divider runs along `angle_deg` with total length `line_len`; adjacent
    dividers are offset perpendicular by `pitch`. Dashes are auto-sized to clear
    the detector's area/elongation/separation gates (needs line_len >= ~180).
    Lets us test the Tier 1 aspect (depth/width) classifier: depth=line_len,
    width=pitch.
    """
    img = np.zeros((720, 1000, 3), np.uint8)
    th = np.deg2rad(angle_deg)
    ux, uy = np.cos(th), np.sin(th)          # along the divider
    vx, vy = -uy, ux                         # perpendicular (line-to-line)
    n_dashes = max(3, int(round(line_len / 60.0)))
    spacing = line_len / n_dashes            # ~60 px between dash centres
    dash = 0.55 * spacing                    # full length < spacing => separated
    for li in range(n_lines):
        poff = (li - (n_lines - 1) / 2) * pitch
        lx, ly = cx + vx * poff, cy + vy * poff
        for k in range(n_dashes):
            doff = (k - (n_dashes - 1) / 2) * spacing
            px, py = lx + ux * doff, ly + uy * doff
            a = (int(px - ux * dash / 2), int(py - uy * dash / 2))
            b = (int(px + ux * dash / 2), int(py + uy * dash / 2))
            cv2.line(img, a, b, (255, 255, 255), thick)
    return img


def _classify_dividers(node, **kw):
    img = _synthetic_dividers(**kw)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    white = cv2.inRange(lab, node.white_lo, node.white_hi)
    return node._classify(node._find_dashes(white))


def test_tier1_aspect_classifies_perpendicular_when_deep_and_narrow():
    """Two long dividers spaced closely => depth>width => perpendicular,
    decided by aspect (not the angle method)."""
    node = _stub_node()
    bay_type, _, info = _classify_dividers(
        node, angle_deg=90, n_lines=2, line_len=300, pitch=90)
    assert info['classified_by'] == 'aspect'
    assert info['aspect'] > 1.0
    assert bay_type == 'perpendicular'


def test_tier1_aspect_classifies_parallel_when_long_and_shallow():
    """Two short dividers spaced far apart => depth<width => parallel."""
    node = _stub_node()
    bay_type, _, info = _classify_dividers(
        node, angle_deg=90, n_lines=2, line_len=190, pitch=400)
    assert info['classified_by'] == 'aspect'
    assert info['aspect'] < 1.0
    assert bay_type == 'parallel'


def test_tier1_single_divider_falls_back_to_angle():
    """One divider has no pitch => no aspect => angle method, dims unset."""
    node = _stub_node()
    bay_type, _, info = _classify_dividers(
        node, angle_deg=90, n_lines=1, line_len=300, pitch=90)
    assert info['classified_by'] == 'angle'
    assert info['aspect'] == -1.0 and info['bay_w_m'] == -1.0
    assert bay_type == 'perpendicular'        # angle method still works


def test_info_schema_complete_when_no_bay():
    """The 'none' info dict must carry EVERY key the debug overlay reads, or the
    node crashes on the first armed frame with no divider (regression guard)."""
    node = _stub_node()
    bay_type, group, info = node._classify([])       # no dashes -> 'none'
    assert bay_type == 'none' and group == []
    for key in ('angle_deg', 'angle_world_deg', 'world_frame', 'confidence',
                'n_dividers', 'pitch_m', 'bay_w_m', 'bay_d_m', 'aspect',
                'classified_by', 'bay_w_px'):
        assert key in info, f'missing key {key!r} in none-case info'


def test_tier1_measures_bay_width_from_pitch():
    """bay width tracks the divider pitch (in fictitious BEV metres)."""
    node = _stub_node()
    _, _, info = _classify_dividers(
        node, angle_deg=90, n_lines=3, line_len=300, pitch=100)
    assert info['bay_w_m'] == pytest.approx(100 * node.mpp_x, rel=0.15)
