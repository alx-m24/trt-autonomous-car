#!/usr/bin/env python3
"""Wheel-encoder odometry publisher — the real-car replacement for the sim's
libgazebo_ros_planar_move /odom source.

In simulation, planar_move (my_car.urdf) synthesises a perfect signed /odom by
integrating the commanded cmd_vel. On the physical RC car nothing does that for
free: this node reproduces the SAME message contract the EKF already consumes,
so ekf.yaml needs no change.

Contract this node must satisfy (see config/ekf.yaml, odom0 = /odom):
  - nav_msgs/Odometry on /odom @ ~30 Hz
  - frame_id = odom, child_frame_id = base_footprint
  - fills absolute pose x, y, yaw  +  body-frame vx  +  yaw-rate
    (exactly the fields odom0_config marks true; everything else stays 0)

Model: rear-axle differential drive from TWO quadrature encoders. Even though
the car steers Ackermann up front, the two rear wheels give an honest vx and a
wheel-derived yaw-rate. The EKF then blends this with the IMU yaw, so small
wheel-slip errors in heading get corrected — which is the whole reason the EKF
exists in this stack.

Direction matters: the encoder driver feeding /left_ticks and /right_ticks MUST
be QUADRATURE (A/B), so the tick counts are SIGNED. A single-channel sensor
(e.g. HK-020K) can only report magnitude and will corrupt x/y/vx the instant the
car reverses into a parallel bay. That is the reason this node takes signed
cumulative counts rather than pulse rates.

TF ownership: in sim, planar_move published the odom->base_footprint TF and so
ekf.yaml sets publish_tf:false. On the real car, prefer letting the EKF own that
TF (flip ekf.yaml publish_tf:true) and leave publish_tf:false here (default), so
you never get two nodes fighting over the same transform. publish_tf here is
provided only for bring-up/debugging without the EKF running.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Int64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class WheelOdomNode(Node):
    def __init__(self):
        super().__init__('wheel_odom_node')

        # ── Geometry (defaults match my_car.urdf) ────────────────────────────
        # Rear wheel radius 0.05 m; rear track = 2 * 0.11 = 0.22 m.
        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('wheel_separation', 0.22)
        # Encoder resolution AFTER gearing and AFTER quadrature x4 decoding, i.e.
        # counts observed per one full wheel revolution. Set this to whatever your
        # MCU/counter reports per wheel-rev, NOT the raw disk PPR.
        self.declare_parameter('counts_per_rev', 1200.0)
        # EMPIRICAL CALIBRATION OVERRIDE. For a compound/unknown drivetrain
        # (e.g. motor gearbox x solid-axle ring/pinion x wheel dia), don't try to
        # derive scale analytically — roll the car a MEASURED distance, read the
        # total tick delta, and set this = distance_metres / total_ticks. If > 0
        # it overrides wheel_radius/counts_per_rev entirely.
        self.declare_parameter('meters_per_count', 0.0)
        # Flip if a wheel's A/B wiring makes forward motion count negative.
        self.declare_parameter('invert_left', False)
        self.declare_parameter('invert_right', False)

        # Single-encoder mode: ONE quadrature encoder on the drive-motor shaft
        # (rear-drive Ackermann has a single motor, so this is the real-car
        # default). Subscribes to /motor_ticks, publishes body vx ONLY; heading
        # and position are left to the IMU via the EKF. See the ekf.yaml note in
        # the class-level comment below.
        self.declare_parameter('single_encoder', False)
        self.declare_parameter('invert_motor', False)

        # ── Framing / topics ─────────────────────────────────────────────────
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('publish_tf', False)   # let the EKF own the TF
        self.declare_parameter('publish_rate', 30.0)

        # ── Noise model (feeds the EKF's trust in this source) ───────────────
        # planar_move used 1e-4 on x/y and 1e-2 on yaw; real wheels slip, so be a
        # touch more honest/pessimistic than the perfect sim integrator.
        self.declare_parameter('var_xy', 2.0e-3)
        self.declare_parameter('var_yaw', 2.0e-2)
        self.declare_parameter('var_vx', 5.0e-3)
        self.declare_parameter('var_wz', 2.0e-2)

        g = self.get_parameter
        self.wheel_radius = float(g('wheel_radius').value)
        self.wheel_sep = float(g('wheel_separation').value)
        self.counts_per_rev = float(g('counts_per_rev').value)
        self.inv_l = -1.0 if g('invert_left').value else 1.0
        self.inv_r = -1.0 if g('invert_right').value else 1.0
        self.single = bool(g('single_encoder').value)
        self.inv_m = -1.0 if g('invert_motor').value else 1.0
        self.odom_frame = str(g('odom_frame').value)
        self.base_frame = str(g('base_frame').value)
        self.publish_tf = bool(g('publish_tf').value)
        self.var_xy = float(g('var_xy').value)
        self.var_yaw = float(g('var_yaw').value)
        self.var_vx = float(g('var_vx').value)
        self.var_wz = float(g('var_wz').value)

        # metres of wheel travel per encoder count. Prefer the empirical
        # calibration if given; else derive from wheel geometry.
        mpc_override = float(g('meters_per_count').value)
        if mpc_override > 0.0:
            self.m_per_count = mpc_override
        else:
            self.m_per_count = (2.0 * math.pi * self.wheel_radius) / self.counts_per_rev

        # ── Integrated pose state (odom frame) ───────────────────────────────
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Last cumulative counts; None until the first message from each side, so
        # we never integrate a bogus delta against the encoder's power-on value.
        self.last_left = None
        self.last_right = None
        self.last_motor = None
        self.last_time = None

        # ── I/O ──────────────────────────────────────────────────────────────
        # SIGNED cumulative counts from the encoder driver (MCU / micro-ROS /
        # LS7366R bridge). Int64 so long runs never wrap.
        if self.single:
            self.create_subscription(Int64, '/motor_ticks', self._motor_cb, qos_profile_sensor_data)
        else:
            self.create_subscription(Int64, '/left_ticks', self._left_cb, qos_profile_sensor_data)
            self.create_subscription(Int64, '/right_ticks', self._right_cb, qos_profile_sensor_data)

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_bc = TransformBroadcaster(self) if self.publish_tf else None

        self.pending_left = None
        self.pending_right = None
        self.pending_motor = None

        period = 1.0 / max(1.0, float(g('publish_rate').value))
        self.create_timer(period, self._tick)

        self.get_logger().info(
            f'wheel_odom: r={self.wheel_radius} sep={self.wheel_sep} '
            f'cpr={self.counts_per_rev} m/count={self.m_per_count:.6g} '
            f'publish_tf={self.publish_tf}')

    def _left_cb(self, msg: Int64):
        self.pending_left = self.inv_l * int(msg.data)

    def _right_cb(self, msg: Int64):
        self.pending_right = self.inv_r * int(msg.data)

    def _motor_cb(self, msg: Int64):
        self.pending_motor = self.inv_m * int(msg.data)

    def _tick(self):
        now = self.get_clock().now()

        if self.single:
            # One motor-shaft encoder: we can only measure signed distance ds
            # (hence vx). There is NO wheel heading, so dyaw stays 0 and we do
            # not dead-reckon x/y — the EKF integrates position along the IMU
            # heading instead.
            if self.pending_motor is None:
                return
            if self.last_motor is None:
                self.last_motor = self.pending_motor
                self.last_time = now
                self._publish(now, 0.0, 0.0)
                return
            ds = (self.pending_motor - self.last_motor) * self.m_per_count
            self.last_motor = self.pending_motor
            dyaw = 0.0
        else:
            # Two wheel encoders: differential-drive. ds/dyaw both observed.
            if self.pending_left is None or self.pending_right is None:
                return  # wait until both encoders have reported at least once
            if self.last_left is None:
                # First real sample: latch baselines, emit a zero-motion odom so
                # the EKF gets a seed without a spurious jump.
                self.last_left = self.pending_left
                self.last_right = self.pending_right
                self.last_time = now
                self._publish(now, 0.0, 0.0)
                return
            dl = (self.pending_left - self.last_left) * self.m_per_count
            dr = (self.pending_right - self.last_right) * self.m_per_count
            self.last_left = self.pending_left
            self.last_right = self.pending_right
            ds = 0.5 * (dl + dr)
            dyaw = (dr - dl) / self.wheel_sep

        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0.0:
            return

        # Dead-reckon pose only in dual mode, where we have a wheel heading.
        # Midpoint (2nd-order) integration reduces heading error on arcs — the
        # reverse parking arcs are precisely where this pays off. ds is signed
        # (reverse -> negative), which is exactly why quadrature is mandatory.
        if not self.single:
            yaw_mid = self.yaw + 0.5 * dyaw
            self.x += ds * math.cos(yaw_mid)
            self.y += ds * math.sin(yaw_mid)
            self.yaw = math.atan2(math.sin(self.yaw + dyaw), math.cos(self.yaw + dyaw))

        vx = ds / dt
        wz = dyaw / dt
        self._publish(now, vx, wz)

    def _publish(self, stamp_time, vx, wz):
        stamp = stamp_time.to_msg()
        qz = math.sin(self.yaw * 0.5)
        qw = math.cos(self.yaw * 0.5)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = vx
        odom.twist.twist.angular.z = wz

        # Diagonal covariances on the fields the EKF actually fuses
        # (x=0, y=7, yaw=35 in the 6x6 row-major pose block;
        #  vx=0, wz=35 in the twist block). Big values elsewhere = "don't trust".
        # In single-encoder mode ONLY vx is real, so everything else is BIG and
        # the EKF leans entirely on the IMU for heading (and integrates position
        # from vx along that heading).
        BIG = 1e6
        pc = [BIG] * 36
        tc = [BIG] * 36
        tc[0] = self.var_vx          # vx (trusted in both modes)
        if not self.single:
            pc[0] = self.var_xy      # x
            pc[7] = self.var_xy      # y
            pc[35] = self.var_yaw    # yaw
            tc[35] = self.var_wz     # wz
        odom.pose.covariance = pc
        odom.twist.covariance = tc

        self.odom_pub.publish(odom)

        if self.tf_bc is not None:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self.tf_bc.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = WheelOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
