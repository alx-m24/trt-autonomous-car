import math

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32


class TerrainNode(Node):
    def __init__(self):
        super().__init__('terrain_node')

        # ──────────────────────────────────────────────────────────────────
        # Subscribers
        # ──────────────────────────────────────────────────────────────────

        self.create_subscription(
            Imu,
            '/imu',
            self.imu_cb,
            20
        )

        # ──────────────────────────────────────────────────────────────────
        # Publishers
        # ──────────────────────────────────────────────────────────────────

        self.pitch_pub = self.create_publisher(
            Float32,
            '/terrain_pitch_deg',
            10
        )

        self.shock_pub = self.create_publisher(
            Float32,
            '/terrain_shock',
            10
        )

        self.speed_scale_pub = self.create_publisher(
            Float32,
            '/terrain_speed_scale',
            10
        )

        self.steering_scale_pub = self.create_publisher(
            Float32,
            '/terrain_steering_scale',
            10
        )

        # ──────────────────────────────────────────────────────────────────
        # Parameters
        # ──────────────────────────────────────────────────────────────────

        # Separate EMA filters:
        # Pitch = sustained low-frequency event
        # Shock = transient high-frequency event

        self.declare_parameter('pitch_alpha', 0.08)
        self.declare_parameter('shock_alpha', 0.18)

        # Scaling gains

        self.declare_parameter('pitch_speed_gain', 0.025)
        self.declare_parameter('shock_speed_gain', 0.12)
        self.declare_parameter('shock_steering_gain', 0.18)

        # Ignore tiny road/sensor noise

        self.declare_parameter('pitch_deadband_deg', 2.0)

        # Safety clamps

        self.declare_parameter('max_pitch_deg', 20.0)
        self.declare_parameter('max_shock', 8.0)

        # Minimum scaling

        self.declare_parameter('min_speed_scale', 0.55)
        self.declare_parameter('min_steering_scale', 0.65)

        # Gravity compensation

        self.declare_parameter('gravity', 9.81)

        # ──────────────────────────────────────────────────────────────────
        # Load parameters
        # ──────────────────────────────────────────────────────────────────

        self.load_parameters()

        self.add_on_set_parameters_callback(self.param_cb)

        # ──────────────────────────────────────────────────────────────────
        # Filter state
        # ──────────────────────────────────────────────────────────────────

        self.filtered_pitch_deg = 0.0
        self.filtered_shock = 0.0

        self.get_logger().info('TerrainNode started.')
        self.get_logger().info(
            'Using hybrid IMU-response architecture.'
        )

    # ──────────────────────────────────────────────────────────────────────
    # Parameter loading
    # ──────────────────────────────────────────────────────────────────────

    def load_parameters(self):
        self.pitch_alpha = float(
            self.get_parameter('pitch_alpha').value
        )

        self.shock_alpha = float(
            self.get_parameter('shock_alpha').value
        )

        self.pitch_speed_gain = float(
            self.get_parameter('pitch_speed_gain').value
        )

        self.shock_speed_gain = float(
            self.get_parameter('shock_speed_gain').value
        )

        self.shock_steering_gain = float(
            self.get_parameter('shock_steering_gain').value
        )

        self.pitch_deadband_deg = float(
            self.get_parameter('pitch_deadband_deg').value
        )

        self.max_pitch_deg = float(
            self.get_parameter('max_pitch_deg').value
        )

        self.max_shock = float(
            self.get_parameter('max_shock').value
        )

        self.min_speed_scale = float(
            self.get_parameter('min_speed_scale').value
        )

        self.min_steering_scale = float(
            self.get_parameter('min_steering_scale').value
        )

        self.gravity = float(
            self.get_parameter('gravity').value
        )

    # ──────────────────────────────────────────────────────────────────────
    # Runtime parameter updates
    # ──────────────────────────────────────────────────────────────────────

    def param_cb(self, params):
        for p in params:
            self.get_logger().info(
                f'Updated parameter: {p.name} = {p.value}'
            )

        self.load_parameters()

        return SetParametersResult(successful=True)

    # ──────────────────────────────────────────────────────────────────────
    # IMU callback
    # ──────────────────────────────────────────────────────────────────────

    def imu_cb(self, msg: Imu):

        # ──────────────────────────────────────────────────────────────
        # Read IMU
        # ──────────────────────────────────────────────────────────────

        ax = float(msg.linear_acceleration.x)
        az = float(msg.linear_acceleration.z)

        # ──────────────────────────────────────────────────────────────
        # Pitch estimation
        #
        # NOTE:
        # This is lightweight estimation from gravity direction.
        # It is NOT full attitude estimation.
        #
        # Under hard acceleration/braking:
        # accel forces can distort pitch estimate.
        #
        # Acceptable for adaptive speed scaling.
        # ──────────────────────────────────────────────────────────────

        raw_pitch_rad = math.atan2(
            ax,
            max(1e-6, abs(az))
        )

        raw_pitch_deg = math.degrees(raw_pitch_rad)

        # Separate low-frequency EMA filter

        self.filtered_pitch_deg = (
            (1.0 - self.pitch_alpha) * self.filtered_pitch_deg +
            self.pitch_alpha * raw_pitch_deg
        )

        # Clamp unrealistic spikes

        self.filtered_pitch_deg = max(
            -self.max_pitch_deg,
            min(self.max_pitch_deg, self.filtered_pitch_deg)
        )

        # Deadband

        effective_pitch = abs(self.filtered_pitch_deg)

        if effective_pitch < self.pitch_deadband_deg:
            effective_pitch = 0.0

        # ──────────────────────────────────────────────────────────────
        # Shock estimation
        #
        # Generic terrain disturbance estimate.
        #
        # NOT semantic classification.
        # ──────────────────────────────────────────────────────────────

        accel_mag = math.sqrt(
            ax * ax +
            az * az
        )

        raw_shock = abs(accel_mag - self.gravity)

        # Separate high-frequency EMA filter

        self.filtered_shock = (
            (1.0 - self.shock_alpha) * self.filtered_shock +
            self.shock_alpha * raw_shock
        )

        # Clamp unrealistic spikes

        self.filtered_shock = min(
            self.filtered_shock,
            self.max_shock
        )

        # ──────────────────────────────────────────────────────────────
        # Dynamic response scaling
        # ──────────────────────────────────────────────────────────────

        hill_factor = (
            1.0 -
            effective_pitch * self.pitch_speed_gain
        )

        shock_speed_factor = (
            1.0 -
            self.filtered_shock * self.shock_speed_gain
        )

        shock_steering_factor = (
            1.0 -
            self.filtered_shock * self.shock_steering_gain
        )

        speed_scale = (
            hill_factor *
            shock_speed_factor
        )

        steering_scale = shock_steering_factor

        # Safety clamps

        speed_scale = max(
            self.min_speed_scale,
            min(1.0, speed_scale)
        )

        steering_scale = max(
            self.min_steering_scale,
            min(1.0, steering_scale)
        )

        # ──────────────────────────────────────────────────────────────
        # Publish outputs
        # ──────────────────────────────────────────────────────────────

        pitch_msg = Float32()
        pitch_msg.data = float(self.filtered_pitch_deg)
        self.pitch_pub.publish(pitch_msg)

        shock_msg = Float32()
        shock_msg.data = float(self.filtered_shock)
        self.shock_pub.publish(shock_msg)

        speed_msg = Float32()
        speed_msg.data = float(speed_scale)
        self.speed_scale_pub.publish(speed_msg)

        steering_msg = Float32()
        steering_msg.data = float(steering_scale)
        self.steering_scale_pub.publish(steering_msg)


def main(args=None):
    rclpy.init(args=args)

    node = TerrainNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()