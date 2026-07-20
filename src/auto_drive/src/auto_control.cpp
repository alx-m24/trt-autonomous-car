#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include <cmath>
#include <algorithm>

class ForwardTargetController : public rclcpp::Node {
public:
    ForwardTargetController() : Node("forward_target_controller") {
        target_sub_ = this->create_subscription<visualization_msgs::msg::Marker>(
            "/forward_target_viz", 10,
            std::bind(&ForwardTargetController::targetCallback, this, std::placeholders::_1));

        cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    }

private:
    // Measured on this sim's gazebo_ros_planar_move setup: commanded angular.z
    // only realizes ~0.13x actual yaw rate (vs ~0.6x for linear velocity) —
    // see the URDF's wheel-friction comment. Compensate here rather than in
    // k_angular, so the gain itself stays physically meaningful.
    static constexpr double YAW_REALIZATION_FACTOR = 0.13;
    static constexpr double LINEAR_REALIZATION_FACTOR = 0.6;

    void targetCallback(const visualization_msgs::msg::Marker::SharedPtr msg) {
        bool valid = msg->color.g > 0.5f;

        geometry_msgs::msg::Twist cmd;
        if (!valid) {
            cmd_pub_->publish(cmd); // all-zero Twist — stop
            return;
        }

        double x = msg->pose.position.x;
        double y = msg->pose.position.y;
        double angle = std::atan2(y, x);
        double dist = std::sqrt(x*x + y*y);

        const double k_angular = 2.0;   // tune against REALIZED yaw, not commanded
        const double k_linear = 0.5;
        const double max_linear = 0.3;
        const double max_angular = 1.5;

        double desired_angular = std::clamp(k_angular * angle, -max_angular, max_angular);
        cmd.angular.z = desired_angular / YAW_REALIZATION_FACTOR;

        double turn_factor = std::max(0.0, 1.0 - std::abs(angle) / (M_PI / 2.0));
        double desired_linear = std::clamp(k_linear * dist * turn_factor, 0.0, max_linear);
        cmd.linear.x = desired_linear / LINEAR_REALIZATION_FACTOR;

        cmd_pub_->publish(cmd);
    }

    rclcpp::Subscription<visualization_msgs::msg::Marker>::SharedPtr target_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ForwardTargetController>());
    rclcpp::shutdown();
    return 0;
}
