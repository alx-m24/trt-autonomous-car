#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

class SensorListener : public rclcpp::Node {
public:
    SensorListener() : Node("sensor_listener") {

        RCLCPP_INFO(this->get_logger(), "=== Initializing Live Sensor Listener Node ===");

        // 1. Odometry — reliable is fine here (default nav topics are reliable)
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&SensorListener::odomCallback, this, std::placeholders::_1));

        // 2. IMU — Gazebo IMU plugin also typically uses sensor-data QoS
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu", rclcpp::SensorDataQoS(),
            std::bind(&SensorListener::imuCallback, this, std::placeholders::_1));

        // 3. Camera — sensor-data QoS (best-effort), this is the fix
        camera_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/front_camera/image_raw", rclcpp::SensorDataQoS(),
            std::bind(&SensorListener::cameraCallback, this, std::placeholders::_1));


        // 4. LiDAR — was missing entirely
        lidar_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", rclcpp::SensorDataQoS(),
            std::bind(&SensorListener::lidarCallback, this, std::placeholders::_1));
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) const {
        RCLCPP_INFO(this->get_logger(),
            "[ODOM] Speed Lin: %.2f m/s | Ang Z: %.2f rad/s",
            msg->twist.twist.linear.x,
            msg->twist.twist.angular.z);
    }

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) const {
        RCLCPP_INFO(this->get_logger(),
            "[IMU]  Acc Z (Shock): %.2f m/s² | Gyro Y (Pitch Rate): %.2f rad/s",
            msg->linear_acceleration.z,
            msg->angular_velocity.y);
    }

    void cameraCallback(const sensor_msgs::msg::Image::SharedPtr msg) const {
        try {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
            RCLCPP_INFO(this->get_logger(),
                "[CAM]  Frame received! Size: %dx%d | Encoding: %s",
                cv_ptr->image.cols,
                cv_ptr->image.rows,
                msg->encoding.c_str());
            cv::imshow("Camera Feed", cv_ptr->image);
            cv::waitKey(1);
        }
        catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge frame decoding failure: %s", e.what());
        }
    }

    void lidarCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) const {
        RCLCPP_INFO(this->get_logger(),
            "[LIDAR] Ranges: %zu | Front: %.2fm | Left: %.2fm | Right: %.2fm",
            msg->ranges.size(),
            msg->ranges[msg->ranges.size()/2],   // roughly "front" depending on angle convention
            msg->ranges[0],
            msg->ranges.back());;
    }

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr lidar_sub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SensorListener>());
    rclcpp::shutdown();
    return 0;
}
