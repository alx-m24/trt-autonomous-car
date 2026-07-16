#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include "occupancy_grid.hpp"

// Camera Projection and TF2 libraries
#include <image_geometry/pinhole_camera_model.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

static vec2f computeDefaultOrigin(const vec2f& gridSize, const vec2f& cellSize) {
    return {
        cellSize.x / 2.0f,                      // X starts at 0.0
        -(gridSize.y - cellSize.y) / 2.0f       // Y is perfectly centered
    };
}

class OccupancyGridNode : public rclcpp::Node {
private:
    const vec2f gridSize = vec2f{1.0f, 1.0f};
    const vec2f cellSize = vec2f{0.05f, 0.05f};

    // Grid mapping structures
    Grid grid_;
    image_geometry::PinholeCameraModel cam_model_;
    bool camera_info_received_ = false;

    // TF2 listeners
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Subscriptions & Publishers
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr lidar_sub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // Scan memory cache
    sensor_msgs::msg::LaserScan::SharedPtr last_scan_;

public:
    OccupancyGridNode()
        : Node("occupancy_grid_node"), 
          grid_(gridSize, cellSize, computeDefaultOrigin(gridSize, cellSize))
    {
        // 1. Initialize TF2 Buffer and Listener
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "/occupancy_grid_viz", 10);

        // 2. Subscribe to CameraInfo
        info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/camera_info", 10,
            [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
                cam_model_.fromCameraInfo(msg);
                camera_info_received_ = true;
            });

        // 3. Subscribe to Image
        camera_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10,
            std::bind(&OccupancyGridNode::imageCallback, this, std::placeholders::_1));

        // 4. Subscribe to Lidar
        lidar_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", rclcpp::SensorDataQoS(),
            std::bind(&OccupancyGridNode::lidarCallback, this, std::placeholders::_1));

        // 5. Grid processing & publish loop
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(200),
            std::bind(&OccupancyGridNode::publishGrid, this));

        RCLCPP_INFO(this->get_logger(), "Occupancy grid node with TF2 camera mapping started.");
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        if (!camera_info_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for camera_info...");
            return;
        }

        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            // Pass the image timestamp and frame ID so we can get highly accurate transforms
            processImage(cv_ptr->image, msg->header.frame_id, msg->header.stamp);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }
    
    void processImage(const cv::Mat& frame, const std::string& camera_frame, const rclcpp::Time& stamp) {
        // Look up the coordinate transform from the camera sensor frame to the floor base frame
        geometry_msgs::msg::TransformStamped tf_stamped;
        try {
            tf_stamped = tf_buffer_->lookupTransform(
                "base_footprint", camera_frame, stamp, rclcpp::Duration::from_seconds(0.05));
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                                 "TF lookup failed: %s", ex.what());
            return;
        }

        // Get camera position relative to the ground
        double tx = tf_stamped.transform.translation.x;
        double ty = tf_stamped.transform.translation.y;
        double tz = tf_stamped.transform.translation.z;

        // Convert camera rotation to rotation matrix
        tf2::Quaternion q(
            tf_stamped.transform.rotation.x,
            tf_stamped.transform.rotation.y,
            tf_stamped.transform.rotation.z,
            tf_stamped.transform.rotation.w
        );
        tf2::Matrix3x3 rotation_matrix(q);

        cv::Mat gray, binary;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, binary, 200, 255, cv::THRESH_BINARY);

        int step = 4; // Downsample pixels to keep simulation fast
        for (int v = 0; v < binary.rows; v += step) {
            for (int u = 0; u < binary.cols; u += step) {
                if (binary.at<uchar>(v, u) == 255) { 
                    
                    // Cast standard 3D ray out of pixel in camera frame
                    cv::Point3d ray = cam_model_.projectPixelTo3dRay(cv::Point2d(u, v));

                    // Transform the direction vector to base_footprint frame
                    tf2::Vector3 ray_cam(ray.x, ray.y, ray.z);
                    tf2::Vector3 ray_ground = rotation_matrix * ray_cam;

                    double vz = ray_ground.z();
                    
                    // Make sure ray points downwards (vz < 0) and avoid division by zero
                    if (vz < -1e-5) {
                        double t = -tz / vz;
                        float x = static_cast<float>(tx + t * ray_ground.x());
                        float y = static_cast<float>(ty + t * ray_ground.y());

                        // Map world coordinates to relative cell index
                        vec2i relative_idx = grid_.getRelativeIndex({x, y});

                        try {
                            grid_.getFromCenter(relative_idx).value = CellValue::OCCUPIED;
                        } catch (const std::out_of_range& e) {
                            // Point landed outside the bounds of our current grid size
                            continue;
                        }
                    }
                }
            }
        }
        publishGrid();
    }

    void lidarCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        last_scan_ = msg;
    }

    void publishGrid() {
        // Process the latest LiDAR scan before drawing
        if (last_scan_) {
            updateGridFromScan(last_scan_);
        }

        auto marker = grid_.toMarker("base_footprint", this->now());
        marker_pub_->publish(marker);
        
        clearGrid();
    }

    void clearGrid() {
        grid_.forEach([](Cell& cell, const vec2u&) {
            cell.value = CellValue::FREE;
        });
    }

    void updateGridFromScan(const sensor_msgs::msg::LaserScan::SharedPtr& scan) {
        for (size_t i = 0; i < scan->ranges.size(); ++i) {
            float range = scan->ranges[i];
            
            // Check for valid scan ranges
            if (range >= scan->range_min && range <= scan->range_max) {
                float angle = scan->angle_min + i * scan->angle_increment;
                
                // Polar to Cartesian conversion relative to the sensor origin (standard ROS coordinate frame)
                float x = range * std::cos(angle);
                float y = range * std::sin(angle);

                // Note: If your lidar sensor is offset from 'base_footprint', you may want
                // to include a quick translation offset here (e.g. adding lidar_offset_x to x)
                vec2i relative_idx = grid_.getRelativeIndex({x, y});
                try {
                    grid_.getFromCenter(relative_idx).value = CellValue::OCCUPIED;
                } catch (const std::out_of_range& e) {
                    continue;
                }
            }
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OccupancyGridNode>());
    rclcpp::shutdown();
    return 0;
}
