#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include "occupancy_grid.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"

// Camera Projection and TF2 libraries
#include <image_geometry/pinhole_camera_model.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
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
    const vec2f gridSize = vec2f{2.0f, 2.0f};
    const vec2f cellSize = vec2f{0.025f, 0.025f};

    // Grid mapping structures
    Grid grid_;
    bool camera_info_received_ = false;
    bool calibrated_cam = false;
    cv::Mat K{};
    cv::Mat D{};
    cv::Mat H{};
    uint32_t image_width_, image_height_;
    cv::Mat latest_white_mask_;
    cv::Mat map1_, map2_;   // add as members

    vec2f CamerePos = vec2f{ -0.13f, -0.13f }; // from config
    float cameraPitch = 20.0f; // about y-axis

    float ground_x_near = 0.3f, ground_x_far = 2.3f;
    float ground_y_left = -1.0f, ground_y_right = 1.0f;
    int warp_w = 400, warp_h = 400;

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
        marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "/occupancy_grid_viz", 10);

         // 1. Initialize TF2 Buffer and Listener
         tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
         tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // 2. Subscribe to CameraInfo
        info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/front_camera/camera_info", 10,
            std::bind(&OccupancyGridNode::cameraInfoCallback, this, std::placeholders::_1));

        // 3. Subscribe to Image
        camera_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/front_camera/image_raw", 10,
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
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        camera_info_received_ = true;
        RCLCPP_INFO(this->get_logger(), "Camera Info received");
        info_sub_.reset(); // unsubscribind from callback

        cv::Mat msg_K(3, 3, CV_64F, const_cast<double*>(msg->k.data()));
        K = msg_K.clone();
        cv::Mat msg_D = cv::Mat(msg->d.size(), 1, CV_64F, const_cast<double*>(msg->d.data())).clone();
        D = msg_D.clone();

        image_width_ = msg->width;
        image_height_ = msg->height;

        calibrate_cam();
    }

    void calibrate_cam() {
        RCLCPP_INFO(this->get_logger(), "Calibrating camera...");

        cv::initUndistortRectifyMap(
            K, D, cv::Mat(),           // no rectification rotation needed (monocular)
            K,                         // new camera matrix -- reuse K, no black-border cropping
            cv::Size(image_width_, image_height_),  // see note below on getting this
            CV_16SC2,                  // fast fixed-point maps
            map1_, map2_);

        geometry_msgs::msg::TransformStamped tf_base_to_optical = tf_buffer_->lookupTransform("camera_optical_frame", "base_link", tf2::TimePointZero);

std::vector<cv::Point3d> ground_pts_base = {
    {0.3,  1.0, 0.0},  // was -1.0 -> now near-left
    {0.3, -1.0, 0.0},  // was  1.0 -> now near-right
    {2.3, -1.0, 0.0},  // was  1.0 -> now far-right
    {2.3,  1.0, 0.0}   // was -1.0 -> now far-left
};

        std::vector<cv::Point2f> src_pixels;
        for (auto& gp : ground_pts_base) {
            geometry_msgs::msg::PointStamped pt_base, pt_optical;
            pt_base.header.frame_id = "base_link";
            pt_base.point.x = gp.x;
            pt_base.point.y = gp.y;
            pt_base.point.z = gp.z;
        
            tf2::doTransform(pt_base, pt_optical, tf_base_to_optical);
        
            cv::Mat p3 = (cv::Mat_<double>(3,1) << pt_optical.point.x, pt_optical.point.y, pt_optical.point.z);
            cv::Mat uv = K * p3;
            double u = uv.at<double>(0) / uv.at<double>(2);
            double v = uv.at<double>(1) / uv.at<double>(2);
        
            src_pixels.push_back(cv::Point2f(u, v));
        }

        std::vector<cv::Point2f> dst_pixels = {
            {0,   400},  // near-left  -> bottom-left
            {400, 400},  // near-right -> bottom-right
            {400, 0},    // far-right  -> top-right
            {0,   0}     // far-left   -> top-left
        };

        H = cv::getPerspectiveTransform(src_pixels, dst_pixels);


        calibrated_cam = true;
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        if (!camera_info_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for camera_info...");
            return;
        }
        if (!calibrated_cam) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for calibration...");
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

    cv::Point2i cellToWarpedPixel(const vec2f& pos) {
        if (pos.x < ground_x_near || pos.x > ground_x_far ||
            pos.y < ground_y_left || pos.y > ground_y_right) {
            return {-1, -1};
        }
    
        // u now decreases as pos.y increases, matching the corrected
        // ground_pts_base <-> dst_pixels pairing
        float u = (ground_y_right - pos.y) / (ground_y_right - ground_y_left) * warp_w;
        float v = (ground_x_far - pos.x) / (ground_x_far - ground_x_near) * warp_h;
    
        return { static_cast<int>(u), static_cast<int>(v) };
    }
    
    void processImage(const cv::Mat& frame, const std::string& camera_frame, const rclcpp::Time& stamp) {
        cv::Mat undistorted;
        cv::remap(frame, undistorted, map1_, map2_, cv::INTER_LINEAR);

        cv::Mat birds_eye;
        cv::warpPerspective(undistorted, birds_eye, H, cv::Size(400, 400));

        cv::Mat hsv, white_mask;
        cv::cvtColor(birds_eye, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(0, 0, 200), cv::Scalar(180, 30, 255), white_mask);

        float px_per_m_x = warp_w / (ground_y_right - ground_y_left);  // 400/2 = 200
        float px_per_m_y = warp_h / (ground_x_far - ground_x_near);    // 400/2 = 200
        
        int half_x = static_cast<int>((cellSize.x * px_per_m_x) / 2.0f);
        int half_y = static_cast<int>((cellSize.y * px_per_m_y) / 2.0f);

        latest_white_mask_ = white_mask; // cache for use in the grid loop below
        grid_.forEach([&](Cell& cell, const vec2u&) {
            vec2f corner_pos = cell.getPosition();
            vec2f center_pos = corner_pos + cellSize * 0.5f; // shift to cell center
        
            cv::Point2i px = cellToWarpedPixel(center_pos);
            if (px.x < 0) return; // outside camera's visible ground region, skip
        
            // sample a small neighborhood rather than a single pixel, since a 0.05m
            // cell covers ~10px at this warp's scale (400px / 2m = 200px/m)
            cv::Rect roi(std::max(0, px.x - half_x), std::max(0, px.y - half_y),
                std::min(2*half_x, latest_white_mask_.cols - px.x + half_x),
                std::min(2*half_y, latest_white_mask_.rows - px.y + half_y));
        
            if (roi.width <= 0 || roi.height <= 0) return;
        
            cv::Mat patch = latest_white_mask_(roi);
            int white_count = cv::countNonZero(patch);
            float white_ratio = static_cast<float>(white_count) / (patch.rows * patch.cols);
        
            if (white_ratio > 0.3f) { // tune threshold
                cell.value = CellValue::OCCUPIED; // you'll need to add this to CellValue enum
            }
        });

        // temporary debug — draw a dot at every cell's projected pixel location
        cv::Mat debug_view;
        cv::cvtColor(white_mask, debug_view, cv::COLOR_GRAY2BGR);
        
        grid_.forEach([&](Cell& cell, const vec2u&) {
            vec2f center_pos = cell.getPosition()+ cellSize * 0.5f;
            cv::Point2i px = cellToWarpedPixel(center_pos);
            if (px.x < 0) {
                cv::circle(debug_view, px, 2, cv::Scalar(255, 0, 0), -1);
            }
            else {
                cv::circle(debug_view, px, 2, cell.value == CellValue::FREE ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), -1);
            }
        });
        
        cv::imshow("birds_eye_debug", debug_view);
        cv::waitKey(1);
    }

    void lidarCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        last_scan_ = msg;
    }

    void publishGrid() {
        // Process the latest LiDAR scan before drawing
        if (last_scan_) {
            updateGridFromScan(last_scan_);
        }

        grid_.get(vec2u(1, 1)).value = CellValue::PLAUSIBLE_PARKING;
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
        // TODO 
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OccupancyGridNode>());
    rclcpp::shutdown();
    return 0;
}
