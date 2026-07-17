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
    const vec2f gridSize = vec2f{2.5f, 2.5f};
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
    cv::Mat valid_mask_;
    cv::Mat map1_, map2_;   // add as members

    vec2f CamerePos = vec2f{ -0.13f, -0.13f }; // from config
    float cameraPitch = 20.0f; // about y-axis

    float ground_x_near = 1.0f, ground_x_far = 2.3f;
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

        cv::namedWindow("mask_tuning");
        cv::createTrackbar("V_min", "mask_tuning", nullptr, 255);
        cv::createTrackbar("S_max", "mask_tuning", nullptr, 255);
        cv::setTrackbarPos("V_min", "mask_tuning", 200);
        cv::setTrackbarPos("S_max", "mask_tuning", 30);

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

    double findHorizonRow(const tf2::Matrix3x3& rot_optical_to_base, double cx) {
        double prev_z = 0.0;
        bool have_prev = false;
    
        for (int v = 0; v < static_cast<int>(image_height_); ++v) {
            double x = (cx - K.at<double>(0, 2)) / K.at<double>(0, 0);
            double y = (v  - K.at<double>(1, 2)) / K.at<double>(1, 1);
            tf2::Vector3 ray_optical(x, y, 1.0);
            tf2::Vector3 ray_base = rot_optical_to_base * ray_optical;
    
            if (have_prev && ((prev_z < 0) != (ray_base.z() < 0))) {
                // Sign change between v-1 and v: horizon crossing is between them.
                // Linear interpolate for a sub-pixel-accurate row.
                double frac = prev_z / (prev_z - ray_base.z());
                return (v - 1) + frac;
            }
            prev_z = ray_base.z();
            have_prev = true;
        }
        return -1.0; // no sign change found in-frame
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

        tf2::Transform tf2_base_to_optical;
        tf2::fromMsg(tf_base_to_optical.transform, tf2_base_to_optical);
        tf2::Transform tf2_optical_to_base = tf2_base_to_optical.inverse();

        // Camera origin expressed in base_link (just the inverse transform's translation)
        tf2::Vector3 cam_origin_base = tf2_optical_to_base.getOrigin();
        
        double cx = K.at<double>(0, 2);
        double v_horizon = findHorizonRow(tf2_optical_to_base.getBasis(), cx);
        
        double top_row;
        if (v_horizon < 0) {
            // Entire image sees ground (steep pitch) or entire image sees sky (shallow pitch).
            // Fall back to full image height, but log it so it's obvious which case you're in.
            RCLCPP_WARN(this->get_logger(), "No horizon found in-frame — using full image height");
            top_row = 0.0;
        } else {
            // Add a small margin so you're not calibrating literally at the horizon line,
            // where ground pixels become extremely compressed/inaccurate anyway.
            top_row = v_horizon + 5.0;
            RCLCPP_INFO(this->get_logger(), "Computed horizon row: %.2f (using %.2f as top edge)", v_horizon, top_row);
        }
        
        std::vector<cv::Point2f> image_corners = {
            {0.f, static_cast<float>(image_height_ - 1)},
            {static_cast<float>(image_width_ - 1), static_cast<float>(image_height_ - 1)},
            {static_cast<float>(image_width_ - 1), static_cast<float>(top_row)},
            {0.f, static_cast<float>(top_row)}
        };

        
        std::vector<cv::Point3d> ground_pts_base; // will be filled in with real ground intersections
        std::vector<cv::Point2f> src_pixels;      // corresponding image pixels (just the corners)
        
        for (auto& corner : image_corners) {
            // Back-project pixel -> ray direction in optical frame (pinhole inverse)
            double x = (corner.x - K.at<double>(0, 2)) / K.at<double>(0, 0);
            double y = (corner.y - K.at<double>(1, 2)) / K.at<double>(1, 1);
            tf2::Vector3 ray_optical(x, y, 1.0);
        
            // Rotate ray direction into base_link (rotation only, no translation)
            tf2::Vector3 ray_base = tf2_optical_to_base.getBasis() * ray_optical;
        
            // Intersect with ground plane z = 0 in base_link:
            // cam_origin_base.z + t * ray_base.z == 0
            if (std::abs(ray_base.z()) < 1e-6) {
                RCLCPP_WARN(this->get_logger(), "Ray parallel to ground plane, skipping corner");
                continue; // ray points at horizon, never hits ground — skip (or clamp to far distance)
            }
            double t = -cam_origin_base.z() / ray_base.z();
        
            if (t < 0) {
                RCLCPP_WARN(this->get_logger(), "Ground intersection behind camera, skipping corner");
                continue; // intersection is behind the camera, not physically meaningful
            }
        
            tf2::Vector3 ground_pt = cam_origin_base + t * ray_base;
        
            ground_pts_base.push_back({ground_pt.x(), ground_pt.y(), 0.0});
            src_pixels.push_back(corner); // src pixel IS the image corner itself now
        }
        
        if (ground_pts_base.size() != 4) {
            RCLCPP_ERROR(this->get_logger(), "Could not compute all 4 ground corners — check camera pitch/mount");
            return; // don't calibrate with incomplete data
        }
        
        // Log the computed real-world footprint so you can sanity check it
        for (size_t i = 0; i < ground_pts_base.size(); ++i) {
            RCLCPP_INFO(this->get_logger(), "Ground corner %zu: x=%.3f y=%.3f",
                i, ground_pts_base[i].x, ground_pts_base[i].y);
        }
        
        // Derive metric bounds directly from what the camera actually sees,
        // instead of hardcoding ground_x_near/far/left/right
        ground_x_near = std::min({ground_pts_base[0].x, ground_pts_base[1].x, ground_pts_base[2].x, ground_pts_base[3].x});
        ground_x_far  = std::max({ground_pts_base[0].x, ground_pts_base[1].x, ground_pts_base[2].x, ground_pts_base[3].x});
        ground_y_left  = std::min({ground_pts_base[0].y, ground_pts_base[1].y, ground_pts_base[2].y, ground_pts_base[3].y});
        ground_y_right = std::max({ground_pts_base[0].y, ground_pts_base[1].y, ground_pts_base[2].y, ground_pts_base[3].y});

        // Clamp to the region actually useful for the grid, even though the camera sees further
        ground_x_far  = std::min(ground_x_far, gridSize.x);
        ground_y_left  = std::max(ground_y_left, -1.1f);
        ground_y_right = std::min(ground_y_right, 1.1f);

        RCLCPP_INFO(this->get_logger(), "Bounds: x[%.3f, %.3f] y[%.3f, %.3f] | grid x[0, %.3f]",
            ground_x_near, ground_x_far, ground_y_left, ground_y_right, gridSize.x);
        
        std::vector<cv::Point2f> dst_pixels = {
            {0.f, static_cast<float>(warp_h)},               // bottom-left  -> near-left
            {static_cast<float>(warp_w), static_cast<float>(warp_h)}, // bottom-right -> near-right
            {static_cast<float>(warp_w), 0.f},               // top-right    -> far-right
            {0.f, 0.f}                                        // top-left     -> far-left
        };
        
        H = cv::getPerspectiveTransform(src_pixels, dst_pixels);

        cv::Mat full_white(image_height_, image_width_, CV_8UC1, cv::Scalar(255));
        cv::warpPerspective(full_white, valid_mask_, H, cv::Size(warp_w, warp_h));

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

        cv::Point2i px = cv::Point2i(u, v);
        if (px.x < 0 || px.x >= warp_w || px.y < 0 || px.y >= warp_h) return {-1,-1};
        if (valid_mask_.at<uchar>(px.y, px.x) == 0) return {-1,-1}; // not actually visible
    
        return { static_cast<int>(u), static_cast<int>(v) };
    }
    
    void processImage(const cv::Mat& frame, const std::string& camera_frame, const rclcpp::Time& stamp) {
        cv::Mat undistorted;
        cv::remap(frame, undistorted, map1_, map2_, cv::INTER_LINEAR);

        cv::Mat birds_eye;
        cv::warpPerspective(undistorted, birds_eye, H, cv::Size(400, 400));

        cv::Mat hsv, white_mask;
        cv::cvtColor(birds_eye, hsv, cv::COLOR_BGR2HSV);
        int v_min = cv::getTrackbarPos("V_min", "mask_tuning");
        int s_max = cv::getTrackbarPos("S_max", "mask_tuning");
        cv::inRange(hsv, cv::Scalar(0, 0, v_min), cv::Scalar(180, s_max, 255), white_mask);

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

        // draw a dot at every cell's projected pixel location
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
