#!/usr/bin/env python3
"""
One-time perspective calibration tool.
Adjust trackbars to calibrate the bird's eye view, then copy the final values
into perception_node.py as hardcoded constants.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class PerspectiveCalibrator(Node):
    def __init__(self):
        super().__init__('perspective_calibrator')
        self.bridge = CvBridge()
        
        self.sub = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_cb, 10)
        
        # Initial perspective points (same as perception_node defaults)
        self.tl_x, self.tl_y = 100, 240
        self.bl_x, self.bl_y = 20, 480
        self.tr_x, self.tr_y = 540, 240
        self.br_x, self.br_y = 620, 480
        
        self.bird_w = 320
        self.bird_h = 480
        
        # White color detection range
        self.lower_white = np.array([0,   0,   200])
        self.upper_white = np.array([255, 50,  255])
        
        # Create window and trackbars
        cv2.namedWindow('Calibrator')
        
        cv2.createTrackbar('TL_X', 'Calibrator', self.tl_x, 640, self.on_trackbar)
        cv2.createTrackbar('TL_Y', 'Calibrator', self.tl_y, 480, self.on_trackbar)
        cv2.createTrackbar('BL_X', 'Calibrator', self.bl_x, 640, self.on_trackbar)
        cv2.createTrackbar('BL_Y', 'Calibrator', self.bl_y, 480, self.on_trackbar)
        
        cv2.createTrackbar('TR_X', 'Calibrator', self.tr_x, 640, self.on_trackbar)
        cv2.createTrackbar('TR_Y', 'Calibrator', self.tr_y, 480, self.on_trackbar)
        cv2.createTrackbar('BR_X', 'Calibrator', self.br_x, 640, self.on_trackbar)
        cv2.createTrackbar('BR_Y', 'Calibrator', self.br_y, 480, self.on_trackbar)
        
        self.get_logger().info('Calibrator started. Adjust trackbars to calibrate bird\'s eye view.')
        self.get_logger().info('Press "s" to save and print calibration values.')
        self.get_logger().info('Press "q" to quit.')

    def on_trackbar(self, val):
        """Trackbar callback — update perspective points."""
        self.tl_x = cv2.getTrackbarPos('TL_X', 'Calibrator')
        self.tl_y = cv2.getTrackbarPos('TL_Y', 'Calibrator')
        self.bl_x = cv2.getTrackbarPos('BL_X', 'Calibrator')
        self.bl_y = cv2.getTrackbarPos('BL_Y', 'Calibrator')
        
        self.tr_x = cv2.getTrackbarPos('TR_X', 'Calibrator')
        self.tr_y = cv2.getTrackbarPos('TR_Y', 'Calibrator')
        self.br_x = cv2.getTrackbarPos('BR_X', 'Calibrator')
        self.br_y = cv2.getTrackbarPos('BR_Y', 'Calibrator')

    def get_bird_eye(self, frame):
        """Apply perspective transformation."""
        pts1 = np.float32([
            [self.tl_x, self.tl_y],
            [self.bl_x, self.bl_y],
            [self.tr_x, self.tr_y],
            [self.br_x, self.br_y]
        ])
        pts2 = np.float32([[0, 0], [0, self.bird_h], [self.bird_w, 0], [self.bird_w, self.bird_h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(frame, matrix, (self.bird_w, self.bird_h))

    def image_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        frame = cv2.resize(frame, (640, 480))
        
        # Draw trapezoid on original frame to show what will be captured
        frame_with_trap = frame.copy()
        pts = np.array([
            [self.tl_x, self.tl_y],
            [self.bl_x, self.bl_y],
            [self.br_x, self.br_y],
            [self.tr_x, self.tr_y]
        ], np.int32)
        cv2.polylines(frame_with_trap, [pts], True, (0, 255, 0), 2)
        
        # Add point labels
        cv2.circle(frame_with_trap, (self.tl_x, self.tl_y), 5, (255, 0, 0), -1)
        cv2.putText(frame_with_trap, 'TL', (self.tl_x-15, self.tl_y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.circle(frame_with_trap, (self.bl_x, self.bl_y), 5, (0, 255, 0), -1)
        cv2.putText(frame_with_trap, 'BL', (self.bl_x-15, self.bl_y+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.circle(frame_with_trap, (self.tr_x, self.tr_y), 5, (0, 0, 255), -1)
        cv2.putText(frame_with_trap, 'TR', (self.tr_x+5, self.tr_y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.circle(frame_with_trap, (self.br_x, self.br_y), 5, (255, 255, 0), -1)
        cv2.putText(frame_with_trap, 'BR', (self.br_x+5, self.br_y+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Get bird's eye view
        bird = self.get_bird_eye(frame)
        
        # Create lane mask
        hsv = cv2.cvtColor(bird, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # Resize all to same height for side-by-side display
        bird_resized = cv2.resize(bird, (320, 240))
        mask_resized = cv2.resize(mask, (320, 240))
        mask_3ch = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        
        frame_resized = cv2.resize(frame_with_trap, (320, 240))
        
        # Create display: camera (left) + bird's eye (right) on top row
        #               original (left) + mask (right) on bottom row
        top_row = np.hstack([frame_resized, bird_resized])
        bottom_row = np.hstack([frame_resized, mask_3ch])
        display = np.vstack([top_row, bottom_row])
        
        cv2.imshow('Calibrator', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('Quitting calibrator.')
            rclpy.shutdown()
        elif key == ord('s'):
            self.print_calibration()

    def print_calibration(self):
        """Print current calibration values to copy into perception_node.py"""
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE - Copy these values into perception_node.py:")
        print("="*60)
        print(f"self.tl = ({self.tl_x}, {self.tl_y})   # Top-left (far, left)")
        print(f"self.bl = ({self.bl_x}, {self.bl_y})   # Bottom-left (near, left)")
        print(f"self.tr = ({self.tr_x}, {self.tr_y})   # Top-right (far, right)")
        print(f"self.br = ({self.br_x}, {self.br_y})   # Bottom-right (near, right)")
        print("="*60 + "\n")


def main():
    rclpy.init()
    node = PerspectiveCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
