# project_node.py
# ROS2 Project - COMP3631
# Author: Sreeja Chowdary Tulluru (sc23sct)
#
# Description:
# This node implements an autonomous robot that:
# 1. Explores the environment using Nav2 waypoint navigation
# 2. Detects red, green and blue coloured boxes using computer vision
# 3. Approaches and stops ~1 metre from the blue box once all colours are seen


import math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """
    Convert a yaw angle (radians) to a Quaternion for use in ROS poses.
    Only z and w components are needed for 2D rotation.
    """
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class ProjectNode(Node):
    """
    Main ROS2 node for the COMP3631 project.

    State machine:
        EXPLORE       -- Navigate through waypoints using Nav2
        APPROACH_BLUE -- Drive toward blue box using camera feedback
        DONE          -- Task complete, robot stopped near blue box
    """
    
    def __init__(self):
        super().__init__('project_node')

        # Camera
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Robot movement
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Nav2 action client
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # Colour detection flags
        self.red_seen   = False
        self.green_seen = False
        self.blue_seen  = False

        # Current frame detection info (None if colour not visible this frame)
        self.red_info   = None
        self.green_info = None
        self.blue_info  = None

        self.sensitivity = 10

        # State machine
        self.state = 'EXPLORE'

        # Nav2 goal tracking
        self.goal_handle  = None
        self.goal_active  = False
        self.goal_pending = False
        self.current_waypoint_index = 0
        self.waypoints_completed    = 0

        # Log counter
        self.log_counter = 0

        # Exploration waypoints
        # Heuristic poses selected from the map using RViz Publish Point tool.
        # Waypoint 1: exits the starting compartment, robot sees red and green
        # Waypoint 2: moves robot near blue box area
        # Nav2 plans a collision-free path to each waypoint automatically.
        self.waypoints = [
            (-2.11, -6.11, 0.0),   # waypoint 1 - exit bottom right compartment
            (-5.84, -9.01, 0.0),   # waypoint 2 - near blue box
        ]

        # Control loop timer
        # Runs every 0.5 seconds to check state and issue commands
        self.timer = self.create_timer(0.5, self.control_loop)
        self.get_logger().info('=' * 50)
        self.get_logger().info('ProjectNode initialised. Starting exploration...')
        self.get_logger().info('=' * 50)

    # -------------------------------------------------------
    # IMAGE PROCESSING
    # -------------------------------------------------------
    def image_callback(self, msg: Image):
        """
        Called every time a new camera frame is received.
        Converts the ROS image to OpenCV HSV format, applies colour masks
        for red, green and blue, detects the largest contour for each colour,
        draws bounding boxes, and updates detection flags.
        """
        
        # Convert ROS Image to OpenCV BGR format
        image   = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        display = image.copy()
        hsv     = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Blue: hue centred at 110 in OpenCV HSV (0-180 range)
        lower_blue  = np.array([110 - self.sensitivity, 100, 100])
        upper_blue  = np.array([110 + self.sensitivity, 255, 255])

        # Green: hue centred at 60 in OpenCV HSV
        lower_green = np.array([60 - self.sensitivity, 100, 100])
        upper_green = np.array([60 + self.sensitivity, 255, 255])

        # Red: wraps around HSV spectrum so needs two ranges (0-10 and 170-180)
        lower_red1  = np.array([0,   100, 100])
        upper_red1  = np.array([10,  255, 255])
        lower_red2  = np.array([170, 100, 100])
        upper_red2  = np.array([180, 255, 255])

        # Create binary masks for each colour
        blue_mask  = cv2.inRange(hsv, lower_blue,  upper_blue)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        red_mask   = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )

        self.blue_info  = self.draw_largest_contour(
            display, blue_mask,  (255,   0,   0), 'BLUE')
        self.green_info = self.draw_largest_contour(
            display, green_mask, (0,   255,   0), 'GREEN')
        self.red_info   = self.draw_largest_contour(
            display, red_mask,   (0,     0, 255), 'RED')

        # Update persistent seen flags — once True, stays True
        if self.red_info   is not None: self.red_seen   = True
        if self.green_info is not None: self.green_seen = True
        if self.blue_info  is not None: self.blue_seen  = True

        # Display the annotated camera feed
        cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
        cv2.imshow('Camera', display)
        cv2.resizeWindow('Camera', 640, 480)
        cv2.waitKey(3)

    def draw_largest_contour(self, image, mask, colour, label):
        """
        Finds the largest contour in the binary mask, draws a bounding box
        and centre of mass marker on the display image.

        Returns a dict with contour centre (cx, cy) and area,
        or None if no significant contour found (area < 250 pixels).
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        c    = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 250:
            return None

        # Draw bounding rectangle around detected colour
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(
            image,
            f'{label} area={int(area)}',
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, colour, 2
        )

        # Calculate and draw centre of mass (used for steering toward blue)
        M = cv2.moments(c)
        if M['m00'] == 0:
            return None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(image, (cx, cy), 5, colour, -1)

        return {'cx': cx, 'cy': cy, 'area': area}

    # -------------------------------------------------------
    # NAV2 NAVIGATION
    # -------------------------------------------------------
    def send_waypoint_goal(self, x: float, y: float, yaw: float):
        """
        Sends a navigation goal to Nav2.
        Nav2 plans a collision-free path using the provided map
        and drives the robot to the specified (x, y, yaw) pose.
        """
        # Wait for Nav2 action server to be available
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('Nav2 not ready, retrying...')
            return

        # Build the goal message with map frame pose
        goal_msg = NavigateToPose.Goal()
        pose     = PoseStamped()
        pose.header.frame_id  = 'map'
        pose.header.stamp     = self.get_clock().now().to_msg()
        pose.pose.position.x  = x
        pose.pose.position.y  = y
        pose.pose.orientation = yaw_to_quaternion(yaw)
        goal_msg.pose = pose

        # Mark as active before sending to prevent duplicate goals
        self.goal_active  = True
        self.goal_pending = True

        self.get_logger().info(
            f'>>> Sending waypoint {self.current_waypoint_index}: '
            f'x={x:.2f}, y={y:.2f}'
        )
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Called when Nav2 accepts or rejects the goal."""
        goal_handle       = future.result()
        self.goal_pending = False

        if not goal_handle.accepted:
            self.get_logger().warn(
                f'>>> Waypoint {self.current_waypoint_index} REJECTED'
            )
            self.goal_active = False
            return

        self.get_logger().info(
            f'>>> Waypoint {self.current_waypoint_index} ACCEPTED'
        )
        self.goal_handle = goal_handle
        
        # Register callback for when navigation finishes
        result_future    = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        """Called when Nav2 finishes navigating to the waypoint."""
        status            = future.result().status
        self.goal_active  = False
        self.goal_handle  = None
        self.goal_pending = False

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(
                f'>>> Waypoint {self.current_waypoint_index} REACHED'
            )
        else:
            self.get_logger().warn(
                f'>>> Waypoint {self.current_waypoint_index} FAILED, skipping'
            )

        # Move to next waypoint (loops back if all visited)
        self.current_waypoint_index = (
            self.current_waypoint_index + 1
        ) % len(self.waypoints)

        self.waypoints_completed += 1
        self.get_logger().info(
            f'>>> Progress: {self.waypoints_completed} waypoints done | '
            f'R={self.red_seen} G={self.green_seen} B={self.blue_seen}'
        )

    def cancel_nav_goal(self):
        """Cancels the currently active Nav2 navigation goal."""
        if self.goal_handle is not None and self.goal_active:
            self.get_logger().info('>>> Cancelling Nav2 goal...')
            future = self.goal_handle.cancel_goal_async()
            future.add_done_callback(self.cancel_done_callback)

    def cancel_done_callback(self, future):
        """Called when Nav2 confirms the goal has been cancelled."""
        self.get_logger().info('>>> Nav2 goal cancelled')
        self.goal_active  = False
        self.goal_pending = False
        self.goal_handle  = None

    # -------------------------------------------------------
    # HIGH-LEVEL CONTROL LOOP
    # -------------------------------------------------------
    def control_loop(self):
        """
        Main control loop running every 0.5 seconds.

        State transitions:
            EXPLORE       -> APPROACH_BLUE  (when all colours seen + blue visible
                                             + current waypoint finished)
            APPROACH_BLUE -> DONE           (when robot is ~1m from blue box)
            DONE          -> timer cancelled (robot stops permanently)
        """

        # If task is complete, cancel the timer and stop the robot
        if self.state == 'DONE':
            self.timer.cancel()
            self.stop_robot()
            return

        # Print status every 5 seconds
        self.log_counter += 1
        if self.log_counter % 10 == 0:
            self.get_logger().info(
                f'[STATE={self.state}] '
                f'wp={self.current_waypoint_index} | '
                f'R={self.red_seen} G={self.green_seen} B={self.blue_seen} | '
                f'blue_visible={self.blue_info is not None}'
            )

        # Switch to APPROACH_BLUE only when:
        # 1. Not currently executing a Nav2 goal (waypoint finished)
        # 2. All 3 colours have been seen at least once
        # 3. Blue box is currently visible in camera
        if (self.state == 'EXPLORE'
                and not self.goal_active        
                and not self.goal_pending     
                and self.blue_info  is not None
                and self.red_seen
                and self.green_seen
                and self.blue_seen):
            self.get_logger().info('-' * 50)
            self.get_logger().info(
                '*** All 3 colours seen + blue visible -> APPROACH_BLUE ***'
            )
            self.get_logger().info('-' * 50)
            self.state = 'APPROACH_BLUE'
            self.cancel_nav_goal()
            return

        if self.state == 'EXPLORE':
            self.explore_logic()
        elif self.state == 'APPROACH_BLUE':
            self.approach_blue_logic()

    def explore_logic(self):
        """
        Sends the next waypoint to Nav2 when the robot is idle.
        Loops through waypoints continuously until all colours are seen.
        """
        if not self.goal_active and not self.goal_pending:
            x, y, yaw = self.waypoints[self.current_waypoint_index]
            self.send_waypoint_goal(x, y, yaw)

    def approach_blue_logic(self):
        """
        Camera-based control to drive toward the blue box and stop ~1m away.

        Uses contour centre of mass to steer left/right (error from image centre)
        and contour area as a proxy for distance (larger area = closer).

        Thresholds:
            error > 40 pixels  -> turn to centre blue in frame
            area  < 50000      -> drive forward toward blue
            area  >= 50000     -> stopped ~1m away, task complete
        """
        twist = Twist()

        # Blue not currently visible — rotate slowly to search
        if self.blue_info is None:
            twist.angular.z = 0.3
            twist.linear.x  = 0.0
            self.cmd_pub.publish(twist)
            return

        cx             = self.blue_info['cx']
        area           = self.blue_info['area']
        image_center_x = 320
        error          = cx - image_center_x   # positive = blue is to the right

        if abs(error) > 40:
            # Blue not centred — turn toward it before moving forward
            twist.linear.x  = 0.0
            twist.angular.z = -0.003 * error

        elif area < 50000:
            # Drive forward toward blue
            # Stops when area reaches 50000 = roughly 1 metre away
            # If robot stops too far: increase this number
            # If robot gets too close: decrease this number
            twist.linear.x  = 0.10
            twist.angular.z = -0.002 * error
            self.get_logger().info(
                f'Approaching blue | area={int(area)} | error={int(error)}'
            )

        else:
            # Area >= 50000 means ~1 metre away — stop
            twist.linear.x  = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            self.get_logger().info('=' * 50)
            self.get_logger().info(
                '*** TASK COMPLETE — stopped ~1m from blue box ***'
            )
            self.get_logger().info(f'Red seen   : {self.red_seen}')
            self.get_logger().info(f'Green seen : {self.green_seen}')
            self.get_logger().info(f'Blue seen  : {self.blue_seen}')
            self.get_logger().info('=' * 50)
            self.state = 'DONE'
            return

        self.cmd_pub.publish(twist)

    def stop_robot(self):
        """Publishes zero velocity to ensure the robot is stationary."""
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = ProjectNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()