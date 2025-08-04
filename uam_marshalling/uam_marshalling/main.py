############################################################################################
#  PROJECT  : Aircraft Marshalling Recognition and UAV Guidance                            #
#                                                                                          #                                      
#  AUTHOR   : Jeonghun Lee                                                                 #
#  INSTITUTE: Cheongju University                                                          #
#                                                                                          #
#  DESCRIPTION:                                                                            #
#      This module integrates YOLOv8-based object detection and                            #
#      MediaPipe Pose Landmarks to recognize aircraft marshalling.                         #
#      Recognized marshalling are translated into control                                  #
#      commands, which are then used to guide unmanned aerial vehicles                     #
#      (UAVs) in simulation environments using ROS2/Gazebo.                                #
#                                                                                          #
#  MAIN FEATURES:                                                                          #
#      - Real-time detection of marshalling using YOLOv8 amd MediaPipe Pose Landmark       #
#      - Result integration for guidance commmand generationn                              #
#      - ROS2 Humble-compatible control interface for UAV simulation(PX4-SITL)             #
#                                                                                          #
#  REQUIREMENTS:                                                                           # 
#      - Python 3.8                                                                        #
#      - ROS2 Humble                                                                       #
#      - OpenCV                                                                            #
#      - Ultralytics YOLOv8                                                                #
#      - MediaPipe                                                                         #
#      - Gazebo-Classic                                                                    #
#                                                                                          #
#  LICENSE:                                                                                #
#      For academic research use. Copyright © 2025 Jeonghun Lee, BSD-3-Clause              #
#                                                                                          #
############################################################################################

import rclpy, cv2, time
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from pose.detectPose import detectPose
from pose.classifyPose import classifyPose
from pose.mpose import pose_frame
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleOdometry
from px4_msgs.msg import VehicleLocalPosition


bridge = CvBridge()
cv_image = np.empty(shape=[0])
classify = classifyPose()
color = (0, 255, 0)


class UamMarshalling(Node):

    def __init__(self):
        super().__init__('UamMarshalling')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )

        self.offboard_control_mode_publisher_ = self.create_publisher(
            OffboardControlMode,
            "/fmu/in/offboard_control_mode",
            qos_profile)
        self.trajectory_setpoint_publisher_ = self.create_publisher(
            TrajectorySetpoint,
            "/fmu/in/trajectory_setpoint",
            qos_profile)
        self.vehicle_command_publisher_ = self.create_publisher(
            VehicleCommand,
            "/fmu/in/vehicle_command",
            qos_profile)

        self.vehicle_odometry_subscriber_ = self.create_subscription(
            VehicleOdometry,
            "/fmu/out/vehicle_odometry",
            self.odometry_callback,
            qos_profile)

        self.vehicle_local_position_subscriber_ = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self.get_vehicle_position,
            qos_profile)

        self.vehicle_status_subscriber_ = self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status",
            self.get_vehicle_status,
            qos_profile)

        self.frame_subscriber_ = self.create_subscription(
            Image, '/camera/camera/color/image_raw',
            self.frame_callback,
            1)

        timer_period = 0.1  # 100 milliseconds
        self.timer_ = self.create_timer(timer_period, self.timer_callback)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.dt = timer_period
        self.offboard_setpoint_counter_ = 0

        # Odometry position (m)
        self.current_position_x = 0.0
        self.current_position_y = 0.0
        self.current_position_z= 0.0

        # Vehicle position (m)
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.state = "INIT"


    def get_vehicle_status(self, msg):
        # TODO: handle NED->ENU transformation
        # print("NAV_STATUS: ", msg.nav_state)
        # print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state

    def timer_callback(self):
        pass

    # Arm the vehicle
    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info("Arm command send")

    def disarm(self):
        print('Disarm command sent')
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        self.publish_vehicle_command(msg)

    def takeoff(self):
        print("Takeoff command sent")
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF, 6.0)
        self.get_logger().info("Takeoff command send")

    def land(self):
        print('Land command sent')
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND, 6.0)
        self.get_logger().info("Land command send")

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = True
        msg.acceleration = True
        msg.attitude = True
        msg.body_rate = True
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.offboard_control_mode_publisher_.publish(msg)

    def engage_offBoard_mode(self):
        print('Offboard mode command sent')
        msg = VehicleCommand()
        msg.param1 = 1.0
        msg.param2 = 6.0
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.publish_vehicle_command(msg)

    def publish_trajectory_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position[0] = self.current_position_x + 2.0
        msg.position[1] = self.current_position_y + 0.0
        msg.position[2] = self.current_position_z -2.0
        msg.yaw = -3.14 # [-PI:PI]
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.trajectory_setpoint_publisher_.publish(msg)

    def publish_trajectory_setpoint1(self):
        msg = TrajectorySetpoint()
        msg.position[0] = self.current_position_x + 38.0
        msg.position[1] = self.current_position_y + 40.0
        msg.position[2] = self.current_position_z + 0.0
        msg.yaw = 0.8424 # [-PI:PI]
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.trajectory_setpoint_publisher_.publish(msg)

        if self.current_position_x > 1050 and self.current_position_y > 1050:
            self.publish_hover_setpoint()
            msg.yaw = 0.8424 # [-PI:PI]


    def publish_right_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position[0] = self.current_position_x + 0.0
        msg.position[1] = self.current_position_y + 2.0
        msg.position[2] = self.current_position_z + 0.0
        msg.yaw = 0.0  # [-PI:PI]
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.trajectory_setpoint_publisher_.publish(msg)

    def publish_hover_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position[0] = self.current_position_x + 0.0
        msg.position[1] = self.current_position_y + 0.0
        msg.position[2] = self.current_position_z + 0.0
        msg.yaw = 0.0  # [-PI:PI]
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.trajectory_setpoint_publisher_.publish(msg)

    def publish_up_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position[0] = self.current_position_x + 0.0
        msg.position[1] = self.current_position_y + 0.0
        msg.position[2] = self.current_position_z -2.0
        msg.yaw = 0.0  # [-PI:PI]
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.trajectory_setpoint_publisher_.publish(msg)

    def publish_left_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position[0] = self.current_position_x + 0.0
        msg.position[1] = self.current_position_y - 2.0
        msg.position[2] = self.current_position_z + 0.0
        msg.yaw = 0.0  # [-PI:PI]
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.trajectory_setpoint_publisher_.publish(msg)

    def publish_down_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position[0] = self.current_position_x + 0.0
        msg.position[1] = self.current_position_y + 0.0
        msg.position[2] = self.current_position_z + 2.0
        msg.yaw = 0.0  # [-PI:PI]
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.trajectory_setpoint_publisher_.publish(msg)

    def publish_ahead_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position[0] = self.current_position_x + 2.0
        msg.position[1] = self.current_position_y + 0.0
        msg.position[2] = self.current_position_z + 0.0
        msg.yaw = 0.0  # [-PI:PI]
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.trajectory_setpoint_publisher_.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command  # command ID
        msg.target_system = 1  # system which should execute the command
        msg.target_component = 1  # component which should execute the command, 0 for all components
        msg.source_system = 1  # system sending the command
        msg.source_component = 1  # component sending the command
        msg.from_external = True
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.vehicle_command_publisher_.publish(msg)

    def get_vehicle_position(self, msg):
        #self.offboard_setpoint_counter_ = msg.timestamp
        self.x = msg.x
        self.y = msg.y
        self.z = msg.z

    def odometry_callback(self, msg):
        #self.offboard_setpoint_counter_ = msg.timestamp
        self.current_position_x = msg.position[0]
        self.current_position_y = msg.position[1]
        self.current_position_z = msg.position[2]

    def frame_callback(self, data):

        if self.offboard_setpoint_counter_ == 10:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1., 6.)
            # Arm the vehicle
            self.arm()
        if (self.offboard_setpoint_counter_ <10e+10**10):
            self.publish_offboard_control_mode()
        # if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
        #     self.publish_trajectory_setpoint()


        global cv_image

        cv_image = bridge.imgmsg_to_cv2(data)

        cv_image.size == (640*480*3)

        frame = cv_image

        # frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Check FPS
        start_t = time.time()

        # Detect pose in frame
        frame, marshal = detectPose(frame, pose_frame, display=False)

        frame_count = 0
        total_fps = 0

        if marshal:

            classify.pose_landmarks(marshal, frame, display=False)

            ## Engage rotor recognition and result
            # if classify.arm_angle(angle = True): # and classify.arm_angle(angle = True) classify.arm_detect(detect=True) and
            #     cv2.putText(frame, 'ARM', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
            #     self.get_logger().info('ARM')
            #     self.offboard_setpoint_counter_ +=1

            ## Move back recognition and result
            # elif classify.back_angle(angle=True): # and classify.back_angle(angle=True)  classify.back_detect(detect=True) and
            #     cv2.putText(frame, 'MOVE BACK', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
            #     self.get_logger().info('MOVE BACK')
            #     if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            #         self.publish_hover_setpoint()
            #     self.offboard_setpoint_counter_ +=1

            if classify.ahead_angle(angle=True) and classify.ahead_detect(detect=True): # c and
                cv2.putText(frame, 'STRAIGHT AHEAD', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                self.get_logger().info('STRAIGHT AHEAD')
                self.publish_ahead_setpoint()
                self.offboard_setpoint_counter_ +=1

            # Hover recognition and result
            elif classify.hover_angle(angle=True) and classify.hover_detect(detect=True): #  and
                cv2.putText(frame, 'HOVER', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                self.get_logger().info('HOVER')
                self.publish_hover_setpoint()
                self.offboard_setpoint_counter_ +=1

            # Land recognition and result
            elif classify.land_angle(angle=True) and classify.land_detect(detect=True): # and
                cv2.putText(frame, 'LAND', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                self.get_logger().info('LAND')
                self.land()

            # Move Downward recognition and result
            elif classify.mv_down_angle(angle=True) and classify.mv_down_detect(detect=True):  # and
                cv2.putText(frame, 'MOVE DOWNWARD', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                self.get_logger().info('MOVE DOWNWARD')
                self.publish_down_setpoint()
                self.offboard_setpoint_counter_ += 1

            # Move Left recognition and result
            elif classify.mv_left_angle(angle=True) and classify.mv_left_detect(detect=True): # and
                cv2.putText(frame, 'MOVE LEFT', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                self.get_logger().info('MOVE LEFT')
                self.publish_left_setpoint()
                self.offboard_setpoint_counter_ += 1

            # Move Right recognition and result
            elif classify.mv_right_angle(angle=True) and classify.mv_right_detect(detect=True): # and
                cv2.putText(frame, 'MOVE RIGHT', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                self.get_logger().info('MOVE RIGHT')
                self.publish_right_setpoint()
                self.offboard_setpoint_counter_ += 1

            # Move Upward recognition and result
            elif classify.mv_up_angle(angle=True) and classify.mv_up_detect(detect=True): # : #
                cv2.putText(frame, 'MOVE UPWARD', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                self.get_logger().info('MOVE UPWARD')
                self.publish_up_setpoint()
                self.offboard_setpoint_counter_ += 1

            else:
                cv2.putText(frame, 'CHECKING POSE', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                self.get_logger().info('CHECKING POSE', once = True)

        elif self.current_position_z <= -13:
            self.publish_trajectory_setpoint1()
            self.offboard_setpoint_counter_ += 1

        # Check FPS and Show FPS
        terminate_t = time.time()
        fps = 1/(terminate_t - start_t)

        total_fps += fps
        frame_count += 1

        avg_fps = total_fps / frame_count

        str = "FPS: %0.1f" %avg_fps
        cv2.putText(frame, str, (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        # Display the frame.
        cv2.imshow('Marshalling Recognition', frame)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()

        self.offboard_setpoint_counter_ += 1

def main(args=None):
    rclpy.init(args=args)
    print("Starting marshalling recognition node...\n")
    node = UamMarshalling()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
