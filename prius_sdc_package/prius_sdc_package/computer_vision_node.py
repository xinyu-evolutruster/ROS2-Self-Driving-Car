import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from .drive_bot import Car

class VideoGetter(Node):
    def __init__(self):
        super().__init__('VideoGetter')
        # create a subscriber
        self.subscriber = self.create_subscription(Image, '/camera/image_raw', self.process_data, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 40)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.send_cmd_vel)

        self.velocity = Twist()
        self.bridge = CvBridge() # converting ros images to opencv data
        
        self.car = Car()

    def send_cmd_vel(self):
        self.publisher.publish(self.velocity)

    def process_data(self, data):
        # perform conversions
        frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        angle, speed, img = self.car.drive_car(frame)

        self.velocity.angular.z = angle
        self.velocity.linear.x = speed

        cv2.imshow("Frame", img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = VideoGetter()
    rclpy.spin(image_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()