import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class VideoGetter(Node):
    def __init__(self):
        super().__init__('VideoGetter')
        # create a subscriber
        self.subscriber = self.create_subscription(Image, '/camera/image_raw', self.process_data, 10)
        self.bridge = CvBridge() # converting ros images to opencv data
        self.out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1280, 720))

    def process_data(self, data):
        # perform conversions
        frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        # self.get_logger().info("write a frame")
        # write the frame to a video
        self.out.write(frame)
        cv2.imshow('output', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = VideoGetter()
    rclpy.spin(image_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()