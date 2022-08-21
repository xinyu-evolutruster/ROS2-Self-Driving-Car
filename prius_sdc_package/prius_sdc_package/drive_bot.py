import cv2
from numpy import interp
from .detection.lanes.lane_detection import detect_lanes
from .detection.signs.sign_detection import detect_signs

from .config import config

class Control():
    def __init__(self):
        self._angle = 0.0
        self._speed = 80.0
        # cruise control variable
        self._prev_mode = "detection"
        self._increase_tire_speed_in_turns = False

        self._class_to_speed = {
            "speed_sign_30": 30,
            "speed_sign_60": 60,
            "speed_sign_90": 90,
            "stop": 0
        }

    def follow_lane(self, max_allowed_distance, distance, curvature, mode, tracked_class):
        # self._speed = 80.0
        
        # cruise control speed adjusted to match road speed limit
        if tracked_class != 0 and self._prev_mode == "tracking" and mode == "detection":
            if tracked_class == "left_turn":
                pass
            else:
                self._speed = self._class_to_speed[tracked_class]
                if tracked_class == "stop":
                    print("Stopping the car!!")
        self._prev_mode = mode

        max_turn_angle = 90
        max_turn_angle_neg = -90
        required_turn_angle = 0

        if abs(distance) > max_allowed_distance:
            if distance > max_allowed_distance:
                required_turn_angle = max_turn_angle + curvature
            else:
                required_turn_angle = max_turn_angle_neg + curvature
        else:
            car_offset = interp(distance, [-max_allowed_distance, max_allowed_distance], [max_turn_angle_neg, max_turn_angle])
            required_turn_angle = car_offset + curvature
        
        # handle overflow
        if required_turn_angle > max_turn_angle:
            required_turn_angle = max_turn_angle
        elif required_turn_angle < max_turn_angle_neg:
            required_turn_angle = max_turn_angle_neg
        
        # handle max car turnablity
        self._angle = interp(required_turn_angle, [max_turn_angle_neg, max_turn_angle], [-45, 45])
        if self._increase_tire_speed_in_turns and tracked_class != "left_turn":
            if self._angle > 30:
                car_speed_turn = interp(self._angle, [30, 45], [80, 100])
                self._speed = car_speed_turn
            elif self. _angle < -30:
                car_speed_turn = interp(self._angle, [-45, -30], [100, 80])
                self._speed = car_speed_turn

    def drive(self, current_state):
        [distance, curvature, image, mode, tracked_class] = current_state
        if distance != config.infinity and curvature != config.infinity:
            self.follow_lane(int(image.shape[1] / 4), distance, curvature, mode, tracked_class)
        else:
            self._speed = 0.0  # stop the car
        
        # interpolating the angle and speed from real world to motor world
        angle_motor = interp(self._angle, [-45, 45], [0.5, -0.5])
        if self._speed != 0:
            speed_motor = interp(self._speed, [30.0, 90.0], [1.0, 2.0])
        else:
            speed_motor = 0.0
        
        return angle_motor, speed_motor

class Car():

    def __init__(self):
        self._control = Control()

    def display_states(self, frame_display, angle_car, current_speed):
        # translate ros car control range ==> real world angle and speed
        angle_car = interp(angle_car, [-0.5, 0.5], [45, -45])
        if current_speed != 0.0:
            current_speed = interp(current_speed, [1, 2], [30, 90])
        
        if angle_car < -10:
            direction_str = "[ Left ]"
            color_direction = (120, 0, 255)
        elif angle_car > 10:
            direction_str = "[ Right ]"
            color_direction = (120, 0, 255)
        else:
            direction_str = "[ Straight ]"
            color_direction = (0, 255, 0)
        
        if current_speed > 0:
            direction_str = "Moving --> " + direction_str
        else:
            color_direction = (0, 0, 255)
        
        cv2.putText(frame_display, str(direction_str), (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.4, color_direction, 1)

        angle_speed_str = "[ Angle, Speed ] = [ " + str(int(angle_car)) + "deg, " + str(int(current_speed)) + "mph ]"
        cv2.putText(frame_display, str(angle_speed_str), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1)

    def drive_car(self, frame):
        img = frame[0:640, 238:1042]

        # resizing to minimize computation time
        img = cv2.resize(img, (320, 240))

        img_orig = img.copy()

        distance, curvature, out_img = detect_lanes(img)
        mode, tracked_class = detect_signs(img_orig, out_img)

        current_state = [distance, curvature, out_img, mode, tracked_class]
        
        angle_motor, speed_motor = self._control.drive(current_state)

        self.display_states(out_img, angle_motor, speed_motor)

        return float(angle_motor), float(speed_motor), out_img
