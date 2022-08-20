import cv2
import numpy as np

def sign_detection_and_tracking(gray, frame, frame_draw):
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=250, param2=30, minRadius=10, maxRadius=100)

    # checking if any circular regions were localized
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # looping over each localized circle and extract its center and radius
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            # extracting ROI from localized circle
            try:
                start_point = (center[0] - radius, center[1] - radius)
                end_point = (center[0] + radius, center[1] + radius)
                print("start point: ", start_point, "end point: ", end_point)
                localized_sign = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]
                # indicating localized potential sign on frame and also displaying separately
                cv2.circle(frame_draw, center, i[2], (0,255,0), 1)
                cv2.circle(frame_draw, center, 2, (0, 0, 255), 3)
                cv2.imshow("ROI", localized_sign)
            except Exception as e:
                print(e)
        cv2.imshow("signss localized", frame_draw)

def detect_signs(frame, frame_draw):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sign_detection_and_tracking(gray, frame, frame_draw)
