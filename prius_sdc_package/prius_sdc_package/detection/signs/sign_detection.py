import os
import cv2
import numpy as np

from tensorflow.keras.models import load_model
import tensorflow as tf

save_dataset = True
iter_num = 0
save_num = 0

model_loaded = False
model = 0
sign_classes = ["speed_sign_30", "speed_sign_60", "speed_sign_90", "stop", "left_turn", "no_sign"]
sign_classes_id = {
    "speed_sign_30": 0,
    "speed_sign_60": 1,
    "speed_sign_90": 2,
    "stop": 3,
    "left_turn": 4,
    "no_sign": 5
}

def image_for_keras(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (30, 30))
    image = np.expand_dims(image, axis=0)
    return image

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

                # saving dataset
                sign = sign_classes[np.argmax(model(image_for_keras(localized_sign)))]

                if sign != "no_sign":
                    # display class
                    cv2.putText(frame_draw, sign, (end_point[0] - 90, start_point[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                    # display confirmed sign in green circle
                    cv2.circle(frame_draw, center, radius, (0,255,0), 2)

                if save_dataset:
                    class_id = "/{}".format(sign_classes_id[sign])
                    global iter_num, save_num
                    iter_num += 1
                    # save every 5 image
                    if iter % 5 == 0:
                        save_num += 1
                        save_dir = os.path.abspath("prius_sdc_package/prius_sdc_package/data/live_dataset")
                        if not os.path.exists(save_dir):
                            os.mkdir(save_dir)
                        img_dir = save_dir + class_id
                        img_name = os.path.join(img_dir, str(save_num), ".png")
                        cv2.imwrite(img_name, localized_sign)

            except Exception as e:
                print(e)
        cv2.imshow("signs localized", frame_draw)

def detect_signs(frame, frame_draw):
    global model_loaded
    if not model_loaded:
        print(tf.__version__)
        print("*********** LOADING MODEL ************")
        # load CNN model
        global model
        model = load_model(os.path.join(os.getcwd(), "prius_sdc_package/prius_sdc_package", "data", "saved_model_5_sign.h5"))
        # summarize model
        model.summary()
        model_loaded = True

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sign_detection_and_tracking(gray, frame, frame_draw)
