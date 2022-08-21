from tracemalloc import start
import cv2
import numpy as np

from ..utils import calculate_distance

class Tracker:
    # class variables
    max_allowed_dist = 100
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    # lukas-kanade parameters
    lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def __init__(self):
        print("Initialized object of signTracking class")
        
        # state variables
        self.mode = "detection"
        self.tracked_class = 0

        # proximity variables
        self.known_centers = []
        self.known_centers_confidence = []
        self.known_centers_classes_confidence = []
        # self.sign_determined = False

        # init variables
        self.old_gray = 0
        self.p0 = np.array([])

        # draw variables
        self.mask = 0
        self.color = np.random.randint(0, 255, (100, 3))
    
    def match_cur_center_to_known(self, center):
        match_found = False
        match_idx = 0
        for i in range(len(self.known_centers)):
            if calculate_distance(center, self.known_centers[i]) < self.max_allowed_dist:
                match_found = True
                match_idx = i
                return match_found, match_idx
        # if no match found, return default values
        return match_found, match_idx
        
    def init_tracker(self, sign, gray, frame_draw, start_point, end_point):
        self.mode = "tracking"
        self.tracked_class = sign
        self.old_gray = gray
        self.mask = np.zeros_like(frame_draw)

        sign_mask = np.zeros_like(gray)
        sign_mask[start_point[1]:end_point[1], start_point[0]:end_point[0]] = 255

        self.p0 = cv2.goodFeaturesToTrack(gray, mask=sign_mask, **self.feature_params)

    def track(self, gray, frame_draw):
        # use lukas-kanade method
        p1, state, _ = cv2.calcOpticalFlowPyrLK(self.old_gray, gray, self.p0, None, **self.lk_params)

        # if no flow, look for new points
        if p1 is None or len(p1[state == 1]) < 3:
            self.mode = "detection"
            self.mask = np.zeros_like(frame_draw)
            self.reset()
        else:
            # select good points
            good_new = p1[state == 1]
            good_old = self.p0[state == 1]
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                new_point_x, new_point_y = (int(x) for x in new.ravel())
                old_point_x, old_point_y = (int(x) for x in old.ravel())
                new_point = (new_point_x, new_point_y)
                old_point = (old_point_x, old_point_y)
                self.mask = cv2.line(self.mask, new_point, old_point, self.color[i].tolist(), 2)
                frame_draw = cv2.circle(frame_draw, new_point, 5, self.color[i].tolist(), -1)
            frame_draw_ = frame_draw + self.mask  # display the image with the flow lines
            
            # test
            # cv2.imshow("frame_draw_", frame_draw_)

            np.copyto(frame_draw, frame_draw_)
            self.old_gray = gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)

    def reset(self):
        self.known_centers = []
        self.known_centers_confidence = []
        self.known_centers_classes_confidence = []
        self.old_gray = 0
        self.p0 = np.array([])
