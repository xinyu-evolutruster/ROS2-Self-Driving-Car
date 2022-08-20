import cv2
import numpy as np
from .lane_utils import approx_dist_between_centers, ret_largest_contour

from ...config import config

def estimate_midlane(midlane_patches, max_dist):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    midlane_patches = cv2.morphologyEx(midlane_patches, cv2.MORPH_DILATE, kernel)

    # 1. keep a midlane_draw for displaying shrtest connectivity later on
    midlane_connectivity_bgr = cv2.cvtColor(midlane_patches, cv2.COLOR_GRAY2BGR)

    # 2. extract the contours that define each object
    contours = cv2.findContours(midlane_patches, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    
    # 3. keep only those contours that are not lines
    min_area = 0.5
    legit_contours = []
    for _, contour in enumerate(contours):
        contour_area = cv2.contourArea(contour)
        if contour_area > min_area:
            legit_contours.append(contour)
    contours = legit_contours
   
    # 4. connect each contours with its closest
    #                     &
    #    disconnecting any that may be farther than x distance
    contour_index_best_match = dict()
    for index, contour in enumerate(contours):
        min_dist = config.infinity
        best_index_cmp = 0
        best_centroid = None
        best_centroid_cmp = None
        for index_cmp in range(len(contours)):
            if index != index_cmp:
                contour_cmp = contours[index_cmp]
                dist, centroid, centroid_cmp = approx_dist_between_centers(contour, contour_cmp)
                if dist < min_dist:
                    min_dist = dist
                    best_index_cmp = index_cmp
                    best_centroid = centroid
                    best_centroid_cmp = centroid_cmp
        if len(contour_index_best_match) == 0:
            if min_dist > max_dist:
                continue
            else:
                contour_index_best_match[index] = best_index_cmp
                cv2.line(midlane_connectivity_bgr, best_centroid, best_centroid_cmp, (0, 255, 0), 2)
        else:
            if best_centroid != None:
                for idx, match in contour_index_best_match.items():
                    if (idx == index and match == best_index_cmp) or (match == index and idx == best_index_cmp):
                        continue
                contour_index_best_match[index] = best_index_cmp
                cv2.line(midlane_connectivity_bgr, best_centroid, best_centroid_cmp, (0, 255, 0), 2)
    midlane_connectivity = cv2.cvtColor(midlane_connectivity_bgr, cv2.COLOR_BGR2GRAY)

    # 5. get estimated midlane by returning the largest contour
    estimated_midlane, largest_found = ret_largest_contour(midlane_connectivity)

    # 6. return estimated midlane if found otherwise send original
    if largest_found:
        return estimated_midlane
    else:
        return midlane_patches