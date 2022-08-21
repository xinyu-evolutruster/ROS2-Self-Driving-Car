from turtle import pos, position
import cv2
import math
import numpy as np
from ..utils import calculate_distance
from ...config import config

# about opencv column and row organizations:
# https://stackoverflow.com/questions/25642532/opencv-pointx-y-represent-column-row-or-row-column

def bw_area_open(img, min_area):
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]
    # filter using contour area and remove small noise
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours_too_small = []
    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area:
            contours_too_small.append(contour)
    thresh = cv2.drawContours(thresh, contours_too_small, -1, 0, -1)

    return thresh

def find_extremas(img):
    positions = np.nonzero(img)
    if len(positions) != 0:
        top = positions[0].min()
        bottom = positions[0].max()
        return top, bottom
    else:
        return 0, 0

def ROI_extractor(img, start_point, end_point):
    # selecting only ROI from image
    ROI_mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.rectangle(ROI_mask, start_point, end_point, 255, thickness=-1)
    image_ROI = cv2.bitwise_and(img, ROI_mask)
    return image_ROI

def find_lowest_row(img):
    positions = np.nonzero(img)

    if len(positions) != 0:
        bottom = positions[0].max()
        return bottom
    else:
        return img.shape[0]

def extract_point(img, specified_row):
    point = (0, specified_row)
    specified_row_data = img[specified_row - 1, :]

    positions = np.nonzero(specified_row_data)
    if len(positions[0]) != 0:
        min_col = positions[0].min()
        point = (min_col, specified_row)
    return point

def ret_lowest_edge_points(gray):
    outer_points_list = []
    thresh = np.zeros(gray.shape, dtype=gray.dtype)
    lane_one_side = np.zeros(gray.shape, dtype=gray.dtype)
    lane_two_side = np.zeros(gray.shape, dtype=gray.dtype)

    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    # cv2.imshow("bin image", bin_img)

    # find the two contours for which you want to find the min distance between them
    contours = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    thresh = cv2.drawContours(thresh, contours, -1, (255,255,255), 1)

    top_row, bottom_row = find_extremas(thresh)

    contour_top_bottom_portion_cut = ROI_extractor(thresh, (0, top_row + 5), (thresh.shape[1], bottom_row - 5))
    contours2 = cv2.findContours(contour_top_bottom_portion_cut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

    lowest_row_first_lane = -1
    lowest_row_second_lane = -1

    euc_row = 0 # row for the points to be compared

    first_line = np.copy(lane_one_side)
    contours_tmp = []

    if len(contours2) > 1:
        for _, contour_tmp in enumerate(contours2):
            if contour_tmp.shape[0] > 50:
                contours_tmp.append(contour_tmp)
        contours2 = contours_tmp

    for index, contour in enumerate(contours2):
        lane_one_side = np.zeros(gray.shape, dtype=gray.dtype)
        lane_one_side = cv2.drawContours(lane_one_side, contours2, index, (255,255,255), 1)
        lane_two_side = cv2.drawContours(lane_two_side, contours2, index, (255,255,255), 1)

        if len(contours2) == 2:
            if index == 0:
                first_line = np.copy(lane_one_side)
                lowest_row_first_lane = find_lowest_row(lane_one_side)
            elif index == 1:
                lowest_row_second_lane = find_lowest_row(lane_one_side)
                if lowest_row_first_lane < lowest_row_second_lane:  # first index is shorter
                    euc_row = lowest_row_first_lane
                else:
                    euc_row = lowest_row_second_lane
                # euc_row = min(lowest_row_first_lane, lowest_row_second_lane)
                point_first = extract_point(first_line, euc_row)
                point_second = extract_point(lane_one_side, euc_row)
                outer_points_list.append(point_first)
                outer_points_list.append(point_second)
        elif len(contours2) == 1:
            point = extract_point(lane_one_side, euc_row)
            outer_points_list.append(point)

    return lane_two_side, outer_points_list, contours

def approx_dist_between_centers(contour1, contour2):
    # compute the center of contour1
    M1 = cv2.moments(contour1)
    cX1 = int(M1["m10"] / M1["m00"])
    cY1 = int(M1["m01"] / M1["m00"])
    # compute the center of contour2
    M2 = cv2.moments(contour2)
    cX2 = int(M2["m10"] / M2["m00"])
    cY2 = int(M2["m01"] / M2["m00"])
    dist = calculate_distance((cX1, cY1), (cX2, cY2))
    centroid1 = (cX1, cY1)
    centroid2 = (cX2, cY2)
    return dist, centroid1, centroid2

def ret_largest_contour(gray, min_area=None):
    largest_contour_found = False
    contour_img = np.zeros(gray.shape, dtype=gray.dtype)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    # find the two contours for which you want to find the min distance between them
    contours = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    max_contour_area = 0
    max_contour_idx = -1
    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_contour_area:
            max_contour_area = area
            max_contour_idx = index
            largest_contour_found = True

    if max_contour_idx != -1:
        contour_img = cv2.drawContours(contour_img, contours, max_contour_idx, (255,255,255), -1)
    return contour_img, largest_contour_found

def ret_largest_contour_outerlane(gray, min_area):
    largest_contour_found = False
    contour_img = np.zeros(gray.shape, dtype=gray.dtype)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # dilating segmented ROI's
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5,5))
    bin_img_dilated = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel)
    bin_img_ret = cv2.morphologyEx(bin_img_dilated, cv2.MORPH_ERODE, kernel)
    bin_img = bin_img_ret

    contours = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    max_contour_area = 0
    max_contour_index = -1
    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_contour_area:
            max_contour_area = area
            max_contour_index = index
            largest_contour_found = True
    
    if max_contour_area < min_area:
        largest_contour_found = False

    if ((max_contour_index != -1) and largest_contour_found):
        contour_img = cv2.drawContours(contour_img, contours, max_contour_index, (255,255,255), -1)

    return contour_img, largest_contour_found

def find_lane_curvature(traj_bottom_px, traj_bottom_py, traj_up_px, traj_up_py):
    offset_vert = 90
    
    if traj_up_px - traj_bottom_px != 0:
        slope = (traj_up_py - traj_bottom_py) / (traj_up_px - traj_bottom_px)
        y_intercept = traj_up_py - (slope * traj_up_px)
        angle_inclination = math.atan(slope) * (180 / np.pi) 
    else:
        slope = config.infinity
        y_intercept = 0
        angle_inclination = 90
    
    if angle_inclination != 90:
        if angle_inclination < 0:
            angle_wrt_vertical = offset_vert + angle_inclination
        else:
            angle_wrt_vertical = angle_inclination - offset_vert
    else:
        angle_wrt_vertical = 0
    return angle_wrt_vertical