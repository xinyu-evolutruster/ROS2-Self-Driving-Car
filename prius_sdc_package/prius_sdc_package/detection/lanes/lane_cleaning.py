import cv2
import math
import numpy as np
from .utils import calculate_distance, cord_sort
from .lane_utils import ret_lowest_edge_points

from ...config import config

def is_path_crossing_mid(mid_lane, mid_contours, outer_contours):
    is_ref_to_path_left = 0
    ref_to_car_path_image = np.zeros_like(mid_lane)

    mid_lane_copy = mid_lane.copy()

    if not mid_contours:
        print("[Warning!!] No midlane detected")

    mid_contours_row_sorted = cord_sort(mid_contours, "rows")
    outer_contours_row_sorted = cord_sort(outer_contours, "rows")
    mid_rows = mid_contours_row_sorted.shape[0]
    outer_rows = outer_contours_row_sorted.shape[0]

    mid_bottom_point = mid_contours_row_sorted[mid_rows - 1, :]
    outer_bottom_point = outer_contours_row_sorted[outer_rows - 1, :]

    car_trajectory_bottom_point = (int((mid_bottom_point[0] + outer_bottom_point[0]) / 2), int((mid_bottom_point[1] + outer_bottom_point[1]) / 2))
    
    bottom_center_x = int(ref_to_car_path_image.shape[1] / 2)
    bottom_center_y = ref_to_car_path_image.shape[0]
    cv2.line(ref_to_car_path_image, car_trajectory_bottom_point, (bottom_center_x, bottom_center_y), (255, 255, 0), 2)
    cv2.line(mid_lane_copy, tuple(mid_bottom_point), (mid_bottom_point[0], mid_lane_copy.shape[0] - 1), (255, 255, 0), 2)

    is_ref_to_path_left = ((int(ref_to_car_path_image.shape[1] / 2) - car_trajectory_bottom_point[0]) > 0)

    if np.any(cv2.bitwise_and(ref_to_car_path_image, mid_lane_copy) > 0):
        return True, is_ref_to_path_left
    else:
        return False, is_ref_to_path_left

def get_yellow_inner_edge(outer_lanes, mid_lane, outer_lane_points):
    outer_lanes_ret = np.zeros_like(outer_lanes)
    offset_correction = 0
    
    # 1. extract mid and outer lane contours
    mid_contours = cv2.findContours(mid_lane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    # outer_contours = cv2.findContours(outer_lanes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    outer_contours = cv2.findContours(outer_lanes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    # 2. check if outer lane was present initially or not
    if not outer_contours:
        no_outer_lane_before = True
    else:
        no_outer_lane_before = False

    # 3. set the first contour of midlane as reference
    ref = (0, 0)
    if mid_contours:
        ref = tuple(mid_contours[0][0][0])
    # ref = (outer_lanes.shape[0] - 1, int(outer_lanes.shape[1] / 2))

    # 4. condition 1: if both mid lane and outer lane are detected
    if mid_contours:
        # 4.A. len(outer_lane_points) == 2
        # (a) fetch the side of outer lane nearer to the mid lane
        if len(outer_lane_points) == 2:
            point_a = outer_lane_points[0]
            point_b = outer_lane_points[1]

            closest_index = 0
            if calculate_distance(point_a, ref) <= calculate_distance(point_b, ref):
                closest_index = 0
            elif len(outer_contours) > 1:
                closest_index = 1

            outer_lanes_ret = cv2.drawContours(outer_lanes_ret, outer_contours, closest_index, 255, 1)
            outer_contours_ret = [outer_contours[closest_index]]

            # (b) if correct out lane was detected
            is_path_crossing, is_crossing_left = is_path_crossing_mid(mid_lane, mid_contours, outer_contours_ret)
            if is_path_crossing:
                outer_lanes = np.zeros_like(outer_lanes)
            else:
                # outer_contours = cv2.findContours(outer_lanes_ret, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
                # cv2.imshow("outer_lanes_ret", outer_lanes_ret)
                return outer_lanes_ret, outer_contours_ret, mid_contours, 0
        
        # 4.B. 0 < len(outer_lane_points) < 2
        elif np.any(outer_lanes > 0):
            is_path_crossing, is_crossing_left = is_path_crossing_mid(mid_lane, mid_contours, outer_contours)
            if is_path_crossing:
                outer_lanes = np.zeros_like(outer_lanes)
            else:
                return outer_lanes, outer_contours, mid_contours, 0
        
        # 4. condition 2: if mid lane is present but no outlane detected
        #    or outer lane got zeroed because of crossing mid lane
        # action: create outer lane on side that represent the larger lane as seen by camera

        if not np.any(outer_lanes > 0):
            # fetching the column of the lowest point of the mid lane
            mid_contours_row_sorted = cord_sort(mid_contours, "rows")
            mid_rows = mid_contours_row_sorted.shape[0]
            mid_low_point = mid_contours_row_sorted[mid_rows - 1, :]
            mid_high_point = mid_contours_row_sorted[0, :]
            mid_low_col = mid_low_point[0]

            # addressing which side to draw the outerlane considering it was present before or not
            draw_right = True
            if no_outer_lane_before:
                if mid_low_col < int(mid_lane.shape[1] / 2):
                    draw_right = True
            else:
                if is_crossing_left:
                    draw_right = True

            # setting outer lane upper and lower points column to the right if draw right and vice versa
            if draw_right:
                low_col = int(mid_lane.shape[1] - 1)
                high_col = int(mid_lane.shape[1] - 1)
                offset_correction = 20
            else:
                low_col = 0
                high_col = 0
                offset_correction = -20

            mid_low_point[1] = mid_lane.shape[0]
            lane_point_lower = (low_col, int(mid_low_point[1]))
            lane_point_upper = (high_col, int(mid_high_point[1]))
            outer_lanes = cv2.line(outer_lanes, lane_point_lower, lane_point_upper, 255, 1)
            outer_contours = cv2.findContours(outer_lanes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
            
            return outer_lanes, outer_contours, mid_contours, offset_correction

        # else:  # no midlane
        #     return outer_lanes, outer_contours, mid_contours, offset_correction
    else:
        return outer_lanes, outer_contours, mid_contours, offset_correction

def extend_short_lane(mid_lane, mid_contours, outer_contours, outer_lane):
    # 1. sorting the mid and outer contours on basis of rows
    if mid_contours and outer_contours:
        mid_contours_row_sorted = cord_sort(mid_contours, "rows")
        outer_contours_row_sorted = cord_sort(outer_contours, "rows")

        image_bottom = mid_lane.shape[0]
        total_num_of_contours_mid_lane = mid_contours_row_sorted.shape[0]
        total_num_of_contours_outer_lane = outer_contours_row_sorted.shape[0]

        # 2. connect mid lane to image bottom by drawing a vertical line if not already connected
        bottom_point_mid = mid_contours_row_sorted[total_num_of_contours_mid_lane - 1, :]
        if bottom_point_mid[1] < image_bottom:
            mid_lane = cv2.line(mid_lane, tuple(bottom_point_mid), (bottom_point_mid[0], image_bottom), 255, 2)
           
        # 3. connect outer lane to image bottom by performing 2 steps
        # step 1: extend outer lane in the direction of its slope
        ## a) taking last 20 points to estimate slope
        bottom_point_outer = outer_contours_row_sorted[total_num_of_contours_outer_lane - 1, :]
        if bottom_point_outer[1] < image_bottom:
            if total_num_of_contours_outer_lane > 20:
                shift = 20
            else:
                shift = 2
            ref_last_10_points = outer_contours_row_sorted[total_num_of_contours_outer_lane - shift:total_num_of_contours_outer_lane-1:2, :]

            ## b) estimating slope
            if len(ref_last_10_points) > 1:
                ref_x = ref_last_10_points[:, 0] # cols
                ref_y = ref_last_10_points[:, 1] # rows
                ref_parameters = np.polyfit(ref_x, ref_y, 1)
                ref_slope = ref_parameters[0]
                ref_y_intercept = ref_parameters[1]
                if math.fabs(ref_slope) < config.eps:
                    ref_line_touch_point = (bottom_point_outer[0], outer_lane.shape[0] - 1)
                    ref_bottom_point_tup = tuple(bottom_point_outer)
                    outer_lane = cv2.line(outer_lane, ref_line_touch_point, ref_bottom_point_tup, 255, 2)
                else:
                    ref_line_touch_point_y = (int((outer_lane.shape[0] - 1 - ref_y_intercept) / ref_slope), outer_lane.shape[0] - 1)
                    ref_line_touch_point_x = (outer_lane.shape[1] - 1, int(ref_slope * (outer_lane.shape[1] - 1) + ref_y_intercept))

                    if ref_line_touch_point_x[1] < outer_lane.shape[0]:
                        # step 2: if required, connect outer lane to bottom by drawing a vertical line
                        ref_line_touch_point = ref_line_touch_point_x
                        outer_lane = cv2.line(outer_lane, ref_line_touch_point, (outer_lane.shape[1] - 1, outer_lane.shape[0] - 1), 255, 2)
                    else:
                        ref_line_touch_point = ref_line_touch_point_y
                        ref_bottom_point_tup = tuple(bottom_point_outer)
                        outer_lane = cv2.line(outer_lane, ref_line_touch_point, ref_bottom_point_tup, 255, 2)

    return mid_lane, outer_lane
