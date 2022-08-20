from asyncio import proactor_events
import cv2
import numpy as np

from .utils import cord_sort
from .lane_utils import find_lane_curvature

from ...config import config

def lane_points(mid_lane, outer_lane, offset_correction):
    mid_contours = cv2.findContours(mid_lane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    outer_contours = cv2.findContours(outer_lane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    if mid_contours and outer_contours:
        mid_contours_row_sorted = cord_sort(mid_contours, "rows")
        outer_contours_row_sorted = cord_sort(outer_contours, "row")

        mid_rows = mid_contours_row_sorted.shape[0]
        outer_rows = outer_contours_row_sorted.shape[0]

        mid_rows_bottom_point = mid_contours_row_sorted[mid_rows - 1, :]
        outer_rows_bottom_point = outer_contours_row_sorted[outer_rows - 1, :]
        mid_rows_top_point = mid_contours_row_sorted[0, :]
        outer_rows_top_point = outer_contours_row_sorted[0, :]

        trajectory_bottom_point = (
            int((mid_rows_bottom_point[0] + outer_rows_bottom_point[0]) / 2), 
            int((mid_rows_bottom_point[1] + outer_rows_bottom_point[1]) / 2)
        )
        trajectory_top_point = (
            int((mid_rows_top_point[0] + outer_rows_top_point[0]) / 2), 
            int((mid_rows_top_point[1] + outer_rows_top_point[1]) / 2)
        )
        return trajectory_bottom_point, trajectory_top_point
    else:
        return None, None

def estimate_non_mid_mask(mid_lane_edge):
    mid_hull_mask = np.zeros((mid_lane_edge.shape[0], mid_lane_edge.shape[1], 1), dtype=np.uint8)
    contours = cv2.findContours(mid_lane_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    if contours:
        hull_list = []
        contours = np.concatenate(contours)
        hull = cv2.convexHull(contours)
        hull_list.append(hull)
        mid_hull_mask = cv2.drawContours(mid_hull_mask, hull_list, 0, 255, -1)
    non_mid_mask = cv2.bitwise_not(mid_hull_mask)
    return non_mid_mask

def fetch_info_and_display(mid_lane_edge, mid_lane, outer_lane, frame, offset_correction):
    # 1. using both outer and middle information to create probable path
    trajectory_bottom_point, trajectory_up_point = lane_points(mid_lane, outer_lane, offset_correction)

    # 2. compute distance and curvature from trajectory points
    dist_between_central_road_and_car_nose = config.infinity
    if trajectory_bottom_point != None:
        dist_between_central_road_and_car_nose = trajectory_bottom_point[0] - int(mid_lane.shape[1] / 2)
        curvature = find_lane_curvature(trajectory_bottom_point[0], trajectory_bottom_point[1], 
                                        trajectory_up_point[0],     trajectory_up_point[1])
    else:
        return 0, 0, frame

    # 3. keep only those edge that are part of mid lane
    mid_lane_edge = cv2.bitwise_and(mid_lane_edge, mid_lane)

    # 4. combine mid and outer lane to get lanes combined and extract its contours
    lanes_combined = cv2.bitwise_or(outer_lane, mid_lane)
    # cv2.imshow("Lanes_combined", lanes_combined)
    projected_lane = np.zeros(lanes_combined.shape, lanes_combined.dtype)
    contours = cv2.findContours(lanes_combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

    # 5. fill projected lane with fillConvexPoly
    if contours:
        contours = np.concatenate(contours)
        contours = np.array(contours)
        cv2.fillConvexPoly(projected_lane, contours, 255)

    # 6. remove mid lane region from projected lane by extracting the midless mask
    mid_less_mask = estimate_non_mid_mask(mid_lane_edge)
    projected_lane = cv2.bitwise_and(mid_less_mask, projected_lane)

    # 7. draw projected lane
    empty_mat      = np.zeros((config.CropHeightResized, projected_lane.shape[1]), projected_lane.dtype)
    lane_drawn_frame = frame[config.CropHeightResized:, :]
    lane_drawn_frame[projected_lane == 255] = lane_drawn_frame[projected_lane == 255] + (0, 100, 0)
    lane_drawn_frame[outer_lane == 255] = lane_drawn_frame[outer_lane == 255] + (0, 0, 100)  # outer lane colored red
    lane_drawn_frame[mid_lane == 255] = lane_drawn_frame[mid_lane == 255] + (100, 0, 0)  # mid lane colored blue
    out_image = lane_drawn_frame

    # 8.draw car direction and lanes direction and distance between car and lane path
    cv2.line(out_image, (int(out_image.shape[1] / 2), out_image.shape[0]), (int(out_image.shape[1] / 2), out_image.shape[0] - int(out_image.shape[0] / 5)), (0, 0, 255), 2)
    if trajectory_bottom_point != None:
        cv2.line(out_image, trajectory_bottom_point, trajectory_up_point, (255, 0, 0), 2)
        cv2.line(out_image, trajectory_bottom_point, (int(out_image.shape[1] / 2), trajectory_bottom_point[1]), (255, 255, 0), 2)  # distance of car center with lane path
    upper_image = frame[0:config.CropHeightResized, :]
    out_image = np.vstack((upper_image, out_image))

    # 9. draw extracted distance and curvature
    curvature_str = "curvature = " + f"{curvature:.2f}"
    dist_between_central_road_and_car_nose_str = "distance = " + str(dist_between_central_road_and_car_nose)
    text_size_ratio = 0.5
    cv2.putText(out_image, curvature_str, (10, 200), cv2.FONT_HERSHEY_DUPLEX, text_size_ratio, (0, 255, 255), 1)
    cv2.putText(out_image, dist_between_central_road_and_car_nose_str, (10, 220), cv2.FONT_HERSHEY_DUPLEX, text_size_ratio, (0, 255, 255), 1)
    
    return dist_between_central_road_and_car_nose, curvature, out_image
