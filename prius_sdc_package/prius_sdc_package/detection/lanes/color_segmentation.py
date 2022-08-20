import cv2
import numpy as np
from .lane_utils import bw_area_open, ret_lowest_edge_points, ret_largest_contour_outerlane

# white regions
hue_white = 0
lit_white = 225
sat_white = 0
# yellow regions
hue_yellow_lower = 30
hue_yellow_higher = 33
lit_yellow = 160
sat_yellow = 0

cur_frame = None
cur_hls = None

def clr_segment(hls, lower_range, upper_range):
    mask_in_range = cv2.inRange(hls, lower_range, upper_range)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.morphologyEx(mask_in_range, cv2.MORPH_DILATE, kernel)
    return mask_dilated

def mask_extract():
    # segmenting white regions
    mask_white = clr_segment(cur_hls, np.array([hue_white, lit_white, sat_white]), np.array([255, 255, 255]))
    # segmenting yellow regions
    mask_yellow = clr_segment(cur_hls, np.array([hue_yellow_lower, lit_yellow, sat_yellow]), np.array([hue_yellow_higher, 255, 255]))

    mask_white_ = (mask_white != 0)
    dst = cur_frame * (mask_white_[:, :, None].astype(cur_frame.dtype))

    mask_yellow_ = (mask_yellow != 0)
    dst_yellow = cur_frame * (mask_yellow_[:, :, None].astype(cur_frame.dtype))

    cv2.imshow("white_regions", dst)
    cv2.imshow("yellow_regions", dst_yellow)

def on_hue_low_change(val):
    global hue_white
    hue_white = val
    mask_extract()

def on_lit_low_change(val):
    global lit_white
    lit_white = val
    mask_extract()

def on_sat_low_change(val):
    global sat_white
    sat_white = val
    mask_extract()

def on_hue_low_y_change(val):
    global hue_yellow_lower
    hue_yellow_lower = val
    mask_extract()

def on_hue_high_y_change(val):
    global hue_yellow_higher
    hue_yellow_higher = val
    mask_extract()

def on_lit_low_y_change(val):
    global lit_yellow
    lit_yellow = val
    mask_extract()

def on_sat_low_y_change(val):
    global sat_yellow
    sat_yellow = val
    mask_extract()

# cv2.namedWindow("white_regions")
# cv2.namedWindow("yellow_regions")

# # create slidebars
# cv2.createTrackbar("Hue_White_Lower", "white_regions", hue_white, 255, on_hue_low_change)
# cv2.createTrackbar("Lit_White_Lower", "white_regions", lit_white, 255, on_lit_low_change)
# cv2.createTrackbar("Sat_White_Lower", "white_regions", sat_white, 255, on_sat_low_change)

# cv2.createTrackbar("Hue_Yellow_Lower", "yellow_regions", hue_yellow_lower, 255, on_hue_low_y_change)
# cv2.createTrackbar("Hue_Yellow_Lower", "yellow_regions", hue_yellow_higher, 255, on_hue_high_y_change)
# cv2.createTrackbar("Lit_Yellow_Lower", "yellow_regions", lit_yellow, 255, on_lit_low_y_change)
# cv2.createTrackbar("Sat_Yellow_Lower", "yellow_regions", sat_yellow, 255, on_sat_low_y_change)

def get_mask_and_edge_of_larger_objects(frame, mask, min_area):
    # keeping only objects larger than min_area
    frame_roi = cv2.bitwise_and(frame, frame, mask=mask)
    frame_roi_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    mask_of_larger_objects = bw_area_open(frame_roi_gray, min_area)
    frame_roi_gray = cv2.bitwise_and(frame_roi_gray, mask_of_larger_objects)
    # extracting edges of those larger objects
    frame_roi_smoothed = cv2.GaussianBlur(frame_roi_gray, (11, 11), 1)
    # edges_of_larger_objects = cv2.Canny(frame_roi_smoothed, 50, 150, None, 3)
    edges = cv2.findContours(frame_roi_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    edges_of_larger_objects = np.zeros_like(frame_roi_gray)
    edges_of_larger_objects = cv2.drawContours(edges_of_larger_objects, edges, -1 ,255, 1)

    total_cols = edges_of_larger_objects.shape[1]
    total_rows = edges_of_larger_objects.shape[0]
    edges_of_larger_objects[:, total_cols - 1] = 0
    edges_of_larger_objects[0, :] = 0
    edges_of_larger_objects[total_rows - 1, :] = 0
    edges_of_larger_objects[:, 0] = 0

    return mask_of_larger_objects, edges_of_larger_objects

def segment_midlane(frame, white_regions, min_area):
    mid_lane_mask, mid_lane_edge = get_mask_and_edge_of_larger_objects(frame, white_regions, min_area)
    return mid_lane_mask, mid_lane_edge

def segment_outerlane(frame, yellow_regions, min_area):
    outer_points_list = []
    mask, edges = get_mask_and_edge_of_larger_objects(frame, yellow_regions, min_area)

    mask_largest, largest_found = ret_largest_contour_outerlane(mask, min_area)
    if largest_found:
        # keep only edges of largest region
        edge_largest = cv2.bitwise_and(edges, mask_largest)
        # return edge points for identifying closest edge later
        lanes_sides_sep, outer_points_list, _ = ret_lowest_edge_points(edge_largest)
        edges = edge_largest
    else:
        lanes_sides_sep = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)

    return mask, edges, lanes_sides_sep, outer_points_list

def segment_lanes(frame, min_area):
    global cur_frame
    global cur_hls

    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    cur_frame = frame
    cur_hls = hls

    # segmenting white regions
    white_regions = clr_segment(hls, np.array([hue_white, lit_white, sat_white]), np.array([255, 255, 255]))

    # segmenting yellow regions
    yellow_regions = clr_segment(hls, np.array([hue_yellow_lower, lit_yellow, sat_yellow]), np.array([hue_yellow_higher, 255, 255]))

    # segmenting midlane from white regions
    mid_lane_mask, mid_lane_edge = segment_midlane(frame, white_regions, min_area)

    # segmenting outerlane from yellow regions
    # outer_lane_edge, outerlane_side_sep, outerlane_points = segment_outerlane(frame, yellow_regions, min_area)
    outer_lane_mask, outer_lane_edge, outer_lanes_sides_sep, outer_lane_points = segment_outerlane(frame, yellow_regions, min_area + 500)

    return mid_lane_mask, mid_lane_edge, outer_lane_mask, outer_lane_edge, outer_lanes_sides_sep, outer_lane_points