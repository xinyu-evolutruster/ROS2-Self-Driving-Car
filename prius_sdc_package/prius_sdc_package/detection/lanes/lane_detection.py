import cv2
import numpy as np
from .color_segmentation import segment_lanes
from .midlane_estimation import estimate_midlane
from .lane_cleaning import get_yellow_inner_edge, extend_short_lane
from .data_extraction import fetch_info_and_display
from ...config import config

def detect_lanes(frame):
    # cropping the roi (e.g. keeping only below the horizon)
    img_cropped = frame[config.CropHeightResized:, :]

    mid_lane_mask, mid_lane_edge, outer_lane_mask, outer_lane_edge, outer_lane_side_sep, outer_lane_points = segment_lanes(img_cropped, config.MinAreaResized)

    estimated_midlane = estimate_midlane(mid_lane_edge, config.MaxDistResized)

    outer_lane_one_side, outer_contours_one_side, mid_contours, offset_correction = get_yellow_inner_edge(outer_lane_edge, estimated_midlane, outer_lane_points)

    extended_mid_lane, extended_outer_lane = extend_short_lane(estimated_midlane, mid_contours, outer_contours_one_side, outer_lane_one_side.copy())

    distance, curvature, out_image = fetch_info_and_display(mid_lane_edge, extended_mid_lane, extended_outer_lane, frame, offset_correction)

    # cv2.imshow("mid_lane_mask", mid_lane_mask)
    # cv2.imshow("mid_lane_edge", mid_lane_edge)
    # cv2.imshow("outerlane_side_sep", outer_lane_side_sep)
    # cv2.imshow("estimated midlane", estimated_midlane)

    # cv2.imshow("outer_lane_mask", outer_lane_mask)
    # cv2.imshow("outer_lane_edge", outer_lane_edge)

    # cv2.imshow("outer_lane_one_side", outer_lane_one_side)
    # cv2.imshow("extended_mid_lane", extended_mid_lane)
    # cv2.imshow("extended_outer_lane", extended_outer_lane)

    # cv2.imshow("frame", out_image)
    # cv2.waitKey(1)

    return distance, curvature, out_image