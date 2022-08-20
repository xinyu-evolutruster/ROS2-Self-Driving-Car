import os
import cv2

detect = 1  # set to 1 for lane detection

infinity = 1000
eps = 0.01

ResizedWidth = 320
ResizedHeight = 240

# ======================== Parameters for Lane Detection ======================

RefImgWidth = 1920
RefImgHeight = 1080

FramePixels = RefImgHeight * RefImgWidth

ResizeFramePixels = ResizedWidth * ResizedHeight

LaneExtractionMinAreaPer = 1000 / FramePixels
MinAreaResized = int(ResizeFramePixels * LaneExtractionMinAreaPer)

BWContourOpenSpeedMaxDistPer = 600 / RefImgHeight
MaxDistResized = int(ResizedHeight * BWContourOpenSpeedMaxDistPer)

CropHeight = 630
CropHeightResized = int((CropHeight / RefImgHeight) * ResizedHeight)