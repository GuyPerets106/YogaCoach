import cv2 as cv

HEIGHT_RATIO = 0.7
WIDTH_RATIO = 0.8
SUN_SALUTATION = ["Plank", "Chaturanga", "Upward Facing Dog", "Downward Facing Dog"]
JOINT_NAMES = ["WRIST", "ELBOW", "SHOULDER", "HIP", "KNEE", "ANKLE"]
YOGA_POSE_PARAMS = {
    "Plank": [("SHOULDER", "ELBOW", "WRIST", 180), ("SHOULDER", "HIP", "ANKLE", 120)],
    "Chaturanga": [("SHOULDER", "HIP", "ANKLE", 180), ("SHOULDER", "ELBOW", "WRIST", 90)],
    "Upward Facing Dog": [("WRIST", "SHOULDER", "KNEE", 45), ("WRIST", "ELBOW", "SHOULDER", 180)],
    "Downward Facing Dog": [("WRIST", "HIP", "ANKLE", 45), ("WRIST", "ELBOW", "SHOULDER", 180), ("HIP", "KNEE", "ANKLE", 180)],
}
YOGA_POSE_HOLD_TIME = {"Plank": 5, "Chaturanga": 5, "Upward Facing Dog": 5, "Downward Facing Dog": 5}
FADE_FACTOR = 0.6
FADE_THRESHOLD = 10
WIN_SIZE = (15, 15)
MAX_LEVEL = 3
CRITERIA = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
MIN_EIG_THRESHOLD = 1e-4
FLAGS = 0
CAM = 0
STATIC_THRESHOLD = 10
