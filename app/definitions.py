import cv2 as cv

HEIGHT_RATIO = 0.7
WIDTH_RATIO = 0.8
SUN_SALUTATION = ["Plank", "Chaturanga", "Upward Facing Dog", "Downward Facing Dog"]
JOINT_NAMES = ["WRIST", "ELBOW", "SHOULDER", "HIP", "KNEE", "ANKLE"]
YOGA_POSE_PARAMS = {  # POSE : [(JOINT1, JOINT2, JOINT3, DESIRED_ANGLE), ...]
    "Plank": [("WRIST", "ELBOW", "SHOULDER", 180), ("WRIST", "ELBOW", "SHOULDER", "90")],
    "Chaturanga": [("WRIST", "ELBOW", "SHOULDER", 90), ("SHOULDER", "HIP", "ANKLE", 160)],
    "Upward Facing Dog": [("WRIST", "ELBOW", "SHOULDER", 180), ("WRIST", "ELBOW", "SHOULDER", "90")], # TODO KNEE ABOVE THE GROUND
    "Downward Facing Dog": [("WRIST", "ELBOW", "SHOULDER", 180), ("HIP", "KNEE", "ANKLE", 180), ("WRIST", "SHOULDER", "HIP", 170)],
}
YOGA_POSE_HOLD_TIME = {"Plank": 5, "Chaturanga": 5, "Upward Facing Dog": 5, "Downward Facing Dog": 5}
YOGA_POSE_ALIGNMENT = {"Plank": ["Straighten your elbow", "Lean forward with your upper body to align your shoulder with your wrist"],
                       "Chaturanga": ["Bend your elbow to 90 degrees", "Lower your buttocks"],
                       "Upward Facing Dog": ["Straighten your elbow", "Lean forward with your upper body to align your shoulder with your wrist"],
                       "Downward Facing Dog": ["Straighten your elbow", "Straighten your legs", "Straighten your back by trying to push the floor"]}
FADE_FACTOR = 0.6
FADE_THRESHOLD = 10
WIN_SIZE = (15, 15)
MAX_LEVEL = 3
CRITERIA = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
MIN_EIG_THRESHOLD = 1e-4
FLAGS = 0
CAM = 0
STATIC_THRESHOLD = 60
ANGLE_ERROR_THRESHOLD = 20
