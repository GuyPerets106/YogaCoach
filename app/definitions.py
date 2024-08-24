import cv2 as cv

HEIGHT_RATIO = 0.7
WIDTH_RATIO = 0.8
SUN_SALUTATION = ["Plank", "Chaturanga", "Upward Facing Dog", "Downward Facing Dog"]
YOGA_SEQUENCES = {"Sun Salutation": SUN_SALUTATION, "Plank": ["Plank"], "Chaturanga": ["Chaturanga"],
                  "Upward Facing Dog": ["Upward Facing Dog"], "Downward Facing Dog": ["Downward Facing Dog"]}
JOINT_NAMES = ["WRIST", "ELBOW", "SHOULDER", "HIP", "KNEE", "ANKLE"]
YOGA_POSE_VIDEOS = {"Plank": "/Users/guyperets/Documents/IP_Course/Project/PLANK_TUTORIAL.mov",
                    "Chaturanga": "/Users/guyperets/Documents/IP_Course/Project/CHATURANGA_TUTORIAL.mov",
                    "Upward Facing Dog": "/Users/guyperets/Documents/IP_Course/Project/UPWARD_DOG_TUTORIAL.mov",
                    "Downward Facing Dog": "/Users/guyperets/Documents/IP_Course/Project/DOWNWARD_DOG_TUTORIAL.mov"}
CALIBRATION_EXPLAINATION = "We will start with the calibration. Stand in profile position, put your left hand straight in shoulder height"
YOGA_POSE_EXPLAINATION = {"Plank": "Slowly bend your back forward and put your palms on the floor slightly wider than your shoulders. Send your legs backwards and keep your back and elbows straight. Your shoulders should be aligned with your wrists",
                          "Chaturanga": "From the plank position, bend your elbows close to your torso until it reaches 90 degrees",
                          "Upward Facing Dog": "From the chaturanga position, put your toes flat on the ground, straighten your elbows and push your torso upwards. Lift your knees of the floor and hold the position",
                          "Downward Facing Dog": "From the Upward facing dog, tack your toes and push your buttocks towards the ceiling. Your knee, elbow and back should be straight. Make sure your body is creating a triangular shape"}
YOGA_POSE_PARAMS = {  # POSE : [(JOINT1, JOINT2, JOINT3, DESIRED_ANGLE), ...]
    "Plank": [("WRIST", "ELBOW", "SHOULDER", 180), ("WRIST", "ELBOW", "SHOULDER", "90"), ("SHOULDER", "HIP", "ANKLE", 160)],
    "Chaturanga": [("WRIST", "ELBOW", "SHOULDER", 90), ("SHOULDER", "HIP", "ANKLE", 160)],
    "Upward Facing Dog": [("WRIST", "ELBOW", "SHOULDER", 180), ("WRIST", "ELBOW", "SHOULDER", "90")], # TODO KNEE ABOVE THE GROUND
    "Downward Facing Dog": [("WRIST", "ELBOW", "SHOULDER", 180), ("HIP", "KNEE", "ANKLE", 180), ("WRIST", "SHOULDER", "HIP", 170)],
}
YOGA_POSE_HOLD_TIME = {"Plank": 5, "Chaturanga": 5, "Upward Facing Dog": 5, "Downward Facing Dog": 5}
YOGA_POSE_ALIGNMENT = {"Plank": ["Straighten your elbow", "Lean forward with your upper body", "Lower your buttocks"],
                       "Chaturanga": ["Bend your elbow to 90 degrees", "Lower your buttocks"],
                       "Upward Facing Dog": ["Straighten your elbow", "Lean forward with your upper body"],
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
ANGLE_ERROR_THRESHOLD = 10
WAIT_FOR_POSE = 180
