import cv2 as cv
import numpy as np
from calibration import calibration


def create_kalman_filters(p0):
    kalman_filters = []
    for point in p0:
        kf = cv.KalmanFilter(4, 2)  # 4 dynamic params (x, y, vx, vy), 2 measurement params (x, y)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        kf.processNoiseCov = np.eye(4, 4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, 2, dtype=np.float32) * 1
        kf.errorCovPost = np.eye(4, 4, dtype=np.float32) * 0.1
        kf.statePost[:2, 0] = point.ravel()  # Initial position
        kalman_filters.append(kf)
    return kalman_filters


class Detector:
    def __init__(self, trainee):
        self.trainee = trainee
        self.detected_joints = []
        self.current_pose = None
        self.lk_params = {
            "winSize": (15, 15),
            "maxLevel": 3,
            "criteria": (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
            "minEigThreshold": 1e-4,
            "flags": 0,
        }
        self.backSub = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
        self.persistent_mask = None
        self.colors = np.random.randint(0, 255, (100, 3))
        self.cap = cv.VideoCapture(1)

        for pose in trainee.yoga_poses:
            self.track_pose(pose)

    def track_pose(self, pose):
        print(f"Tracking {pose.name}")
        JOINTS_LOCATIONS = calibration()
        for (joint1, joint2, joint3), angle in pose.joints_angles.items():
            print(f"NOTICE: Angle between {joint1}, {joint2} and {joint3} should be {angle}")

        TRACKING_POSE_FLAG = True

        while TRACKING_POSE_FLAG:
            key = cv.waitKey(1)
            if key == 32:  # Space key
                TRACKING_POSE_FLAG = False
