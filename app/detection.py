import cv2 as cv
import numpy as np
from yoga_pose import YogaPose
from calibration import calibration
from definitions import *


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
            "winSize": WIN_SIZE,
            "maxLevel": MAX_LEVEL,
            "criteria": CRITERIA,
            "minEigThreshold": MIN_EIG_THRESHOLD,
            "flags": FLAGS,
        }
        self.backSub = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
        self.persistent_mask = None
        self.colors = np.random.randint(0, 255, (100, 3))
        self.count = 0

        for pose in trainee.yoga_poses:
            self.pose_params = YOGA_POSE_PARAMS[pose.name]  # This is a list of relevant joints and angles between them to the specific pose
            self.track_pose(pose)

        print("DONE")

    def track_pose(self, pose):
        print(f"Tracking {pose.name}")
        for (joint1, joint2, joint3), angle in pose.joints_angles.items():
            print(f"NOTICE: Angle between {joint1}, {joint2} and {joint3} should be {angle}")

        JOINTS_LOCATIONS = calibration()  # This is a dictionary of joint names and their initial locations
        TRACKING_POSE_FLAG = True

        cap = cv.VideoCapture(CAM)
        ret, old_frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            exit()
        joint_coords = list(JOINTS_LOCATIONS.values())
        old_frame = cv.flip(old_frame, 1)
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        old_gray = cv.GaussianBlur(old_gray, (3, 3), 0)
        # segmented_old, backSub, persistent_mask = self.detect_person(old_gray, self.backSub, self.persistent_mask)

        p0 = np.array(joint_coords, dtype=np.float32)
        kfs = create_kalman_filters(p0)
        mask = np.zeros_like(old_frame)
        hsv = np.zeros_like(old_gray)
        hsv[..., 1] = 255

        while TRACKING_POSE_FLAG:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed To Capture Frame")
                exit()
            frame = cv.flip(frame, 1)
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame_gray = cv.GaussianBlur(frame_gray, (3, 3), 0)
            # segmented_frame, self.backSub, self.persistent_mask = self.detect_person(frame_gray, self.backSub, self.persistent_mask)
            predicted_points = []
            for kf in kfs:
                prediction = kf.predict()
                predicted_points.append(prediction[:2].reshape(-1))

            predicted_points = np.array(predicted_points, dtype=np.float32).reshape(-1, 1, 2)
            # calculate optical flow
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.lk_params)

            # Select good points
            if p1 is not None and st is not None:
                good_new = []
                for i, kf in enumerate(kfs):
                    if st[i]:
                        measurement = p1[i].reshape(-1, 1)
                        kf.correct(measurement)
                        good_new.append(p1[i].reshape(2))
                        self.count += 1
                    else:
                        if self.count < 100:
                            self.count += 1
                            good_new.append(p0[i].reshape(2))
                        else:
                            good_new.append(predicted_points[i].reshape(2))
            else:
                print("ERROR: Optical flow calculation failed")
                p0 = np.array(joint_coords, dtype=np.float32)
                old_gray = frame_gray.copy()
                continue
            good_new = np.array(good_new, dtype=np.float32).reshape(-1, 1, 2)
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, p0)):
                a, b = new.ravel()
                c, d = old.ravel()
                a, b, c, d = int(a), int(b), int(c), int(d)
                cv.line(mask, (a, b), (c, d), self.colors[i].tolist(), 2)
                cv.circle(frame, (a, b), 5, self.colors[i].tolist(), -1)
            mask = cv.multiply(mask, FADE_FACTOR)
            mask[mask < FADE_THRESHOLD] = 0
            img = cv.add(frame, mask)
            self.connect_relevant_joints(img, good_new, show_joint_names=True)
            self.check_pose(img, good_new)
            cv.imshow("YogaCoach", img)
            k = cv.waitKey(1)
            if k == 27 or k == ord("q"):
                exit()  # Close the program

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

            key = cv.waitKey(1)
            if key == 32:  # Space key
                TRACKING_POSE_FLAG = False

        cap.release()
        cv.waitKey(1)
        cv.destroyAllWindows()
        cv.waitKey(1)

    def check_pose(self, img, joints_coords):
        for i, (joint1, joint2, joint3, angle) in enumerate(self.pose_params):
            joint1_idx = JOINT_NAMES.index(joint1)
            joint2_idx = JOINT_NAMES.index(joint2)
            joint3_idx = JOINT_NAMES.index(joint3)
            angle = np.deg2rad(angle)
            joint1_coords = joints_coords[joint1_idx].ravel()
            joint2_coords = joints_coords[joint2_idx].ravel()
            joint3_coords = joints_coords[joint3_idx].ravel()
            v1 = joint1_coords - joint2_coords
            v2 = joint3_coords - joint2_coords
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle_between_joints = np.arccos(cos_angle)
            text_position = (10, 30 * (i * 2 + 1))
            if abs(angle_between_joints - angle) < 5:
                cv.putText(  # TODO CHANGE TO TTS
                    img,
                    f"Angle between {joint1}, {joint2} and {joint3} is {np.rad2deg(angle_between_joints):.2f} degrees",
                    text_position,
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )
            else:
                cv.putText(  # TODO CHANGE TO TTS
                    img,
                    "FIX YOUR POSE",
                    text_position,
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv.LINE_AA,
                )

    def connect_relevant_joints(self, img, joints_coords, show_joint_names=False):
        for joint1, joint2, joint3, _ in self.pose_params:
            joint1_idx = JOINT_NAMES.index(joint1)
            joint2_idx = JOINT_NAMES.index(joint2)
            joint3_idx = JOINT_NAMES.index(joint3)
            joint1_coords = joints_coords[joint1_idx].ravel()
            joint2_coords = joints_coords[joint2_idx].ravel()
            joint3_coords = joints_coords[joint3_idx].ravel()
            # Convert to int (pixel coordinates)
            joint1_coords = tuple(map(int, joint1_coords))
            joint2_coords = tuple(map(int, joint2_coords))
            joint3_coords = tuple(map(int, joint3_coords))
            cv.line(img, joint1_coords, joint2_coords, (0, 255, 0), 2)
            cv.line(img, joint2_coords, joint3_coords, (0, 255, 0), 2)

            if show_joint_names:
                cv.putText(img, joint1, joint1_coords, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv.LINE_AA)
                cv.putText(img, joint2, joint2_coords, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv.LINE_AA)
                cv.putText(img, joint3, joint3_coords, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv.LINE_AA)

    def detect_person(self, frame, backSub, persistent_mask):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        height, width = frame.shape[:2]
        roi = frame[int(height * 0.1) : int(height * 0.9), int(width * 0.25) : int(width * 0.75)]
        fg_mask = backSub.apply(roi)
        fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel)
        fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel)
        edges = cv.Canny(roi, 50, 150)
        combined_mask = cv.bitwise_or(fg_mask, edges)
        combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel, iterations=3)
        _, combined_mask = cv.threshold(combined_mask, 127, 255, cv.THRESH_BINARY)

        # If this is the first frame, initialize the persistent mask
        if persistent_mask is None:
            persistent_mask = combined_mask
        else:
            persistent_mask = cv.addWeighted(persistent_mask, 0.2, combined_mask, 0.8, 0)
            _, persistent_mask = cv.threshold(persistent_mask, 127, 255, cv.THRESH_BINARY)

        contours, _ = cv.findContours(persistent_mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        human_mask = np.zeros_like(persistent_mask)
        cv.drawContours(human_mask, contours, -1, (255), thickness=cv.FILLED)

        full_mask = np.zeros_like(frame)
        full_mask[int(height * 0.1) : int(height * 0.9), int(width * 0.25) : int(width * 0.75)] = human_mask

        return full_mask, backSub, persistent_mask
