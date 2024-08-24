import cv2 as cv
import numpy as np
import os
import threading
import subprocess
from gtts import gTTS
from pydub import AudioSegment
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


def create_kalman_filter(p0):
    kf = cv.KalmanFilter(4, 2)  # 4 dynamic params (x, y, vx, vy), 2 measurement params (x, y)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
    kf.processNoiseCov = np.eye(4, 4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, 2, dtype=np.float32) * 1
    kf.errorCovPost = np.eye(4, 4, dtype=np.float32) * 0.1
    kf.statePost[:2, 0] = p0.ravel()  # Initial position
    return kf


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

        for pose in self.trainee.yoga_poses:
            self.curr_pose = pose.name
            self.pose_params = YOGA_POSE_PARAMS[self.curr_pose]  # This is a list of relevant joints and angles between them to the specific pose
            self.track_pose2(pose)

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
        static_frames = 0
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
            self.connect_joints(img, good_new, show_joint_names=True)
            if np.all(p1 - p0 < 2):
                static_frames += 1
            else:
                static_frames = 0
            if static_frames > STATIC_THRESHOLD:
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

    def track_pose2(self, pose):
        print(f"Tracking {pose.name}")
        for (joint1, joint2, joint3), angle in pose.joints_angles.items():
            print(f"NOTICE: Angle between {joint1}, {joint2} and {joint3} should be {angle}")
        TRACKING_POSE_FLAG = True
        JOINTS_LOCATIONS = {}
        selected_joints = []
        p0 = np.array([])
        kfs = []
        good_new = np.array([])

        def select_joint(event, x, y, flags, param):
            nonlocal p0, kfs
            if event == cv.EVENT_LBUTTONDOWN:
                joint_name = JOINT_NAMES[len(selected_joints)]
                JOINTS_LOCATIONS[joint_name] = (x, y)
                selected_joints.append(joint_name)

                new_joint = np.array([(x, y)], dtype=np.float32).reshape(-1, 1, 2)
                if p0.size == 0:
                    p0 = new_joint
                else:
                    p0 = np.vstack((p0, new_joint))
                kf = create_kalman_filter(new_joint)
                kfs.append(kf)

        cap = cv.VideoCapture(CAM)
        ret, old_frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            exit()

        old_frame = cv.flip(old_frame, 1)
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        old_gray = cv.GaussianBlur(old_gray, (3, 3), 0)

        # Set up the mouse callback function
        cv.namedWindow("YogaCoach")
        cv.setMouseCallback("YogaCoach", select_joint)

        mask = np.zeros_like(old_frame)
        hsv = np.zeros_like(old_gray)
        hsv[..., 1] = 255
        static_frames = 0
        POSE_CHECKED = False
        COUNT_TO_START = 0
        while TRACKING_POSE_FLAG:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed To Capture Frame")
                exit()

            frame = cv.flip(frame, 1)
            img = frame.copy()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame_gray = cv.GaussianBlur(frame_gray, (3, 3), 0)

            if p0.size != 0:
                predicted_points = []
                for kf in kfs:
                    prediction = kf.predict()
                    predicted_points.append(prediction[:2].reshape(-1))

                predicted_points = np.array(predicted_points, dtype=np.float32).reshape(-1, 1, 2)

                # Calculate optical flow
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
                    old_gray = frame_gray.copy()
                    continue

                good_new = np.array(good_new, dtype=np.float32).reshape(-1, 1, 2)

                # Draw the tracks
                for i, (new, old) in enumerate(zip(good_new, p0)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    a, b, c, d = int(a), int(b), int(c), int(d)
                    cv.line(mask, (a, b), (c, d), self.colors[i % len(self.colors)].tolist(), 2)
                    cv.circle(frame, (a, b), 5, self.colors[i % len(self.colors)].tolist(), -1)

                mask = cv.multiply(mask, FADE_FACTOR)
                mask[mask < FADE_THRESHOLD] = 0
                img = cv.add(frame, mask)
                if len(selected_joints) == len(JOINT_NAMES):
                    self.connect_joints(img, good_new, show_joint_names=True)

                    if COUNT_TO_START < 120:  # Wait for 4 seconds before starting to check the pose
                        COUNT_TO_START += 1
                        continue
                    if np.all(p1 - p0 < 2):  # Start counting static frames
                        static_frames += 1
                    else:
                        static_frames = 0

                    if static_frames > STATIC_THRESHOLD:
                        if POSE_CHECKED:
                            continue
                        else:
                            # Open a thread to check the pose
                            threading.Thread(target=self.check_pose, args=(img, good_new)).start()
                            POSE_CHECKED = True

                p0 = good_new.reshape(-1, 1, 2) if p0.size != 0 else p0

            cv.imshow("YogaCoach", img)
            key = cv.waitKey(1)
            if key == 27 or key == ord("q"):  # ESC key
                exit()
            elif key == 32:  # Space key
                TRACKING_POSE_FLAG = False
            elif key == ord("r"):  # Reset the joints
                JOINTS_LOCATIONS = {}
                selected_joints = []
                p0 = np.array([])
                kfs = []
                good_new = np.array([])

            old_gray = frame_gray.copy()

        cap.release()
        cv.waitKey(1)
        cv.destroyAllWindows()
        cv.waitKey(1)

    def check_pose(self, img, joints_coords):
        for i, (joint1, joint2, joint3, angle) in enumerate(self.pose_params):

            if angle == "90":  # Special Emphasis
                joint1_idx = JOINT_NAMES.index(joint1)
                joint3_idx = JOINT_NAMES.index(joint3)
                joint1_coords = joints_coords[joint1_idx].ravel()
                joint3_coords = joints_coords[joint3_idx].ravel()
                # Check only the x coordinates
                if abs(joint1_coords[0] - joint3_coords[0]) < ANGLE_ERROR_THRESHOLD:
                    text = f"Good Job {self.trainee.name}! Your {joint3} is 90 degrees to the floor"
                    # cv.putText(
                    #     img,
                    #     text,
                    #     (10, 30 * (i * 2 + 1)),
                    #     cv.FONT_HERSHEY_COMPLEX,
                    #     0.75,
                    #     (0, 255, 0),
                    #     1,
                    #     cv.LINE_AA,
                    # )
                    tts = gTTS(text=text, lang="en")
                    tts.save("speech.mp3")
                    sound = AudioSegment.from_mp3("speech.mp3")
                    faster_sound = sound.speedup(playback_speed=1.2)
                    faster_sound.export("speech_faster.mp3", format="mp3")
                    subprocess.run(["mpg321", "speech_faster.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    os.system("rm speech.mp3 speech_faster.mp3")
                else:
                    text = YOGA_POSE_ALIGNMENT[self.curr_pose][i]
                    # cv.putText(
                    #     img,
                    #     text,
                    #     (10, 30 * (i * 2 + 1)),
                    #     cv.FONT_HERSHEY_COMPLEX,
                    #     0.75,
                    #     (0, 0, 255),
                    #     1,
                    #     cv.LINE_AA
                    # )
                    tts = gTTS(text=text, lang="en")
                    tts.save("speech.mp3")
                    sound = AudioSegment.from_mp3("speech.mp3")
                    faster_sound = sound.speedup(playback_speed=1.2)
                    faster_sound.export("speech_faster.mp3", format="mp3")
                    subprocess.run(["mpg321", "speech_faster.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    os.system("rm speech.mp3 speech_faster.mp3")
                continue

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
            angle_error = abs(angle_between_joints - angle)
            if angle_error < ANGLE_ERROR_THRESHOLD:
                text = f"Good Job {self.trainee.name}! Angle between your {joint1}, {joint2} and {joint3} is in the correct range"
                # cv.putText(
                #     img,
                #     text,
                #     text_position,
                #     cv.FONT_HERSHEY_COMPLEX,
                #     0.75,
                #     (0, 255, 0),
                #     1,
                #     cv.LINE_AA,
                # )
                tts = gTTS(text=text, lang="en")
                tts.save("speech.mp3")
                sound = AudioSegment.from_mp3("speech.mp3")
                faster_sound = sound.speedup(playback_speed=1.2)
                faster_sound.export("speech_faster.mp3", format="mp3")
                subprocess.run(["mpg321", "speech_faster.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.system("rm speech.mp3 speech_faster.mp3")
            else:
                text = YOGA_POSE_ALIGNMENT[self.curr_pose][i]
                # cv.putText(
                #     img,
                #     text,
                #     text_position,
                #     cv.FONT_HERSHEY_COMPLEX,
                #     0.75,
                #     (0, 0, 255),
                #     1,
                #     cv.LINE_AA,
                # )
                tts = gTTS(text=text, lang="en")
                tts.save("speech.mp3")
                sound = AudioSegment.from_mp3("speech.mp3")
                faster_sound = sound.speedup(playback_speed=1.2)
                faster_sound.export("speech_faster.mp3", format="mp3")
                subprocess.run(["mpg321", "speech_faster.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.system("rm speech.mp3 speech_faster.mp3")

    def connect_joints(self, img, joints_coords, show_joint_names=False):
        # Connect all joint one after the other in the order of JOINT_NAMES
        for i in range(len(joints_coords) - 1):
            joint1 = joints_coords[i].ravel()
            joint2 = joints_coords[i + 1].ravel()
            # Convert to integers (pixel coordinates)
            joint1 = tuple(map(int, joint1))
            joint2 = tuple(map(int, joint2))
            cv.line(img, joint1, joint2, (0, 255, 0), 2)
            if show_joint_names:
                cv.putText(img, JOINT_NAMES[i], joint1, cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
                if i == len(joints_coords) - 2:  # Last joint
                    cv.putText(img, JOINT_NAMES[i + 1], joint2, cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

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
