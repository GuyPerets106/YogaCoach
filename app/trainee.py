from definitions import YOGA_POSE_HOLD_TIME, YOGA_POSE_PARAMS
from detection import Detector
from yoga_pose import YogaPose


class Trainee:
    def __init__(self, name: str, selected_yoga_training: list[YogaPose]):
        print(f"Creating Trainee {name} for {selected_yoga_training}")
        self.name = name
        self.detected_joints = []
        self.yoga_poses = []
        self.selected_yoga_training = selected_yoga_training
        for yoga_pose in self.selected_yoga_training:
            self.add_yoga_pose(yoga_pose)
        self.detector = Detector(self)

        # calib_params = calibrate()
        # self.detect_joints(calib_params)

    def add_yoga_pose(self, yoga_pose: str):
        yoga_pose = YogaPose(yoga_pose, YOGA_POSE_PARAMS[yoga_pose], YOGA_POSE_HOLD_TIME[yoga_pose])
        self.yoga_poses.append(yoga_pose)

    def __str__(self):
        training = ", ".join([yoga_pose.name for yoga_pose in self.selected_yoga_training])
        return f"{self.name}: Selected Training: {training}"

    def __repr__(self):
        return f"Trainee(name={self.name}, Selected Training={self.selected_yoga_training})"

    def detect_joints(self):
        pass
