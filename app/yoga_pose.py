from joint import Joint


class YogaPose:
    def __init__(self, name: str, joints_angles_tuples: list[tuple], hold_time: int):
        self.name = name
        self.joints_angles = {}
        self.hold_time = hold_time
        for joint1, joint2, joint3, angle in joints_angles_tuples:
            self.add_joint_angle(joint1, joint2, joint3, angle)

        print(f"Creating Yoga Pose {self.name} with {self.joints_angles}")

    def __str__(self):
        return f"Pose {self.name}"

    def add_joint_angle(self, joint1: Joint, joint2: Joint, joint3: Joint, angle: float):  # Joint is an abstract class
        assert joint1 != joint2 and joint2 != joint3 and joint1 != joint3, "Joints must be different"
        assert angle >= 0 and angle <= 180, "Angle must be between 0 and 180 degrees"
        self.joints_angles[(joint1, joint2, joint3)] = angle
