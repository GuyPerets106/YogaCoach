import cv2
from definitions import *


def calibration():
    cap = cv2.VideoCapture(CAM)
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        return
    first_frame = cv2.flip(first_frame, 1)
    width = first_frame.shape[1]
    height = first_frame.shape[0]
    w = 70
    h = 70
    joints = {
        "SHOULDER": (width // 2 - w // 2, height // 5, w, h),  # (x, y, w, h) - xy is the top left corner
        "ELBOW": (int(width // 2 + 1 * w), height // 5, w, h),
        "WRIST": (int(width // 2 + 2 * w), height // 5, w, h),
        "HIP": (width // 2 - w // 2, int(height // 5 + 2.75 * h), w, h),
        "KNEE": (width // 2 - w // 2, int(height // 5 + 4.5 * h), w, h),
        "ANKLE": (width // 2 - w // 2, int(height // 5 + 5.5 * h + 50), w, h),
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        frame = cv2.flip(frame, 1)
        for _, rect in joints.items():
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)  # When user is ready (fits his joints in the boxes), press space
        if key == 32:  # Space ASCII code
            break
    cap.release()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    centers = {joint: (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2) for joint, rect in joints.items()}
    # print(f"Centers Are: {centers}")
    return centers


if __name__ == "__main__":
    calibration()
