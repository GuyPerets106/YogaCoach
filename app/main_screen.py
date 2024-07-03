import customtkinter as ctk
import cv2
from definitions import *


def create_main_window():
    window = ctk.CTk()
    window._set_appearance_mode("dark")
    screen_width = int(window.winfo_screenwidth() * WIDTH_RATIO)
    screen_height = int(window.winfo_screenheight() * HEIGHT_RATIO)
    font_size = int(screen_width * 0.02)
    window.geometry(f"{screen_width}x{screen_height}")
    window.title("YogaCoach")

    main_frame = ctk.CTkScrollableFrame(master=window, corner_radius=0, fg_color="#1C1C1C")
    main_frame.pack(fill="both", expand=True, side="left")

    # Add a button to the main_frame, that will trigger a function when clicked

    button = ctk.CTkButton(master=main_frame, text="Start The Game", command=start_game)
    button.pack(pady=20)

    return window


def start_game():
    cap = cv2.VideoCapture(1)  # 0 for the first webcam
    while True:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    window = create_main_window()
    window.mainloop()
