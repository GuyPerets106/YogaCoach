import customtkinter as ctk
from definitions import *
from trainee import Trainee

yoga_sequences = {"Sun Salutation": SUN_SALUTATION}


def create_main_window():
    window = ctk.CTk()
    window._set_appearance_mode("dark")
    screen_width = int(window.winfo_screenwidth() * WIDTH_RATIO)
    screen_height = int(window.winfo_screenheight() * HEIGHT_RATIO)
    font_size = int(screen_width * 0.01)
    window.geometry(f"{screen_width}x{screen_height}")
    window.title("YogaCoach")

    main_frame = ctk.CTkScrollableFrame(master=window, corner_radius=0, fg_color="#1C1C1C")
    main_frame.pack(fill="both", expand=True, side="left")

    # Trainee Selected Yoga Sequence
    yoga_sequence_var = ctk.StringVar()
    yoga_sequence_frame = ctk.CTkFrame(master=main_frame, bg_color="#1C1C1C", fg_color="#1C1C1C")
    yoga_sequence_frame.pack(pady=20)
    yoga_sequence_frame.grid_columnconfigure(0, minsize=100)
    resolution_title = ctk.CTkLabel(
        master=yoga_sequence_frame,
        text="Yoga Sequence:",
        font=("SFPro", font_size, "bold"),
        bg_color="#1C1C1C",
        text_color="white",
    )
    yoga_sequence_optionbox = ctk.CTkOptionMenu(
        master=yoga_sequence_frame,
        values=["Sun Salutation"],
        variable=yoga_sequence_var,
        font=("Roboto", 14),
        bg_color="black",
        dropdown_fg_color="#1C1C1C",
        dropdown_text_color="white",
    )
    resolution_title.grid(row=0, column=0, padx=5, pady=8, sticky="e")
    yoga_sequence_optionbox.grid(row=0, column=1, padx=5, pady=8, sticky="w")
    yoga_sequence_optionbox.set("Sun Salutation")

    # Trainee Name
    name_frame = ctk.CTkFrame(master=main_frame, bg_color="#1C1C1C", fg_color="#1C1C1C")
    name_frame.pack(pady=20)
    name_frame.grid_columnconfigure(0, minsize=100)
    name_title = ctk.CTkLabel(
        master=name_frame, text="Name:", font=("Roboto", font_size, "bold"), bg_color="#1C1C1C", text_color="white"
    )
    name_entry = ctk.CTkEntry(
        master=name_frame, placeholder_text="Default Student", font=("Roboto", 14), bg_color="black", width=180
    )
    name_title.grid(row=0, column=0, padx=5, pady=5, sticky="e")
    name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    # Trainee Height
    height_frame = ctk.CTkFrame(master=main_frame, bg_color="#1C1C1C", fg_color="#1C1C1C")
    height_frame.pack(pady=20)
    height_frame.grid_columnconfigure(0, minsize=100)
    height_title = ctk.CTkLabel(
        master=height_frame, text="Height (cm):", font=("Roboto", font_size, "bold"), bg_color="#1C1C1C", text_color="white"
    )
    height_entry = ctk.CTkEntry(master=height_frame, placeholder_text="170", font=("Roboto", 14), bg_color="black", width=180)
    height_title.grid(row=0, column=0, padx=5, pady=5, sticky="e")
    height_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    # Start Yoga Sequence Button
    button = ctk.CTkButton(
        master=main_frame,
        text="Start Yoga Sequence",
        command=lambda: start_yoga_sequence(name_entry.get(), height_entry.get(), yoga_sequence_var.get()),
    )
    button.pack(pady=20)

    return window


def start_yoga_sequence(name, height, yoga_sequence):
    trainee_selected_yoga_sequence = yoga_sequences[yoga_sequence]
    print(f"Starting {yoga_sequence} for {name} (Height {height}cm)")
    print(f"Selected Yoga Sequence: {trainee_selected_yoga_sequence}")
    trainee = Trainee(name, height, trainee_selected_yoga_sequence)
    assert 1 == 0
    print(f"Created {trainee}")


if __name__ == "__main__":
    window = create_main_window()
    window.mainloop()
