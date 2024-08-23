import customtkinter as ctk
from definitions import *
from trainee import Trainee

yoga_sequences = {"Sun Salutation": SUN_SALUTATION}


def create_main_window():
    window = ctk.CTk()
    window._set_appearance_mode("dark")

    # Adjust the screen ratio to make the window smaller but proportional
    screen_width = int(window.winfo_screenwidth() * WIDTH_RATIO * 0.7)
    screen_height = int(window.winfo_screenheight() * HEIGHT_RATIO * 0.7)
    font_size = int(screen_width * 0.015)  # Adjust font size for better visibility
    button_font_size = int(screen_width * 0.02)  # Larger font size for the button

    window.geometry(f"{screen_width}x{screen_height}")
    window.title("YogaCoach")

    main_frame = ctk.CTkFrame(master=window, corner_radius=0, fg_color="#1C1C1C")
    main_frame.pack(fill="both", expand=True)

    # Centering all elements within the main frame
    container_frame = ctk.CTkFrame(master=main_frame, corner_radius=0, fg_color="#1C1C1C")
    container_frame.place(relx=0.5, rely=0.5, anchor="center")

    # Configure grid layout to ensure proper alignment
    container_frame.grid_columnconfigure(0, weight=1)
    container_frame.grid_columnconfigure(1, weight=1)

    # Trainee Selected Yoga Sequence
    yoga_sequence_var = ctk.StringVar()
    yoga_sequence_frame = ctk.CTkFrame(master=container_frame, bg_color="#1C1C1C", fg_color="#1C1C1C")
    yoga_sequence_frame.grid(row=0, column=0, pady=10, sticky="e")
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
        font=("Roboto", font_size),
        bg_color="black",
        dropdown_fg_color="#1C1C1C",
        dropdown_text_color="white",
    )
    resolution_title.grid(row=0, column=0, pady=10, sticky="e")
    yoga_sequence_optionbox.grid(row=0, column=1, padx=5, pady=8, sticky="w")
    yoga_sequence_optionbox.set("Sun Salutation")

    # Trainee Name
    name_frame = ctk.CTkFrame(master=container_frame, bg_color="#1C1C1C", fg_color="#1C1C1C")
    name_frame.grid(row=1, column=0, pady=10, sticky="e")
    name_title = ctk.CTkLabel(master=name_frame, text="Name:", font=("Roboto", font_size, "bold"), bg_color="#1C1C1C", text_color="white")
    name_entry = ctk.CTkEntry(master=name_frame, placeholder_text="Default Student", font=("Roboto", font_size), bg_color="black", width=200)
    name_title.grid(row=0, column=0, padx=5, pady=5, sticky="e")
    name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    # Start Yoga Sequence Button
    button = ctk.CTkButton(
        master=container_frame,
        text="Start Yoga Sequence",
        font=("Roboto", button_font_size, "bold"),
        width=screen_width * 0.4,  # Make the button wider
        height=screen_height * 0.1,  # Make the button taller
        command=lambda: start_yoga_sequence(name_entry.get(), yoga_sequence_var.get()),
    )
    button.grid(row=2, column=0, columnspan=2, pady=20, sticky="n")  # Centered and aligned below the input fields

    return window


def start_yoga_sequence(name, yoga_sequence):
    if name == "":
        name = "Default Student"
    trainee_selected_yoga_sequence = yoga_sequences[yoga_sequence]
    print(f"Starting {yoga_sequence} for {name}")
    print(f"Selected Yoga Sequence: {trainee_selected_yoga_sequence}")
    Trainee(name, trainee_selected_yoga_sequence)


if __name__ == "__main__":
    window = create_main_window()
    window.mainloop()
