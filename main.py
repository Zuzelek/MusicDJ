import os
import librosa
import numpy as np
import soundfile as sf
from tkinter import Tk, filedialog, Listbox, Button, Scrollbar, messagebox, Label, Scale, HORIZONTAL
import pygame
import time


# Function to import playlist
def import_playlist():
    folder_path = filedialog.askdirectory(title="Select Folder Containing MP3 Files")
    if folder_path:
        mp3_files = [file for file in os.listdir(folder_path) if file.endswith('.mp3')]
        if mp3_files:
            return folder_path, mp3_files
        else:
            messagebox.showerror("Error", "No MP3 files found in the selected folder.")
            return None, None
    else:
        messagebox.showerror("Error", "No folder selected.")
        return None, None


# Function to analyze the audio files
def analyze_audio(file_path):
    try:
        y, sr = librosa.load(file_path)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
        key = np.argmax(np.mean(chroma, axis=1))

        return tempo, key
    except Exception as e:
        return None, None


# Function to analyze playlist in selected folder
def analyze_playlist(folder_path, mp3_files, listbox):
    listbox.delete(0, "end")
    track_info = []
    for file in mp3_files:
        file_path = os.path.join(folder_path, file)
        tempo, key = analyze_audio(file_path)

        if tempo and key is not None:
            result = f"File: {file} | BPM: {tempo} | Key: {key}"
            track_info.append((file, result, file_path))
            listbox.insert("end", result)
        else:
            result = f"File: {file} | Error processing file"
            track_info.append((file, result, None))
            listbox.insert("end", result)

    return track_info


# Audio Playback Functions
current_file_path = None
pygame.mixer.init()


def play_audio(file_path):
    global current_file_path
    if file_path and file_path != current_file_path:
        pygame.mixer.music.load(file_path)
        current_file_path = file_path
    if pygame.mixer.music.get_busy() == 0:  # Check if music is not already playing
        pygame.mixer.music.play()


def pause_audio():
    pygame.mixer.music.pause()


def resume_audio():
    pygame.mixer.music.unpause()


# Function to adjust BPM of an audio file
def adjust_bpm(file_path, target_bpm):
    try:

        y, sr = librosa.load(file_path)

        original_tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        if isinstance(original_tempo, np.ndarray):
            original_tempo = original_tempo.item()

        stretch_factor = target_bpm / original_tempo
        y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)

        timestamp = int(time.time() * 1000)
        output_file = f"adjusted_audio_{timestamp}.wav"

        sf.write(output_file, y_stretched, sr)
        return output_file, original_tempo
    except Exception as e:
        messagebox.showerror("Error", f"Failed to adjust BPM: {e}")
        return None, None


# GUI Functionality for BPM Matching and Playback
def create_gui():
    root = Tk()
    root.title("Music AI DJ")

    root.geometry("600x600")

    scroll = Scrollbar(root)
    scroll.pack(side="right", fill="y")

    # Listbox for displaying track names
    listbox = Listbox(root, height=15, width=110, selectmode="single")
    listbox.pack(padx=10, pady=10)

    scroll.config(command=listbox.yview)
    listbox.config(yscrollcommand=scroll.set)

    # BPM Adjustment Section
    bpm_label = Label(root, text="Set Target BPM:")
    bpm_label.pack()

    bpm_scale = Scale(root, from_=50, to=200, orient=HORIZONTAL, length=400)
    bpm_scale.set(120)
    bpm_scale.pack()

    analyze_button = Button(root, text="Import Playlist & Analyze", width=25, height=2,
                            command=lambda: start_analysis(listbox))
    analyze_button.pack(pady=10)

    # Playback Buttons
    play_button = Button(root, text="Play Selected File", width=20,
                         command=lambda: play_selected_track(listbox))
    play_button.pack(pady=5)

    pause_button = Button(root, text="Pause", width=10, command=pause_audio)
    pause_button.pack(pady=5)

    resume_button = Button(root, text="Resume", width=10, command=resume_audio)
    resume_button.pack(pady=5)

    bpm_button = Button(root, text="Adjust BPM & Play", width=20,
                        command=lambda: handle_bpm_adjustment(bpm_scale, listbox))
    bpm_button.pack(pady=10)

    root.mainloop()


def start_analysis(listbox):
    folder_path, mp3_files = import_playlist()
    if folder_path and mp3_files:
        track_info = analyze_playlist(folder_path, mp3_files, listbox)
        listbox.track_info = track_info  # Save track info for later use


def play_selected_track(listbox):
    selected_index = listbox.curselection()
    if not selected_index:
        messagebox.showerror("Error", "No track selected.")
        return

    selected_track = listbox.track_info[selected_index[0]]
    file_path = selected_track[2]

    if file_path:
        play_audio(file_path)
    else:
        messagebox.showerror("Error", "Error with the selected track.")


def handle_bpm_adjustment(bpm_scale, listbox):
    target_bpm = bpm_scale.get()
    selected_index = listbox.curselection()

    if not selected_index:
        messagebox.showerror("Error", "No track selected for BPM adjustment.")
        return

    selected_track = listbox.track_info[selected_index[0]]
    file_path = selected_track[2]

    if not file_path:
        messagebox.showerror("Error", "No valid file selected.")
        return

    adjusted_file, original_bpm = adjust_bpm(file_path, target_bpm)
    if adjusted_file:
        result = f"Original BPM: {original_bpm:.2f}, Adjusted BPM: {target_bpm}\nPlaying adjusted audio."
        listbox.insert("end", result)
        play_audio(adjusted_file)



if __name__ == "__main__":
    create_gui()
