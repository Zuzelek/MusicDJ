# gui.py

import os
import csv
import threading
from tkinter import Tk, filedialog, Text, Button, Scrollbar, messagebox, ttk
from audio_analyzer import AudioAnalyzer
from audio_mixer import AudioMixer


class MusicDJApp:
    """Class for the main GUI application."""

    def __init__(self):
        self.root = Tk()
        self.root.title("Music DJ Application")
        self.root.geometry("700x600")

        self.text_area = None
        self.progress = None
        self.playlist = None
        self.analysis_results = []  # Store results for exporting
        self.create_gui()

    def create_gui(self):
        scroll = Scrollbar(self.root)
        scroll.pack(side="right", fill="y")

        self.text_area = Text(self.root, height=20, width=80)
        self.text_area.pack(padx=10, pady=10)

        scroll.config(command=self.text_area.yview)
        self.text_area.config(yscrollcommand=scroll.set)

        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=10)

        analyze_button = Button(
            self.root,
            text="Import Playlist & Analyze",
            width=25,
            height=2,
            command=self.start_analysis_thread
        )
        analyze_button.pack(pady=10)

        play_button = Button(
            self.root,
            text="Play First Track",
            width=25,
            height=2,
            command=self.play_first_track
        )
        play_button.pack(pady=10)

        stop_button = Button(
            self.root,
            text="Stop Audio",
            width=25,
            height=2,
            command=AudioMixer.stop_audio
        )
        stop_button.pack(pady=10)

        export_txt_button = Button(
            self.root,
            text="Export Analysis (Text)",
            width=25,
            height=2,
            command=self.export_analysis_to_text
        )
        export_txt_button.pack(pady=10)

        export_csv_button = Button(
            self.root,
            text="Export Analysis (CSV)",
            width=25,
            height=2,
            command=self.export_analysis_to_csv
        )
        export_csv_button.pack(pady=10)

    def import_playlist(self):
        """Imports a playlist folder and returns the list of MP3 files."""
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

    def analyze_playlist(self, folder_path, mp3_files):
        """Analyzes the playlist and displays the results."""
        self.text_area.delete(1.0, "end")
        self.analysis_results = []  # Reset results for new analysis
        self.progress["maximum"] = len(mp3_files)
        self.progress["value"] = 0

        for i, file in enumerate(mp3_files):
            file_path = os.path.join(folder_path, file)
            result = AudioAnalyzer.analyze_audio(file_path)

            if result is not None:
                try:
                    tempo, key, camelot = result
                    artist_title = file.split(' - ')
                    if len(artist_title) == 2:
                        artist, title = artist_title
                    else:
                        artist, title = 'Unknown', 'Unknown'

                    result_line = f"Analyzing file: {file_path}\n"
                    result_line += f"Artist: {artist}\n"
                    result_line += f"Title: {title}\n"
                    result_line += f"BPM: {tempo}\n"
                    result_line += f"Key: {key}\n\n"
                except Exception as e:
                    result_line = f"Error analyzing file {file_path}: {e}\n"

                self.text_area.insert("end", result_line)
                self.analysis_results.append({"File": file, "BPM": float(tempo), "Key": key})
            else:
                result_line = f"Error processing file: {file}\n\n"
                self.text_area.insert("end", result_line)

            self.progress["value"] += 1
            self.root.update_idletasks()

    def start_analysis_thread(self):
        folder_path, mp3_files = self.import_playlist()
        if folder_path and mp3_files:
            threading.Thread(target=self.analyze_playlist, args=(folder_path, mp3_files), daemon=True).start()

    def play_first_track(self):
        """Plays the first track in the analyzed playlist."""
        folder_path, mp3_files = self.import_playlist()
        if folder_path and mp3_files:
            first_track = os.path.join(folder_path, mp3_files[0])
            AudioMixer.play_audio(first_track)

    def export_analysis_to_text(self):
        """Exports the analysis results to a text file."""
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            with open(file_path, "w") as f:
                f.write(self.text_area.get(1.0, "end"))
            messagebox.showinfo("Export Successful", f"Analysis saved to {file_path}")

    def export_analysis_to_csv(self):
        """Exports the analysis results to a CSV file."""
        if not self.analysis_results:
            messagebox.showerror("Error", "No analysis results to export.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            with open(file_path, "w", newline="") as csvfile:
                fieldnames = ["File", "BPM", "Key"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.analysis_results)
            messagebox.showinfo("Export Successful", f"Analysis saved to {file_path}")

    def run(self):
        """Runs the main event loop."""
        self.root.mainloop()
