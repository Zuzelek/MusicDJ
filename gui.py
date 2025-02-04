import os
import csv
import threading
from tkinter import Tk, filedialog, Text, Button, Scrollbar, messagebox, ttk
from tkinterdnd2 import TkinterDnD, DND_FILES  # For drag-and-drop support
from audio_analyzer import AudioAnalyzer, AudioFeatures
from audio_mixer import AudioMixer


class MusicDJApp:
    """Class for the main GUI application."""

    def __init__(self):
        self.root = TkinterDnD.Tk()
        self.root.title("Music DJ Application")
        self.root.geometry("800x700")

        self.text_area = None
        self.progress = None
        self.playlist = None
        self.analysis_results = []
        self.analyzer = AudioAnalyzer()
        self.create_gui()

    def create_gui(self):
        # main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # scrollbar
        scroll = Scrollbar(main_frame)
        scroll.pack(side="right", fill="y")

        self.text_area = Text(main_frame, height=20, width=80, font=("Courier", 10))
        self.text_area.pack(side="left", fill="both", expand=True)

        scroll.config(command=self.text_area.yview)
        self.text_area.config(yscrollcommand=scroll.set)

        # control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=10, pady=5)

        # progress bar
        self.progress = ttk.Progressbar(
            control_frame,
            orient="horizontal",
            length=400,
            mode="determinate"
        )
        self.progress.pack(pady=10)

        # buttons
        buttons = [
            ("Import Playlist & Analyze", self.start_analysis_thread),
            ("Play First Track", self.play_first_track),
            ("Play All Tracks", self.play_all_tracks),
            ("Stop Audio", AudioMixer.stop_audio),
            ("Clear Playlist", self.clear_playlist),
            ("Export Analysis (Text)", self.export_analysis_to_text),
            ("Export Analysis (CSV)", self.export_analysis_to_csv),
        ]

        for text, command in buttons:
            button = ttk.Button(control_frame, text=text, command=command, width=25)
            button.pack(pady=5)

        # Add drag-and-drop support
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.handle_drop)

    def handle_drop(self, event):
        folder_path = event.data.strip('{}')
        if os.path.isdir(folder_path):
            self.start_analysis_thread(folder_path)
        else:
            messagebox.showerror("Error", "Please drop a folder containing MP3 files.")

    def import_playlist(self, folder_path=None):
        """Imports a playlist function"""
        if not folder_path:
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
        """Analyze a playlist function"""
        self.text_area.delete(1.0, "end")
        self.analysis_results = []
        self.progress["maximum"] = len(mp3_files)
        self.progress["value"] = 0

        for i, file in enumerate(mp3_files):
            file_path = os.path.join(folder_path, file)
            features = self.analyzer.analyze_audio(file_path)

            if features is not None:
                try:
                    # Parse artist and title from filename
                    artist_title = file.split(' - ')
                    if len(artist_title) == 2:
                        artist, title = artist_title
                    else:
                        artist, title = 'Unknown', 'Unknown'

                    camelot_number = features.camelot_key[:-1]
                    key_type = "Major" if features.camelot_key[-1] == "B" else "Minor"

                    result_line = f"""
{'=' * 50}
File: {file}
Artist: {artist}
Title: {title}
BPM: {features.tempo:.1f}
Key: {features.key} (Camelot: {features.camelot_key})
Energy Level: {features.energy:.2f}
Segments: {features.segments['count']}
"""
                    self.text_area.insert("end", result_line)

                    # results
                    self.analysis_results.append({
                        "File": file,
                        "Artist": artist,
                        "Title": title,
                        "BPM": float(features.tempo),
                        "Key": features.key,
                        "Camelot Full": features.camelot_key,
                        "Camelot Number": camelot_number,
                        "Key Type": key_type,
                        "Energy": float(features.energy),
                        "Segments": features.segments['count']
                    })

                except Exception as e:
                    result_line = f"Error analyzing file {file_path}: {str(e)}\n\n"
                    self.text_area.insert("end", result_line)
            else:
                result_line = f"Error processing file: {file}\n\n"
                self.text_area.insert("end", result_line)

            self.progress["value"] += 1
            self.root.update_idletasks()

    def start_analysis_thread(self, folder_path=None):
        """Seperate thread for analysis so program can continue running"""
        folder_path, mp3_files = self.import_playlist(folder_path)
        if folder_path and mp3_files:
            threading.Thread(target=self.analyze_playlist, args=(folder_path, mp3_files), daemon=True).start()

    def play_first_track(self):
        """Play first track"""
        folder_path, mp3_files = self.import_playlist()
        if folder_path and mp3_files:
            first_track = os.path.join(folder_path, mp3_files[0])
            AudioMixer.play_audio(first_track)

    def play_all_tracks(self):
        """Plays all tracks """
        folder_path, mp3_files = self.import_playlist()
        if folder_path and mp3_files:
            for file in mp3_files:
                track = os.path.join(folder_path, file)
                AudioMixer.play_audio(track)

    def clear_playlist(self):
        """Clear playlist"""
        self.text_area.delete(1.0, "end")
        self.analysis_results = []
        self.progress["value"] = 0
        messagebox.showinfo("Info", "Playlist and analysis results cleared.")

    def export_analysis_to_text(self):
        """Exports the analysis results to txt"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, "w") as f:
                f.write(self.text_area.get(1.0, "end"))
            messagebox.showinfo("Export Successful", f"Analysis saved to {file_path}")

    def export_analysis_to_csv(self):
        """Exports the analysis CSV file."""
        if not self.analysis_results:
            messagebox.showerror("Error", "No analysis results to export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, "w", newline="") as csvfile:
                fieldnames = [
                    "File", "Artist", "Title", "BPM", "Key",
                    "Camelot Full", "Camelot Number", "Key Type",
                    "Energy", "Segments"
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.analysis_results)
            messagebox.showinfo("Export Successful", f"Analysis saved to {file_path}")

    def run(self):
        self.root.mainloop()