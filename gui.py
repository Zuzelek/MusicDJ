import os
import csv
import threading
from tkinter import Tk, filedialog, Text, Button, Scrollbar, messagebox, ttk
from audio_analyzer import AudioAnalyzer, AudioFeatures
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
        self.analyzer = AudioAnalyzer()  # Create analyzer instance
        self.create_gui()

    def create_gui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Add scrollbar
        scroll = Scrollbar(main_frame)
        scroll.pack(side="right", fill="y")

        # Create text area with improved styling
        self.text_area = Text(main_frame, height=20, width=80, font=("Courier", 10))
        self.text_area.pack(side="left", fill="both", expand=True)

        scroll.config(command=self.text_area.yview)
        self.text_area.config(yscrollcommand=scroll.set)

        # Create control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=10, pady=5)

        # Add progress bar
        self.progress = ttk.Progressbar(
            control_frame,
            orient="horizontal",
            length=400,
            mode="determinate"
        )
        self.progress.pack(pady=10)

        # Create buttons with improved styling
        button_style = {"width": 25, "height": 2}

        analyze_button = ttk.Button(
            control_frame,
            text="Import Playlist & Analyze",
            command=self.start_analysis_thread,
            width=25
        )
        analyze_button.pack(pady=5)

        play_button = ttk.Button(
            control_frame,
            text="Play First Track",
            command=self.play_first_track,
            width=25
        )
        play_button.pack(pady=5)

        stop_button = ttk.Button(
            control_frame,
            text="Stop Audio",
            command=AudioMixer.stop_audio,
            width=25
        )
        stop_button.pack(pady=5)

        export_txt_button = ttk.Button(
            control_frame,
            text="Export Analysis (Text)",
            command=self.export_analysis_to_text,
            width=25
        )
        export_txt_button.pack(pady=5)

        export_csv_button = ttk.Button(
            control_frame,
            text="Export Analysis (CSV)",
            command=self.export_analysis_to_csv,
            width=25
        )
        export_csv_button.pack(pady=5)

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
            features = self.analyzer.analyze_audio(file_path)

            if features is not None:
                try:
                    # Parse artist and title from filename
                    artist_title = file.split(' - ')
                    if len(artist_title) == 2:
                        artist, title = artist_title
                    else:
                        artist, title = 'Unknown', 'Unknown'

                    # Format the analysis results
                    result_line = f"""
{'=' * 50}
File: {file}
Artist: {artist}
Title: {title}
BPM: {features.tempo:.1f}
Key: {features.key}
Energy Level: {features.energy:.2f}
Number of Segments: {features.segments['count']}
{'=' * 50}
"""
                    self.text_area.insert("end", result_line)

                    # Store comprehensive analysis results
                    self.analysis_results.append({
                        "File": file,
                        "Artist": artist,
                        "Title": title,
                        "BPM": float(features.tempo),
                        "Key": features.key,
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

    def start_analysis_thread(self):
        """Starts the analysis in a separate thread."""
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
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, "w") as f:
                f.write(self.text_area.get(1.0, "end"))
            messagebox.showinfo("Export Successful", f"Analysis saved to {file_path}")

    def export_analysis_to_csv(self):
        """Exports the analysis results to a CSV file."""
        if not self.analysis_results:
            messagebox.showerror("Error", "No analysis results to export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, "w", newline="") as csvfile:
                fieldnames = ["File", "Artist", "Title", "BPM", "Key", "Energy", "Segments"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.analysis_results)
            messagebox.showinfo("Export Successful", f"Analysis saved to {file_path}")

    def run(self):
        """Runs the main event loop."""
        self.root.mainloop()