import csv
import os
import json
import threading
import time
from tkinter import filedialog, Scrollbar, messagebox, ttk, Scale, Toplevel
from tkinterdnd2 import TkinterDnD, DND_FILES
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from audioProcessing.audio_analyzer import AudioAnalyzer
from audioProcessing.audio_mixer import AudioMixer


class MusicDJApp:
    """Class for the main GUI application."""

    def __init__(self):
        self.root = TkinterDnD.Tk()
        self.root.title("Music DJ Application")
        self.root.geometry("1000x800")

        self.playlist_tree = None
        self.progress = None
        self.analysis_results = []
        self.section_data = None
        self.analyzer = AudioAnalyzer()
        self.current_track_index = 0
        self.playlist_folder = None

        try:
            with open('results.json', 'r') as f:
                self.section_data = json.load(f)
                print(f"Loaded section data for {len(self.section_data)} tracks")
        except (FileNotFoundError, json.JSONDecodeError):
            print("No pre-analyzed section data found")

        self.create_gui()

    def create_gui(self):
        """Create GUI components."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.playlist_tree = ttk.Treeview(main_frame,
                                          columns=("File", "Artist", "Title", "BPM", "Key", "Energy", "Sections"),
                                          show="headings")

        # Configure column headings
        self.playlist_tree.heading("File", text="File")
        self.playlist_tree.heading("Artist", text="Artist")
        self.playlist_tree.heading("Title", text="Title")
        self.playlist_tree.heading("BPM", text="BPM")
        self.playlist_tree.heading("Key", text="Key")
        self.playlist_tree.heading("Energy", text="Energy")
        self.playlist_tree.heading("Sections", text="Sections")

        # Configure column widths
        self.playlist_tree.column("File", width=200)
        self.playlist_tree.column("Artist", width=150)
        self.playlist_tree.column("Title", width=150)
        self.playlist_tree.column("BPM", width=50)
        self.playlist_tree.column("Key", width=80)
        self.playlist_tree.column("Energy", width=80)
        self.playlist_tree.column("Sections", width=150)

        self.playlist_tree.pack(fill="both", expand=True, pady=10)

        scroll = Scrollbar(main_frame, orient="vertical", command=self.playlist_tree.yview)
        scroll.pack(side="right", fill="y")
        self.playlist_tree.configure(yscrollcommand=scroll.set)

        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=10, pady=5)

        # Store control_frame for later access
        self.control_frame = control_frame

        self.progress = ttk.Progressbar(control_frame, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=10)

        volume_label = ttk.Label(control_frame, text="Volume:")
        volume_label.pack(side="left", padx=5)
        self.volume_slider = Scale(control_frame, from_=0.0, to=1.0, resolution=0.1, orient="horizontal",
                                   command=AudioMixer.set_volume)
        self.volume_slider.set(1.0)
        self.volume_slider.pack(side="left", padx=5)

        # Create a frame for buttons to organize them better
        buttons_frame = ttk.Frame(self.root)
        buttons_frame.pack(fill="x", padx=10, pady=5)

        # First row of buttons
        buttons_row1 = [
            ("Import Playlist & Analyze", self.start_analysis_thread),
            ("Play Selected Track", self.play_selected_track),
            ("Play All Tracks", self.play_all_tracks),
            ("Stop Audio", self.stop_audio),
        ]

        for text, command in buttons_row1:
            button = ttk.Button(buttons_frame, text=text, command=command, width=20)
            button.pack(side="left", padx=5, pady=5)

        # Second row of buttons
        buttons_frame2 = ttk.Frame(self.root)
        buttons_frame2.pack(fill="x", padx=10, pady=5)

        buttons_row2 = [
            ("Export Analysis (Text)", self.export_analysis_to_text),
            ("Export Analysis (CSV)", self.export_analysis_to_csv),
            ("Visualize Sections", self.visualize_selected_track),
            ("Run Section Analyzer", self.run_section_analyzer),
        ]

        for text, command in buttons_row2:
            button = ttk.Button(buttons_frame2, text=text, command=command, width=20)
            button.pack(side="left", padx=5, pady=5)

        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.handle_drop)

    def handle_drop(self, event):
        """Handle drag-and-drop events."""
        folder_path = event.data.strip('{}')
        if os.path.isdir(folder_path):
            self.start_analysis_thread(folder_path)
        else:
            messagebox.showerror("Error", "Please drop a valid folder.")

    def import_playlist(self, folder_path=None):
        if not folder_path:
            folder_path = filedialog.askdirectory(title="Select Folder Containing Audio Files")
        if folder_path:
            audio_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.mp3', '.wav', '.WAV'))]
            if audio_files:
                return folder_path, audio_files
            else:
                messagebox.showerror("Error", "No audio files found.")
        return None, None

    def analyze_playlist(self, folder_path, audio_files):
        """Analyzes audio files and updates the GUI."""
        self.playlist_tree.delete(*self.playlist_tree.get_children())
        self.analysis_results = []
        self.progress["maximum"] = len(audio_files)
        self.progress["value"] = 0

        self.playlist_folder = folder_path

        for i, file in enumerate(audio_files):
            file_path = os.path.join(folder_path, file)
            features = self.analyzer.analyze_audio(file_path)

            if features:
                artist, title = file.split(' - ') if ' - ' in file else ('Unknown', file.split('.')[0])

                # Check for sections in the external section data
                section_display = ""
                if self.section_data:
                    for track_data in self.section_data:
                        if track_data["file_name"] == file:
                            section_counts = {}
                            for section in track_data["sections"]:
                                section_type = section["type"]
                                section_counts[section_type] = section_counts.get(section_type, 0) + 1

                            section_display = ", ".join([f"{k}: {v}" for k, v in section_counts.items()])
                            break

                self.playlist_tree.insert("", "end", values=(file, artist, title, f"{features.tempo:.1f}",
                                                             features.key, f"{features.energy:.2f}", section_display))

                # Add to analysis results
                result_dict = {
                    "File": file,
                    "Artist": artist,
                    "Title": title,
                    "BPM": features.tempo,
                    "Key": features.key,
                    "Energy": features.energy,
                    "Segments": features.segments['count']
                }

                self.analysis_results.append(result_dict)

            self.progress["value"] += 1
            self.root.update_idletasks()

    def start_analysis_thread(self, folder_path=None):
        folder_path, audio_files = self.import_playlist(folder_path)
        if folder_path and audio_files:
            threading.Thread(target=self.analyze_playlist, args=(folder_path, audio_files), daemon=True).start()

    def play_selected_track(self):
        selected_item = self.playlist_tree.selection()
        if selected_item:
            # Get the selected file name from the Treeview
            file = self.playlist_tree.item(selected_item, "values")[0]

            if not hasattr(self, 'playlist_folder') or not self.playlist_folder:
                messagebox.showerror("Error", "No playlist folder selected. Please import a playlist first.")
                return

            track = os.path.join(self.playlist_folder, file)

            threading.Thread(target=AudioMixer.play_audio, args=(track, 3), daemon=True).start()

    def play_all_tracks(self):
        if not hasattr(self, 'playlist_folder') or not self.playlist_folder:
            messagebox.showerror("Error", "No playlist folder selected. Please import a playlist first.")
            return

        audio_files = [file for file in os.listdir(self.playlist_folder) if
                       file.lower().endswith(('.mp3', '.wav', '.WAV'))]
        if audio_files:
            threading.Thread(target=self._play_all_tracks, args=(audio_files,), daemon=True).start()

    def _play_all_tracks(self, audio_files):
        for file in audio_files:
            track = os.path.join(self.playlist_folder, file)
            AudioMixer.play_audio(track, fade_in_duration=3)
            time.sleep(2)

    def stop_audio(self):
        AudioMixer.stop_audio(fade_out_duration=2)

    def skip_track(self, position):
        AudioMixer.skip_to_position(float(position))

    def export_analysis_to_text(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            with open(file_path, "w") as f:
                for result in self.analysis_results:
                    f.write(f"{result}\n")
            messagebox.showinfo("Export Successful", f"Analysis saved to {file_path}")

    def export_analysis_to_csv(self):
        if not self.analysis_results:
            messagebox.showerror("Error", "No analysis results to export.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            with open(file_path, "w", newline="") as csvfile:
                fieldnames = ["File", "Artist", "Title", "BPM", "Key", "Energy", "Segments"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for result in self.analysis_results:
                    writer.writerow(result)
            messagebox.showinfo("Export Successful", f"Analysis saved to {file_path}")

    def visualize_selected_track(self):
        """Visualize the sections of the selected track"""
        selected_item = self.playlist_tree.selection()
        if not selected_item:
            messagebox.showerror("Error", "Please select a track to visualize")
            return

        # Get the selected file name from the Treeview
        file = self.playlist_tree.item(selected_item, "values")[0]

        # Check if the playlist folder is already set
        if not hasattr(self, 'playlist_folder') or not self.playlist_folder:
            messagebox.showerror("Error", "No playlist folder selected. Please import a playlist first.")
            return

        # Look up section data
        track_sections = None
        if self.section_data:
            for track in self.section_data:
                if track['file_name'] == file:
                    track_sections = track['sections']
                    break

        if not track_sections:
            messagebox.showerror("Error", "No section data available for this track.")
            messagebox.showinfo("Tip", "Run the beat_detection.py script first to analyze tracks.")
            return

        # Construct the full path to the selected track
        track_path = os.path.join(self.playlist_folder, file)

        # Create visualization
        self.visualize_track_sections(track_path, track_sections, file)

    def visualize_track_sections(self, track_path, sections, file_name):
        """Create a visualization window for the track sections"""
        # Create a new window for visualization
        viz_window = Toplevel(self.root)
        viz_window.title(f"Section Analysis: {file_name}")
        viz_window.geometry("1000x700")

        # Create matplotlib figure
        fig = plt.figure(figsize=(14, 10))

        # Load audio
        import librosa
        import numpy as np
        y, sr = librosa.load(track_path, sr=44100)

        # Plot waveform
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(y, sr=sr, alpha=0.6)
        plt.title(f"Waveform with Sections: {file_name}")

        # Add colored sections to waveform
        section_colors = {
            "intro": "#3498db",
            "buildup": "#f39c12",
            "drop": "#e74c3c",
            "breakdown": "#9b59b6",
            "verse": "#2ecc71",
            "outro": "#34495e"
        }

        y_min, y_max = plt.ylim()

        for section in sections:
            color = section_colors.get(section['type'], "#7f8c8d")
            plt.axvspan(section['start'], section['end'], alpha=0.3, color=color)
            middle_x = (section['start'] + section['end']) / 2
            plt.text(middle_x, y_max * 0.8, section['type'], horizontalalignment='center',
                     fontweight='bold', color='black', bbox=dict(facecolor='white', alpha=0.7))

        # Plot spectrogram
        plt.subplot(3, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogram with Sections")

        # Add colored sections to spectrogram
        for section in sections:
            color = section_colors.get(section['type'], "#7f8c8d")
            plt.axvspan(section['start'], section['end'], alpha=0.3, color=color)

        # Plot energy and bass content
        plt.subplot(3, 1, 3)

        # Get energy curve
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

        # Get bass content
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
        bass_band = spec_contrast[0]  # First band is lowest frequencies

        # Plot curves
        plt.plot(times, librosa.util.normalize(rms), label='Energy', color='blue', linewidth=2)
        plt.plot(times[:len(bass_band)], librosa.util.normalize(bass_band), label='Bass', color='red', linewidth=1)

        plt.title("Energy and Bass Content")
        plt.legend()

        # Add colored sections
        for section in sections:
            color = section_colors.get(section['type'], "#7f8c8d")
            plt.axvspan(section['start'], section['end'], alpha=0.3, color=color)

        plt.tight_layout()

        # Embed the figure in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=viz_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Add controls
        controls_frame = ttk.Frame(viz_window)
        controls_frame.pack(fill="x", padx=10, pady=5)

        # Add button to play specific sections
        def play_section(section_type):
            for section in sections:
                if section['type'] == section_type:
                    start_time = section['start']
                    end_time = section['end']
                    threading.Thread(
                        target=AudioMixer.play_section,
                        args=(track_path, start_time, end_time),
                        daemon=True
                    ).start()
                    break

        # Create buttons for each section type
        section_types = set(section['type'] for section in sections)
        for section_type in section_types:
            btn = ttk.Button(
                controls_frame,
                text=f"Play {section_type}",
                command=lambda s=section_type: play_section(s)
            )
            btn.pack(side="left", padx=5)

        # Add a stop button
        stop_btn = ttk.Button(controls_frame, text="Stop", command=AudioMixer.stop_audio)
        stop_btn.pack(side="left", padx=5)

    def run_section_analyzer(self):
        """Run the section analyzer script on the playlist"""
        if not hasattr(self, 'playlist_folder') or not self.playlist_folder:
            messagebox.showerror("Error", "No playlist folder selected. Please import a playlist first.")
            return

        response = messagebox.askyesno("Run Section Analyzer",
                                       "This will run the external section analyzer script on your playlist. Continue?")
        if not response:
            return

        try:
            import subprocess
            cmd = ["python", "beat_detection.py", self.playlist_folder, "--output", "results.json"]

            progress_dialog = Toplevel(self.root)
            progress_dialog.title("Running Section Analyzer")
            progress_dialog.geometry("400x150")
            progress_dialog.transient(self.root)
            progress_dialog.grab_set()

            message_label = ttk.Label(progress_dialog,
                                      text="Running section analysis...\nThis may take a while depending on your playlist size.")
            message_label.pack(pady=20)

            progress = ttk.Progressbar(progress_dialog, orient="horizontal", length=350, mode="indeterminate")
            progress.pack(pady=10)
            progress.start()

            # Function to run the process and update the UI
            def run_analyzer():
                try:
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = process.communicate()

                    if process.returncode == 0:
                        # Success - reload section data
                        try:
                            with open('results.json', 'r') as f:
                                self.section_data = json.load(f)

                            # Update section display in treeview
                            for item in self.playlist_tree.get_children():
                                file = self.playlist_tree.item(item, "values")[0]
                                for track_data in self.section_data:
                                    if track_data["file_name"] == file:
                                        section_counts = {}
                                        for section in track_data["sections"]:
                                            section_type = section["type"]
                                            section_counts[section_type] = section_counts.get(section_type, 0) + 1

                                        section_display = ", ".join([f"{k}: {v}" for k, v in section_counts.items()])
                                        values = list(self.playlist_tree.item(item, "values"))
                                        values[6] = section_display
                                        self.playlist_tree.item(item, values=values)
                                        break

                            messagebox.showinfo("Analysis Complete",
                                                f"Section analysis completed successfully.\nAnalyzed {len(self.section_data)} tracks.")
                        except (FileNotFoundError, json.JSONDecodeError) as e:
                            messagebox.showerror("Error", f"Could not load results file: {e}")
                    else:
                        messagebox.showerror("Error", f"Section analyzer failed with error:\n{stderr.decode()}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to run section analyzer: {e}")
                finally:
                    progress_dialog.destroy()

            # Run in separate thread to keep UI responsive
            threading.Thread(target=run_analyzer, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to run section analyzer: {e}")

    def run(self):
        self.root.mainloop()