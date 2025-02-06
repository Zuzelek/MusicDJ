import os
import csv
import threading
from datetime import time
from tkinter import Tk, filedialog, Text, Button, Scrollbar, messagebox, ttk, Scale
from tkinterdnd2 import TkinterDnD, DND_FILES
from audio_analyzer import AudioAnalyzer, AudioFeatures
from audio_mixer import AudioMixer


class MusicDJApp:
    """Class for the main GUI application."""

    def __init__(self):
        self.root = TkinterDnD.Tk()
        self.root.title("Music DJ Application")
        self.root.geometry("1000x800")

        self.playlist_tree = None
        self.progress = None
        self.playlist = None
        self.analysis_results = []
        self.analyzer = AudioAnalyzer()
        self.current_track_index = 0
        self.playlist_folder = None  # Store the playlist folder path
        self.create_gui()

    def create_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Playlist display (Treeview)
        self.playlist_tree = ttk.Treeview(main_frame, columns=("File", "Artist", "Title", "BPM", "Key", "Energy"),
                                          show="headings")
        self.playlist_tree.heading("File", text="File")
        self.playlist_tree.heading("Artist", text="Artist")
        self.playlist_tree.heading("Title", text="Title")
        self.playlist_tree.heading("BPM", text="BPM")
        self.playlist_tree.heading("Key", text="Key")
        self.playlist_tree.heading("Energy", text="Energy")
        self.playlist_tree.pack(fill="both", expand=True, pady=10)

        # Scrollbar for playlist
        scroll = Scrollbar(main_frame, orient="vertical", command=self.playlist_tree.yview)
        scroll.pack(side="right", fill="y")
        self.playlist_tree.configure(yscrollcommand=scroll.set)

        # Control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=10, pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(control_frame, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=10)

        # Volume control
        volume_label = ttk.Label(control_frame, text="Volume:")
        volume_label.pack(side="left", padx=5)
        self.volume_slider = Scale(control_frame, from_=0.0, to=1.0, resolution=0.1, orient="horizontal",
                                   command=AudioMixer.set_volume)
        self.volume_slider.set(1.0)
        self.volume_slider.pack(side="left", padx=5)

        # Skip slider
        skip_label = ttk.Label(control_frame, text="Skip (s):")
        skip_label.pack(side="left", padx=5)
        self.skip_slider = Scale(control_frame, from_=0, to=600, resolution=1, orient="horizontal",
                                 command=self.skip_track)
        self.skip_slider.pack(side="left", padx=5)

        # Buttons
        buttons = [
            ("Import Playlist & Analyze", self.start_analysis_thread),
            ("Play Selected Track", self.play_selected_track),
            ("Play All Tracks", self.play_all_tracks),
            ("Stop Audio", self.stop_audio),
            ("Clear Playlist", self.clear_playlist),
            ("Export Analysis (Text)", self.export_analysis_to_text),
            ("Export Analysis (CSV)", self.export_analysis_to_csv),
        ]

        for text, command in buttons:
            button = ttk.Button(control_frame, text=text, command=command, width=25)
            button.pack(side="left", padx=5, pady=5)

        # Drag-and-drop support
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.handle_drop)

    def handle_drop(self, event):
        """Handle drag-and-drop events"""
        folder_path = event.data.strip('{}')
        if os.path.isdir(folder_path):
            self.start_analysis_thread(folder_path)
        else:
            messagebox.showerror("Error", "Please drop a folder containing MP3 files.")

    def import_playlist(self, folder_path=None):
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
        self.playlist_tree.delete(*self.playlist_tree.get_children())  # Clear previous entries
        self.analysis_results = []
        self.progress["maximum"] = len(mp3_files)
        self.progress["value"] = 0

        self.playlist_folder = folder_path

        for i, file in enumerate(mp3_files):
            file_path = os.path.join(folder_path, file)
            features = self.analyzer.analyze_audio(file_path)

            if features is not None:
                try:
                    artist_title = file.split(' - ')
                    if len(artist_title) == 2:
                        artist, title = artist_title
                    else:
                        artist, title = 'Unknown', 'Unknown'

                    self.playlist_tree.insert("", "end", values=(
                        file, artist, title, f"{features.tempo:.1f}", features.key, f"{features.energy:.2f}"
                    ))

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
                    print(f"Error analyzing file {file_path}: {e}")
            else:
                print(f"Error processing file: {file}")

            self.progress["value"] += 1
            self.root.update_idletasks()

    def start_analysis_thread(self, folder_path=None):
        folder_path, mp3_files = self.import_playlist(folder_path)
        if folder_path and mp3_files:
            threading.Thread(target=self.analyze_playlist, args=(folder_path, mp3_files), daemon=True).start()

    def play_selected_track(self):
        selected_item = self.playlist_tree.selection()
        if selected_item:
            # Get the selected file name from the Treeview
            file = self.playlist_tree.item(selected_item, "values")[0]

            # Check if the playlist folder is already set
            if not hasattr(self, 'playlist_folder') or not self.playlist_folder:
                messagebox.showerror("Error", "No playlist folder selected. Please import a playlist first.")
                return

            # Construct the full path to the selected track
            track = os.path.join(self.playlist_folder, file)

            # Play the selected track in a separate thread
            threading.Thread(target=AudioMixer.play_audio, args=(track, 3), daemon=True).start()

    def play_all_tracks(self):
        if not hasattr(self, 'playlist_folder') or not self.playlist_folder:
            messagebox.showerror("Error", "No playlist folder selected. Please import a playlist first.")
            return

        mp3_files = [file for file in os.listdir(self.playlist_folder) if file.endswith('.mp3')]
        if mp3_files:
            threading.Thread(target=self._play_all_tracks, args=(mp3_files,), daemon=True).start()

    def _play_all_tracks(self, mp3_files):
        for file in mp3_files:
            track = os.path.join(self.playlist_folder, file)
            AudioMixer.play_audio(track, fade_in_duration=3)
            time.sleep(2)  # Wait for fade-out before starting the next track

    def stop_audio(self):
        AudioMixer.stop_audio(fade_out_duration=2)

    def skip_track(self, position):
        AudioMixer.skip_to_position(float(position))

    def clear_playlist(self):
        self.playlist_tree.delete(*self.playlist_tree.get_children())
        self.analysis_results = []
        self.progress["value"] = 0
        self.playlist_folder = None  # Reset the playlist folder
        messagebox.showinfo("Info", "Playlist and analysis results cleared.")

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
                writer.writerows(self.analysis_results)
            messagebox.showinfo("Export Successful", f"Analysis saved to {file_path}")

    def run(self):
        self.root.mainloop()