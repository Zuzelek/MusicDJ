import csv
import os
import threading
from datetime import time
from tkinter import filedialog, Scrollbar, messagebox, ttk, Scale
from tkinterdnd2 import TkinterDnD, DND_FILES
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
        self.analyzer = AudioAnalyzer()
        self.current_track_index = 0
        self.playlist_folder = None  # Store the playlist folder path
        self.create_gui()

    def create_gui(self):
        """Create GUI components."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.playlist_tree = ttk.Treeview(main_frame, columns=("File", "Artist", "Title", "BPM", "Key", "Energy"),
                                          show="headings")
        for col in ["File", "Artist", "Title", "BPM", "Key", "Energy"]:
            self.playlist_tree.heading(col, text=col)

        self.playlist_tree.pack(fill="both", expand=True, pady=10)

        scroll = Scrollbar(main_frame, orient="vertical", command=self.playlist_tree.yview)
        scroll.pack(side="right", fill="y")
        self.playlist_tree.configure(yscrollcommand=scroll.set)

        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=10, pady=5)

        self.progress = ttk.Progressbar(control_frame, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=10)

        volume_label = ttk.Label(control_frame, text="Volume:")
        volume_label.pack(side="left", padx=5)
        self.volume_slider = Scale(control_frame, from_=0.0, to=1.0, resolution=0.1, orient="horizontal",
                                   command=AudioMixer.set_volume)
        self.volume_slider.set(1.0)
        self.volume_slider.pack(side="left", padx=5)

        buttons = [
            ("Import Playlist & Analyze", self.start_analysis_thread),
            ("Play Selected Track", self.play_selected_track),
            ("Play All Tracks", self.play_all_tracks),
            ("Stop Audio", self.stop_audio),
            ("Export Analysis (Text)", self.export_analysis_to_text),
            ("Export Analysis (CSV)", self.export_analysis_to_csv),
        ]

        for text, command in buttons:
            button = ttk.Button(control_frame, text=text, command=command, width=25)
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
            folder_path = filedialog.askdirectory(title="Select Folder Containing MP3 Files")
        if folder_path:
            mp3_files = [file for file in os.listdir(folder_path) if file.endswith('.mp3')]
            if mp3_files:
                return folder_path, mp3_files
            else:
                messagebox.showerror("Error", "No MP3 files found.")
        return None, None

    def analyze_playlist(self, folder_path, mp3_files):
        """Analyzes audio files and updates the GUI."""
        self.playlist_tree.delete(*self.playlist_tree.get_children())
        self.analysis_results = []
        self.progress["maximum"] = len(mp3_files)
        self.progress["value"] = 0

        self.playlist_folder = folder_path

        for i, file in enumerate(mp3_files):
            file_path = os.path.join(folder_path, file)
            features = self.analyzer.analyze_audio(file_path)

            if features:
                artist, title = file.split(' - ') if ' - ' in file else ('Unknown', 'Unknown')
                self.playlist_tree.insert("", "end", values=(file, artist, title, f"{features.tempo:.1f}",
                                                              features.key, f"{features.energy:.2f}"))
                self.analysis_results.append({
                    "File": file, "Artist": artist, "Title": title,
                    "BPM": features.tempo, "Key": features.key, "Energy": features.energy,
                    "Segments": features.segments['count']
                })

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
