import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import os
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import librosa
import soundfile as sf
import json

# Import my custom modules
from transition_point_detector import TransitionPointDetector
from audio_analyzer import AudioAnalyzer
from frequency_processor import FrequencyProcessor
from section_analyzer import SectionAnalyzer
import utils


class DJMixerApp:
    """
    Main app for my EDM DJ Mixer project.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("AI DJ Mixer for EDM")
        self.root.geometry("1000x800")

        # Initialize my components
        self.transition_detector = TransitionPointDetector()
        self.audio_analyzer = AudioAnalyzer()
        self.frequency_processor = FrequencyProcessor()
        self.section_analyzer = SectionAnalyzer()

        # Track storage
        self.tracks = []
        self.analyzed_tracks = {}
        self.current_visualization = None
        self.current_audio = None
        self.current_sample_rate = None

        # Set up the interface
        self.create_ui()

    def create_ui(self):
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.create_track_analysis_tab()
        self.create_transition_tab()
        self.create_mixing_tab()

        # Status bar at bottom
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_track_analysis_tab(self):
        track_tab = ttk.Frame(self.notebook)
        self.notebook.add(track_tab, text="Track Analysis")

        # Upper frame for buttons
        button_frame = tk.Frame(track_tab)
        button_frame.pack(fill=tk.X, pady=5)

        # Main buttons
        import_btn = tk.Button(button_frame, text="Import Tracks", command=self.import_tracks)
        import_btn.pack(side=tk.LEFT, padx=5)

        analyze_btn = tk.Button(button_frame, text="Analyze Tracks", command=self.analyze_tracks)
        analyze_btn.pack(side=tk.LEFT, padx=5)

        visualize_btn = tk.Button(button_frame, text="Visualize Track", command=self.visualize_selected_track)
        visualize_btn.pack(side=tk.LEFT, padx=5)

        export_btn = tk.Button(button_frame, text="Export Analysis", command=self.export_analysis)
        export_btn.pack(side=tk.LEFT, padx=5)

        # Split view
        paned_window = tk.PanedWindow(track_tab, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=10)

        # Left panel - Track list
        left_frame = tk.Frame(paned_window, width=300)
        paned_window.add(left_frame)

        tk.Label(left_frame, text="Imported Tracks:").pack(anchor=tk.W)

        self.track_listbox = tk.Listbox(left_frame, width=40, height=20)
        self.track_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right panel - Results and visualization
        right_frame = tk.Frame(paned_window, width=600)
        paned_window.add(right_frame)

        tk.Label(right_frame, text="Analysis Results:").pack(anchor=tk.W)
        self.output_text = scrolledtext.ScrolledText(right_frame, width=60, height=10)
        self.output_text.pack(fill=tk.X, padx=5, pady=5)

        # Visualization area
        self.visualization_frame = tk.Frame(right_frame, bg='white', height=400)
        self.visualization_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_transition_tab(self):
        transition_tab = ttk.Frame(self.notebook)
        self.notebook.add(transition_tab, text="Transition Points")

        # Button frame
        button_frame = tk.Frame(transition_tab)
        button_frame.pack(fill=tk.X, pady=5)

        transitions_btn = tk.Button(button_frame, text="Find All Transitions", command=self.find_transitions)
        transitions_btn.pack(side=tk.LEFT, padx=5)

        compare_btn = tk.Button(button_frame, text="Compare Selected Tracks", command=self.compare_selected_tracks)
        compare_btn.pack(side=tk.LEFT, padx=5)

        # Split view
        paned_window = tk.PanedWindow(transition_tab, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=10)

        # Left panel - Track selection
        left_frame = tk.Frame(paned_window, width=300)
        paned_window.add(left_frame)

        tk.Label(left_frame, text="Select Tracks:").pack(anchor=tk.W)

        self.transition_track_listbox = tk.Listbox(left_frame, width=40, height=20, selectmode=tk.MULTIPLE)
        self.transition_track_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right panel - Transition results
        right_frame = tk.Frame(paned_window, width=600)
        paned_window.add(right_frame)

        tk.Label(right_frame, text="Transition Analysis:").pack(anchor=tk.W)
        self.transition_text = scrolledtext.ScrolledText(right_frame, width=60, height=30)
        self.transition_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_mixing_tab(self):
        mixing_tab = ttk.Frame(self.notebook)
        self.notebook.add(mixing_tab, text="Mixing")

        # Button frame
        button_frame = tk.Frame(mixing_tab)
        button_frame.pack(fill=tk.X, pady=5)

        mix_btn = tk.Button(button_frame, text="Create Auto Mix", command=self.create_auto_mix)
        mix_btn.pack(side=tk.LEFT, padx=5)

        play_btn = tk.Button(button_frame, text="Play Preview", command=self.play_preview)
        play_btn.pack(side=tk.LEFT, padx=5)

        stop_btn = tk.Button(button_frame, text="Stop", command=self.stop_playback)
        stop_btn.pack(side=tk.LEFT, padx=5)

        export_mix_btn = tk.Button(button_frame, text="Export Mix", command=self.export_mix)
        export_mix_btn.pack(side=tk.LEFT, padx=5)

        # Mix sequence area
        lower_frame = tk.Frame(mixing_tab)
        lower_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        sequence_frame = tk.LabelFrame(lower_frame, text="Mix Sequence")
        sequence_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.mix_sequence_text = scrolledtext.ScrolledText(sequence_frame, width=80, height=20)
        self.mix_sequence_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def import_tracks(self):
        # Let user pick audio files
        filetypes = [("Audio files", "*.wav;*.mp3")]
        filenames = filedialog.askopenfilenames(title="Select audio files", filetypes=filetypes)

        if not filenames:
            return

        # Clear existing tracks
        self.tracks = []
        self.track_listbox.delete(0, tk.END)
        self.transition_track_listbox.delete(0, tk.END)

        # Add tracks to list
        for file in filenames:
            track_name = os.path.basename(file)
            self.tracks.append({
                'name': track_name,
                'path': file
            })
            self.track_listbox.insert(tk.END, track_name)
            self.transition_track_listbox.insert(tk.END, track_name)

        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, f"Imported {len(self.tracks)} tracks.\n")
        self.output_text.insert(tk.END, "Ready for analysis.\n")

        self.status_var.set(f"Imported {len(self.tracks)} tracks")

    def analyze_tracks(self):
        if not self.tracks:
            messagebox.showinfo("Info", "Please import tracks first.")
            return

        # Run in separate thread to keep UI responsive
        threading.Thread(target=self._analyze_tracks_thread, daemon=True).start()

    def _analyze_tracks_thread(self):
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Starting analysis...\n")
        self.status_var.set("Analyzing tracks...")

        # Reset previous results
        self.analyzed_tracks = {}

        for i, track in enumerate(self.tracks):
            self.output_text.insert(tk.END, f"Analyzing {track['name']}...\n")
            self.output_text.see(tk.END)

            try:
                # Analyze the track
                analysis = self.transition_detector.analyze_track(track['path'])

                # Store results
                self.analyzed_tracks[track['name']] = {
                    'path': track['path'],
                    'analysis': analysis
                }

                # Show results in UI
                self.output_text.insert(tk.END, f"✓ Completed: {track['name']}\n")
                self.output_text.insert(tk.END, f"   BPM: {analysis['bpm']:.1f}, Key: {analysis['key']}\n")
                self.output_text.insert(tk.END,
                                        f"   Found {len(analysis['transition_points']['entries'])} entry points and {len(analysis['transition_points']['exits'])} exit points\n")
                self.output_text.see(tk.END)

            except Exception as e:
                self.output_text.insert(tk.END, f"⚠ Error analyzing {track['name']}: {str(e)}\n")
                self.output_text.see(tk.END)

        self.output_text.insert(tk.END, "Analysis complete. Ready to find transitions.\n")
        self.output_text.see(tk.END)
        self.status_var.set("Analysis complete")

    def visualize_selected_track(self):
        # Get selected track
        selection = self.track_listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a track to visualize.")
            return

        selected_track = self.tracks[selection[0]]['name']

        if selected_track not in self.analyzed_tracks:
            messagebox.showinfo("Info", f"Track '{selected_track}' has not been analyzed yet.")
            return

        # Clear existing visualization
        for widget in self.visualization_frame.winfo_children():
            widget.destroy()

        # Get track data
        track_path = self.analyzed_tracks[selected_track]['path']
        track_analysis = self.analyzed_tracks[selected_track]['analysis']

        # Create figure
        fig = plt.Figure(figsize=(8, 6), dpi=100)

        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=self.visualization_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create subplots
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        # Try to load audio for waveform
        try:
            y, sr = librosa.load(track_path, sr=None, duration=60)  # Just load first minute for preview
            librosa.display.waveshow(y, sr=sr, ax=ax1)
            ax1.set_title(f"Waveform: {selected_track}")
        except Exception:
            ax1.set_title(f"Analysis of '{selected_track}'")
            ax1.text(0.5, 0.5, "Waveform not available", ha='center', va='center')

        # Show track details
        ax1.text(0.02, 0.9, f"BPM: {track_analysis['bpm']:.1f}", transform=ax1.transAxes)
        ax1.text(0.02, 0.8, f"Key: {track_analysis['key']} ({track_analysis['camelot_key']})", transform=ax1.transAxes)

        # Show sections and transition points
        ax2.set_title('Sections and Transition Points')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Section Type')
        ax2.set_xlim(0, track_analysis['duration'])
        ax2.set_ylim(0, 1)

        # Color-code the sections
        intro = track_analysis['sections']['intro']
        ax2.axvspan(intro['start'], intro['end'], alpha=0.2, color='green', label='Intro')

        for i, drop in enumerate(track_analysis['sections']['drops']):
            ax2.axvspan(drop['start'], drop['end'], alpha=0.3, color='red', label=f'Drop {i + 1}' if i == 0 else "")

        for i, breakdown in enumerate(track_analysis['sections']['breakdowns']):
            ax2.axvspan(breakdown['start'], breakdown['end'], alpha=0.3, color='blue',
                        label=f'Breakdown {i + 1}' if i == 0 else "")

        outro = track_analysis['sections']['outro']
        ax2.axvspan(outro['start'], outro['end'], alpha=0.2, color='purple', label='Outro')

        # Mark entry points
        for point in track_analysis['transition_points']['entries']:
            ax2.axvline(x=point['time'], linestyle='--', color='green', alpha=0.7)
            ax2.text(point['time'], 0.9, 'IN', horizontalalignment='center', verticalalignment='center',
                     transform=ax2.get_xaxis_transform())

        # Mark exit points
        for point in track_analysis['transition_points']['exits']:
            ax2.axvline(x=point['time'], linestyle='--', color='red', alpha=0.7)
            ax2.text(point['time'], 0.1, 'OUT', horizontalalignment='center', verticalalignment='center',
                     transform=ax2.get_xaxis_transform())

        ax2.legend(loc='upper right')

        fig.tight_layout()
        canvas.draw()

    def find_transitions(self):
        # Find transitions between all analyzed tracks
        if len(self.analyzed_tracks) < 2:
            messagebox.showinfo("Info", "Please analyze at least two tracks first.")
            return

        self.transition_text.delete(1.0, tk.END)
        self.transition_text.insert(tk.END, "Finding optimal transitions...\n\n")
        self.status_var.set("Finding transitions...")

        # Get all track names
        track_names = list(self.analyzed_tracks.keys())

        # Check all track pairs
        for i in range(len(track_names)):
            for j in range(len(track_names)):
                if i != j:  # Don't compare a track with itself
                    track1 = track_names[i]
                    track2 = track_names[j]

                    # Get analysis results
                    track1_analysis = self.analyzed_tracks[track1]['analysis']
                    track2_analysis = self.analyzed_tracks[track2]['analysis']

                    # Find optimal transition
                    transition = self.transition_detector.find_optimal_transition(track1_analysis, track2_analysis)

                    # Show results
                    self.transition_text.insert(tk.END, f"From '{track1}' to '{track2}':\n")
                    self.transition_text.insert(tk.END, f"  Score: {transition['score']:.2f}/1.0\n")
                    self.transition_text.insert(tk.END,
                                                f"  Exit: {transition['exit_point']['time']:.1f}s ({transition['exit_point']['type']})\n")
                    self.transition_text.insert(tk.END,
                                                f"  Entry: {transition['entry_point']['time']:.1f}s ({transition['entry_point']['type']})\n")
                    self.transition_text.insert(tk.END, f"  BPM ratio: {transition['bpm_ratio']:.2f}\n")
                    self.transition_text.insert(tk.END,
                                                f"  Component scores: Rhythm={transition['component_scores']['rhythm']:.2f}, " +
                                                f"Energy={transition['component_scores']['energy']:.2f}, " +
                                                f"Structure={transition['component_scores']['structure']:.2f}, " +
                                                f"Harmonic={transition['component_scores']['harmonic']:.2f}\n\n")
                    self.transition_text.see(tk.END)

        self.status_var.set("Transition analysis complete")

    def compare_selected_tracks(self):
        # Compare just two selected tracks
        selection = self.transition_track_listbox.curselection()
        if len(selection) != 2:
            messagebox.showinfo("Info", "Please select exactly two tracks to compare.")
            return

        track1_name = self.tracks[selection[0]]['name']
        track2_name = self.tracks[selection[1]]['name']

        if track1_name not in self.analyzed_tracks or track2_name not in self.analyzed_tracks:
            messagebox.showinfo("Info", "Both tracks must be analyzed first.")
            return

        # Get analysis results
        track1_analysis = self.analyzed_tracks[track1_name]['analysis']
        track2_analysis = self.analyzed_tracks[track2_name]['analysis']

        # Find optimal transition
        transition = self.transition_detector.find_optimal_transition(track1_analysis, track2_analysis)

        # Clear previous results
        self.transition_text.delete(1.0, tk.END)

        # Show detailed results with my DJ tips
        self.transition_text.insert(tk.END, f"TRANSITION ANALYSIS: '{track1_name}' to '{track2_name}'\n\n")
        self.transition_text.insert(tk.END, f"OVERALL SCORE: {transition['score']:.2f}/1.0\n\n")

        self.transition_text.insert(tk.END, "TRACK INFO:\n")
        self.transition_text.insert(tk.END, f"  Track 1: {track1_name}\n")
        self.transition_text.insert(tk.END, f"    BPM: {track1_analysis['bpm']:.1f}\n")
        self.transition_text.insert(tk.END, f"    Key: {track1_analysis['key']} ({track1_analysis['camelot_key']})\n")
        self.transition_text.insert(tk.END, f"    Energy: {track1_analysis['energy_profile']:.3f}\n\n")

        self.transition_text.insert(tk.END, f"  Track 2: {track2_name}\n")
        self.transition_text.insert(tk.END, f"    BPM: {track2_analysis['bpm']:.1f}\n")
        self.transition_text.insert(tk.END, f"    Key: {track2_analysis['key']} ({track2_analysis['camelot_key']})\n")
        self.transition_text.insert(tk.END, f"    Energy: {track2_analysis['energy_profile']:.3f}\n\n")

        self.transition_text.insert(tk.END, "TRANSITION DETAILS:\n")
        self.transition_text.insert(tk.END, f"  Exit point in '{track1_name}':\n")
        self.transition_text.insert(tk.END, f"    Time: {transition['exit_point']['time']:.1f} seconds\n")
        self.transition_text.insert(tk.END, f"    Type: {transition['exit_point']['type']}\n")
        self.transition_text.insert(tk.END, f"    Quality: {transition['exit_point']['quality']:.2f}\n\n")

        self.transition_text.insert(tk.END, f"  Entry point in '{track2_name}':\n")
        self.transition_text.insert(tk.END, f"    Time: {transition['entry_point']['time']:.1f} seconds\n")
        self.transition_text.insert(tk.END, f"    Type: {transition['entry_point']['type']}\n")
        self.transition_text.insert(tk.END, f"    Quality: {transition['entry_point']['quality']:.2f}\n\n")

        self.transition_text.insert(tk.END, "COMPATIBILITY SCORES:\n")
        self.transition_text.insert(tk.END,
                                    f"  Rhythmic compatibility: {transition['component_scores']['rhythm']:.2f}/1.0\n")
        self.transition_text.insert(tk.END,
                                    f"  Energy compatibility: {transition['component_scores']['energy']:.2f}/1.0\n")
        self.transition_text.insert(tk.END,
                                    f"  Structural compatibility: {transition['component_scores']['structure']:.2f}/1.0\n")
        self.transition_text.insert(tk.END,
                                    f"  Harmonic compatibility: {transition['component_scores']['harmonic']:.2f}/1.0\n\n")

        self.transition_text.insert(tk.END, "MIXING TIPS:\n")

        # Give tailored DJ advice
        if transition['component_scores']['rhythm'] < 0.6:
            self.transition_text.insert(tk.END, "  ⚠ BPM adjustment needed - tracks have different tempos\n")
            if track1_analysis['bpm'] > track2_analysis['bpm']:
                self.transition_text.insert(tk.END,
                                            f"  → Slow down track 1 from {track1_analysis['bpm']:.1f} to {track2_analysis['bpm']:.1f} BPM\n")
            else:
                self.transition_text.insert(tk.END,
                                            f"  → Speed up track 1 from {track1_analysis['bpm']:.1f} to {track2_analysis['bpm']:.1f} BPM\n")

        if transition['component_scores']['harmonic'] < 0.6:
            self.transition_text.insert(tk.END, "  ⚠ Keys are not fully compatible\n")
            self.transition_text.insert(tk.END, f"  → Use EQ to reduce melodic elements during transition\n")
            self.transition_text.insert(tk.END, f"  → Focus on percussive elements for smoother blending\n")

        # Recommend crossfade length based on section type
        crossfade_length = 8  # Default
        if "breakdown" in transition['exit_point']['type'] and "breakdown" in transition['entry_point']['type']:
            crossfade_length = 16
            self.transition_text.insert(tk.END,
                                        f"  → Use long crossfade ({crossfade_length} seconds) between breakdowns\n")
        elif "drop" in transition['exit_point']['type'] and "drop" in transition['entry_point']['type']:
            crossfade_length = 4
            self.transition_text.insert(tk.END, f"  → Use short crossfade ({crossfade_length} seconds) between drops\n")
        else:
            self.transition_text.insert(tk.END, f"  → Use medium crossfade ({crossfade_length} seconds)\n")

        self.transition_text.insert(tk.END,
                                    f"  → Apply EQ: Reduce low frequencies in track 1 while increasing in track 2\n")

        self.status_var.set("Transition analysis complete")

    def create_auto_mix(self):
        # Create a full automated DJ mix
        if len(self.analyzed_tracks) < 2:
            messagebox.showinfo("Info", "Please analyze at least two tracks first.")
            return

        self.mix_sequence_text.delete(1.0, tk.END)
        self.mix_sequence_text.insert(tk.END, "Generating automated mix sequence...\n\n")
        self.status_var.set("Creating mix sequence...")

        # Get all track names
        track_names = list(self.analyzed_tracks.keys())

        # Start with a random track
        import random
        current_track = random.choice(track_names)
        remaining_tracks = [t for t in track_names if t != current_track]

        # Build my mix sequence
        mix_sequence = [{
            'track': current_track,
            'path': self.analyzed_tracks[current_track]['path'],
            'exit_point': None,
            'next_track': None,
            'next_entry_point': None
        }]

        # Find best next track for each position
        while remaining_tracks:
            best_score = -1
            best_next_track = None
            best_transition = None

            for next_track in remaining_tracks:
                # Get analysis results
                current_analysis = self.analyzed_tracks[current_track]['analysis']
                next_analysis = self.analyzed_tracks[next_track]['analysis']

                # Find optimal transition
                transition = self.transition_detector.find_optimal_transition(current_analysis, next_analysis)

                if transition['score'] > best_score:
                    best_score = transition['score']
                    best_next_track = next_track
                    best_transition = transition

            # Update current track's exit point
            mix_sequence[-1]['exit_point'] = best_transition['exit_point']['time']
            mix_sequence[-1]['next_track'] = best_next_track
            mix_sequence[-1]['next_entry_point'] = best_transition['entry_point']['time']

            # Add next track to sequence
            mix_sequence.append({
                'track': best_next_track,
                'path': self.analyzed_tracks[best_next_track]['path'],
                'exit_point': None,
                'next_track': None,
                'next_entry_point': None
            })

            # Update for next iteration
            current_track = best_next_track
            remaining_tracks.remove(best_next_track)

        # Show the mix sequence
        self.mix_sequence_text.insert(tk.END, "AUTOMATED MIX SEQUENCE:\n\n")

        for i, item in enumerate(mix_sequence):
            self.mix_sequence_text.insert(tk.END, f"{i + 1}. {item['track']}\n")

            if item['exit_point'] is not None:
                self.mix_sequence_text.insert(tk.END, f"   Exit at: {item['exit_point']:.1f} seconds\n")
                self.mix_sequence_text.insert(tk.END, f"   Transition to: {item['next_track']}\n")
                self.mix_sequence_text.insert(tk.END, f"   Entry at: {item['next_entry_point']:.1f} seconds\n\n")

        # Store for later
        self.current_mix_sequence = mix_sequence

        self.status_var.set("Mix sequence created")

    def play_preview(self):
        # Play a preview of the mix
        if not hasattr(self, 'current_mix_sequence'):
            messagebox.showinfo("Info", "Please create a mix first.")
            return

        messagebox.showinfo("Info", "Play functionality would go here - I'll implement this in Sprint 5.")

    def stop_playback(self):
        # Stop playback
        messagebox.showinfo("Info", "Stop functionality would go here - I'll implement this in Sprint 5.")

    def export_mix(self):
        # Export the mix to a file
        if not hasattr(self, 'current_mix_sequence'):
            messagebox.showinfo("Info", "Please create a mix first.")
            return

        # Ask where to save
        output_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("MP3 files", "*.mp3"), ("All files", "*.*")]
        )

        if not output_path:
            return

        # Prepare transition information for export
        transitions = []
        for item in self.current_mix_sequence:
            if item['exit_point'] is not None:
                transitions.append({
                    'track_path': item['path'],
                    'exit_point': item['exit_point'],
                    'entry_point': 0,  # For first track
                    'crossfade_duration': 8.0  # Default
                })

        # Create and export the mix
        try:
            self.status_var.set("Creating mix file...")
            utils.create_mix_from_transitions(transitions, output_path)
            self.status_var.set(f"Mix exported to {output_path}")
            messagebox.showinfo("Success", f"Mix exported to {output_path}")
        except Exception as e:
            self.status_var.set("Error creating mix")
            messagebox.showerror("Error", f"Failed to create mix: {str(e)}")

    def export_analysis(self):
        # Export analysis results
        if not self.analyzed_tracks:
            messagebox.showinfo("Info", "No analysis results to export.")
            return

        # Ask for format
        format_choice = messagebox.askquestion("Export Format", "Export as JSON? (No for CSV)")

        if format_choice == 'yes':
            # JSON export
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if file_path:
                # Prepare data
                export_data = {}
                for track_name, track_info in self.analyzed_tracks.items():
                    export_data[track_name] = track_info['analysis']

                # Export
                utils.export_to_json(export_data, file_path)
                messagebox.showinfo("Success", f"Analysis exported to {file_path}")
        else:
            # CSV export
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if file_path:
                # Prepare data
                export_data = []
                for track_name, track_info in self.analyzed_tracks.items():
                    analysis = track_info['analysis']
                    export_data.append({
                        'file': track_name,
                        'bpm': analysis['bpm'],
                        'key': analysis['key'],
                        'energy': analysis['energy_profile']
                    })

                # Export
                utils.export_to_csv(export_data, file_path)
                messagebox.showinfo("Success", f"Analysis exported to {file_path}")