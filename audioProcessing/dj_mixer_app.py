import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import os
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
from transition_point_detector import TransitionPointDetector
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

        self.transition_detector = TransitionPointDetector()
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
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.create_track_analysis_tab()
        self.create_transition_tab()
        self.create_mixing_tab()

        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_track_analysis_tab(self):
        track_tab = ttk.Frame(self.notebook)
        self.notebook.add(track_tab, text="Track Analysis")

        button_frame = tk.Frame(track_tab)
        button_frame.pack(fill=tk.X, pady=5)

        import_btn = tk.Button(button_frame, text="Import Tracks", command=self.import_tracks)
        import_btn.pack(side=tk.LEFT, padx=5)

        analyze_btn = tk.Button(button_frame, text="Analyze Tracks", command=self.analyze_tracks)
        analyze_btn.pack(side=tk.LEFT, padx=5)

        visualize_btn = tk.Button(button_frame, text="Visualize Track", command=self.visualize_selected_track)
        visualize_btn.pack(side=tk.LEFT, padx=5)

        export_btn = tk.Button(button_frame, text="Export Analysis", command=self.export_analysis)
        export_btn.pack(side=tk.LEFT, padx=5)

        paned_window = tk.PanedWindow(track_tab, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=10)

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

        self.visualization_frame = tk.Frame(right_frame, bg='white', height=400)
        self.visualization_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_transition_tab(self):
        transition_tab = ttk.Frame(self.notebook)
        self.notebook.add(transition_tab, text="Transition Points")

        button_frame = tk.Frame(transition_tab)
        button_frame.pack(fill=tk.X, pady=5)

        transitions_btn = tk.Button(button_frame, text="Find All Transitions", command=self.find_transitions)
        transitions_btn.pack(side=tk.LEFT, padx=5)

        paned_window = tk.PanedWindow(transition_tab, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=10)

        left_frame = tk.Frame(paned_window, width=300)
        paned_window.add(left_frame)

        tk.Label(left_frame, text="Select Tracks:").pack(anchor=tk.W)

        self.transition_track_listbox = tk.Listbox(left_frame, width=40, height=20, selectmode=tk.MULTIPLE)
        self.transition_track_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        right_frame = tk.Frame(paned_window, width=600)
        paned_window.add(right_frame)

        tk.Label(right_frame, text="Transition Analysis:").pack(anchor=tk.W)
        self.transition_text = scrolledtext.ScrolledText(right_frame, width=60, height=30)
        self.transition_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_mixing_tab(self):
        mixing_tab = ttk.Frame(self.notebook)
        self.notebook.add(mixing_tab, text="Mixing")

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

        lower_frame = tk.Frame(mixing_tab)
        lower_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        sequence_frame = tk.LabelFrame(lower_frame, text="Mix Sequence")
        sequence_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.mix_sequence_text = scrolledtext.ScrolledText(sequence_frame, width=80, height=20)
        self.mix_sequence_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def import_tracks(self):
        filetypes = [("Audio files", "*.wav;*.mp3")]
        filenames = filedialog.askopenfilenames(title="Select audio files", filetypes=filetypes)

        if not filenames:
            return

        self.tracks = []
        self.track_listbox.delete(0, tk.END)
        self.transition_track_listbox.delete(0, tk.END)

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

                # Check for vocals
                vocal_segments = analysis.get('vocal_segments', [])
                num_vocals = len(vocal_segments)
                vocal_status = "No vocals detected"
                if num_vocals > 0:
                    total_vocal_duration = sum(end - start for start, end in vocal_segments)
                    vocal_status = f"Found {num_vocals} vocal segments ({total_vocal_duration:.1f}s total)"

                # Show results in UI
                self.output_text.insert(tk.END, f"✓ Completed: {track['name']}\n")
                self.output_text.insert(tk.END, f"   BPM: {analysis['bpm']:.1f}, Key: {analysis['key']}\n")
                self.output_text.insert(tk.END,
                                        f"   Found {len(analysis['transition_points']['entries'])} entry points and {len(analysis['transition_points']['exits'])} exit points\n")
                self.output_text.insert(tk.END,
                                        f"   Vocals: {vocal_status}\n")
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

        for widget in self.visualization_frame.winfo_children():
            widget.destroy()

        track_path = self.analyzed_tracks[selected_track]['path']
        track_analysis = self.analyzed_tracks[selected_track]['analysis']

        fig = plt.Figure(figsize=(8, 6), dpi=100)

        canvas = FigureCanvasTkAgg(fig, master=self.visualization_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        try:
            y, sr = librosa.load(track_path, sr=None, duration=60)  # Just load first minute for preview
            librosa.display.waveshow(y, sr=sr, ax=ax1)
            ax1.set_title(f"Waveform: {selected_track}")
        except Exception:
            ax1.set_title(f"Analysis of '{selected_track}'")
            ax1.text(0.5, 0.5, "Waveform not available", ha='center', va='center')

        ax1.text(0.02, 0.9, f"BPM: {track_analysis['bpm']:.1f}", transform=ax1.transAxes)
        ax1.text(0.02, 0.8, f"Key: {track_analysis['key']} ({track_analysis['camelot_key']})", transform=ax1.transAxes)

        ax2.set_title('Sections and Transition Points')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Section Type')
        ax2.set_xlim(0, track_analysis['duration'])
        ax2.set_ylim(0, 1)

        intro = track_analysis['sections']['intro']
        ax2.axvspan(intro['start'], intro['end'], alpha=0.2, color='green', label='Intro')

        for i, drop in enumerate(track_analysis['sections']['drops']):
            ax2.axvspan(drop['start'], drop['end'], alpha=0.3, color='red', label=f'Drop {i + 1}' if i == 0 else "")

        for i, breakdown in enumerate(track_analysis['sections']['breakdowns']):
            ax2.axvspan(breakdown['start'], breakdown['end'], alpha=0.3, color='blue',
                        label=f'Breakdown {i + 1}' if i == 0 else "")

        outro = track_analysis['sections']['outro']
        ax2.axvspan(outro['start'], outro['end'], alpha=0.2, color='purple', label='Outro')

        for point in track_analysis['transition_points']['entries']:
            ax2.axvline(x=point['time'], linestyle='--', color='green', alpha=0.7)
            ax2.text(point['time'], 0.9, 'IN', horizontalalignment='center', verticalalignment='center',
                     transform=ax2.get_xaxis_transform())

        for point in track_analysis['transition_points']['exits']:
            ax2.axvline(x=point['time'], linestyle='--', color='red', alpha=0.7)
            ax2.text(point['time'], 0.1, 'OUT', horizontalalignment='center', verticalalignment='center',
                     transform=ax2.get_xaxis_transform())

        ax2.legend(loc='upper right')

        fig.tight_layout()
        canvas.draw()

    def find_transitions(self):
        if len(self.analyzed_tracks) < 2:
            messagebox.showinfo("Info", "Please analyze at least two tracks first.")
            return

        self.transition_text.delete(1.0, tk.END)
        self.transition_text.insert(tk.END, "Finding optimal transitions...\n\n")
        self.status_var.set("Finding transitions...")

        track_names = list(self.analyzed_tracks.keys())

        for i in range(len(track_names)):
            for j in range(len(track_names)):
                if i != j:
                    track1 = track_names[i]
                    track2 = track_names[j]

                    track1_analysis = self.analyzed_tracks[track1]['analysis']
                    track2_analysis = self.analyzed_tracks[track2]['analysis']

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

    def create_auto_mix(self):
        # Check for enough tracks
        if len(self.analyzed_tracks) < 2:
            messagebox.showinfo("Info", "Please analyze at least two tracks first.")
            return

        self.mix_sequence_text.delete(1.0, tk.END)
        self.mix_sequence_text.insert(tk.END, "Generating automated mix sequence...\n\n")
        self.status_var.set("Creating mix sequence...")

        # Get all track names
        track_names = list(self.analyzed_tracks.keys())

        # Start with best track (detect if it has vocals at the start)
        first_track_scores = {}
        for track in track_names:
            analysis = self.analyzed_tracks[track]['analysis']

            # Check for vocals in the intro
            has_vocals_in_intro = False
            if 'vocal_segments' in analysis:
                for start, end in analysis['vocal_segments']:
                    if start < analysis['sections']['intro']['end']:
                        has_vocals_in_intro = True
                        break

            intro_score = 1.0
            if has_vocals_in_intro:
                intro_score *= 0.7

            # Prefer tracks with defined intros
            intro_length = analysis['sections']['intro']['end']
            if intro_length > 20:
                intro_score *= 1.3

            # Prefer tracks in common BPM ranges
            bpm = analysis['bpm']
            bpm_score = 1.0
            if 120 <= bpm <= 130:  # Ideal house range
                bpm_score = 1.2
            elif 170 <= bpm <= 180:  # DnB range
                bpm_score = 1.1

            first_track_scores[track] = intro_score * bpm_score

        # Choose best starting track
        current_track = max(first_track_scores, key=first_track_scores.get)
        remaining_tracks = [t for t in track_names if t != current_track]

        # Build the mix sequence
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

                # Check for vocal clash
                vocal_compatibility = 1.0  # Default good compatibility
                if 'component_scores' in transition and 'vocal' in transition['component_scores']:
                    vocal_compatibility = transition['component_scores']['vocal']

                # Adjust score based on vocal compatibility
                adjusted_score = transition['score']
                if vocal_compatibility < 0.7:  # Significant vocal clash
                    adjusted_score *= 0.8  # Reduce score

                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_next_track = next_track
                    best_transition = transition

            # Update the mix sequence
            mix_sequence[-1]['exit_point'] = best_transition['exit_point']['time']
            mix_sequence[-1]['next_track'] = best_next_track
            mix_sequence[-1]['next_entry_point'] = best_transition['entry_point']['time']
            mix_sequence[-1]['crossfade_duration'] = best_transition.get('recommended_crossfade', 12.0)

            # Check for vocal clashes to display warning
            has_vocal_clash = False
            if 'component_scores' in best_transition and 'vocal' in best_transition['component_scores']:
                has_vocal_clash = best_transition['component_scores']['vocal'] < 0.7

            mix_sequence[-1]['has_vocal_clash'] = has_vocal_clash

            mix_sequence.append({
                'track': best_next_track,
                'path': self.analyzed_tracks[best_next_track]['path'],
                'exit_point': None,
                'next_track': None,
                'next_entry_point': None
            })

            current_track = best_next_track
            remaining_tracks.remove(best_next_track)

        self.mix_sequence_text.insert(tk.END, "AUTOMATED MIX SEQUENCE:\n\n")

        for i, item in enumerate(mix_sequence):
            self.mix_sequence_text.insert(tk.END, f"{i + 1}. {item['track']}\n")

            if item['exit_point'] is not None:
                self.mix_sequence_text.insert(tk.END, f"   Exit at: {item['exit_point']:.1f} seconds\n")
                self.mix_sequence_text.insert(tk.END, f"   Transition to: {item['next_track']}\n")
                self.mix_sequence_text.insert(tk.END, f"   Entry at: {item['next_entry_point']:.1f} seconds\n")

                crossfade = item.get('crossfade_duration', 12.0)
                self.mix_sequence_text.insert(tk.END, f"   Crossfade: {crossfade:.1f} seconds\n")

                if item.get('has_vocal_clash', False):
                    self.mix_sequence_text.insert(tk.END, f"   ⚠️ Note: Potential vocal clash detected\n")

                self.mix_sequence_text.insert(tk.END, "\n")

        self.current_mix_sequence = mix_sequence
        self.status_var.set("Mix sequence created")

    def play_preview(self):
        if not hasattr(self, 'current_mix_sequence'):
            messagebox.showinfo("Info", "Please create a mix first.")
            return

        messagebox.showinfo("Info", "Play functionality would go here - I'll implement this in Sprint 5.")

    def stop_playback(self):

        messagebox.showinfo("Info", "Stop functionality would go here - I'll implement this in Sprint 5.")

    def export_mix(self):
        if not hasattr(self, 'current_mix_sequence'):
            messagebox.showinfo("Info", "Please create a mix first.")
            return

        output_path = filedialog.asksaveasfilename(
            title="Save Mix As",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("MP3 files", "*.mp3"), ("All files", "*.*")]
        )

        if not output_path:
            return

        # Clear any previous statu
        self.status_var.set("Preparing to create mix...")
        self.root.update()

        transitions = []

        self.mix_sequence_text.delete(1.0, tk.END)
        self.mix_sequence_text.insert(tk.END, "Creating mix with the following tracks:\n\n")

        for i, item in enumerate(self.current_mix_sequence):
            if item['track'] not in self.analyzed_tracks:
                messagebox.showerror("Error", f"Track '{item['track']}' has not been analyzed.")
                return

            track_analysis = self.analyzed_tracks[item['track']]['analysis']
            track_info_str = f"{i + 1}. {item['track']}"

            if i < len(self.current_mix_sequence) - 1:
                track_info_str += f"\n   Exit at: {item['exit_point']:.1f} seconds"
                track_info_str += f"\n   Transition to: {self.current_mix_sequence[i + 1]['track']}"

                # Add vocal segments if available
                vocal_segments = track_analysis.get('vocal_segments', [])

                transition_info = {
                    'track_path': item['path'],
                    'exit_point': item['exit_point'],
                    'bpm': track_analysis['bpm'],
                    'key': track_analysis.get('key', 'Unknown'),
                    'crossfade_duration': item.get('crossfade_duration', 16.0),
                    'vocal_segments': vocal_segments  # Add vocal segments
                }
            else:
                transition_info = {
                    'track_path': item['path'],
                    'exit_point': None,
                    'bpm': track_analysis['bpm'],
                    'key': track_analysis.get('key', 'Unknown'),
                    'vocal_segments': track_analysis.get('vocal_segments', [])
                }

            transitions.append(transition_info)

            self.mix_sequence_text.insert(tk.END, f"{track_info_str}\n\n")

        self.mix_sequence_text.insert(tk.END, "Analyzing vocals for enhanced transitions...\n")
        self.status_var.set("Analyzing vocals...")
        self.root.update()

        try:
            enhanced_transitions = utils.analyze_and_enhance_transitions(transitions)
            self.mix_sequence_text.insert(tk.END, "Vocal analysis complete.\n\n")
        except Exception as e:
            print(f"Error in vocal analysis: {e}")
            enhanced_transitions = transitions

        try:
            self.status_var.set("Creating mix file...")
            self.root.update()

            utils.create_robust_mix(enhanced_transitions, output_path, min_crossfade=12.0)

            self.status_var.set(f"Mix exported to {output_path}")
            self.mix_sequence_text.insert(tk.END, f"Mix successfully exported to:\n{output_path}\n")
            messagebox.showinfo("Success", f"Mix exported to {output_path}")
        except Exception as e:
            error_msg = f"Failed to create mix: {str(e)}"
            print(error_msg)
            self.status_var.set("Error creating mix")
            self.mix_sequence_text.insert(tk.END, f"ERROR: {error_msg}\n")
            messagebox.showerror("Error", error_msg)

    def export_analysis(self):
        if not self.analyzed_tracks:
            messagebox.showinfo("Info", "No analysis results to export.")
            return

        format_choice = messagebox.askquestion("Export Format", "Export as JSON? (No for CSV)")

        if format_choice == 'yes':
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if file_path:
                export_data = {}
                for track_name, track_info in self.analyzed_tracks.items():
                    export_data[track_name] = track_info['analysis']

                utils.export_to_json(export_data, file_path)
                messagebox.showinfo("Success", f"Analysis exported to {file_path}")
        else:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if file_path:
                export_data = []
                for track_name, track_info in self.analyzed_tracks.items():
                    analysis = track_info['analysis']
                    export_data.append({
                        'file': track_name,
                        'bpm': analysis['bpm'],
                        'key': analysis['key'],
                        'energy': analysis['energy_profile']
                    })

                utils.export_to_csv(export_data, file_path)
                messagebox.showinfo("Success", f"Analysis exported to {file_path}")