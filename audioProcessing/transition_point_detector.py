import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, savgol_filter
from sklearn.preprocessing import MinMaxScaler


class TransitionPointDetector:
    """
    My AI-driven transition point detection for EDM tracks.
    """

    def __init__(self):
        # Parameters I selected after testing
        self.MIN_SECTION_LENGTH = 4
        self.DROP_ENERGY_RATIO = 1.5
        self.BREAKDOWN_ENERGY_RATIO = 0.6
        self.INTRO_LENGTH_RATIO = 0.15
        self.OUTRO_LENGTH_RATIO = 0.15

        # Weights for scoring potential transitions
        self.SCORE_WEIGHTS = {
            'rhythmic_stability': 0.3,
            'energy_compatibility': 0.25,
            'structural_position': 0.25,
            'harmonic_compatibility': 0.2
        }

        # Camelot wheel mapping for harmonic mixing
        self.camelot_wheel = {
            "C Major": "8B", "A Minor": "8A",
            "G Major": "9B", "E Minor": "9A",
            "D Major": "10B", "B Minor": "10A",
            "A Major": "11B", "F# Minor": "11A",
            "E Major": "12B", "C# Minor": "12A",
            "B Major": "1B", "G# Minor": "1A",
            "F# Major": "2B", "D# Minor": "2A",
            "C# Major": "3B", "A# Minor": "3A",
            "F Major": "4B", "D Minor": "4A",
            "Bb Major": "5B", "G Minor": "5A",
            "Eb Major": "6B", "C Minor": "6A",
            "Ab Major": "7B", "F Minor": "7A",
            "Db Major": "8B", "Bb Minor": "8A"
        }

    def analyze_track(self, audio_path):
        # Load audio file
        y, sr = librosa.load(audio_path, sr=44100)

        # Get track duration
        duration = librosa.get_duration(y=y, sr=sr)

        # Extract basic features
        bpm = self._get_tempo(y, sr)
        key = self._get_key(y, sr)
        energy_profile = self._get_energy_profile(y)

        # Extract dynamic features
        energy_curve = self._get_energy_curve(y)

        # Smooth the energy curve for better section detection
        energy_curve_smooth = savgol_filter(energy_curve,
                                            int(sr / 512 * 8) if len(energy_curve) > sr / 512 * 8 else 3,
                                            2)

        # Normalize energy curve for easier analysis
        scaler = MinMaxScaler()
        energy_curve_norm = scaler.fit_transform(energy_curve_smooth.reshape(-1, 1)).flatten()

        # Detect musical sections
        sections = self._detect_sections(energy_curve_norm, sr, duration)

        # Find potential transition points
        transition_points = self._detect_transition_points(sections, energy_curve_norm, sr)

        # Package results
        return {
            'duration': duration,
            'bpm': bpm,
            'key': key,
            'camelot_key': self._get_camelot_key(key),
            'energy_profile': float(energy_profile),
            'sections': sections,
            'transition_points': transition_points
        }

    def find_optimal_transition(self, track1_analysis, track2_analysis):
        # Check harmonic compatibility
        harmonic_score = self._harmonic_compatibility_score(
            track1_analysis['camelot_key'],
            track2_analysis['camelot_key']
        )

        # Get all potential exit points from track1
        exit_points = track1_analysis['transition_points']['exits']

        # Get all potential entry points to track2
        entry_points = track2_analysis['transition_points']['entries']

        # Find the best transition combination
        best_score = -1
        best_transition = None

        for exit_point in exit_points:
            for entry_point in entry_points:
                # Calculate transition score components
                rhythm_score = self._rhythm_compatibility_score(
                    track1_analysis['bpm'],
                    track2_analysis['bpm']
                )

                energy_score = self._energy_compatibility_score(
                    exit_point['energy'],
                    entry_point['energy']
                )

                structure_score = self._structure_compatibility_score(
                    exit_point['type'],
                    entry_point['type']
                )

                # Weighted total score
                total_score = (
                        self.SCORE_WEIGHTS['rhythmic_stability'] * rhythm_score +
                        self.SCORE_WEIGHTS['energy_compatibility'] * energy_score +
                        self.SCORE_WEIGHTS['structural_position'] * structure_score +
                        self.SCORE_WEIGHTS['harmonic_compatibility'] * harmonic_score
                )

                if total_score > best_score:
                    best_score = total_score
                    best_transition = {
                        'exit_point': exit_point,
                        'entry_point': entry_point,
                        'score': total_score,
                        'bpm_ratio': track1_analysis['bpm'] / track2_analysis['bpm'],
                        'harmonic_compatibility': harmonic_score,
                        'component_scores': {
                            'rhythm': rhythm_score,
                            'energy': energy_score,
                            'structure': structure_score,
                            'harmonic': harmonic_score
                        }
                    }

        return best_transition

    def _get_tempo(self, y, sr):
        # Extract tempo (BPM) from audio signal
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        return float(tempo)

    def _get_key(self, y, sr):
        # Extract musical key from audio signal
        y_harmonic = librosa.effects.harmonic(y)
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

        # Placeholder - will need to integrate with my key detection function
        key = "C Major"
        return key

    def _get_camelot_key(self, key):
        return self.camelot_wheel.get(key, "8B")  # Default if not found

    def _get_energy_profile(self, y):
        return np.mean(librosa.feature.rms(y=y))

    def _get_energy_curve(self, y):
        return librosa.feature.rms(y=y).flatten()

    def _detect_sections(self, energy_curve, sr, duration):
        # Convert curve from frames to seconds
        time_axis = np.linspace(0, duration, len(energy_curve))

        # Calculate average energy
        avg_energy = np.mean(energy_curve)

        # Find potential drops (high energy segments)
        drop_threshold = avg_energy * self.DROP_ENERGY_RATIO
        drop_candidates = np.where(energy_curve > drop_threshold)[0]

        # Find potential breakdowns (low energy segments)
        breakdown_threshold = avg_energy * self.BREAKDOWN_ENERGY_RATIO
        breakdown_candidates = np.where(energy_curve < breakdown_threshold)[0]

        # Detect drops (group nearby frames)
        drops = self._group_frames_to_sections(drop_candidates, time_axis)

        # Detect breakdowns
        breakdowns = self._group_frames_to_sections(breakdown_candidates, time_axis)

        # Estimate intro based on duration
        intro_end = duration * self.INTRO_LENGTH_RATIO

        # Estimate outro based on duration
        outro_start = duration * (1 - self.OUTRO_LENGTH_RATIO)

        # Compile results
        return {
            'intro': {
                'start': 0,
                'end': intro_end
            },
            'drops': drops,
            'breakdowns': breakdowns,
            'outro': {
                'start': outro_start,
                'end': duration
            }
        }

    def _group_frames_to_sections(self, candidates, time_axis):
        if len(candidates) == 0:
            return []

        sections = []
        section_start = candidates[0]

        for i in range(1, len(candidates)):
            # If there's a gap, end the current section
            if candidates[i] - candidates[i - 1] > 1:
                # Only add sections longer than minimum length
                section_length = time_axis[candidates[i - 1]] - time_axis[section_start]
                if section_length >= self.MIN_SECTION_LENGTH:
                    sections.append({
                        'start': float(time_axis[section_start]),
                        'end': float(time_axis[candidates[i - 1]]),
                        'energy': float(np.mean(time_axis[section_start:candidates[i - 1]]))
                    })
                section_start = candidates[i]

        # Add the last section if it meets the criteria
        section_length = time_axis[-1] if len(candidates) == len(time_axis) else time_axis[candidates[-1]] - time_axis[
            section_start]
        if section_length >= self.MIN_SECTION_LENGTH:
            sections.append({
                'start': float(time_axis[section_start]),
                'end': float(time_axis[candidates[-1]]),
                'energy': float(np.mean(time_axis[section_start:candidates[-1]]))
            })

        return sections

    def _detect_transition_points(self, sections, energy_curve, sr):
        entry_points = []
        exit_points = []

        # Intro as entry point
        entry_points.append({
            'time': float(sections['intro']['end']),
            'type': 'intro_end',
            'quality': 0.8,
            'energy': float(
                np.mean(energy_curve[:int(sections['intro']['end'] * len(energy_curve) / len(energy_curve))]))
        })

        # Drops as potential entry points
        for i, drop in enumerate(sections['drops']):
            entry_points.append({
                'time': float(drop['start']),
                'type': f'drop_start_{i + 1}',
                'quality': 0.9,
                'energy': float(drop['energy'])
            })

        # Breakdowns as potential entry points
        for i, breakdown in enumerate(sections['breakdowns']):
            entry_points.append({
                'time': float(breakdown['end']),
                'type': f'breakdown_end_{i + 1}',
                'quality': 0.7,
                'energy': float(breakdown['energy'])
            })

        # Breakdowns as potential exit points
        for i, breakdown in enumerate(sections['breakdowns']):
            exit_points.append({
                'time': float(breakdown['start']),
                'type': f'breakdown_start_{i + 1}',
                'quality': 0.8,
                'energy': float(breakdown['energy'])
            })

        # Drops as potential exit points
        for i, drop in enumerate(sections['drops']):
            exit_points.append({
                'time': float(drop['end']),
                'type': f'drop_end_{i + 1}',
                'quality': 0.7,
                'energy': float(drop['energy'])
            })

        # Outro as exit point
        exit_points.append({
            'time': float(sections['outro']['start']),
            'type': 'outro_start',
            'quality': 0.9,
            'energy': float(
                np.mean(energy_curve[int(sections['outro']['start'] * len(energy_curve) / len(energy_curve)):]))
        })

        return {
            'entries': entry_points,
            'exits': exit_points
        }

    def _harmonic_compatibility_score(self, camelot_key1, camelot_key2):
        # Perfect match
        if camelot_key1 == camelot_key2:
            return 1.0

        # Extract number and letter
        try:
            num1 = int(camelot_key1[:-1])
            letter1 = camelot_key1[-1]
            num2 = int(camelot_key2[:-1])
            letter2 = camelot_key2[-1]
        except:
            # Invalid format, assume low compatibility
            return 0.3

        # Same letter, adjacent number
        if letter1 == letter2 and abs(num1 - num2) == 1:
            return 0.9

        # Same number, different letter (
        if num1 == num2 and letter1 != letter2:
            return 0.8

        # Adjacent number, different letter or non-adjacent but same letter
        if (abs(num1 - num2) == 1 and letter1 != letter2) or (letter1 == letter2):
            return 0.6

        # Otherwise not very comaptible
        return 0.3

    def _rhythm_compatibility_score(self, bpm1, bpm2):
        # Calculate ratio for comparison
        if bpm1 > bpm2:
            ratio = bpm1 / bpm2
        else:
            ratio = bpm2 / bpm1

        # Check for double/half time relationships
        if ratio > 2:
            if abs(ratio - 2) < 0.05:
                return 0.8

        # Perfect match (within 1%)
        if abs(ratio - 1) < 0.01:
            return 1.0

        # Very close (within 3%)
        if abs(ratio - 1) < 0.03:
            return 0.9

        # Close (within 5%)
        if abs(ratio - 1) < 0.05:
            return 0.8

        # Moderate difference (within 10%)
        if abs(ratio - 1) < 0.1:
            return 0.6

        # Larger difference
        return 0.4

    def _energy_compatibility_score(self, energy1, energy2):
        # Calculate ratio of energies
        if energy1 > energy2:
            ratio = energy2 / energy1
        else:
            ratio = energy1 / energy2

        # Score based on how close energy levels are
        if ratio > 0.95:
            return 1.0
        elif ratio > 0.9:
            return 0.9
        elif ratio > 0.8:
            return 0.8
        elif ratio > 0.7:
            return 0.7
        else:
            return 0.5

    def _structure_compatibility_score(self, exit_type, entry_type):
        # Best transitions based on my DJing experience
        best_pairs = [
            ('outro_start', 'intro_end'),
            ('breakdown_start', 'breakdown_end'),
            ('drop_end', 'drop_start')
        ]

        # Good transitions
        good_pairs = [
            ('drop_end', 'breakdown_end'),
            ('breakdown_start', 'intro_end'),
            ('outro_start', 'drop_start')
        ]

        # Check exact pairs
        if any(exit_type == pair[0] and entry_type == pair[1] for pair in best_pairs):
            return 1.0

        # Check good pairs
        if any(exit_type == pair[0] and entry_type == pair[1] for pair in good_pairs):
            return 0.8

        # Check partial matches by prefix
        exit_prefix = exit_type.split('_')[0]
        entry_prefix = entry_type.split('_')[0]

        if exit_prefix == 'drop' and entry_prefix == 'drop':
            return 0.7

        if exit_prefix == 'breakdown' and entry_prefix == 'breakdown':
            return 0.7

        # Default compatibility
        return 0.5

    def visualize_analysis(self, track_analysis, audio_path=None):
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Show waveform if audio path is provided
        if audio_path:
            y, sr = librosa.load(audio_path, sr=44100)
            librosa.display.waveshow(y, sr=sr, ax=ax1)
            ax1.set_title('Waveform')
        else:
            ax1.set_title('Track Analysis')

        # Display track details
        ax1.text(0.02, 0.9, f"BPM: {track_analysis['bpm']:.1f}", transform=ax1.transAxes)
        ax1.text(0.02, 0.85, f"Key: {track_analysis['key']} ({track_analysis['camelot_key']})", transform=ax1.transAxes)
        ax1.text(0.02, 0.8, f"Duration: {track_analysis['duration']:.1f}s", transform=ax1.transAxes)

        # Second subplot for sections and transition points
        ax2.set_title('Sections and Transition Points')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Section Type')
        ax2.set_xlim(0, track_analysis['duration'])

        # Plot intro section
        intro = track_analysis['sections']['intro']
        ax2.axvspan(intro['start'], intro['end'], alpha=0.2, color='green', label='Intro')

        # Plot drops
        for i, drop in enumerate(track_analysis['sections']['drops']):
            ax2.axvspan(drop['start'], drop['end'], alpha=0.3, color='red', label=f'Drop {i + 1}' if i == 0 else "")

        # Plot breakdowns
        for i, breakdown in enumerate(track_analysis['sections']['breakdowns']):
            ax2.axvspan(breakdown['start'], breakdown['end'], alpha=0.3, color='blue',
                        label=f'Breakdown {i + 1}' if i == 0 else "")

        # Plot outro section
        outro = track_analysis['sections']['outro']
        ax2.axvspan(outro['start'], outro['end'], alpha=0.2, color='purple', label='Outro')

        # Plot entry points
        for point in track_analysis['transition_points']['entries']:
            ax2.axvline(x=point['time'], linestyle='--', color='green', alpha=0.7)
            ax2.text(point['time'], 0.9, 'IN', horizontalalignment='center', verticalalignment='center',
                     transform=ax2.get_xaxis_transform())

        # Plot exit points
        for point in track_analysis['transition_points']['exits']:
            ax2.axvline(x=point['time'], linestyle='--', color='red', alpha=0.7)
            ax2.text(point['time'], 0.1, 'OUT', horizontalalignment='center', verticalalignment='center',
                     transform=ax2.get_xaxis_transform())

        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.show()