import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler


class SectionAnalyzer:
    """
    My module to detect structure in EDM tracks.
    """

    def __init__(self):
        # Parameters I tuned through testing
        self.MIN_SECTION_LENGTH = 4  # Anything shorter isn't a real section
        self.DROP_ENERGY_RATIO = 1.5  # Drops are typically this much higher than average
        self.BREAKDOWN_ENERGY_RATIO = 0.6  # Breakdowns are typically this much lower
        self.INTRO_LENGTH_RATIO = 0.15  # Intro usually 15% of track
        self.OUTRO_LENGTH_RATIO = 0.15  # Same for outro

    def analyze_sections(self, y, sr):
        # Get track duration
        duration = librosa.get_duration(y=y, sr=sr)

        # Extract energy curve
        energy_curve = librosa.feature.rms(y=y).flatten()

        # Smooth it for better detection
        energy_curve_smooth = savgol_filter(energy_curve,
                                            int(sr / 512 * 8) if len(energy_curve) > sr / 512 * 8 else 3,
                                            2)

        # Normalize to 0-1 range
        scaler = MinMaxScaler()
        energy_curve_norm = scaler.fit_transform(energy_curve_smooth.reshape(-1, 1)).flatten()

        # Time axis for mapping frames to seconds
        time_axis = np.linspace(0, duration, len(energy_curve_norm))

        # Calculate average energy
        avg_energy = np.mean(energy_curve_norm)

        # Find drops (high energy sections)
        drop_threshold = avg_energy * self.DROP_ENERGY_RATIO
        drop_candidates = np.where(energy_curve_norm > drop_threshold)[0]
        drops = self._group_frames_to_sections(drop_candidates, time_axis)

        # Find breakdowns (low energy sections)
        breakdown_threshold = avg_energy * self.BREAKDOWN_ENERGY_RATIO
        breakdown_candidates = np.where(energy_curve_norm < breakdown_threshold)[0]
        breakdowns = self._group_frames_to_sections(breakdown_candidates, time_axis)

        # Estimate intro and outro
        intro_end = duration * self.INTRO_LENGTH_RATIO
        outro_start = duration * (1 - self.OUTRO_LENGTH_RATIO)

        # Find buildups (sections before drops)
        buildups = []
        for drop in drops:
            # Typical buildup length in EDM is ~16 seconds
            buildup_start = drop['start'] - 16
            buildup_start = max(0, buildup_start)

            # Only add if not part of intro
            if buildup_start > intro_end:
                buildups.append({
                    'start': float(buildup_start),
                    'end': float(drop['start']),
                    'energy': float(np.mean(energy_curve_norm[
                                            int(buildup_start * len(energy_curve_norm) / duration):
                                            int(drop['start'] * len(energy_curve_norm) / duration)
                                            ]))
                })

        # Package everything together
        return {
            'intro': {
                'start': 0,
                'end': intro_end,
                'energy': float(np.mean(energy_curve_norm[:int(intro_end * len(energy_curve_norm) / duration)]))
            },
            'drops': drops,
            'breakdowns': breakdowns,
            'buildups': buildups,
            'outro': {
                'start': outro_start,
                'end': duration,
                'energy': float(np.mean(energy_curve_norm[int(outro_start * len(energy_curve_norm) / duration):]))
            },
            'energy_curve': energy_curve_norm.tolist(),
            'time_axis': time_axis.tolist()
        }

    def _group_frames_to_sections(self, candidates, time_axis):
        # Handle empty case
        if len(candidates) == 0:
            return []

        sections = []
        section_start = candidates[0]

        for i in range(1, len(candidates)):
            # If gap bigger than 1 frame, end section
            if candidates[i] - candidates[i - 1] > 1:
                # Only keep sections that are long enough
                section_length = time_axis[candidates[i - 1]] - time_axis[section_start]
                if section_length >= self.MIN_SECTION_LENGTH:
                    sections.append({
                        'start': float(time_axis[section_start]),
                        'end': float(time_axis[candidates[i - 1]]),
                        'energy': float(np.mean(time_axis[section_start:candidates[i - 1]]))
                    })
                section_start = candidates[i]

        # Don't forget the last section
        if len(candidates) > 0:
            section_length = time_axis[-1] if len(candidates) == len(time_axis) else time_axis[candidates[-1]] - \
                                                                                     time_axis[section_start]
            if section_length >= self.MIN_SECTION_LENGTH:
                sections.append({
                    'start': float(time_axis[section_start]),
                    'end': float(time_axis[candidates[-1]]),
                    'energy': float(np.mean(time_axis[section_start:candidates[-1]]))
                })

        return sections

    def visualize_sections(self, sections, audio_path=None):
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # If audio is provided, show waveform
        if audio_path:
            y, sr = librosa.load(audio_path, sr=44100)
            librosa.display.waveshow(y, sr=sr, ax=ax1)
            ax1.set_title('Waveform')
        else:
            # Otherwise just plot energy
            ax1.plot(sections['time_axis'], sections['energy_curve'])
            ax1.set_title('Energy Curve')

        # Plot section markers
        ax2.set_title('Detected Sections')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Section Type')
        ax2.set_xlim(0, sections['outro']['end'])
        ax2.set_ylim(0, 1)

        # Color-code each section type
        intro = sections['intro']
        ax2.axvspan(intro['start'], intro['end'], alpha=0.2, color='green', label='Intro')

        for i, buildup in enumerate(sections['buildups']):
            ax2.axvspan(buildup['start'], buildup['end'], alpha=0.3, color='yellow',
                        label=f'Buildup {i + 1}' if i == 0 else "")

        for i, drop in enumerate(sections['drops']):
            ax2.axvspan(drop['start'], drop['end'], alpha=0.3, color='red', label=f'Drop {i + 1}' if i == 0 else "")

        for i, breakdown in enumerate(sections['breakdowns']):
            ax2.axvspan(breakdown['start'], breakdown['end'], alpha=0.3, color='blue',
                        label=f'Breakdown {i + 1}' if i == 0 else "")

        outro = sections['outro']
        ax2.axvspan(outro['start'], outro['end'], alpha=0.2, color='purple', label='Outro')

        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.show()