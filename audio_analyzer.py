import librosa
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict


@dataclass
class AudioFeatures:
    """Class to store audio analysis results"""
    tempo: float
    key: str
    segments: Dict
    energy: float
    beats: np.ndarray
    onset_env: np.ndarray


class AudioAnalyzer:
    """Enhanced class for analyzing audio features."""

    def __init__(self):
        self.sr = 44100  # Standard sample rate
        self.KEY_MAPPING = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def analyze_audio(self, file_path: str) -> Optional[AudioFeatures]:
        """Analyzes the audio file and returns comprehensive features."""
        try:
            # Load the audio file
            y, sr = librosa.load(file_path, sr=self.sr)

            # Basic feature extraction
            tempo, beats = self.get_tempo_and_beats(y, sr)
            key = self.get_key(y, sr)
            segments = self.get_song_segments(y, sr)
            energy = self.get_energy_profile(y)
            onset_env = self.get_onset_envelope(y, sr)

            return AudioFeatures(
                tempo=tempo,
                key=key,
                segments=segments,
                energy=energy,
                beats=beats,
                onset_env=onset_env
            )

        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")
            return None

    def get_tempo_and_beats(self, y: np.ndarray, sr: int) -> Tuple[float, np.ndarray]:
        """Extract tempo and beat frames."""
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]  # Use librosa.beat.tempo directly
        beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[1]
        return tempo, beats

    def get_key(self, y: np.ndarray, sr: int) -> str:
        # Extract harmonic signal
        y_harmonic = librosa.effects.harmonic(y)
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

        # Smooth chroma values
        chroma_mean = np.mean(chroma, axis=1)

        # Get the key index
        key_idx = np.argmax(chroma_mean)
        return self.KEY_MAPPING[key_idx]

    def get_song_segments(self, y: np.ndarray, sr: int) -> Dict:
        # Get onset strength and reduce sensitivity
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=1024)

        # Detect segment boundaries
        boundary_frames = librosa.segment.agglomerative(onset_env, k=8)  # Group into 8 sections
        boundary_times = librosa.frames_to_time(boundary_frames, sr=sr)

        # Analyze segments
        segments = {
            'start_times': boundary_times,
            'durations': np.diff(boundary_times),
            'count': len(boundary_times)
        }
        return segments

    def get_energy_profile(self, y: np.ndarray) -> float:
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)

        # Normalize to maximum RMS value
        rms_normalized = rms / np.max(rms)
        return np.mean(rms_normalized)

    def get_onset_envelope(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Get the onset strength envelope."""
        return librosa.onset.onset_strength(y=y, sr=sr)
