import librosa
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from scipy.stats import pearsonr


@dataclass
class AudioFeatures:
    tempo: float
    key: str
    camelot_key: str  # New field for Camelot Wheel key
    segments: Dict
    energy: float
    beats: np.ndarray
    onset_env: np.ndarray


class AudioAnalyzer:

    def __init__(self):

        self.sr = 44100
        self.KEY_MAPPING = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        # Key profiles based on Krumhansl-Schmuckler key-finding algorithm
        self.major_template = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        self.minor_template = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        self.camelot_wheel = {
            "C Major": "8B", "A Minor": "8A",
            "G Major": "9B", "E Minor": "9A",
            "D Major": "10B", "B Minor": "10A",
            "A Major": "11B", "F# Minor": "11A",
            "E Major": "12B", "C# Minor": "12A",
            "B Major": "1B", "G# Minor": "1A",
            "F# Major": "2B", "D# Minor": "2A",
            "F Major": "3B", "D Minor": "3A",
            "Bb Major": "6B", "G Minor": "6A",
            "Eb Major": "5B", "C Minor": "5A",
            "Ab Major": "4B", "F Minor": "4A",
            "Db Major": "3B", "Bb Minor": "3A"
        }

    def analyze_audio(self, file_path: str) -> Optional[AudioFeatures]:
        """Analyzes the audio and return features"""
        try:
            # Load the audio file
            y, sr = librosa.load(file_path, sr=self.sr)

            # Basic feature extraction
            tempo, beats = self.get_tempo_and_beats(y, sr)
            key = self.get_key(y, sr)
            camelot_key = self.camelot_wheel.get(key, "Unknown")  # Get Camelot Wheel key
            segments = self.get_song_segments(y, sr)
            energy = self.get_energy_profile(y)
            onset_env = self.get_onset_envelope(y, sr)

            return AudioFeatures(
                tempo=tempo,
                key=key,
                camelot_key=camelot_key,  # Include Camelot Wheel key
                segments=segments,
                energy=energy,
                beats=beats,
                onset_env=onset_env
            )

        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")
            return None

    def get_tempo_and_beats(self, y: np.ndarray, sr: int) -> Tuple[float, np.ndarray]:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]  # Use librosa.beat.tempo directly
        beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[1]
        return tempo, beats

    def get_bpm(self, file_path):
        """BPM analyzing"""
        try:
            y, sr = librosa.load(file_path, sr=self.sr)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
            return tempo
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None

    def get_key(self, y: np.ndarray, sr: int) -> str:
        """Detects key and distinguishes between major and minor"""
        # Extract harmonic signal
        y_harmonic = librosa.effects.harmonic(y)

        # Compute chroma features
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, n_chroma=12, n_octaves=7)

        # Average chroma features over time
        chroma_mean = np.mean(chroma, axis=1)
        chroma_mean = chroma_mean / np.linalg.norm(chroma_mean)  # Normalize

        best_score = -np.inf
        detected_key = ""

        # Iterating over possible keys (C to B)
        for i, base_key in enumerate(self.KEY_MAPPING):

            major_profile = np.roll(self.major_template, i)
            minor_profile = np.roll(self.minor_template, i)

            # Compute similarity
            major_similarity = pearsonr(chroma_mean, major_profile)[0]
            minor_similarity = pearsonr(chroma_mean, minor_profile)[0]

            # Compare the results and select the best
            if major_similarity > best_score:
                best_score = major_similarity
                detected_key = f"{base_key} Major"
            if minor_similarity > best_score:
                best_score = minor_similarity
                detected_key = f"{base_key} Minor"

        return detected_key

    def get_song_segments(self, y: np.ndarray, sr: int) -> Dict:
        """Detect song segments (needs improving)"""

        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=1024)

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

        return librosa.onset.onset_strength(y=y, sr=sr)