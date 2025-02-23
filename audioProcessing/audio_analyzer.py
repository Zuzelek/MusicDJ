import librosa
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
from scipy.stats import pearsonr


@dataclass
class AudioFeatures:
    tempo: float
    key: str
    camelot_key: str
    segments: Dict
    energy: float
    onset_env: np.ndarray


class AudioAnalyzer:

    def __init__(self):
        self.sr = 44100
        self.KEY_MAPPING = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

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
        try:
            y, sr = librosa.load(file_path, sr=self.sr)

            tempo_val = self.get_bpm(y, sr)
            key = self.get_key(y, sr)
            camelot_key = self.camelot_wheel.get(key, "Unknown")
            segments = self.get_song_segments(y, sr)
            energy = self.get_energy_profile(y)
            onset_env = self.get_onset_envelope(y, sr)

            return AudioFeatures(
                tempo=tempo_val,
                key=key,
                camelot_key=camelot_key,
                segments=segments,
                energy=energy,
                onset_env=onset_env
            )

        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")
            return None

    def get_bpm(self, y, sr):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env, trim=True)
        tempo_scalar = tempo.item() if isinstance(tempo, np.ndarray) else tempo

        beat_times = librosa.frames_to_time(beats, sr=sr)
        beat_intervals = np.diff(beat_times)
        avg_beat_interval = np.mean(beat_intervals)
        detected_bpm = 60 / avg_beat_interval


        if detected_bpm < 90:
            doubled_bpm = detected_bpm * 2
            onset_env_fast = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256, max_size=1)
            tempo_fast, _ = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env_fast, bpm=doubled_bpm)
            tempo_fast_scalar = tempo_fast.item() if isinstance(tempo_fast, np.ndarray) else tempo_fast
            if abs(tempo_fast_scalar - doubled_bpm) < 10:
                detected_bpm = doubled_bpm

        # Round to nearest integer
        return round(detected_bpm)

    def get_key(self, y: np.ndarray, sr: int) -> str:
        y_harmonic = librosa.effects.harmonic(y)
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        best_score = -np.inf
        detected_key = ""

        for i, base_key in enumerate(self.KEY_MAPPING):
            major_profile = np.roll(self.major_template, i)
            minor_profile = np.roll(self.minor_template, i)

            major_similarity = pearsonr(chroma_mean, major_profile)[0]
            minor_similarity = pearsonr(chroma_mean, minor_profile)[0]

            if major_similarity > best_score:
                best_score = major_similarity
                detected_key = f"{base_key} Major"
            if minor_similarity > best_score:
                best_score = minor_similarity
                detected_key = f"{base_key} Minor"

        return detected_key

    def get_song_segments(self, y: np.ndarray, sr: int) -> Dict:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        boundary_frames = librosa.segment.agglomerative(onset_env, k=8)
        boundary_times = librosa.frames_to_time(boundary_frames, sr=sr)

        return {
            'start_times': boundary_times.tolist(),
            'durations': np.diff(boundary_times).tolist(),
            'count': len(boundary_times)
        }

    def get_energy_profile(self, y: np.ndarray) -> float:
        rms = librosa.feature.rms(y=y)
        return np.mean(rms)

    def get_onset_envelope(self, y: np.ndarray, sr: int) -> np.ndarray:
        return librosa.onset.onset_strength(y=y, sr=sr)
