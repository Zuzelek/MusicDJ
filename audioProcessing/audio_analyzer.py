import numpy as np
import librosa
from scipy.stats import pearsonr


class AudioAnalyzer:
    """
    Core audio analysis - detects BPM, key signatures and energy profiles.
    """

    def __init__(self):
        # Templates I'm using for key detection
        self.major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        self.minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])

        # Keys in Western music theory
        self.KEY_MAPPING = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # My Camelot wheel mapping for harmonic mixing
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

    def analyze_audio(self, file_path):
        # Load audio file
        y, sr = librosa.load(file_path, sr=44100)

        # Get key features
        tempo = self.get_tempo(y, sr)
        key = self.get_key(y, sr)
        camelot_key = self.get_camelot_key(key)
        energy = self.get_energy_profile(y)
        beats = self.get_beats(y, sr)
        downbeats = self.get_downbeats(y, sr)

        return {
            'tempo': tempo,
            'key': key,
            'camelot_key': camelot_key,
            'energy': energy,
            'beats': beats,
            'downbeats': downbeats
        }

    def get_tempo(self, y, sr):
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
        return round(detected_bpm), beat_times

    def get_key(self, y, sr):
        # Extract harmonic component (filtering out drums)
        y_harmonic = librosa.effects.harmonic(y)

        # Compute chroma features
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, n_chroma=12, n_octaves=7)

        # Average and normalize
        chroma_mean = np.mean(chroma, axis=1)
        chroma_mean = chroma_mean / np.linalg.norm(chroma_mean)

        # Find best correlation with key templates
        best_score = -np.inf
        detected_key = ""

        for i, base_key in enumerate(self.KEY_MAPPING):
            # Roll template to each possible key
            major_profile = np.roll(self.major_template, i)
            minor_profile = np.roll(self.minor_template, i)

            # Calculate correlation
            major_similarity = pearsonr(chroma_mean, major_profile)[0]
            minor_similarity = pearsonr(chroma_mean, minor_profile)[0]

            # Update if better correlation found
            if major_similarity > best_score:
                best_score = major_similarity
                detected_key = f"{base_key} Major"

            if minor_similarity > best_score:
                best_score = minor_similarity
                detected_key = f"{base_key} Minor"

        return detected_key

    def get_camelot_key(self, key):
        return self.camelot_wheel.get(key, "8B")  # Default if not found

    def get_energy_profile(self, y):
        # Overall loudness/energy
        return float(np.mean(librosa.feature.rms(y=y)))

    def get_beats(self, y, sr):
        # Get beat timings
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        return beat_times.tolist()

    def get_downbeats(self, y, sr):
        # Find strong beats using PLP
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        plp = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
        downbeats = np.where(plp > 0.5)[0]  # Threshold I found works well
        return librosa.frames_to_time(downbeats, sr=sr).tolist()

    def get_onset_envelope(self, y, sr):
        return librosa.onset.onset_strength(y=y, sr=sr)

    def get_energy_curve(self, y):
        return librosa.feature.rms(y=y).flatten()