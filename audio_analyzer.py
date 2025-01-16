import librosa
import numpy as np

class AudioAnalyzer:
    """Class for analyzing audio features."""

    KEY_MAPPING = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    ]

    CAMELOT_MAPPING = {
        "C": 0,
        "C#": 1,
        "D": 2,
        "D#": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G": 7,
        "G#": 8,
        "A": 9,
        "A#": 10,
        "B": 11
    }

    @staticmethod
    def analyze_audio(file_path):
        """Analyzes the tempo, key, and Camelot notation of an audio file."""
        try:
            y, sr = librosa.load(file_path)

            # Tempo detection
            tempo = librosa.beat.beat_track(y=y, sr=sr)[0]

            # Key detection
            chroma = librosa.feature.chroma_cens(y=y, sr=sr)
            key_index = np.argmax(np.mean(chroma, axis=1))
            key_label = AudioAnalyzer.KEY_MAPPING[key_index]

            camelot = AudioAnalyzer.CAMELOT_MAPPING[key_label]

            return tempo, key_label, camelot
        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")
            return None, None, None
