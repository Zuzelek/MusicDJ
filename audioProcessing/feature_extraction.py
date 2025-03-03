import numpy as np
import librosa
import os
import json
import warnings
from scipy.stats import skew, kurtosis
from librosa.feature.spectral import spectral_bandwidth, spectral_centroid
from librosa.feature import mfcc, chroma_stft, spectral_contrast
from librosa.onset import onset_strength
from librosa.effects import harmonic, percussive

# Ignore specific warnings
warnings.filterwarnings('ignore', category=UserWarning)


class FeatureExtractor:
    """
    A comprehensive feature extraction pipeline for EDM tracks to support AI DJ mixing.
    """

    def __init__(self, sample_rate=22050, hop_length=512):
        """
        Initialize the feature extractor.

        Parameters:
        -----------
        sample_rate : int
            Sample rate for audio analysis
        hop_length : int
            Hop length for FFT-based features
        """
        self.sr = sample_rate
        self.hop_length = hop_length

    def extract_all_features(self, audio_path, output_folder=None):
        """
        Extract a comprehensive set of features from an audio file.

        Parameters:
        -----------
        audio_path : str
            Path to the audio file
        output_folder : str, optional
            If provided, saves features to JSON file in this folder

        Returns:
        --------
        features : dict
            Dictionary containing all extracted features
        """
        print(f"Extracting features from {os.path.basename(audio_path)}")

        # Load audio file
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None

        # Calculate duration
        duration = librosa.get_duration(y=y, sr=sr)

        # Extract rhythmic features
        rhythmic_features = self.extract_rhythmic_features(y, sr)

        # Extract tonal features
        tonal_features = self.extract_tonal_features(y, sr)

        # Extract spectral features
        spectral_features = self.extract_spectral_features(y, sr)

        # Extract energy features
        energy_features = self.extract_energy_features(y, sr)

        # Extract structural features
        structural_features = self.extract_structural_features(y, sr)

        # Combine all features
        all_features = {
            'metadata': {
                'filename': os.path.basename(audio_path),
                'duration': duration,
            },
            'rhythmic': rhythmic_features,
            'tonal': tonal_features,
            'spectral': spectral_features,
            'energy': energy_features,
            'structure': structural_features
        }

        # Save features if output folder is specified
        if output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(
                output_folder,
                os.path.splitext(os.path.basename(audio_path))[0] + "_features.json"
            )

            with open(output_file, 'w') as f:
                json.dump(all_features, f, indent=2)

            print(f"Features saved to {output_file}")

        return all_features

    def extract_rhythmic_features(self, y, sr):
        """Extract rhythmic features including tempo, beat positions, and rhythm patterns."""
        # Extract onset envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)

        # Tempo and beat information
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=self.hop_length)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length)

        # Calculate beat intervals and statistics
        beat_intervals = np.diff(beat_times)

        # Handle empty beat intervals
        if len(beat_intervals) > 0:
            beat_regularity = 1.0 - np.std(beat_intervals) / np.mean(beat_intervals)
            beat_stats = {
                'mean': float(np.mean(beat_intervals)),
                'std': float(np.std(beat_intervals)),
                'skewness': float(skew(beat_intervals)) if len(beat_intervals) > 2 else 0,
                'kurtosis': float(kurtosis(beat_intervals)) if len(beat_intervals) > 2 else 0
            }
        else:
            beat_regularity = 0
            beat_stats = {
                'mean': 0,
                'std': 0,
                'skewness': 0,
                'kurtosis': 0
            }

        # Extract percussive component for rhythm analysis
        y_perc = percussive(y)

        # Calculate pulse clarity (strength of beats)
        onset_env_perc = librosa.onset.onset_strength(y=y_perc, sr=sr)
        pulse_clarity = np.mean(onset_env_perc) / np.max(onset_env_perc) if np.max(onset_env_perc) > 0 else 0

        # Calculate rhythmic patterns using a tempogram
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=self.hop_length)

        # Calculate rhythmic centroid as indicator of rhythmic complexity
        if tempogram.size > 0:
            rhythm_centroid = np.sum(tempogram * np.arange(tempogram.shape[0]).reshape(-1, 1)) / np.sum(tempogram)
            rhythm_centroid = float(np.mean(rhythm_centroid))
        else:
            rhythm_centroid = 0

        # Calculate tempo range for BPM sliding
        if 0.5 * tempo >= 85 and 2 * tempo <= 175:
            tempo_half = 0.5 * tempo
            tempo_double = 2 * tempo
        else:
            tempo_half = tempo
            tempo_double = tempo

        return {
            'bpm': float(tempo),
            'bpm_range': {'min': float(tempo_half), 'max': float(tempo_double)},
            'beat_positions': beat_times.tolist(),
            'beat_regularity': float(beat_regularity),
            'beat_statistics': beat_stats,
            'pulse_clarity': float(pulse_clarity),
            'rhythm_complexity': float(rhythm_centroid)
        }

    def extract_tonal_features(self, y, sr):
        """Extract tonal features including key and harmonic content."""
        # Extract harmonic component
        y_harm = harmonic(y)

        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)

        # Estimate key using chroma
        chroma_avg = np.mean(chroma, axis=1)
        key_index = np.argmax(chroma_avg)

        # Map key index to musical key
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_note = note_names[key_index]

        # Estimate mode (major/minor)
        # Using the method described in Lartillot's implementation
        major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]) / 7.0
        minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]) / 7.0

        # Rotate profiles to match current key
        maj_corr = np.corrcoef(chroma_avg, np.roll(major_profile, key_index))[0, 1]
        min_corr = np.corrcoef(chroma_avg, np.roll(minor_profile, key_index))[0, 1]

        mode = 'major' if maj_corr >= min_corr else 'minor'
        key = f"{key_note} {mode}"

        # Key certainty (correlation with best matching profile)
        key_certainty = max(maj_corr, min_corr)

        # Calculate key strength for all keys
        key_strengths = {}
        for i, note in enumerate(note_names):
            maj_prof = np.roll(major_profile, i)
            min_prof = np.roll(minor_profile, i)

            maj_strength = np.corrcoef(chroma_avg, maj_prof)[0, 1]
            min_strength = np.corrcoef(chroma_avg, min_prof)[0, 1]

            key_strengths[f"{note} major"] = float(maj_strength)
            key_strengths[f"{note} minor"] = float(min_strength)

        # Calculate tonal centroid as a measure of harmonic complexity
        tonal_centroid = librosa.feature.tonnetz(y=y_harm, sr=sr)
        tonal_complexity = float(np.mean(np.std(tonal_centroid, axis=1)))

        # Harmonic change detection function
        hcdf = np.sqrt(np.sum(np.diff(chroma, axis=1) ** 2, axis=0))
        harmonic_change_rate = float(np.mean(hcdf))

        return {
            'key': key,
            'key_certainty': float(key_certainty),
            'key_strengths': key_strengths,
            'harmonic_complexity': float(tonal_complexity),
            'harmonic_change_rate': float(harmonic_change_rate),
            'average_chroma': chroma_avg.tolist()
        }

    def extract_spectral_features(self, y, sr):
        """Extract spectral features describing timbre and frequency characteristics."""
        # Compute STFT
        S = np.abs(librosa.stft(y, hop_length=self.hop_length))

        # Compute spectral centroid
        sc = spectral_centroid(y=y, sr=sr, hop_length=self.hop_length).ravel()

        # Compute spectral bandwidth
        sb = spectral_bandwidth(y=y, sr=sr, hop_length=self.hop_length).ravel()

        # Compute spectral contrast
        contrast = spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)

        # Compute MFCCs for timbre representation
        mfccs = mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)

        # Compute spectral roll-off
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length).ravel()

        # Compute spectral flatness
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=self.hop_length).ravel()

        # Process features for JSON serialization
        spectral_stats = {
            'centroid': {
                'mean': float(np.mean(sc)),
                'std': float(np.std(sc)),
                'skewness': float(skew(sc)) if len(sc) > 2 else 0,
            },
            'bandwidth': {
                'mean': float(np.mean(sb)),
                'std': float(np.std(sb)),
            },
            'contrast': {
                'mean': [float(np.mean(contrast[i])) for i in range(contrast.shape[0])],
                'std': [float(np.std(contrast[i])) for i in range(contrast.shape[0])],
            },
            'rolloff': {
                'mean': float(np.mean(rolloff)),
                'std': float(np.std(rolloff)),
            },
            'flatness': {
                'mean': float(np.mean(flatness)),
                'std': float(np.std(flatness)),
            },
            'mfcc': {
                'mean': [float(np.mean(mfccs[i])) for i in range(mfccs.shape[0])],
                'std': [float(np.std(mfccs[i])) for i in range(mfccs.shape[0])],
            }
        }

        # Calculate brightness (ratio of high-frequency to total energy)
        brightness_freq = 2000  # Threshold in Hz
        brightness_bin = int(brightness_freq * S.shape[0] / (sr / 2))
        brightness = np.sum(S[brightness_bin:, :]) / np.sum(S) if np.sum(S) > 0 else 0

        return {
            'statistics': spectral_stats,
            'brightness': float(brightness)
        }

    def extract_energy_features(self, y, sr):
        """Extract energy-related features for track intensity analysis."""
        # Compute RMS energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]

        # Compute energy over time
        energy = np.sum(np.abs(librosa.stft(y, hop_length=self.hop_length)) ** 2, axis=0)

        # Compute low/mid/high frequency band energies
        S = np.abs(librosa.stft(y, hop_length=self.hop_length))

        # Define frequency bands
        n_fft = 2 * (S.shape[0] - 1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Low: 20-250 Hz (sub-bass and bass)
        low_band = np.where((freqs >= 20) & (freqs <= 250))[0]
        # Mid: 250-4000 Hz (mid-range)
        mid_band = np.where((freqs > 250) & (freqs <= 4000))[0]
        # High: 4000-20000 Hz (high-end)
        high_band = np.where((freqs > 4000) & (freqs <= 20000))[0]

        low_energy = np.sum(S[low_band, :], axis=0) if len(low_band) > 0 else np.zeros_like(energy)
        mid_energy = np.sum(S[mid_band, :], axis=0) if len(mid_band) > 0 else np.zeros_like(energy)
        high_energy = np.sum(S[high_band, :], axis=0) if len(high_band) > 0 else np.zeros_like(energy)

        # Normalize energies
        total_energy = low_energy + mid_energy + high_energy
        low_energy_ratio = low_energy / total_energy if np.sum(total_energy) > 0 else np.zeros_like(low_energy)
        mid_energy_ratio = mid_energy / total_energy if np.sum(total_energy) > 0 else np.zeros_like(mid_energy)
        high_energy_ratio = high_energy / total_energy if np.sum(total_energy) > 0 else np.zeros_like(high_energy)

        # Calculate dynamic range
        energy_db = librosa.power_to_db(energy, ref=np.max)
        dynamic_range = np.max(energy_db) - np.min(energy_db) if len(energy_db) > 0 else 0

        # Calculate energy envelope
        energy_env = librosa.feature.rms(y=y, frame_length=2048, hop_length=self.hop_length)[0]
        energy_env_smooth = librosa.util.normalize(energy_env)

        # Calculate energy peaks
        energy_peaks = librosa.util.peak_pick(energy_env_smooth, pre_max=10, post_max=10,
                                              pre_avg=10, post_avg=10, delta=0.2, wait=20)
        peak_times = librosa.frames_to_time(energy_peaks, sr=sr, hop_length=self.hop_length)

        # Calculate energy dynamics
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
        energy_skew = float(skew(energy)) if len(energy) > 2 else 0
        energy_change = np.mean(np.abs(np.diff(energy_env_smooth)))

        return {
            'rms': {
                'mean': float(np.mean(rms)),
                'std': float(np.std(rms)),
                'max': float(np.max(rms)) if len(rms) > 0 else 0,
                'min': float(np.min(rms)) if len(rms) > 0 else 0
            },
            'dynamic_range': float(dynamic_range),
            'frequency_bands': {
                'low_ratio': {
                    'mean': float(np.mean(low_energy_ratio)),
                    'std': float(np.std(low_energy_ratio))
                },
                'mid_ratio': {
                    'mean': float(np.mean(mid_energy_ratio)),
                    'std': float(np.std(mid_energy_ratio))
                },
                'high_ratio': {
                    'mean': float(np.mean(high_energy_ratio)),
                    'std': float(np.std(high_energy_ratio))
                }
            },
            'energy_peaks': peak_times.tolist(),
            'energy_dynamics': {
                'mean': float(energy_mean),
                'std': float(energy_std),
                'skewness': energy_skew,
                'change_rate': float(energy_change)
            }
        }

    def extract_structural_features(self, y, sr):
        """Extract features related to track structure and segmentation."""
        # Compute MFCC features for segmentation
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)

        # Compute chroma features for harmonic content
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)

        # Combine features for segmentation
        features = np.vstack([mfccs, chroma])

        # Compute self-similarity matrix
        S = librosa.segment.recurrence_matrix(features, width=3, mode='affinity', sym=True)

        # Enhance diagonals
        S_enhance = librosa.segment.path_enhance(S, 15)

        # Find segment boundaries
        boundary_frames = librosa.segment.agglomerative(S_enhance, 8)
        boundary_times = librosa.frames_to_time(boundary_frames, sr=sr, hop_length=self.hop_length)

        # Calculate novelty curve (for change detection)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        novelty = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr,
                                             hop_length=self.hop_length, backtrack=False)
        novelty_times = librosa.frames_to_time(novelty, sr=sr, hop_length=self.hop_length)

        # Calculate homogeneity for each segment
        segment_homogeneity = []
        for i in range(len(boundary_frames) - 1):
            start = boundary_frames[i]
            end = boundary_frames[i + 1]

            if start < features.shape[1] and end <= features.shape[1]:
                segment_features = features[:, start:end]
                if segment_features.size > 0:
                    # Calculate the variance within each segment
                    segment_var = np.mean(np.var(segment_features, axis=1))
                    segment_homogeneity.append(float(1.0 / (1.0 + segment_var)))

        # Estimate repetitive structure
        if S.shape[0] > 0:
            repetitiveness = float(np.mean(S))
        else:
            repetitiveness = 0.0

        return {
            'segment_boundaries': boundary_times.tolist(),
            'novelty_points': novelty_times.tolist(),
            'segment_homogeneity': segment_homogeneity,
            'repetitiveness': repetitiveness
        }


def compare_tracks(track1_features, track2_features):
    """
    Compare two tracks for mixing compatibility.

    Parameters:
    -----------
    track1_features : dict
        Features of the first track
    track2_features : dict
        Features of the second track

    Returns:
    --------
    compatibility : dict
        Dictionary of compatibility scores and mixing recommendations
    """
    compatibility = {}

    # Tempo compatibility
    tempo1 = track1_features['rhythmic']['bpm']
    tempo2 = track2_features['rhythmic']['bpm']

    # Check for tempo doubling/halving relationships
    tempo_ratios = [
        abs(tempo1 - tempo2) / max(tempo1, tempo2),
        abs(tempo1 - 2 * tempo2) / max(tempo1, 2 * tempo2),
        abs(2 * tempo1 - tempo2) / max(2 * tempo1, tempo2)
    ]

    tempo_compatibility = 1.0 - min(tempo_ratios)

    # Tonal compatibility (key matching)
    key1 = track1_features['tonal']['key']
    key2 = track2_features['tonal']['key']

    # Get Camelot key compatibility
    key_compatibility = calculate_key_compatibility(key1, key2)

    # Energy compatibility
    energy1 = track1_features['energy']['rms']['mean']
    energy2 = track2_features['energy']['rms']['mean']

    energy_ratio = min(energy1, energy2) / max(energy1, energy2)

    # Spectral compatibility
    bright1 = track1_features['spectral']['brightness']
    bright2 = track2_features['spectral']['brightness']

    brightness_diff = abs(bright1 - bright2)
    brightness_compatibility = 1.0 - min(brightness_diff, 1.0)

    # Overall compatibility score (weighted sum)
    overall_compatibility = (
            0.35 * tempo_compatibility +
            0.35 * key_compatibility +
            0.15 * energy_ratio +
            0.15 * brightness_compatibility
    )

    # Calculate recommended crossfade duration based on compatibility
    if overall_compatibility > 0.8:
        # Very compatible tracks can have shorter transitions
        min_crossfade = 4  # seconds
    elif overall_compatibility > 0.6:
        # Moderately compatible tracks
        min_crossfade = 8  # seconds
    else:
        # Less compatible tracks need longer crossfades
        min_crossfade = 16  # seconds

    # Mixing recommendations
    if overall_compatibility > 0.7:
        mix_technique = "Harmonic mixing (blend)"
    elif tempo_compatibility > 0.9:
        mix_technique = "Beat matching"
    else:
        mix_technique = "Effect transition (echo out/in)"

    compatibility = {
        'overall_score': float(overall_compatibility),
        'tempo_compatibility': float(tempo_compatibility),
        'key_compatibility': float(key_compatibility),
        'energy_compatibility': float(energy_ratio),
        'spectral_compatibility': float(brightness_compatibility),
        'recommended_min_crossfade': float(min_crossfade),
        'recommended_technique': mix_technique
    }

    return compatibility


def calculate_key_compatibility(key1, key2):
    """
    Calculate musical key compatibility using the Camelot wheel (Circle of Fifths).

    Parameters:
    -----------
    key1 : str
        First key (e.g. "C major", "A minor")
    key2 : str
        Second key

    Returns:
    --------
    compatibility : float
        Compatibility score between 0 and 1
    """
    # Parse keys
    k1_parts = key1.split()
    k2_parts = key2.split()

    if len(k1_parts) < 2 or len(k2_parts) < 2:
        return 0.5  # Default compatibility for unparseable keys

    k1_note = k1_parts[0]
    k1_mode = k1_parts[1]
    k2_note = k2_parts[0]
    k2_mode = k2_parts[1]

    # Convert to Camelot notation
    camelot1 = key_to_camelot(k1_note, k1_mode)
    camelot2 = key_to_camelot(k2_note, k2_mode)

    if camelot1 is None or camelot2 is None:
        return 0.5  # Default compatibility

    # Extract number and letter
    num1, letter1 = int(camelot1[:-1]), camelot1[-1]
    num2, letter2 = int(camelot2[:-1]), camelot2[-1]

    # Calculate compatibility
    if num1 == num2 and letter1 == letter2:
        # Exact same key
        return 1.0
    elif num1 == num2 and letter1 != letter2:
        # Relative major/minor
        return 0.9
    elif abs(num1 - num2) % 12 == 1 and letter1 == letter2:
        # Adjacent key (perfect fifth)
        return 0.8
    elif (abs(num1 - num2) % 12 == 1 and letter1 != letter2) or (abs(num1 - num2) % 12 == 2 and letter1 == letter2):
        # Near keys
        return 0.6
    else:
        # Calculate distance on the Camelot wheel (1-12)
        distance = min(abs(num1 - num2), 12 - abs(num1 - num2))
        # Convert to compatibility score
        return max(0.0, 1.0 - (distance / 6.0))


def key_to_camelot(note, mode):
    """Convert musical key to Camelot wheel notation."""
    # Camelot notation
    camelot_map = {
        'B': {'major': '1B', 'minor': '10A'},
        'F#': {'major': '2B', 'minor': '11A'},
        'Gb': {'major': '2B', 'minor': '11A'},
        'Db': {'major': '3B', 'minor': '12A'},
        'C#': {'major': '3B', 'minor': '12A'},
        'Ab': {'major': '4B', 'minor': '1A'},
        'G#': {'major': '4B', 'minor': '1A'},
        'Eb': {'major': '5B', 'minor': '2A'},
        'D#': {'major': '5B', 'minor': '2A'},
        'Bb': {'major': '6B', 'minor': '3A'},
        'A#': {'major': '6B', 'minor': '3A'},
        'F': {'major': '7B', 'minor': '4A'},
        'C': {'major': '8B', 'minor': '5A'},
        'G': {'major': '9B', 'minor': '6A'},
        'D': {'major': '10B', 'minor': '7A'},
        'A': {'major': '11B', 'minor': '8A'},
        'E': {'major': '12B', 'minor': '9A'}
    }

    try:
        return camelot_map[note][mode.lower()]
    except KeyError:
        return None


# Example usage function
def extract_and_compare_tracks(track1_path, track2_path, output_folder=None):
    """
    Extract features from two tracks and compare their compatibility for mixing.

    Parameters:
    -----------
    track1_path : str
        Path to first audio file
    track2_path : str
        Path to second audio file
    output_folder : str, optional
        Folder to save feature files

    Returns:
    --------
    result : dict
        Contains features of both tracks and their compatibility analysis
    """
    extractor = FeatureExtractor()

    # Extract features
    track1_features = extractor.extract_all_features(track1_path, output_folder)
    track2_features = extractor.extract_all_features(track2_path, output_folder)

    if track1_features is None or track2_features is None:
        return None

    # Compare tracks
    compatibility = compare_tracks(track1_features, track2_features)

    return {
        'track1': {
            'path': track1_path,
            'features': track1_features
        },
        'track2': {
            'path': track2_path,
            'features': track2_features
        },
        'compatibility': compatibility
    }