import numpy as np
import librosa
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def enhanced_beat_detection(audio_path, sr=None, start_time=0, duration=None):
    """
    Advanced beat detection using multiple methods with confidence scoring.

    Parameters:
    -----------
    audio_path : str
        Path to audio file
    sr : int, optional
        Sample rate for audio loading (None uses default)
    start_time : float, optional
        Start time in seconds
    duration : float, optional
        Duration in seconds (None for full file)

    Returns:
    --------
    beat_times : np.ndarray
        Array of beat timestamps in seconds
    confidence : np.ndarray
        Confidence values for each detected beat (0-1)
    bpm : float
        Estimated BPM
    """
    # Load audio with optional start and duration
    if duration:
        offset = start_time
        y, sr = librosa.load(audio_path, sr=sr, offset=offset, duration=duration)
    else:
        y, sr = librosa.load(audio_path, sr=sr, offset=start_time)

    # Method 1: Standard librosa beat tracking
    tempo1, beats1 = librosa.beat.beat_track(y=y, sr=sr)

    # Method 2: Enhanced onset detection with dynamic threshold
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Smooth onset envelope
    onset_env_smooth = gaussian_filter1d(onset_env, sigma=2)

    # Dynamic thresholding based on local context
    local_avg = librosa.util.localmax(onset_env_smooth)
    peaks, _ = find_peaks(onset_env_smooth, height=0.5 * np.mean(onset_env_smooth), distance=sr / tempo1 / 4)

    # Method 3: Tempogram-based beat tracking
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    tempo3 = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]

    # Get frames that correspond to tempo-spaced intervals
    tempo_period = 60 / tempo3  # Convert BPM to seconds per beat
    frames_per_beat = tempo_period * sr / float(librosa.get_duration(y=y, sr=sr) / len(onset_env))

    # Create a consistent grid of beats based on the strongest periodicity
    beat_phase = np.argmax(np.sum(tempogram, axis=1))
    beats3 = np.arange(beat_phase, len(onset_env), frames_per_beat, dtype=int)
    beats3 = beats3[beats3 < len(onset_env)]

    # Convert all beat frames to times
    beat_times1 = librosa.frames_to_time(beats1, sr=sr)
    beat_times2 = librosa.frames_to_time(peaks, sr=sr)
    beat_times3 = librosa.frames_to_time(beats3, sr=sr)

    # Combine all methods - find consensus beats
    # Create a time grid with high resolution
    time_grid = np.linspace(0, librosa.get_duration(y=y, sr=sr), 1000)

    # Create onset functions for each method
    def gaussian_window(center, width=0.05):
        return np.exp(-0.5 * ((time_grid - center) / width) ** 2)

    beat_functions = []
    for beat_set in [beat_times1, beat_times2, beat_times3]:
        beat_function = np.zeros_like(time_grid)
        for beat in beat_set:
            beat_function += gaussian_window(beat)
        beat_functions.append(beat_function)

    # Average the beat functions
    combined_function = np.mean(beat_functions, axis=0)

    # Find peaks in the combined function
    consensus_peaks, properties = find_peaks(combined_function, height=0.4, distance=sr / tempo1 / 4)
    consensus_beat_times = time_grid[consensus_peaks]

    # Get confidence values from peak heights
    confidence = properties['peak_heights'] / np.max(properties['peak_heights'])

    # Get the final BPM estimate by averaging intervals between beats
    if len(consensus_beat_times) > 1:
        beat_intervals = np.diff(consensus_beat_times)
        mean_interval = np.mean(beat_intervals)
        bpm = 60.0 / mean_interval
    else:
        # Fall back to librosa estimate if consensus beats are insufficient
        bpm = tempo1

    return consensus_beat_times, confidence, bpm


def find_beat_drops(audio_path, sr=None):
    """
    Detect significant drops in EDM music (sudden energy changes after buildups).

    Parameters:
    -----------
    audio_path : str
        Path to audio file
    sr : int, optional
        Sample rate for audio loading

    Returns:
    --------
    drop_times : np.ndarray
        Timestamps of detected drops in seconds
    drop_confidence : np.ndarray
        Confidence scores for each drop (0-1)
    """
    y, sr = librosa.load(audio_path, sr=sr)

    # Extract various features for drop detection
    # 1. RMS energy with larger frames to capture energy trends
    hop_length = 512
    frame_length = 2048
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # 2. Spectral flux to detect sudden changes in spectrum
    spec = np.abs(librosa.stft(y, hop_length=hop_length))
    spec_flux = np.sum(np.diff(spec, axis=1) ** 2, axis=0)
    spec_flux = np.concatenate(([0], spec_flux))  # Match dimensions with RMS

    # 3. Low frequency energy ratio (more bass during drops)
    spec_bands = librosa.amplitude_to_db(spec)
    bass_band = np.mean(spec_bands[:int(spec_bands.shape[0] * 0.1)], axis=0)
    mid_band = np.mean(spec_bands[int(spec_bands.shape[0] * 0.1):int(spec_bands.shape[0] * 0.5)], axis=0)
    bass_mid_ratio = bass_band - mid_band

    # Normalize features
    rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-10)
    flux_norm = (spec_flux - np.min(spec_flux)) / (np.max(spec_flux) - np.min(spec_flux) + 1e-10)
    bass_ratio_norm = (bass_mid_ratio - np.min(bass_mid_ratio)) / (
                np.max(bass_mid_ratio) - np.min(bass_mid_ratio) + 1e-10)

    # Compute first-order derivatives
    rms_diff = np.diff(gaussian_filter1d(rms_norm, sigma=5))
    rms_diff = np.concatenate(([0], rms_diff))

    # Create a drop detection function
    drop_function = (flux_norm * 0.4 + rms_diff * 0.3 + bass_ratio_norm * 0.3)
    drop_function = gaussian_filter1d(drop_function, sigma=2)

    # Find peaks in the detection function with minimum distance and height constraints
    min_distance = int(sr / hop_length * 5)  # Minimum 5 seconds between drops
    min_height = np.percentile(drop_function, 90)  # Only consider top 10% as potential drops
    drop_peaks, properties = find_peaks(drop_function, height=min_height, distance=min_distance)

    # Convert frame indices to time
    drop_times = librosa.frames_to_time(drop_peaks, sr=sr, hop_length=hop_length)

    # Calculate confidence based on peak height
    drop_confidence = (properties['peak_heights'] - min_height) / (np.max(properties['peak_heights']) - min_height)

    return drop_times, drop_confidence


def analyze_beat_structure(audio_path, sr=None):
    """
    Analyze the beat structure to identify phrases, sections, and musical structure.

    Parameters:
    -----------
    audio_path : str
        Path to audio file
    sr : int, optional
        Sample rate for audio loading

    Returns:
    --------
    section_boundaries : np.ndarray
        Timestamps of detected section changes in seconds
    phrase_boundaries : np.ndarray
        Timestamps of detected musical phrases in seconds
    structure_labels : list
        Estimated structure labels (e.g., 'A', 'B', 'C')
    """
    y, sr = librosa.load(audio_path, sr=sr)

    # Get beat times
    beat_times, _, _ = enhanced_beat_detection(audio_path, sr)

    # Compute chromagram for harmonic content analysis
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    # Compute MFCCs for timbre analysis
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Perform beat-synchronous feature extraction
    beat_frames = librosa.time_to_frames(beat_times, sr=sr)
    beat_frames = beat_frames[beat_frames < chroma.shape[1]]

    # Handle empty beat frames
    if len(beat_frames) < 2:
        # Return default values if not enough beats detected
        section_boundaries = np.array([])
        phrase_boundaries = beat_times
        structure_labels = ['A']
        return section_boundaries, phrase_boundaries, structure_labels

    beat_chroma = librosa.util.sync(chroma, beat_frames)
    beat_mfcc = librosa.util.sync(mfcc, beat_frames)

    # Concatenate features for segmentation
    beat_features = np.vstack([beat_chroma, beat_mfcc])

    # Compute self-similarity matrix
    S = librosa.segment.recurrence_matrix(beat_features, mode='affinity', width=3)

    # Enhance diagonal
    S = librosa.segment.path_enhance(S, 15)

    # Detect section boundaries using spectral clustering
    # Use a smaller number of segments for shorter tracks
    num_segments = min(8, max(2, len(beat_frames) // 8))
    section_idx = librosa.segment.agglomerative(S, num_segments)

    # Convert to boundaries - FIXED to handle array dimension mismatch
    if len(section_idx) > 0 and len(beat_times) > len(section_idx):
        # Ensure beat_times and section_idx have compatible dimensions
        section_boundaries = beat_times[1:len(section_idx) + 1][np.diff(section_idx) != 0]
    else:
        # Fallback if dimensions don't match or no sections detected
        section_boundaries = np.array([beat_times[len(beat_times) // 2]]) if len(beat_times) > 1 else np.array([])

    # Estimate smaller phrases (typically 4, 8, or 16 beats in EDM)
    if len(beat_times) >= 16:
        # Start with assumption of 4-beat phrases
        phrase_length = 4

        # Check if 8-beat phrases are more likely
        beat_intervals = np.diff(beat_times)
        if len(beat_intervals) >= 16:
            auto_corr = np.correlate(beat_intervals, beat_intervals, mode='full')
            center = len(auto_corr) // 2
            auto_corr = auto_corr[center:]

            # Look for peaks at 4, 8, and 16 positions
            peak_positions = [4, 8, 16]
            peak_values = [auto_corr[min(p, len(auto_corr) - 1)] for p in peak_positions]
            phrase_length = peak_positions[np.argmax(peak_values)]

        # Create phrase boundaries every phrase_length beats
        phrase_indices = np.arange(0, len(beat_times), phrase_length)
        phrase_boundaries = beat_times[phrase_indices]
    else:
        # Fall back for very short clips
        phrase_boundaries = beat_times

    # Assign structure labels (A, B, C, etc.)
    unique_sections = np.unique(section_idx)
    structure_labels = [chr(65 + i) for i in range(len(unique_sections))]  # A, B, C...

    return section_boundaries, phrase_boundaries, structure_labels


def analyze_transition_points(audio_path, sr=None):
    """
    Find optimal transition points for DJ mixing.

    Parameters:
    -----------
    audio_path : str
        Path to audio file
    sr : int, optional
        Sample rate for audio loading

    Returns:
    --------
    transition_points : dict
        Dictionary with:
        - 'intro_end': Time when intro ends (seconds)
        - 'outro_start': Time when outro begins (seconds)
        - 'drop_points': List of drop times (seconds)
        - 'breakdown_points': List of breakdown points (seconds)
        - 'transition_quality': Score for each beat (0-1)
    """
    y, sr = librosa.load(audio_path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    # Get beat times and section boundaries
    beat_times, beat_confidence, bpm = enhanced_beat_detection(audio_path, sr)

    try:
        section_boundaries, phrase_boundaries, _ = analyze_beat_structure(audio_path, sr)
    except Exception as e:
        print(f"Warning: Beat structure analysis failed, using simplified analysis: {e}")
        # Provide simplified fallback values
        section_boundaries = np.array([])
        phrase_boundaries = beat_times[::4] if len(beat_times) >= 4 else beat_times

    try:
        drop_times, _ = find_beat_drops(audio_path, sr)
    except Exception as e:
        print(f"Warning: Drop detection failed: {e}")
        drop_times = np.array([])

    # Extract energy profile
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Smooth RMS
    rms_smooth = gaussian_filter1d(rms, sigma=10)

    # Find intro end (first significant energy increase)
    rms_norm = (rms_smooth - np.min(rms_smooth)) / (np.max(rms_smooth) - np.min(rms_smooth) + 1e-10)
    energy_thresh = 0.2
    intro_frames = np.where(rms_norm > energy_thresh)[0]
    intro_end = rms_times[intro_frames[0]] if len(intro_frames) > 0 else 0

    # Find outro start (last significant energy decrease)
    outro_start = duration - 30  # Default to 30 sec before end

    # Refine outro start using section boundaries
    if len(section_boundaries) > 0:
        last_sections = section_boundaries[section_boundaries > 0.7 * duration]
        if len(last_sections) > 0:
            outro_start = last_sections[0]

    # Find breakdowns (energy dips between drops)
    breakdown_points = []
    if len(drop_times) >= 2:
        for i in range(len(drop_times) - 1):
            start_idx = np.argmin(np.abs(rms_times - drop_times[i])) + int(8 * sr / hop_length)  # 8 sec after drop
            end_idx = np.argmin(np.abs(rms_times - drop_times[i + 1]))

            if end_idx - start_idx > 0:
                segment_rms = rms_norm[start_idx:end_idx]
                local_min_idx = np.argmin(segment_rms) + start_idx
                breakdown_points.append(rms_times[local_min_idx])

    # Calculate transition quality for each beat
    transition_quality = np.zeros_like(beat_times)

    for i, beat_time in enumerate(beat_times):
        # Factors that make a good transition point:

        # 1. Phrase boundary bonus (phrase start/end are good transition points)
        phrase_factor = 0
        if np.min(np.abs(phrase_boundaries - beat_time)) < 0.1:
            phrase_factor = 1.0

        # 2. Beat confidence
        conf_factor = beat_confidence[i] if i < len(beat_confidence) else 0.5

        # 3. Energy level (mid-energy is best for transitions)
        beat_idx = np.argmin(np.abs(rms_times - beat_time))
        energy_level = rms_norm[beat_idx] if beat_idx < len(rms_norm) else 0.5
        energy_factor = 1.0 - 2.0 * abs(energy_level - 0.5)  # Peak at 0.5, lower at 0 and 1

        # 4. Proximity to drops (avoid transitions near drops)
        if len(drop_times) > 0:
            drop_distance = np.min(np.abs(drop_times - beat_time))
            drop_factor = min(drop_distance / 16.0, 1.0)  # Avoid 16 sec around drops
        else:
            drop_factor = 1.0

        # 5. Not too close to intro/outro
        position_factor = min(beat_time / 16.0, 1.0) * min((duration - beat_time) / 16.0, 1.0)

        # Weighted sum
        transition_quality[i] = (
                0.3 * phrase_factor +
                0.2 * conf_factor +
                0.2 * energy_factor +
                0.2 * drop_factor +
                0.1 * position_factor
        )

    return {
        'intro_end': intro_end,
        'outro_start': outro_start,
        'drop_points': drop_times.tolist(),
        'breakdown_points': breakdown_points,
        'transition_quality': transition_quality.tolist(),
        'beat_times': beat_times.tolist(),
        'bpm': bpm
    }