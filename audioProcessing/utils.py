import os
import numpy as np
import librosa
import soundfile as sf
import csv
import json
from pathlib import Path
from scipy.signal import butter, lfilter
import scipy.ndimage


def load_audio(file_path, sr=44100):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None, None


def match_target_tempo(y, sr, source_bpm, target_bpm):
    ratio = target_bpm / source_bpm

    return librosa.effects.time_stretch(y, rate=ratio)


def tempo_ramp(y, sr, source_bpm, target_bpm, ramp_duration=32.0):
    initial_ratio = target_bpm / source_bpm

    ramp_samples = int(ramp_duration * sr)

    # If the audio is shorter than the ramp duration
    #  use regular time stretching
    if len(y) <= ramp_samples:
        return librosa.effects.time_stretch(y, rate=initial_ratio)

    transition_section = y[:ramp_samples]
    stretched_transition = librosa.effects.time_stretch(transition_section, rate=initial_ratio)

    ramping_section = y[ramp_samples:ramp_samples * 2]

    num_segments = 8
    segment_size = len(ramping_section) // num_segments

    ramped_segments = []
    for i in range(num_segments):
        segment_ratio = initial_ratio + (1.0 - initial_ratio) * (i / (num_segments - 1))

        segment = ramping_section[i * segment_size:(i + 1) * segment_size]
        stretched_segment = librosa.effects.time_stretch(segment, rate=segment_ratio)
        ramped_segments.append(stretched_segment)

    stretched_ramping = np.concatenate(ramped_segments)

    if len(y) > ramp_samples * 2:
        remainder_section = y[ramp_samples * 2:]
        stretched_remainder = remainder_section
    else:
        stretched_remainder = np.array([])

    return np.concatenate([stretched_transition, stretched_ramping, stretched_remainder])


def improved_crossfade(y1, y2, sr, fade_duration=8.0, curve_type="logarithmic"):
    fade_samples = int(fade_duration * sr)

    if len(y1) < fade_samples or len(y2) < fade_samples:
        raise ValueError("Audio segments too short for this crossfade length")

    if curve_type == "logarithmic":
        # Curved crossfade
        fade_out = np.logspace(0, -2, fade_samples) / 3.16  # Normalized
        fade_in = 1 - np.logspace(0, -2, fade_samples)[::-1] / 3.16
    else:
        # Linear fades
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        fade_in = np.linspace(0.0, 1.0, fade_samples)

    # Apply fades
    y1_end = y1[-fade_samples:]
    y1_end_faded = y1_end * fade_out

    y2_start = y2[:fade_samples]
    y2_start_faded = y2_start * fade_in

    crossfaded = y1_end_faded + y2_start_faded

    result = np.concatenate([y1[:-fade_samples], crossfaded, y2[fade_samples:]])

    return result


def multiband_crossfade(y1, y2, sr, fade_duration=16.0, vocal_aware=False, vocal_segments1=None, vocal_segments2=None):
    fade_samples = int(fade_duration * sr)

    if len(y1) < fade_samples or len(y2) < fade_samples:
        # Fall back to basic crossfade if segments are too short
        return improved_crossfade(y1, y2, sr, fade_duration / 2)

    # Split into frequency bands
    y1_low = butter_lowpass_filter(y1, 300, sr)
    y1_mid = y1 - y1_low - butter_highpass_filter(y1, 4000, sr)
    y1_high = butter_highpass_filter(y1, 4000, sr)

    y2_low = butter_lowpass_filter(y2, 300, sr)
    y2_mid = y2 - y2_low - butter_highpass_filter(y2, 4000, sr)
    y2_high = butter_highpass_filter(y2, 4000, sr)

    if vocal_aware:
        y1_vocal = butter_bandpass_filter(y1, 300, 3500, sr)
        y2_vocal = butter_bandpass_filter(y2, 300, 3500, sr)

    y1_transition = y1[-fade_samples:]
    y2_transition = y2[:fade_samples]

    # Creating crossfade curves for different frequency bands
    # Low frequencies (bass)  short crossfade to maintain impact
    fade_out_bass = np.linspace(1.0, 0.0, fade_samples) ** 2
    fade_in_bass = np.linspace(0.0, 1.0, fade_samples) ** 2

    # Mid frequencies  standard crossfade
    fade_out_mid = np.linspace(1.0, 0.0, fade_samples)
    fade_in_mid = np.linspace(0.0, 1.0, fade_samples)

    # High frequencies slightly faster to maintain clarity
    fade_out_high = np.linspace(1.0, 0.0, fade_samples) ** 0.8
    fade_in_high = np.linspace(0.0, 1.0, fade_samples) ** 0.8

    if vocal_aware and vocal_segments1 and vocal_segments2:
        # Create fade masks based on vocal presence
        vocal_mask1 = np.ones(fade_samples)
        vocal_mask2 = np.ones(fade_samples)

        # Convert time ranges to sample indices
        for start, end in vocal_segments1:
            # Map to the crossfade region
            start_idx = max(0, int((start - (len(y1) - fade_samples) / sr) * sr))
            end_idx = min(fade_samples, int((end - (len(y1) - fade_samples) / sr) * sr))

            if end_idx > start_idx and start_idx < fade_samples:
                # Apply a deeper fade during vocal segments
                vocal_mask1[start_idx:end_idx] = 0.3

        for start, end in vocal_segments2:
            # Map to the crossfade region
            start_idx = max(0, int(start * sr))
            end_idx = min(fade_samples, int(end * sr))

            if end_idx > start_idx and start_idx < fade_samples:
                # Apply a deeper fade during vocal segments
                vocal_mask2[start_idx:end_idx] = 0.3

        # Smooth the masks to avoid abrupt changes
        vocal_mask1 = scipy.ndimage.gaussian_filter1d(vocal_mask1, sigma=sr / 100)
        vocal_mask2 = scipy.ndimage.gaussian_filter1d(vocal_mask2, sigma=sr / 100)

        # Apply to mid-band where vocals mostly live
        fade_out_mid = fade_out_mid * vocal_mask1
        fade_in_mid = fade_in_mid * vocal_mask2

    # Apply fades to each frequency band
    y1_low_faded = y1_low[-fade_samples:] * fade_out_bass
    y2_low_faded = y2_low[:fade_samples] * fade_in_bass

    y1_mid_faded = y1_mid[-fade_samples:] * fade_out_mid
    y2_mid_faded = y2_mid[:fade_samples] * fade_in_mid

    y1_high_faded = y1_high[-fade_samples:] * fade_out_high
    y2_high_faded = y2_high[:fade_samples] * fade_in_high

    # Combine all bands
    crossfaded = (y1_low_faded + y2_low_faded) + (y1_mid_faded + y2_mid_faded) + (y1_high_faded + y2_high_faded)

    result = np.concatenate([y1[:-fade_samples], crossfaded, y2[fade_samples:]])

    # Normalize if necessary
    max_val = np.max(np.abs(result))
    if max_val > 0.98:
        result = result * (0.95 / max_val)

    return result


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    """Apply bandpass filter to audio"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Prevent issues with extreme values
    low = max(0.001, min(0.99, low))
    high = max(low + 0.001, min(0.99, high))

    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y


def butter_highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y


def create_robust_mix(transitions, output_path, min_crossfade=8.0):
    mix = np.array([])
    sr = 44100

    print(f"Creating mix with {len(transitions)} tracks")

    print("Validating audio files...")
    valid_transitions = []
    for i, transition in enumerate(transitions):
        y, sr_check = load_audio(transition['track_path'])
        if y is None:
            print(f"Could not load track {transition['track_path']} - skipping")
            continue
        valid_transitions.append(transition)

    if not valid_transitions:
        raise ValueError("No valid audio files found in transition list")

    print(f"Found {len(valid_transitions)} valid tracks")
    transitions = valid_transitions

    # Track to store the last audio for verification
    last_audio_segment = None
    last_audio_path = None

    # Process each track
    for i, transition in enumerate(transitions):
        track_name = os.path.basename(transition['track_path'])
        print(f"Processing track {i + 1}/{len(transitions)}: {track_name}")

        # Load current track
        try:
            y_current, sr = librosa.load(transition['track_path'], sr=sr)
        except Exception as e:
            print(f"Error loading {transition['track_path']}: {e}")
            continue

        # Normalize audio to prevent volume jumps
        if np.max(np.abs(y_current)) > 0:
            # Only normalize if audio isn't silent
            target_peak = 0.9
            current_peak = np.max(np.abs(y_current))
            y_current = y_current * (target_peak / current_peak)

        track_duration = len(y_current) / sr
        print(f"  Track duration: {track_duration:.2f} seconds")

        # First track - just add it up to its exit point
        if i == 0:
            # First track has an exit point (will be mixed with next track)
            if transition['exit_point'] is not None and transition['exit_point'] < track_duration:
                exit_sample = int(transition['exit_point'] * sr)
                print(f"  Adding track from beginning to exit point ({transition['exit_point']:.1f}s)")
                current_segment = y_current[:exit_sample]

                # Verify not adding silence
                if np.max(np.abs(current_segment)) < 0.01:
                    print("Warning: First track is nearly silent. Using more of the next track.")
                    # Find next non-silent section
                    for test_point in [0.25, 0.5, 0.75]:
                        test_sample = int(track_duration * test_point * sr)
                        if test_sample < len(y_current):
                            test_segment = y_current[:test_sample]
                            if np.max(np.abs(test_segment)) >= 0.01:
                                current_segment = test_segment
                                print(f"  Using first {test_point * 100:.0f}% of track instead")
                                break

                mix = current_segment
            else:
                print(f"  Adding entire track")
                mix = y_current

            last_audio_segment = mix
            last_audio_path = transition['track_path']
        else:
            # For next tracks, create a transition
            prev_transition = transitions[i - 1]
            prev_track_path = prev_transition['track_path']

            if prev_transition['exit_point'] is not None:
                # Verify there is a valid mix so far
                if len(mix) == 0:
                    print("Warning: Mix is empty. Starting with current track.")
                    if transition['exit_point'] is not None:
                        mix = y_current[:int(transition['exit_point'] * sr)]
                    else:
                        mix = y_current
                    continue

                # Verify the last segment isn't silent
                if np.max(np.abs(mix[-sr:])) < 0.01:
                    print("Warning: End of mix is silent. Finding better transition point.")
                    # Find last non-silent section
                    for test_offset in [2, 4, 8, 16]:
                        if len(mix) > test_offset * sr:
                            test_segment = mix[-test_offset * sr:]
                            if np.max(np.abs(test_segment)) >= 0.01:
                                # Trim mix any silent points in the mix
                                mix = mix[:-test_offset * sr]
                                print(f"  Trimmed {test_offset}s of silence from end of mix")
                                break

                try:
                    y_prev, sr_prev = librosa.load(prev_track_path, sr=sr)
                except Exception as e:
                    print(f"  Error reloading previous track: {e}")
                    print(f"  Falling back to using existing mix")
                    if transition['exit_point'] is not None:
                        mix = np.concatenate([mix, y_current[:int(transition['exit_point'] * sr)]])
                    else:
                        mix = np.concatenate([mix, y_current])
                    continue

                # Get BPMs for tempo matching
                prev_bpm = prev_transition.get('bpm', 120)
                current_bpm = transition.get('bpm', 120)

                # Check for vocal information
                has_vocal_info = 'vocal_segments' in prev_transition and 'vocal_segments' in transition

                # longer crossfades for larger BPM differences
                # and when vocal clashes are detected
                bpm_ratio = max(prev_bpm, current_bpm) / min(prev_bpm, current_bpm)

                # Longer crossfade for BPM differences or vocal clashes
                if bpm_ratio > 1.08 or (has_vocal_info and len(prev_transition.get('vocal_segments', [])) > 0):
                    base_crossfade = 16.0
                else:
                    base_crossfade = 12.0

                crossfade_duration = max(min_crossfade,
                                         transition.get('crossfade_duration', base_crossfade))

                # Increase crossfade further if there are detected vocal clashes
                if has_vocal_info and 'vocal_clash_score' in transition and transition['vocal_clash_score'] < 0.7:
                    crossfade_duration *= 1.5  # 50% longer for vocal clashes

                print(f"  Creating {crossfade_duration:.1f}s crossfade (BPM ratio: {bpm_ratio:.2f})")

                # Get exit point from previous track
                exit_sample = int(prev_transition['exit_point'] * sr_prev)

                if exit_sample >= len(y_prev):
                    print(
                        f"  ⚠️ Exit point {prev_transition['exit_point']:.1f}s beyond track length ({len(y_prev) / sr_prev:.1f}s)")
                    exit_sample = len(y_prev) - 1

                prev_crossfade_start = max(0, exit_sample - int(crossfade_duration * sr_prev))

                # Apply tempo adjustment if BPMs differ significantly
                if abs(prev_bpm - current_bpm) > 3.0:
                    print(f"  Adjusting tempo from {current_bpm} to {prev_bpm} BPM with ramping")
                    try:
                        y_current_adjusted = tempo_ramp(y_current, sr, current_bpm, prev_bpm,
                                                        ramp_duration=32.0)
                    except Exception as e:
                        print(f"  Error in tempo adjustment: {e}")
                        y_current_adjusted = y_current
                else:
                    y_current_adjusted = y_current

                current_segment = y_current_adjusted[:int(crossfade_duration * sr)]

                if prev_crossfade_start >= exit_sample:
                    print(f"Track too short for crossfade")
                    # Just use what we have
                    prev_segment = y_prev[max(0, exit_sample - int(sr_prev)):exit_sample]
                else:
                    prev_segment = y_prev[prev_crossfade_start:exit_sample]

                min_segment_length = int(crossfade_duration * sr * 0.5)
                if len(prev_segment) < min_segment_length or len(current_segment) < min_segment_length:
                    crossfade_length = min(len(prev_segment), len(current_segment))
                    crossfade_duration = crossfade_length / sr
                    print(f"  ⚠️ Short segments, using {crossfade_duration:.1f}s crossfade")

                # Verify segments aren't silent
                if np.max(np.abs(prev_segment)) < 0.01:
                    print("  ⚠️ Warning: Previous segment is nearly silent")
                if np.max(np.abs(current_segment)) < 0.01:
                    print("  ⚠️ Warning: Current segment is nearly silent")

                # Find where the previous track segment starts in the mix
                mix_length = len(mix)

                # If r=theres more than the previous track in the mix already
                # need to find where to start the crossfade
                if mix_length > 0:
                    # Calculate how many samples to keep from existing mix
                    mix_to_keep = mix_length - len(prev_segment)

                    if mix_to_keep <= 0:
                        print("  ⚠️ Mix is shorter than needed for crossfade, using all of it")
                        mix_to_keep = 0

                    # Keep the portion of the mix up to the crossfade point
                    mix_part = mix[:mix_to_keep]
                else:
                    mix_part = np.array([])

                # Create enhanced crossfade between previous and current segments
                try:
                    if has_vocal_info:
                        # Extract vocal segments for the transition regions
                        prev_vocals = prev_transition.get('vocal_segments', [])
                        current_vocals = transition.get('vocal_segments', [])

                        crossfaded = multiband_crossfade(
                            prev_segment,
                            current_segment,
                            sr,
                            crossfade_duration,
                            vocal_aware=True,
                            vocal_segments1=prev_vocals,
                            vocal_segments2=current_vocals
                        )
                    else:
                        crossfaded = improved_crossfade(
                            prev_segment,
                            current_segment,
                            sr,
                            crossfade_duration,
                            curve_type="logarithmic"
                        )

                except Exception as e:
                    print(f"  ⚠️ Error in crossfade: {e}")
                    # Just append as fallback
                    crossfaded = np.concatenate([prev_segment, current_segment])

                # Verify crossfade isn't silent
                if np.max(np.abs(crossfaded)) < 0.01:
                    print("  ⚠️ Warning: Crossfade is silent, fix")
                    # Just concatenate as a last resort
                    crossfaded = np.concatenate([prev_segment, current_segment])

                # Add existing mix, then crossfade
                mix = np.concatenate([mix_part, crossfaded])

                # Add rest of current playing track
                if len(current_segment) < len(y_current_adjusted):
                    rest_of_track = y_current_adjusted[len(current_segment):]

                    # If this track has an exit point, trim it
                    if transition['exit_point'] is not None and transition['exit_point'] < len(y_current_adjusted) / sr:
                        end_sample = int(transition['exit_point'] * sr)
                        if len(current_segment) < end_sample < len(y_current_adjusted):
                            rest_of_track = y_current_adjusted[len(current_segment):end_sample]

                    mix = np.concatenate([mix, rest_of_track])
            else:
                print("  No exit point in previous track, adding current track directly")
                if transition['exit_point'] is not None and transition['exit_point'] < len(y_current) / sr:
                    segment_to_add = y_current[:int(transition['exit_point'] * sr)]
                else:
                    segment_to_add = y_current

                if np.max(np.abs(segment_to_add)) < 0.01:
                    print("  ⚠️ Warning: Segment is nearly silent. Using full track.")
                    segment_to_add = y_current

                mix = np.concatenate([mix, segment_to_add])

        if np.max(np.abs(mix[-sr:])) < 0.01:
            print("  ⚠️ Warning: End of mix is silent after adding track.")

        last_audio_segment = y_current
        last_audio_path = transition['track_path']

    # Final safety check - ensure there is audio
    if len(mix) == 0:
        print("❌ ERROR: Mix is empty! Last valid track as fallback.")
        if last_audio_segment is not None:
            mix = last_audio_segment
        else:
            raise ValueError("No valid audio to create mix")

    rms_values = librosa.feature.rms(y=mix, frame_length=4096, hop_length=2048)[0]
    silent_frames = np.where(rms_values < 0.001)[0]

    if len(silent_frames) > 0:
        if len(silent_frames) > 0:
            silent_groups = []
            current_group = [silent_frames[0]]

            for i in range(1, len(silent_frames)):
                if silent_frames[i] == silent_frames[i - 1] + 1:
                    current_group.append(silent_frames[i])
                else:
                    if len(current_group) > int(2 * sr / 2048):  # More than 2 seconds
                        silent_groups.append(current_group)
                    current_group = [silent_frames[i]]

            if len(current_group) > int(2 * sr / 2048):
                silent_groups.append(current_group)

            if silent_groups:
                print(f"⚠Warning: Found {len(silent_groups)} extended silent sections in final mix")

    if np.max(np.abs(mix)) > 0:
        if np.max(np.abs(mix)) < 0.1:
            print("Warning: Mix is very quiet. Applying gain.")
            mix = mix * (0.9 / np.max(np.abs(mix)))

    # Print final mix stats
    mix_duration_sec = len(mix) / sr
    mix_duration_min = mix_duration_sec / 60
    print(f"Final mix length: {mix_duration_min:.2f} minutes ({mix_duration_sec:.2f} seconds)")
    print(f"Peak amplitude: {np.max(np.abs(mix)):.3f}")

    try:
        sf.write(output_path, mix, sr)
        print(f"Mix successfully exported to {output_path}")
    except Exception as e:
        print(f"❌ Error saving mix: {e}")
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            # Try writing with scipy instead
            from scipy.io import wavfile
            wavfile.write(output_path, sr, (mix * 32767).astype(np.int16))
            print(f"Mix saved using alternate method to {output_path}")
        except Exception as e2:
            print(f"❌ Failed to save mix with alternate method: {e2}")
            raise

    return mix


def detect_vocals_in_track(audio_path, sr=44100):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"Error loading audio for vocal detection: {e}")
        return []

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)

    y_harmonic, y_percussive = librosa.effects.hpss(y)

    mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)

    contrast = librosa.feature.spectral_contrast(y=y_harmonic, sr=sr)

    y_vocals = butter_bandpass_filter(y_harmonic, 200, 3500, sr)

    # RMS energy of the vocal range
    rms_vocals = librosa.feature.rms(y=y_vocals)[0]

    # overall RMS energy
    rms_full = librosa.feature.rms(y=y)[0]

    # Calculate the ratio, higher for vocal sections
    vocal_ratio = np.zeros_like(rms_vocals)
    nonzero_indices = rms_full > 0
    vocal_ratio[nonzero_indices] = rms_vocals[nonzero_indices] / rms_full[nonzero_indices]

    # Smooth the ratio curve
    hop_length = 512
    vocal_ratio_smooth = scipy.ndimage.gaussian_filter1d(vocal_ratio, sigma=sr / hop_length / 2)

    # vocal detection threshold based on song
    threshold = np.mean(vocal_ratio_smooth) + 0.8 * np.std(vocal_ratio_smooth)

    vocal_frames = np.where(vocal_ratio_smooth > threshold)[0]

    # Convert frames to time
    frame_times = librosa.frames_to_time(np.arange(len(vocal_ratio)), sr=sr, hop_length=hop_length)

    # Group into segments
    vocal_segments = []
    if len(vocal_frames) > 0:
        segment_start = frame_times[vocal_frames[0]]
        prev_frame = vocal_frames[0]

        for i in range(1, len(vocal_frames)):
            if vocal_frames[i] - prev_frame > int(0.5 * sr / hop_length):
                segment_end = frame_times[prev_frame]
                # Only keep segments longer than 0.5 seconds
                if segment_end - segment_start >= 0.5:
                    vocal_segments.append([segment_start, segment_end])
                segment_start = frame_times[vocal_frames[i]]

            prev_frame = vocal_frames[i]

        segment_end = frame_times[prev_frame]
        if segment_end - segment_start >= 0.5:
            vocal_segments.append([segment_start, segment_end])

    return vocal_segments


def analyze_and_enhance_transitions(transitions):
    enhanced_transitions = []

    # Detect vocals in each track
    for i, transition in enumerate(transitions):
        track_path = transition['track_path']
        print(f"Exporting Mix {i + 1}: {os.path.basename(track_path)}")

        # Detect vocals
        vocal_segments = detect_vocals_in_track(track_path)

        # Add vocal information to transitio
        enhanced_transition = transition.copy()
        enhanced_transition['vocal_segments'] = vocal_segments
        enhanced_transition['has_vocals'] = len(vocal_segments) > 0

        if i > 0 and 'vocal_segments' in enhanced_transitions[-1]:
            prev_vocals = enhanced_transitions[-1]['vocal_segments']

            # If either track has vocals, check compatibility
            if len(prev_vocals) > 0 or len(vocal_segments) > 0:
                # Calculate default crossfade duration
                crossfade_duration = transition.get('recommended_crossfade', 12.0)

                prev_exit_point = enhanced_transitions[-1].get('exit_point', 0)
                current_entry_point = 0

                # Findinh vocals that might clash during transition
                exit_vocal_clash = any(
                    prev_exit_point - crossfade_duration <= end and prev_exit_point >= start
                    for start, end in prev_vocals
                )

                entry_vocal_clash = any(
                    current_entry_point <= end and current_entry_point + crossfade_duration >= start
                    for start, end in vocal_segments
                )

                # Calculate a vocal clash score (1.0 = no clash, 0.0 = complete clash)
                vocal_clash_score = 0.2  # default low score

                if not exit_vocal_clash and not entry_vocal_clash:
                    vocal_clash_score = 1.0  # No clash
                elif not exit_vocal_clash or not entry_vocal_clash:
                    vocal_clash_score = 0.7  # Partial clash

                # Add score to transition info
                enhanced_transition['vocal_clash_score'] = vocal_clash_score

                # Adjust recommended crossfade based on vocals
                if vocal_clash_score < 0.8 and 'recommended_crossfade' in enhanced_transition:
                    # Increase crossfade duration for vocals
                    enhanced_transition['recommended_crossfade'] *= 1.5

        enhanced_transitions.append(enhanced_transition)

    return enhanced_transitions


def export_to_csv(data, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["File", "BPM", "Key", "Energy"])

        for item in data:
            writer.writerow([
                item.get('file', ''),
                item.get('bpm', ''),
                item.get('key', ''),
                item.get('energy', '')
            ])


def export_to_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)