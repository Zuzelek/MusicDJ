import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, savgol_filter, find_peaks
import scipy.ndimage
from sklearn.preprocessing import MinMaxScaler


class TransitionPointDetector:
    def __init__(self):
        # basic params for detectin
        self.MIN_SECTION_LENGTH = 4
        self.DROP_ENERGY_RATIO = 1.7
        self.BREAKDOWN_ENERGY_RATIO = 0.55
        self.INTRO_LENGTH_RATIO = 0.12
        self.OUTRO_LENGTH_RATIO = 0.18

        self.SCORE_WEIGHTS = {
            'rhythmic_stability': 0.30,
            'energy_compatibility': 0.25,
            'structural_position': 0.20,
            'harmonic_compatibility': 0.15,
            'vocal_compatibility': 0.10
        }
        self.major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        self.minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])

        self.KEY_MAPPING = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        # camelot wheel - DJs need this for key mixing
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
        # load da file
        y, sr = librosa.load(audio_path, sr=44100)
        duration = librosa.get_duration(y=y, sr=sr)

        bpm = self._get_tempo(y, sr)
        key = self._get_key(y, sr)
        energy_profile = self._get_energy_profile(y)
        energy_curve = self._get_energy_curve(y)

        # smooth it cuz raw is too jumpy
        energy_curve_smooth = savgol_filter(energy_curve,
                                            int(sr / 512 * 8) if len(energy_curve) > sr / 512 * 8 else 3,
                                            2)

        # normalize 0-1 makes it easier
        scaler = MinMaxScaler()
        energy_curve_norm = scaler.fit_transform(energy_curve_smooth.reshape(-1, 1)).flatten()

        beat_times, beat_confidence = self._get_beats_with_confidence(y, sr)
        vocal_segments = self._detect_vocals(y, sr)

        # find all sections & transition points
        sections = self._detect_sections(energy_curve_norm, sr, duration, beat_times)
        transition_points = self._detect_transition_points(sections, energy_curve_norm, sr, beat_times, vocal_segments)
        transition_points = self._improve_transition_points(transition_points, y, sr, beat_times)

        # wrap it up neatly
        return {
            'duration': duration,
            'bpm': bpm,
            'key': key,
            'camelot_key': self._get_camelot_key(key),
            'energy_profile': float(energy_profile),
            'sections': sections,
            'transition_points': transition_points,
            'beat_times': beat_times.tolist() if isinstance(beat_times, np.ndarray) else beat_times,
            'vocal_segments': vocal_segments
        }

    def find_optimal_transition(self, track1_analysis, track2_analysis):
        # check if keys are compatible
        harmonic_score = self._harmonic_compatibility_score(
            track1_analysis['camelot_key'],
            track2_analysis['camelot_key']
        )

        # potential points to mix
        exit_points = track1_analysis['transition_points']['exits']
        entry_points = track2_analysis['transition_points']['entries']

        # check for vocals cuz they can clash
        vocals1 = track1_analysis.get('vocal_segments', [])
        vocals2 = track2_analysis.get('vocal_segments', [])

        best_score = -1
        best_transition = None

        for exit_point in exit_points:
            for entry_point in entry_points:
                # check all the factors
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

                # vocals clashing is bad
                vocal_score = self._vocal_compatibility_score(
                    exit_point['time'], vocals1,
                    entry_point['time'], vocals2,
                    crossfade_duration=12.0
                )

                # boost the score for really nice transitions
                if exit_point['type'] == 'outro_start' and entry_point['type'] == 'intro_end':
                    structure_score *= 1.5

                if 'breakdown' in exit_point['type'] and 'breakdown' in entry_point['type']:
                    if energy_score > 0.8:
                        structure_score *= 1.2

                # calc the total score
                total_score = (
                        self.SCORE_WEIGHTS['rhythmic_stability'] * rhythm_score +
                        self.SCORE_WEIGHTS['energy_compatibility'] * energy_score +
                        self.SCORE_WEIGHTS['structural_position'] * structure_score +
                        self.SCORE_WEIGHTS['harmonic_compatibility'] * harmonic_score +
                        self.SCORE_WEIGHTS['vocal_compatibility'] * vocal_score
                )

                # penalize cutting a track too early - thats rude
                if exit_point['time'] < track1_analysis['duration'] * 0.5:
                    early_exit_penalty = 0.5 * (1 - exit_point['time'] / (track1_analysis['duration'] * 0.5))
                    total_score -= early_exit_penalty

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
                            'harmonic': harmonic_score,
                            'vocal': vocal_score
                        }
                    }

                    # figure out how long to xfade based on type
                    if vocal_score < 0.7:
                        if 'breakdown' in exit_point['type'] or 'breakdown' in entry_point['type']:
                            best_transition['recommended_crossfade'] = 24.0
                        else:
                            best_transition['recommended_crossfade'] = 16.0
                    else:
                        if 'breakdown' in exit_point['type'] and 'breakdown' in entry_point['type']:
                            best_transition['recommended_crossfade'] = 16.0
                        elif 'drop' in exit_point['type'] and 'drop' in entry_point['type']:
                            best_transition['recommended_crossfade'] = 8.0
                        elif exit_point['type'] == 'outro_start' and entry_point['type'] == 'intro_end':
                            best_transition['recommended_crossfade'] = 16.0
                        else:
                            best_transition['recommended_crossfade'] = 12.0

        return best_transition

    def _detect_vocals(self, y, sr):
        # this is magic - uses harmonic content to find vocals
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

        y_harmonic = librosa.effects.harmonic(y)

        #  some features to detect human voice
        # mfcc is short term power spectrum of a sound
        # constrst is repetition and interesting prts
        mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)
        contrast = librosa.feature.spectral_contrast(y=y_harmonic, sr=sr)

        # split into drums vs melodic parts
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # calc the energy profile of the melodic part
        harmonic_rms = librosa.feature.rms(y=y_harmonic)[0]

        # smoooth it out
        harmonic_rms_smooth = savgol_filter(harmonic_rms,
                                            int(sr / 512 * 4) if len(harmonic_rms) > sr / 512 * 4 else 3,
                                            2)

        # if its above threshold, probly vocals
        threshold = np.mean(harmonic_rms_smooth) + 0.5 * np.std(harmonic_rms_smooth)
        vocal_frames = np.where(harmonic_rms_smooth > threshold)[0]

        # cnvrt to real time
        hop_length = 512
        frame_times = librosa.frames_to_time(np.arange(len(harmonic_rms)), sr=sr, hop_length=hop_length)

        # group into chunks so  system knows when singing happens
        vocal_segments = []
        if len(vocal_frames) > 0:
            segment_start = frame_times[vocal_frames[0]]
            prev_frame = vocal_frames[0]

            for i in range(1, len(vocal_frames)):
                if vocal_frames[i] - prev_frame > int(0.5 * sr / hop_length):
                    segment_end = frame_times[prev_frame]
                    if segment_end - segment_start >= 0.3:
                        vocal_segments.append([segment_start, segment_end])
                    segment_start = frame_times[vocal_frames[i]]

                prev_frame = vocal_frames[i]

            # last segment of a song
            segment_end = frame_times[prev_frame]
            if segment_end - segment_start >= 0.3:
                vocal_segments.append([segment_start, segment_end])

        return vocal_segments

    def _vocal_compatibility_score(self, exit_time, track1_vocals, entry_time, track2_vocals, crossfade_duration=12.0):
        if not track1_vocals and not track2_vocals:
            return 1.0

        # define the region where tracks overlap
        exit_start = exit_time - crossfade_duration
        exit_end = exit_time
        entry_start = entry_time
        entry_end = entry_time + crossfade_duration

        # find vocals that would be in the transition
        exit_vocals_in_transition = []
        for start, end in track1_vocals:
            if end > exit_start and start < exit_end:
                overlap_start = max(start, exit_start)
                overlap_end = min(end, exit_end)
                exit_vocals_in_transition.append([overlap_start - exit_start, overlap_end - exit_start])

        entry_vocals_in_transition = []
        for start, end in track2_vocals:
            if end > entry_start and start < entry_end:
                overlap_start = max(start, entry_start)
                overlap_end = min(end, entry_end)
                entry_vocals_in_transition.append([overlap_start - entry_start, overlap_end - entry_start])

        # calc how much vocal overlap
        total_overlap = 0
        for exit_vocal in exit_vocals_in_transition:
            for entry_vocal in entry_vocals_in_transition:
                overlap_start = max(exit_vocal[0], entry_vocal[0])
                overlap_end = min(exit_vocal[1], entry_vocal[1])

                if overlap_end > overlap_start:
                    total_overlap += (overlap_end - overlap_start)

        # turn into %
        vocal_clash_percentage = total_overlap / crossfade_duration if crossfade_duration > 0 else 0

        # higher = better, 1 = perfect
        score = 1.0 - vocal_clash_percentage

        return score

    def _get_tempo(self, y, sr):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env, trim=True)
        tempo_scalar = tempo.item() if isinstance(tempo, np.ndarray) else tempo

        beat_times = librosa.frames_to_time(beats, sr=sr)

        if len(beat_times) > 1:
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

            return float(round(detected_bpm))
        else:
            return float(tempo_scalar)

    def _get_key(self, y, sr):
        y_harmonic = librosa.effects.harmonic(y)
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

        chroma_mean = np.mean(chroma, axis=1)
        key_idx = np.argmax(chroma_mean)

        base_key = self.KEY_MAPPING[key_idx]

        minor_profile = np.roll(self.minor_template, key_idx)
        major_profile = np.roll(self.major_template, key_idx)

        minor_corr = np.corrcoef(chroma_mean, minor_profile)[0, 1]
        major_corr = np.corrcoef(chroma_mean, major_profile)[0, 1]

        key = f"{base_key} {'Minor' if minor_corr > major_corr else 'Major'}"
        return key

    def _get_camelot_key(self, key):
        return self.camelot_wheel.get(key, "8B")

    def _get_energy_profile(self, y):
        return np.mean(librosa.feature.rms(y=y))

    def _get_energy_curve(self, y):
        return librosa.feature.rms(y=y).flatten()

    def _get_beats_with_confidence(self, y, sr):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        beat_onset_strengths = onset_env[beat_frames] if len(beat_frames) > 0 else []
        beat_confidence = []

        for strength in beat_onset_strengths:
            conf = min(1.0, strength / np.max(onset_env) if np.max(onset_env) > 0 else 0)
            beat_confidence.append(float(conf))

        return beat_times, beat_confidence

    def _detect_sections(self, energy_curve, sr, duration, beat_times):
        time_axis = np.linspace(0, duration, len(energy_curve))

        # this is how DJs think of songs - in phrases
        if len(beat_times) >= 16:
            bar_positions = beat_times[::4]  # every 4 beats = 1 bar
            phrase_boundaries = beat_times[::32]  # every 8 bars = phrase typically
        else:
            # not enough data, guess
            phrase_boundaries = np.linspace(0, duration, int(duration / 16) + 1)
            bar_positions = np.linspace(0, duration, int(duration / 4) + 1)

        # find drops and breakdowns based on energy
        avg_energy = np.mean(energy_curve)
        drop_threshold = avg_energy * self.DROP_ENERGY_RATIO
        breakdown_threshold = avg_energy * self.BREAKDOWN_ENERGY_RATIO

        drop_candidates = np.where(energy_curve > drop_threshold)[0]
        breakdown_candidates = np.where(energy_curve < breakdown_threshold)[0]

        # groupiing into proper sections
        drops = self._group_frames_to_sections(drop_candidates, time_axis)
        breakdowns = self._group_frames_to_sections(breakdown_candidates, time_axis)

        # align to phrases so transitions sound good
        aligned_drops = []
        for drop in drops:
            nearest_phrase_idx = np.argmin(np.abs(phrase_boundaries - drop['start']))
            aligned_start = phrase_boundaries[nearest_phrase_idx]

            if abs(aligned_start - drop['start']) < 4.0:
                drop['start'] = float(aligned_start)

            aligned_drops.append(drop)

        # same for breakdowns
        aligned_breakdowns = []
        for breakdown in breakdowns:
            nearest_phrase_idx = np.argmin(np.abs(phrase_boundaries - breakdown['start']))
            aligned_start = phrase_boundaries[nearest_phrase_idx]

            if abs(aligned_start - breakdown['start']) < 4.0:
                breakdown['start'] = float(aligned_start)

            aligned_breakdowns.append(breakdown)

        # guess intro/outro
        intro_end = min(duration * self.INTRO_LENGTH_RATIO, 30.0)
        if len(phrase_boundaries) > 0:
            nearest_boundary_idx = np.argmin(np.abs(phrase_boundaries - intro_end))
            intro_end = float(phrase_boundaries[nearest_boundary_idx])

        outro_start = max(duration * (1 - self.OUTRO_LENGTH_RATIO), duration - 30.0)
        if len(phrase_boundaries) > 0:
            nearest_boundary_idx = np.argmin(np.abs(phrase_boundaries - outro_start))
            outro_start = float(phrase_boundaries[nearest_boundary_idx])

        return {
            'intro': {
                'start': 0,
                'end': intro_end
            },
            'drops': aligned_drops,
            'breakdowns': aligned_breakdowns,
            'outro': {
                'start': outro_start,
                'end': duration
            },
            'phrase_boundaries': phrase_boundaries.tolist() if isinstance(phrase_boundaries,
                                                                          np.ndarray) else phrase_boundaries,
            'bar_positions': bar_positions.tolist() if isinstance(bar_positions, np.ndarray) else bar_positions
        }

    def _group_frames_to_sections(self, candidates, time_axis):
        if len(candidates) == 0:
            return []

        sections = []
        section_start = candidates[0]

        for i in range(1, len(candidates)):
            # if theres a gap, end the section
            if candidates[i] - candidates[i - 1] > 1:
                section_length = time_axis[candidates[i - 1]] - time_axis[section_start]
                if section_length >= self.MIN_SECTION_LENGTH:
                    sections.append({
                        'start': float(time_axis[section_start]),
                        'end': float(time_axis[candidates[i - 1]]),
                        'energy': float(np.mean(time_axis[section_start:candidates[i - 1]]))
                    })
                section_start = candidates[i]

        # add final section
        section_length = time_axis[-1] if len(candidates) == len(time_axis) else time_axis[candidates[-1]] - time_axis[
            section_start]
        if section_length >= self.MIN_SECTION_LENGTH:
            sections.append({
                'start': float(time_axis[section_start]),
                'end': float(time_axis[candidates[-1]]),
                'energy': float(np.mean(time_axis[section_start:candidates[-1]]))
            })

        return sections

    def _detect_transition_points(self, sections, energy_curve, sr, beat_times, vocal_segments=None):
        entry_points = []
        exit_points = []

        time_axis = np.linspace(0, sections['outro']['end'], len(energy_curve))

        # where to start new track
        intro_entry_time = float(sections['intro']['end'])
        if vocal_segments:
            # dont cut in during vocals
            vocal_free_zone = True
            for vs_start, vs_end in vocal_segments:
                if abs(vs_start - intro_entry_time) < 4.0 or abs(vs_end - intro_entry_time) < 4.0:
                    vocal_free_zone = False
                    break

            if not vocal_free_zone:
                # find better spot
                for offset in [4, 8, 12, 16]:
                    candidate_time = intro_entry_time + offset
                    if candidate_time < sections['outro']['start']:
                        is_clear = True
                        for vs_start, vs_end in vocal_segments:
                            if abs(vs_start - candidate_time) < 4.0 or abs(vs_end - candidate_time) < 4.0:
                                is_clear = False
                                break

                        if is_clear:
                            intro_entry_time = candidate_time
                            break

        entry_points.append({
            'time': intro_entry_time,
            'type': 'intro_end',
            'quality': 0.8,
            'energy': float(np.mean(energy_curve[:int(intro_entry_time * len(energy_curve) / len(time_axis))]))
        })

        # add points at phrase boundries
        phrase_boundaries = sections.get('phrase_boundaries', [])
        for i, boundary in enumerate(phrase_boundaries):
            # skip start/end parts
            if boundary < sections['intro']['end'] or boundary > sections['outro']['start']:
                continue

            # check for vocals
            vocal_clash_entry = False
            vocal_clash_exit = False
            if vocal_segments:
                for vs_start, vs_end in vocal_segments:
                    # entry point check
                    if boundary >= vs_start and boundary <= vs_end:
                        vocal_clash_entry = True
                        break
                    if vs_start > boundary and vs_start - boundary < 4.0:
                        vocal_clash_entry = True
                        break

                    # exit point check
                    if boundary >= vs_start and boundary <= vs_end:
                        vocal_clash_exit = True
                        break
                    if vs_end < boundary and boundary - vs_end < 4.0:
                        vocal_clash_exit = True
                        break

            # just use every 4th phrase (thats every 32 bars)
            if i % 4 == 0:
                idx = min(int(boundary * len(energy_curve) / len(time_axis)), len(energy_curve) - 1)
                energy_val = float(energy_curve[idx]) if idx < len(energy_curve) else 0.5

                # score entry points
                quality = 0.75
                if vocal_clash_entry:
                    quality *= 0.7

                entry_points.append({
                    'time': float(boundary),
                    'type': f'phrase_boundary_{i}',
                    'quality': quality,
                    'energy': energy_val
                })

                # also as exit points
                quality = 0.75
                if vocal_clash_exit:
                    quality *= 0.7

                exit_points.append({
                    'time': float(boundary),
                    'type': f'phrase_boundary_{i}',
                    'quality': quality,
                    'energy': energy_val
                })

        # drops work as entry points
        for i, drop in enumerate(sections['drops']):
            # check vocals
            drop_vocal_clash = False
            if vocal_segments:
                for vs_start, vs_end in vocal_segments:
                    if abs(vs_start - drop['start']) < 2.0 or abs(vs_end - drop['start']) < 2.0:
                        drop_vocal_clash = True
                        break

            quality = 0.9
            if drop_vocal_clash:
                quality *= 0.8

            entry_points.append({
                'time': float(drop['start']),
                'type': f'drop_start_{i + 1}',
                'quality': quality,
                'energy': float(drop['energy'])
            })

        # end of breakdown = good entry
        for i, breakdown in enumerate(sections['breakdowns']):

            breakdown_vocal_clash = False
            if vocal_segments:
                for vs_start, vs_end in vocal_segments:
                    if abs(vs_start - breakdown['end']) < 2.0 or abs(vs_end - breakdown['end']) < 2.0:
                        breakdown_vocal_clash = True
                        break

            quality = 0.7
            if breakdown_vocal_clash:
                quality *= 0.8

            entry_points.append({
                'time': float(breakdown['end']),
                'type': f'breakdown_end_{i + 1}',
                'quality': quality,
                'energy': float(breakdown['energy'])
            })

        # start of breakdown = good exit
        for i, breakdown in enumerate(sections['breakdowns']):
            # check vocals
            breakdown_vocal_clash = False
            if vocal_segments:
                for vs_start, vs_end in vocal_segments:
                    if abs(vs_start - breakdown['start']) < 2.0 or abs(vs_end - breakdown['start']) < 2.0:
                        breakdown_vocal_clash = True
                        break

            quality = 0.8
            if breakdown_vocal_clash:
                quality *= 0.8

            exit_points.append({
                'time': float(breakdown['start']),
                'type': f'breakdown_start_{i + 1}',
                'quality': quality,
                'energy': float(breakdown['energy'])
            })

        # end of drop = good exit
        for i, drop in enumerate(sections['drops']):
            # check vocals
            drop_vocal_clash = False
            if vocal_segments:
                for vs_start, vs_end in vocal_segments:
                    if abs(vs_start - drop['end']) < 2.0 or abs(vs_end - drop['end']) < 2.0:
                        drop_vocal_clash = True
                        break

            quality = 0.7
            if drop_vocal_clash:
                quality *= 0.8

            exit_points.append({
                'time': float(drop['end']),
                'type': f'drop_end_{i + 1}',
                'quality': quality,
                'energy': float(drop['energy'])
            })

        # outro makes a good exit point
        outro_exit_time = float(sections['outro']['start'])

        # avoid vocals at outro
        if vocal_segments:
            outro_vocal_clash = False
            for vs_start, vs_end in vocal_segments:
                if abs(vs_start - outro_exit_time) < 4.0 or abs(vs_end - outro_exit_time) < 4.0:
                    outro_vocal_clash = True
                    break

            if outro_vocal_clash:
                # find better spot
                for offset in [4, 8, 12]:
                    candidate_time = outro_exit_time - offset
                    if candidate_time > sections['intro']['end']:
                        is_clear = True
                        for vs_start, vs_end in vocal_segments:
                            if abs(vs_start - candidate_time) < 4.0 or abs(vs_end - candidate_time) < 4.0:
                                is_clear = False
                                break

                        if is_clear:
                            outro_exit_time = candidate_time
                            break

        exit_points.append({
            'time': outro_exit_time,
            'type': 'outro_start',
            'quality': 0.9,
            'energy': float(
                np.mean(energy_curve[int(outro_exit_time * len(energy_curve) / len(time_axis)):]))
        })

        return {
            'entries': entry_points,
            'exits': exit_points
        }

    def _improve_transition_points(self, transition_points, y, sr, beat_times):
        improved_entries = []
        improved_exits = []

        if not len(beat_times):
            # no beats found, use original
            return transition_points

        # align entry points to beats - always sound better
        for point in transition_points['entries']:
            closest_beat_idx = np.argmin(np.abs(beat_times - point['time']))
            aligned_time = beat_times[closest_beat_idx]

            point['time'] = float(aligned_time)
            improved_entries.append(point)

        # same for exits
        for point in transition_points['exits']:
            closest_beat_idx = np.argmin(np.abs(beat_times - point['time']))
            aligned_time = beat_times[closest_beat_idx]

            point['time'] = float(aligned_time)
            improved_exits.append(point)

        return {
            'entries': improved_entries,
            'exits': improved_exits
        }

    def _harmonic_compatibility_score(self, camelot_key1, camelot_key2):
        # if keys are the same its a perfect match
        if camelot_key1 == camelot_key2:
            return 1.0

        # break up key parts
        try:
            num1 = int(camelot_key1[:-1])
            letter1 = camelot_key1[-1]
            num2 = int(camelot_key2[:-1])
            letter2 = camelot_key2[-1]
        except:
            # somethings wrong, assume bad
            return 0.3

        # same letter adj number is very good
        if letter1 == letter2 and abs(num1 - num2) == 1:
            return 0.9

        # same number diff letter is relative maj/min
        if num1 == num2 and letter1 != letter2:
            return 0.8

        # might still work ok but wont sound pleaseing
        if (abs(num1 - num2) == 1 and letter1 != letter2) or (letter1 == letter2):
            return 0.6

        # probly not compatible
        return 0.3

    def _rhythm_compatibility_score(self, bpm1, bpm2):
        if bpm1 > bpm2:
            ratio = bpm1 / bpm2
        else:
            ratio = bpm2 / bpm1

        if ratio > 2:
            if abs(ratio - 2) < 0.05:
                return 0.8

        if abs(ratio - 1) < 0.01:
            return 1.0

        # very close
        if abs(ratio - 1) < 0.03:
            return 0.9

        # close enuf
        if abs(ratio - 1) < 0.05:
            return 0.8

        # bit off but workable
        if abs(ratio - 1) < 0.1:
            return 0.6

        # way out of scale
        return 0.4

    def _energy_compatibility_score(self, energy1, energy2):
        # calc ratio (always 0-1)
        if energy1 > energy2:
            ratio = energy2 / energy1
        else:
            ratio = energy1 / energy2

        # score based on how similar the energies are
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
        # best transitions
        best_pairs = [
            ('outro_start', 'intro_end'),
            ('breakdown_start', 'breakdown_end'),
            ('drop_end', 'drop_start')
        ]

        # good but not perfect
        good_pairs = [
            ('drop_end', 'breakdown_end'),
            ('breakdown_start', 'intro_end'),
            ('outro_start', 'drop_start'),
            ('phrase_boundary', 'phrase_boundary')
        ]

        # exact matches are best
        if any(exit_type == pair[0] and entry_type == pair[1] for pair in best_pairs):
            return 1.0

        # still good pairings
        if any(exit_type == pair[0] and entry_type == pair[1] for pair in good_pairs):
            return 0.8

        # phrase boundry to phrase is good generic choice
        if 'phrase_boundary' in exit_type and 'phrase_boundary' in entry_type:
            return 0.8

        # check partial matches
        exit_prefix = exit_type.split('_')[0]
        entry_prefix = entry_type.split('_')[0]

        if exit_prefix == 'drop' and entry_prefix == 'drop':
            return 0.7

        if exit_prefix == 'breakdown' and entry_prefix == 'breakdown':
            return 0.7

        # default - terrible
        return 0.5

    def find_better_transition_points(self, audio_path):
        y, sr = librosa.load(audio_path, sr=44100)

        # find the beats first
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)

        # find phrases 8 bars usually
        phrase_length = 8
        beats_per_bar = 4

        if len(beat_times) >= phrase_length * beats_per_bar:
            phrase_boundaries = beat_times[::phrase_length * beats_per_bar]
        else:
            duration = librosa.get_duration(y=y, sr=sr)
            phrase_boundaries = np.linspace(0, duration, int(duration / 16) + 1)

        # look at energy changes
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        # smooth it and find peaks
        rms_smooth = scipy.ndimage.gaussian_filter1d(rms, sigma=5)
        rms_diff = np.diff(rms_smooth)
        rms_diff = np.concatenate([[0], rms_diff])

        # peaks = energy going up, troughs energy going down
        peak_indices = find_peaks(rms_diff, height=np.percentile(rms_diff, 90))[0]
        trough_indices = find_peaks(-rms_diff, height=np.percentile(-rms_diff, 90))[0]

        peak_times = rms_times[peak_indices]
        trough_times = rms_times[trough_indices]

        # score each boundry point based on energy
        scored_boundaries = []
        for boundary in phrase_boundaries:
            # good exit = energy droppin
            dist_to_trough = np.min(np.abs(trough_times - boundary)) if len(trough_times) > 0 else float('inf')
            exit_score = 1.0 / (1.0 + dist_to_trough) if dist_to_trough < 8.0 else 0.0

            # good entry = energy rising
            dist_to_peak = np.min(np.abs(peak_times - boundary)) if len(peak_times) > 0 else float('inf')
            entry_score = 1.0 / (1.0 + dist_to_peak) if dist_to_peak < 8.0 else 0.0

            idx = np.argmin(np.abs(rms_times - boundary))
            energy = float(rms_smooth[idx]) if idx < len(rms_smooth) else 0.0

            scored_boundaries.append({
                'time': float(boundary),
                'exit_score': float(exit_score),
                'entry_score': float(entry_score),
                'energy': energy
            })

        # sort by scores
        exit_points = sorted(scored_boundaries, key=lambda x: x['exit_score'], reverse=True)
        entry_points = sorted(scored_boundaries, key=lambda x: x['entry_score'], reverse=True)

        formatted_exits = []
        for i, point in enumerate(exit_points[:10]):
            formatted_exits.append({
                'time': point['time'],
                'type': f'phrase_exit_{i + 1}',
                'quality': point['exit_score'],
                'energy': point['energy']
            })

        formatted_entries = []
        for i, point in enumerate(entry_points[:10]):  # top 10 only
            formatted_entries.append({
                'time': point['time'],
                'type': f'phrase_entry_{i + 1}',
                'quality': point['entry_score'],
                'energy': point['energy']
            })

        return {
            'exits': formatted_exits,
            'entries': formatted_entries
        }

    def visualize_analysis(self, track_analysis, audio_path=None):

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        if audio_path:
            y, sr = librosa.load(audio_path, sr=44100)
            librosa.display.waveshow(y, sr=sr, ax=ax1)
            ax1.set_title('Waveform')
        else:
            ax1.set_title('Track Analysis')

        ax1.text(0.02, 0.9, f"BPM: {track_analysis['bpm']:.1f}", transform=ax1.transAxes)
        ax1.text(0.02, 0.85, f"Key: {track_analysis['key']} ({track_analysis['camelot_key']})",
                 transform=ax1.transAxes)
        ax1.text(0.02, 0.8, f"Duration: {track_analysis['duration']:.1f}s", transform=ax1.transAxes)

        # show structure
        ax2.set_title('Sections and Transition Points')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Section Type')
        ax2.set_xlim(0, track_analysis['duration'])

        # intro part
        intro = track_analysis['sections']['intro']
        ax2.axvspan(intro['start'], intro['end'], alpha=0.2, color='green', label='Intro')

        # show the drops
        for i, drop in enumerate(track_analysis['sections']['drops']):
            ax2.axvspan(drop['start'], drop['end'], alpha=0.3, color='red', label=f'Drop {i + 1}' if i == 0 else "")

        # show breakdowns
        for i, breakdown in enumerate(track_analysis['sections']['breakdowns']):
            ax2.axvspan(breakdown['start'], breakdown['end'], alpha=0.3, color='blue',
                        label=f'Breakdown {i + 1}' if i == 0 else "")

        # outro part
        outro = track_analysis['sections']['outro']
        ax2.axvspan(outro['start'], outro['end'], alpha=0.2, color='purple', label='Outro')

        # add phrase markers
        if 'phrase_boundaries' in track_analysis['sections']:
            for boundary in track_analysis['sections']['phrase_boundaries']:
                ax2.axvline(x=boundary, linestyle=':', color='black', alpha=0.3)

        # entry points
        for point in track_analysis['transition_points']['entries']:
            ax2.axvline(x=point['time'], linestyle='--', color='green', alpha=0.7)
            ax2.text(point['time'], 0.9, 'IN', horizontalalignment='center', verticalalignment='center',
                     transform=ax2.get_xaxis_transform())

        # exit points
        for point in track_analysis['transition_points']['exits']:
            ax2.axvline(x=point['time'], linestyle='--', color='red', alpha=0.7)
            ax2.text(point['time'], 0.1, 'OUT', horizontalalignment='center', verticalalignment='center',
                     transform=ax2.get_xaxis_transform())

        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.show()