import os
import json
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import matplotlib.pyplot as plt
from feature_extraction import FeatureExtractor
from beat_detection import analyze_transition_points, enhanced_beat_detection
import random
import argparse


class AutoMixer:
    def __init__(self, output_folder="output_mix"):
        """Initialize the automatic mixing system."""
        self.output_folder = output_folder
        self.tracks = []
        self.track_info = {}
        self.mix_sequence = []
        self.extractor = FeatureExtractor()

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

    def scan_tracks(self, music_folder):
        """Scan a folder for music tracks and analyze them."""
        print(f"Scanning folder: {music_folder}")

        for filename in os.listdir(music_folder):
            if filename.endswith(('.mp3', '.wav')):
                track_path = os.path.join(music_folder, filename)
                self.tracks.append(track_path)

                # Extract features
                print(f"Analyzing track: {filename}")
                features = self.extractor.extract_all_features(track_path)
                transitions = analyze_transition_points(track_path)

                # Store track information
                self.track_info[track_path] = {
                    'features': features,
                    'transitions': transitions,
                    'name': filename
                }

                print(f"  BPM: {features['rhythmic']['bpm']:.1f}")
                print(f"  Key: {features['tonal']['key']}")
                print(f"  Good transition points: {self._count_good_transition_points(transitions)}")

        print(f"Found and analyzed {len(self.tracks)} tracks")

    def _count_good_transition_points(self, transitions, threshold=0.7):
        """Count the number of good transition points in a track."""
        good_points = [score for score in transitions['transition_quality'] if score > threshold]
        return len(good_points)

    def plan_track_sequence(self, num_tracks=None, energy_flow=None):
        """Plan the sequence of tracks for the mix."""
        if not self.tracks:
            print("No tracks available. Please scan tracks first.")
            return

        if num_tracks is None or num_tracks > len(self.tracks):
            num_tracks = len(self.tracks)

        # Default energy flow: build up, then down
        if energy_flow is None:
            energy_flow = "buildup"  # Options: "buildup", "wave", "flat"

        print(f"Planning a mix with {num_tracks} tracks using {energy_flow} energy flow")

        # Calculate compatibility between all track pairs
        compatibility_matrix = self._calculate_compatibility_matrix()

        # Order tracks based on compatibility and energy flow
        if energy_flow == "buildup":
            self.mix_sequence = self._create_buildup_sequence(compatibility_matrix, num_tracks)
        elif energy_flow == "wave":
            self.mix_sequence = self._create_wave_sequence(compatibility_matrix, num_tracks)
        else:  # flat
            self.mix_sequence = self._create_flat_sequence(compatibility_matrix, num_tracks)

        # Print the planned sequence
        print("\nPlanned mix sequence:")
        for i, track_path in enumerate(self.mix_sequence):
            track_name = self.track_info[track_path]['name']
            track_bpm = self.track_info[track_path]['features']['rhythmic']['bpm']
            track_key = self.track_info[track_path]['features']['tonal']['key']
            print(f"{i + 1}. {track_name} (BPM: {track_bpm:.1f}, Key: {track_key})")

    def _calculate_compatibility_matrix(self):
        """Calculate compatibility scores between all pairs of tracks."""
        n_tracks = len(self.tracks)
        compatibility_matrix = np.zeros((n_tracks, n_tracks))

        for i in range(n_tracks):
            for j in range(n_tracks):
                if i != j:
                    track1 = self.tracks[i]
                    track2 = self.tracks[j]

                    # Calculate compatibility based on BPM, key, and energy
                    compatibility = self._calculate_track_compatibility(
                        self.track_info[track1]['features'],
                        self.track_info[track2]['features']
                    )

                    compatibility_matrix[i, j] = compatibility

        return compatibility_matrix

    def _calculate_track_compatibility(self, features1, features2):
        """Calculate compatibility score between two tracks."""
        # BPM compatibility
        bpm1 = features1['rhythmic']['bpm']
        bpm2 = features2['rhythmic']['bpm']

        # Check for tempo doubling/halving relationships
        tempo_ratios = [
            abs(bpm1 - bpm2) / max(bpm1, bpm2),
            abs(bpm1 - 2 * bpm2) / max(bpm1, 2 * bpm2),
            abs(2 * bpm1 - bpm2) / max(2 * bpm1, bpm2)
        ]

        tempo_compatibility = 1.0 - min(tempo_ratios)

        # Key compatibility
        key1 = features1['tonal']['key']
        key2 = features2['tonal']['key']
        key_compatibility = self._calculate_key_compatibility(key1, key2)

        # Energy compatibility
        energy1 = features1['energy']['rms']['mean']
        energy2 = features2['energy']['rms']['mean']
        energy_ratio = min(energy1, energy2) / max(energy1, energy2)

        # Overall compatibility (weighted sum)
        overall_compatibility = (
                0.4 * tempo_compatibility +
                0.4 * key_compatibility +
                0.2 * energy_ratio
        )

        return overall_compatibility

    def _calculate_key_compatibility(self, key1, key2):
        """Calculate musical key compatibility using the Circle of Fifths."""
        # Simple compatibility rules:
        # 1.0: Same key
        # 0.8: Relative major/minor or perfect fifth
        # 0.6: Close on circle of fifths
        # 0.4: Further away

        # For a basic implementation, we'll just check if keys are identical
        if key1 == key2:
            return 1.0

        # This is a simplified version - a full implementation would use
        # camelot wheel or circle of fifths relationships
        return 0.5  # Default medium compatibility

    def _create_buildup_sequence(self, compatibility_matrix, num_tracks):
        """Create a sequence that builds up in energy."""
        # Get energy levels for all tracks
        energy_levels = [self.track_info[track]['features']['energy']['rms']['mean']
                         for track in self.tracks]

        # Sort tracks by energy
        sorted_indices = np.argsort(energy_levels)

        # Start with low energy tracks, then increase
        sequence = []
        current_idx = sorted_indices[0]
        sequence.append(self.tracks[current_idx])

        # Build the rest of the sequence, considering compatibility
        for _ in range(1, min(num_tracks, len(self.tracks))):
            # Get compatibility scores for all tracks with the current one
            compatibility_scores = compatibility_matrix[current_idx]

            # Create a combined score that weighs both energy progression and compatibility
            combined_scores = np.zeros(len(self.tracks))

            for i in range(len(self.tracks)):
                # Skip tracks already in the sequence
                if self.tracks[i] in sequence:
                    combined_scores[i] = -1
                    continue

                # Energy score: reward tracks with higher energy than current
                energy_diff = energy_levels[i] - energy_levels[current_idx]
                energy_score = 1.0 if energy_diff > 0 else 0.2

                # Combine with compatibility
                combined_scores[i] = 0.7 * compatibility_scores[i] + 0.3 * energy_score

            # Select track with highest combined score
            next_idx = np.argmax(combined_scores)
            sequence.append(self.tracks[next_idx])
            current_idx = next_idx

        return sequence

    def _create_wave_sequence(self, compatibility_matrix, num_tracks):
        """Create a sequence with alternating energy levels."""
        # Implementation similar to buildup, but alternating high/low energy
        energy_levels = [self.track_info[track]['features']['energy']['rms']['mean']
                         for track in self.tracks]

        # Sort tracks by energy
        sorted_indices = np.argsort(energy_levels)
        mid_point = len(sorted_indices) // 2

        # Alternating pattern
        sequence = []
        remaining_indices = list(sorted_indices)
        want_high_energy = False

        while len(sequence) < min(num_tracks, len(self.tracks)) and remaining_indices:
            if want_high_energy:
                # Find highest energy track that's compatible with the last track
                best_idx = self._find_best_compatible_track(
                    sequence[-1] if sequence else None,
                    [self.tracks[i] for i in remaining_indices],
                    prefer_high_energy=True
                )
            else:
                # Find lowest energy track that's compatible with the last track
                best_idx = self._find_best_compatible_track(
                    sequence[-1] if sequence else None,
                    [self.tracks[i] for i in remaining_indices],
                    prefer_high_energy=False
                )

            if best_idx is not None:
                sequence.append(best_idx)
                remaining_indices.remove(self.tracks.index(best_idx))
                want_high_energy = not want_high_energy
            else:
                break

        return sequence

    def _find_best_compatible_track(self, current_track, candidates, prefer_high_energy=True):
        """Find the most compatible track from candidates."""
        if current_track is None:
            # If no current track, just pick based on energy preference
            energy_levels = [self.track_info[track]['features']['energy']['rms']['mean']
                             for track in candidates]
            idx = np.argmax(energy_levels) if prefer_high_energy else np.argmin(energy_levels)
            return candidates[idx]

        current_idx = self.tracks.index(current_track)
        best_score = -1
        best_track = None

        for candidate in candidates:
            candidate_idx = self.tracks.index(candidate)
            compatibility = self._calculate_track_compatibility(
                self.track_info[current_track]['features'],
                self.track_info[candidate]['features']
            )

            # Energy preference factor
            energy_current = self.track_info[current_track]['features']['energy']['rms']['mean']
            energy_candidate = self.track_info[candidate]['features']['energy']['rms']['mean']

            energy_factor = energy_candidate > energy_current if prefer_high_energy else energy_candidate < energy_current
            energy_score = 1.0 if energy_factor else 0.2

            # Combined score
            score = 0.7 * compatibility + 0.3 * energy_score

            if score > best_score:
                best_score = score
                best_track = candidate

        return best_track

    def _create_flat_sequence(self, compatibility_matrix, num_tracks):
        """Create a sequence optimizing for compatibility between adjacent tracks."""
        remaining_tracks = self.tracks.copy()

        # Start with a random track
        sequence = [random.choice(remaining_tracks)]
        remaining_tracks.remove(sequence[0])

        # Build sequence by finding most compatible next track
        while len(sequence) < min(num_tracks, len(self.tracks)) and remaining_tracks:
            last_track = sequence[-1]
            last_idx = self.tracks.index(last_track)

            # Get compatibility scores
            compatibility_scores = np.array([
                compatibility_matrix[last_idx, self.tracks.index(track)]
                if track in remaining_tracks else -1
                for track in self.tracks
            ])

            # Select most compatible track
            best_idx = np.argmax(compatibility_scores)
            sequence.append(self.tracks[best_idx])

            if self.tracks[best_idx] in remaining_tracks:
                remaining_tracks.remove(self.tracks[best_idx])

        return sequence

    def create_mix(self, output_file="final_mix.wav", transition_duration=30):
        """Create the final mix using the planned sequence."""
        if not self.mix_sequence:
            print("No mix sequence planned. Please plan a track sequence first.")
            return

        print(f"\nCreating mix with {len(self.mix_sequence)} tracks")

        # Initialize variables for the mix
        final_mix = np.array([])
        sr = None

        # Process each track in the sequence
        for i in range(len(self.mix_sequence) - 1):  # Process pairs of tracks
            current_track = self.mix_sequence[i]
            next_track = self.mix_sequence[i + 1]

            print(
                f"Mixing tracks {i + 1} and {i + 2}: {os.path.basename(current_track)} → {os.path.basename(next_track)}")

            # Load both tracks
            y_current, sr_current = librosa.load(current_track, sr=sr)
            y_next, sr_next = librosa.load(next_track, sr=sr_current)

            # Set the global sample rate from the first track
            if sr is None:
                sr = sr_current

            # Get track information
            current_features = self.track_info[current_track]['features']
            next_features = self.track_info[next_track]['features']
            current_transitions = self.track_info[current_track]['transitions']
            next_transitions = self.track_info[next_track]['transitions']

            # Get BPM information for time stretching
            bpm_current = current_features['rhythmic']['bpm']
            bpm_next = next_features['rhythmic']['bpm']

            # Time-stretch if BPMs are significantly different
            if abs(bpm_current - bpm_next) / max(bpm_current, bpm_next) > 0.05:
                print(f"Time-stretching track {i + 2} from {bpm_next:.1f} to {bpm_current:.1f} BPM")
                bpm_ratio = bpm_current / bpm_next
                y_next = librosa.effects.time_stretch(y=y_next, rate=bpm_ratio)

            # Determine transition points
            # For first track in the mix, add it completely until transition point
            if i == 0:
                # Find a good transition out point (80% into the track is a safe default)
                if len(current_transitions['transition_quality']) > 0:
                    # Find a good transition point in the last third of the track
                    last_third_idx = max(0, int(2 * len(current_transitions['beat_times']) / 3))
                    quality_scores = current_transitions['transition_quality'][last_third_idx:]
                    beat_times = current_transitions['beat_times'][last_third_idx:]

                    if len(quality_scores) > 0:
                        best_idx = np.argmax(quality_scores)
                        transition_out_time = beat_times[best_idx]
                    else:
                        # Fallback: use 80% of track length
                        transition_out_time = 0.8 * librosa.get_duration(y=y_current, sr=sr)
                else:
                    # Fallback: use 80% of track length
                    transition_out_time = 0.8 * librosa.get_duration(y=y_current, sr=sr)

                transition_out_sample = int(transition_out_time * sr)

                # Add the first track up to transition point
                final_mix = y_current[:transition_out_sample]

            # Find transition in point for next track (around 15-20% into the track)
            if len(next_transitions['transition_quality']) > 0:
                # Find a good entry point in the first third
                first_third_idx = max(1, int(len(next_transitions['beat_times']) / 3))
                quality_scores = next_transitions['transition_quality'][:first_third_idx]
                beat_times = next_transitions['beat_times'][:first_third_idx]

                if len(quality_scores) > 0:
                    best_idx = np.argmax(quality_scores)
                    transition_in_time = beat_times[best_idx]
                else:
                    # Fallback: use 15% of track length
                    transition_in_time = 0.15 * librosa.get_duration(y=y_next, sr=sr)
            else:
                # Fallback: use 15% of track length
                transition_in_time = 0.15 * librosa.get_duration(y=y_next, sr=sr)

            transition_in_sample = int(transition_in_time * sr)

            # Calculate crossfade duration (in samples)
            crossfade_samples = min(int(transition_duration * sr),
                                    len(y_current) - transition_out_sample,
                                    transition_in_sample)

            # Ensure we have enough samples for crossfade
            if crossfade_samples <= 0:
                print(f"Warning: Cannot create crossfade for tracks {i + 1} and {i + 2}. Using direct transition.")
                final_mix = np.concatenate([final_mix, y_next[transition_in_sample:]])
                continue

            # Create crossfade weights
            fade_out = np.linspace(1, 0, crossfade_samples)
            fade_in = np.linspace(0, 1, crossfade_samples)

            # Apply crossfade
            crossfade_out = y_current[transition_out_sample:transition_out_sample + crossfade_samples]
            crossfade_in = y_next[transition_in_sample - crossfade_samples:transition_in_sample]

            # Ensure both segments are the same length
            min_length = min(len(crossfade_out), len(crossfade_in), len(fade_out))
            if min_length == 0:
                print(f"Warning: Cannot create crossfade for tracks {i + 1} and {i + 2}. Using direct transition.")
                final_mix = np.concatenate([final_mix, y_next[transition_in_sample:]])
                continue

            crossfade_out = crossfade_out[:min_length]
            crossfade_in = crossfade_in[:min_length]
            fade_out = fade_out[:min_length]
            fade_in = fade_in[:min_length]

            # Create the mixed segment
            crossfade_mix = crossfade_out * fade_out + crossfade_in * fade_in

            # Add to the final mix
            final_mix = np.concatenate([final_mix, crossfade_mix, y_next[transition_in_sample:]])

        # Add the last track if it wasn't added
        if len(self.mix_sequence) == 1:
            last_track = self.mix_sequence[0]
            y_last, sr_last = librosa.load(last_track, sr=sr)
            final_mix = y_last

        # Normalize the final mix to prevent clipping
        final_mix = librosa.util.normalize(final_mix)

        # Save the final mix
        output_path = os.path.join(self.output_folder, output_file)
        sf.write(output_path, final_mix, sr)

        print(f"Mix created and saved to {output_path}")
        print(f"Total mix duration: {len(final_mix) / sr / 60:.2f} minutes")

        return output_path

    def visualize_mix(self, mix_path):
        """Create a visualization of the mix."""
        y, sr = librosa.load(mix_path, sr=None)

        # Create figure
        plt.figure(figsize=(14, 8))

        # Plot waveform
        plt.subplot(3, 1, 1)
        plt.plot(librosa.times_like(y, sr=sr), y)
        plt.title('Mix Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        # Plot spectrogram
        plt.subplot(3, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mix Spectrogram')

        # Plot RMS energy
        plt.subplot(3, 1, 3)
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        frames = range(len(rms))
        t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
        plt.plot(t, rms)
        plt.title('Mix Energy')
        plt.xlabel('Time (s)')
        plt.ylabel('RMS Energy')

        # Add track markers
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        track_duration = 0

        for i, track_path in enumerate(self.mix_sequence):
            track_info = self.track_info[track_path]
            track_name = os.path.basename(track_path)
            track_duration += librosa.get_duration(filename=track_path)

            color = colors[i % len(colors)]
            for j in range(3):
                plt.subplot(3, 1, j + 1)
                plt.axvline(x=track_duration, color=color, linestyle='--', alpha=0.7)

                # Add track name
                if j == 0:
                    plt.text(track_duration - 60, 0.8, track_name,
                             rotation=90, color=color, alpha=0.7,
                             transform=plt.gca().get_xaxis_transform())

        # Save the visualization
        vis_path = os.path.join(self.output_folder, 'mix_visualization.png')
        plt.tight_layout()
        plt.savefig(vis_path, dpi=150)
        plt.close()

        print(f"Mix visualization saved to {vis_path}")


def main():
    parser = argparse.ArgumentParser(description='Create an automatic DJ mix from a folder of tracks')
    parser.add_argument('--folder', required=True, help='Folder containing music tracks')
    parser.add_argument('--output', default='auto_mix.wav', help='Output file name for the mix')
    parser.add_argument('--tracks', type=int, default=None, help='Number of tracks to include in the mix')
    parser.add_argument('--energy', choices=['buildup', 'wave', 'flat'], default='buildup',
                        help='Energy flow pattern for the mix')
    parser.add_argument('--transition', type=int, default=30,
                        help='Transition duration in seconds')

    args = parser.parse_args()

    mixer = AutoMixer(output_folder="mix_output")
    mixer.scan_tracks(args.folder)
    mixer.plan_track_sequence(num_tracks=args.tracks, energy_flow=args.energy)
    mix_path = mixer.create_mix(output_file=args.output, transition_duration=args.transition)
    mixer.visualize_mix(mix_path)

    print("\nMix complete! Enjoy your automated DJ set.")


if __name__ == "__main__":
    main()