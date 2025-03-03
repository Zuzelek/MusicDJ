import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from beat_detection import analyze_transition_points
from feature_extraction import FeatureExtractor, extract_and_compare_tracks


def analyze_track(audio_path, output_folder):
    """Analyze a single track for DJ mixing purposes."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get track name
    track_name = os.path.basename(audio_path)
    print(f"\n=== Analyzing track: {track_name} ===")

    # Extract all musical features
    print("\nExtracting features...")
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(audio_path, output_folder)

    if features is None:
        print(f"Error extracting features from {audio_path}")
        return

    # Analyze transition points
    print("\nAnalyzing beat structure and transition points...")
    try:
        transition_analysis = analyze_transition_points(audio_path)

        # Save transition analysis
        transition_file = os.path.join(output_folder,
                                       os.path.splitext(track_name)[0] + "_transitions.json")
        with open(transition_file, 'w') as f:
            json.dump(transition_analysis, f, indent=2)

        # Print summary
        print("\n=== Track Analysis Summary ===")
        print(f"BPM: {features['rhythmic']['bpm']:.1f}")
        print(f"Key: {features['tonal']['key']}")
        print(f"Energy (RMS): {features['energy']['rms']['mean']:.3f}")
        print(f"Intro Ends: {transition_analysis['intro_end']:.2f}s")
        print(f"Outro Starts: {transition_analysis['outro_start']:.2f}s")

        if len(transition_analysis['drop_points']) > 0:
            print(f"Drop Points: {', '.join([f'{t:.2f}s' for t in transition_analysis['drop_points']])}")

        # Visualize analysis
        try:
            visualize_analysis(audio_path, features, transition_analysis, output_folder)
            print(
                f"\nVisualization saved to {os.path.join(output_folder, os.path.splitext(track_name)[0] + '_analysis.png')}")
        except Exception as e:
            print(f"Error creating visualization: {e}")

        return features, transition_analysis

    except Exception as e:
        print(f"Error analyzing track structure: {e}")
        return features, None


def compare_tracks_for_mixing(track1_path, track2_path, output_folder):
    """Compare two tracks for mixing compatibility."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get track names
    track1_name = os.path.basename(track1_path)
    track2_name = os.path.basename(track2_path)

    print(f"\n=== Comparing tracks for mixing ===")
    print(f"Track 1: {track1_name}")
    print(f"Track 2: {track2_name}")

    # Extract features and compare
    comparison = extract_and_compare_tracks(track1_path, track2_path, output_folder)

    if comparison is None:
        print("Error comparing tracks")
        return

    # Save comparison results
    comparison_file = os.path.join(
        output_folder,
        f"{os.path.splitext(track1_name)[0]}_vs_{os.path.splitext(track2_name)[0]}_comparison.json"
    )

    with open(comparison_file, 'w') as f:
        json.dump(comparison['compatibility'], f, indent=2)

    # Print summary
    compatibility = comparison['compatibility']
    print("\n=== Mixing Compatibility Summary ===")
    print(f"Overall Compatibility: {compatibility['overall_score']:.2f}/1.00")
    print(f"  ↳ Tempo: {compatibility['tempo_compatibility']:.2f}/1.00")
    print(f"  ↳ Key: {compatibility['key_compatibility']:.2f}/1.00")
    print(f"  ↳ Energy: {compatibility['energy_compatibility']:.2f}/1.00")
    print(f"  ↳ Spectral: {compatibility['spectral_compatibility']:.2f}/1.00")
    print(f"\nRecommended Technique: {compatibility['recommended_technique']}")
    print(f"Minimum Crossfade: {compatibility['recommended_min_crossfade']:.1f} seconds")

    # Create visualization of the comparison
    try:
        visualize_comparison(comparison, output_folder)
    except Exception as e:
        print(f"Error creating visualization: {e}")

    return comparison


def visualize_analysis(audio_path, features, transition_analysis, output_folder):
    """Create visualizations for track analysis."""
    track_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Beat and structure analysis
    beat_times = transition_analysis['beat_times']

    # Convert beat times to a binary representation for visualization
    max_time = features['metadata']['duration']
    time_axis = np.linspace(0, max_time, 1000)
    beat_signal = np.zeros_like(time_axis)

    for beat in beat_times:
        idx = np.argmin(np.abs(time_axis - beat))
        if idx < len(beat_signal):
            beat_signal[idx] = 1.0

    # Plot beats
    ax1.plot(time_axis, beat_signal, 'b-', alpha=0.5)

    # Plot segmentation points
    if 'segment_boundaries' in features['structure']:
        for boundary in features['structure']['segment_boundaries']:
            ax1.axvline(x=boundary, color='g', linestyle='--', alpha=0.5)

    # Plot drops
    for drop in transition_analysis['drop_points']:
        ax1.axvline(x=drop, color='r', linestyle='-', linewidth=2, alpha=0.7)
        ax1.text(drop, 0.5, 'DROP', color='r', rotation=90,
                 verticalalignment='center', horizontalalignment='right')

    # Plot intro/outro
    ax1.axvline(x=transition_analysis['intro_end'], color='m', linestyle='-', alpha=0.7)
    ax1.text(transition_analysis['intro_end'], 0.8, 'INTRO END', color='m', rotation=90,
             verticalalignment='center', horizontalalignment='right')

    ax1.axvline(x=transition_analysis['outro_start'], color='c', linestyle='-', alpha=0.7)
    ax1.text(transition_analysis['outro_start'], 0.8, 'OUTRO START', color='c', rotation=90,
             verticalalignment='center', horizontalalignment='right')

    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('Beat Structure')
    ax1.set_title(f'Track Analysis: {track_name}')

    # Transition Quality Plot
    transition_quality = transition_analysis['transition_quality']

    # Interpolate transition quality to the time axis
    tq_interp = np.interp(time_axis,
                          np.array(beat_times)[:len(transition_quality)],
                          transition_quality)

    ax2.plot(time_axis, tq_interp, 'g-')
    ax2.set_ylabel('Transition Quality')
    ax2.set_ylim(0, 1.1)

    # Highlight best transition regions
    threshold = 0.7
    above_threshold = tq_interp > threshold

    # Find contiguous regions
    region_starts = []
    region_ends = []
    in_region = False

    for i, val in enumerate(above_threshold):
        if val and not in_region:
            region_starts.append(time_axis[i])
            in_region = True
        elif not val and in_region:
            region_ends.append(time_axis[i - 1])
            in_region = False

    # Handle case if we're still in a region at the end
    if in_region:
        region_ends.append(time_axis[-1])

    # Shade the good transition regions
    for start, end in zip(region_starts, region_ends):
        if end - start > 2.0:  # Only highlight regions longer than 2 seconds
            ax2.axvspan(start, end, color='g', alpha=0.3)
            midpoint = (start + end) / 2
            ax2.text(midpoint, 0.8, 'GOOD TRANSITION', color='g', rotation=0,
                     verticalalignment='center', horizontalalignment='center')

    # Energy profile
    energy_peaks = features['energy']['energy_peaks']

    # Extract frequency band ratio average values
    low_ratio = features['energy']['frequency_bands']['low_ratio']['mean']
    mid_ratio = features['energy']['frequency_bands']['mid_ratio']['mean']
    high_ratio = features['energy']['frequency_bands']['high_ratio']['mean']

    # Create mini bar chart in a text box for frequency distribution
    ax3.text(max_time * 0.02, 0.8,
             f'Frequency Distribution:\nLow: {low_ratio:.2f}  Mid: {mid_ratio:.2f}  High: {high_ratio:.2f}',
             bbox=dict(facecolor='white', alpha=0.7))

    # Mark energy peaks
    for peak in energy_peaks:
        ax3.axvline(x=peak, color='orange', linestyle='-', alpha=0.5)

    # Add key annotation
    ax3.text(max_time * 0.02, 0.5, f"Key: {features['tonal']['key']}",
             bbox=dict(facecolor='white', alpha=0.7))

    # Add BPM annotation
    ax3.text(max_time * 0.02, 0.2, f"BPM: {features['rhythmic']['bpm']:.1f}",
             bbox=dict(facecolor='white', alpha=0.7))

    ax3.set_ylim(0, 1)
    ax3.set_ylabel('Energy Profile')
    ax3.set_xlabel('Time (seconds)')

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{track_name}_analysis.png"), dpi=150)
    plt.close(fig)


def visualize_comparison(comparison, output_folder):
    """Create visualization for track comparison."""
    track1_name = os.path.basename(comparison['track1']['path'])
    track2_name = os.path.basename(comparison['track2']['path'])

    compatibility = comparison['compatibility']

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Compatibility scores as a radar chart
    categories = ['Tempo', 'Key', 'Energy', 'Spectral']
    scores = [
        compatibility['tempo_compatibility'],
        compatibility['key_compatibility'],
        compatibility['energy_compatibility'],
        compatibility['spectral_compatibility']
    ]

    # Number of variables
    N = len(categories)

    # We are going to plot the first line of the data frame
    # But we need to repeat the first value to close the circular graph
    values = scores + [scores[0]]

    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Draw one axe per variable + add labels
    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax1.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=7)
    plt.ylim(0, 1)

    # Plot data
    ax1.plot(angles, values, linewidth=1, linestyle='solid')

    # Fill area
    ax1.fill(angles, values, 'b', alpha=0.1)

    # Add track names and overall score
    ax1.set_title(f"Mixing Compatibility: {compatibility['overall_score']:.2f}/1.00", size=11)
    ax1.text(0, 1.35, f"Track 1: {track1_name}", size=9)
    ax1.text(0, 1.25, f"Track 2: {track2_name}", size=9)

    # Mixing recommendation
    ax2.axis('off')
    recommendation_text = (
        f"MIXING RECOMMENDATIONS\n\n"
        f"Technique: {compatibility['recommended_technique']}\n"
        f"Minimum Crossfade Duration: {compatibility['recommended_min_crossfade']:.1f} seconds\n\n"
    )

    # Add specific advice based on compatibility scores
    if compatibility['tempo_compatibility'] < 0.7:
        recommendation_text += "• Tempo difference is significant. Consider tempo adjustment or effect-based transition.\n"

    if compatibility['key_compatibility'] < 0.7:
        recommendation_text += "• Keys are not highly compatible. Use EQ to help mask harmonic clashes or use effects.\n"

    if compatibility['energy_compatibility'] < 0.7:
        recommendation_text += "• Energy levels differ significantly. Consider using longer transitions and volume adjustment.\n"

    ax2.text(0.1, 0.9, recommendation_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Save figure
    fig_path = os.path.join(output_folder,
                            f"{os.path.splitext(track1_name)[0]}_vs_{os.path.splitext(track2_name)[0]}_compatibility.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f"Visualization saved to {fig_path}")


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Analyze and compare tracks for DJ mixing')

    # Create mutually exclusive group for track analysis vs. comparison
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--track', type=str, help='Path to audio track for analysis')
    group.add_argument('--compare', type=str, nargs=2, help='Paths to two audio tracks for comparison')

    # Output folder
    parser.add_argument('--output', type=str, default='output', help='Output folder for analysis results')

    # Parse arguments
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Perform requested operation
    if args.track:
        print(f"Analyzing track: {args.track}")
        analyze_track(args.track, args.output)
    elif args.compare:
        print(f"Comparing tracks: {args.compare[0]} and {args.compare[1]}")
        compare_tracks_for_mixing(args.compare[0], args.compare[1], args.output)