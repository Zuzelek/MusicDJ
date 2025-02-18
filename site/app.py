from flask import Flask, request, redirect, jsonify, render_template, session, Response
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import csv
from io import StringIO
from audio_analyzer import AudioAnalyzer, AudioFeatures

app = Flask(__name__)
app.secret_key = '1bc4f365f4282600bcc405126d8ca0cf'  # Your Flask secret key

# Spotify API credentials
CLIENT_ID = "fceaae445db64dfa9879e3f632fb7d33"  # Your Client ID
CLIENT_SECRET = "821f091dec784c2ebef15865e206fb20"  # Your Client Secret
REDIRECT_URI = "http://localhost:8080/callback"  # Must match Spotify Dashboard

# Spotify OAuth setup
sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope="user-library-read playlist-read-private user-read-private user-read-email user-modify-playback-state"
)

@app.route('/')
def home():
    access_token = session.get('access_token')
    return render_template('index.html', access_token=access_token)

@app.route('/login')
def login():
    print("Login route called")  # Debugging
    auth_url = sp_oauth.get_authorize_url()
    print(f"Redirecting to Spotify auth URL: {auth_url}")  # Debugging
    return redirect(auth_url)

@app.route('/callback')
def callback():
    print("Callback route called")  # Debugging
    code = request.args.get('code')
    if not code:
        return "Error: No code provided by Spotify", 400

    print(f"Received code from Spotify: {code}")  # Debugging
    token_info = sp_oauth.get_access_token(code, as_dict=False)
    if isinstance(token_info, str):  # Handle direct token string
        access_token = token_info
        refresh_token = None
    else:  # Handle dictionary response
        access_token = token_info['access_token']
        refresh_token = token_info.get('refresh_token')

    # Store tokens in the session
    session['access_token'] = access_token
    session['refresh_token'] = refresh_token

    print("Login successful, redirecting to home")  # Debugging
    return redirect('/')

@app.route('/playlists')
def playlists():
    access_token = session.get('access_token')
    if not access_token:
        return jsonify({"error": "Not authenticated"}), 401

    try:
        sp = spotipy.Spotify(auth=access_token)
        playlists = sp.current_user_playlists()
        return jsonify([{'name': playlist['name'], 'id': playlist['id']} for playlist in playlists['items']])
    except spotipy.exceptions.SpotifyException as e:
        print(f"Spotify API error: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/playlists/<playlist_id>/songs')
def playlist_songs(playlist_id):
    access_token = session.get('access_token')
    if not access_token:
        return jsonify({"error": "Not authenticated"}), 401

    try:
        sp = spotipy.Spotify(auth=access_token)
        tracks = sp.playlist_tracks(playlist_id)
        analyzer = AudioAnalyzer()
        songs = []

        for item in tracks['items']:
            track = item['track']
            if not track:  # Skip if track is None
                continue

            # Fetch audio features for the track
            audio_features = sp.audio_features(track['id'])[0] if track['id'] else None

            # Analyze the track using your AudioAnalyzer
            analysis = analyzer.analyze_audio_from_features(audio_features)

            songs.append({
                'name': track['name'],
                'artist': track['artists'][0]['name'] if track['artists'] else 'Unknown',
                'id': track['id'],
                'bpm': analysis.tempo if analysis else None,
                'key': analysis.key if analysis else None,
                'energy': analysis.energy if analysis else None
            })

        return jsonify(songs)
    except spotipy.exceptions.SpotifyException as e:
        print(f"Spotify API error: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/playlists/<playlist_id>/export')
def export_playlist(playlist_id):
    access_token = session.get('access_token')
    if not access_token:
        return jsonify({"error": "Not authenticated"}), 401

    try:
        sp = spotipy.Spotify(auth=access_token)
        tracks = sp.playlist_tracks(playlist_id)
        analyzer = AudioAnalyzer()
        songs = []

        for item in tracks['items']:
            track = item['track']
            if not track:  # Skip if track is None
                continue

            # Fetch audio features for the track
            audio_features = sp.audio_features(track['id'])[0] if track['id'] else None

            # Analyze the track using your AudioAnalyzer
            analysis = analyzer.analyze_audio_from_features(audio_features)

            songs.append({
                'name': track['name'],
                'artist': track['artists'][0]['name'] if track['artists'] else 'Unknown',
                'bpm': analysis.tempo if analysis else None,
                'key': analysis.key if analysis else None,
                'energy': analysis.energy if analysis else None
            })

        # Create a CSV file in memory
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=['name', 'artist', 'bpm', 'key', 'energy'])
        writer.writeheader()
        writer.writerows(songs)

        # Return the CSV file as a download
        output.seek(0)
        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment;filename=playlist_{playlist_id}.csv"}
        )
    except spotipy.exceptions.SpotifyException as e:
        print(f"Spotify API error: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    app.run(port=8080, debug=True)