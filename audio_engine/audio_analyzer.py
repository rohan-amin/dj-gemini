# dj-gemini/audio_engine/audio_analyzer.py

import os
import json
import essentia.standard as es
import numpy as np
import time # Keep for the test block

# Assuming config.py is in the project root for the __main__ block.
# The class itself receives config values via __init__.

class AudioAnalyzer:
    def __init__(self, cache_dir, beats_cache_file_extension, 
                 beat_tracker_algo_name, bpm_estimator_algo_name):
        self.cache_dir = cache_dir
        self.beats_cache_file_extension = beats_cache_file_extension 
        self.beat_tracker_algo_name = beat_tracker_algo_name
        self.bpm_estimator_algo_name = bpm_estimator_algo_name
        
        if not os.path.exists(self.cache_dir):
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                print(f"DEBUG: AudioAnalyzer - Created cache directory: {self.cache_dir}")
            except OSError as e:
                print(f"ERROR: AudioAnalyzer - Could not create cache directory {self.cache_dir}: {e}")

    def _get_beats_cache_filepath(self, audio_filepath):
        """Generates a cache filepath for beat/bpm analysis data."""
        basename = os.path.basename(audio_filepath)
        # Example: starships.mp3 -> starships.mp3.beats (if extension is .beats)
        cache_file = basename + self.beats_cache_file_extension 
        return os.path.join(self.cache_dir, cache_file)

    def _get_cue_filepath(self, audio_filepath):
        """
        Generates the path for the .cue file.
        Example: /path/to/starships.mp3 -> /path/to/starships.mp3.cue
        """
        return audio_filepath + ".cue" # CORRECTED: Keep original full filename and add .cue

    def _load_cue_points(self, audio_filepath):
        """Loads cue points from a .cue JSON file if it exists."""
        cue_filepath = self._get_cue_filepath(audio_filepath)
        if os.path.exists(cue_filepath):
            print(f"DEBUG: AudioAnalyzer - Found cue file: {cue_filepath}")
            try:
                with open(cue_filepath, 'r') as f:
                    cue_data = json.load(f)
                if isinstance(cue_data, dict):
                    print(f"DEBUG: AudioAnalyzer - Successfully loaded {len(cue_data)} cue points from {os.path.basename(cue_filepath)}.")
                    return cue_data
                else:
                    print(f"WARNING: AudioAnalyzer - Cue file {cue_filepath} does not contain a valid JSON object (dictionary).")
            except json.JSONDecodeError:
                print(f"ERROR: AudioAnalyzer - Error decoding JSON from cue file: {cue_filepath}")
            except Exception as e:
                print(f"ERROR: AudioAnalyzer - Could not load or parse cue file {cue_filepath}: {e}")
        else:
            print(f"DEBUG: AudioAnalyzer - No cue file found at: {cue_filepath}")
        return {} 

    def analyze_track(self, audio_filepath):
        print(f"DEBUG: AudioAnalyzer - Analyzing track: {audio_filepath}")
        if not os.path.exists(audio_filepath):
            print(f"ERROR: AudioAnalyzer - Audio file not found: {audio_filepath}")
            return None
            
        beats_cache_filepath = self._get_beats_cache_filepath(audio_filepath)
        analysis_data = None # Will hold beats, bpm, sample_rate

        # 1. Check beats/bpm cache first
        if os.path.exists(beats_cache_filepath):
            print(f"DEBUG: AudioAnalyzer - Attempting to load analysis from beats cache: {beats_cache_filepath}")
            try:
                with open(beats_cache_filepath, 'r') as f:
                    analysis_data = json.load(f)
                if not all(k in analysis_data for k in ['beat_timestamps', 'bpm', 'sample_rate', 'file_path_original_for_beats']):
                    print(f"WARNING: AudioAnalyzer - Beats cache file {beats_cache_filepath} is missing keys. Re-analyzing.")
                    analysis_data = None 
                else:
                    print(f"DEBUG: AudioAnalyzer - Loaded from beats cache: BPM {analysis_data.get('bpm')}, Beats: {len(analysis_data.get('beat_timestamps', []))}")
            except Exception as e:
                print(f"ERROR: AudioAnalyzer - Could not load/parse beats cache {beats_cache_filepath}: {e}. Re-analyzing.")
                analysis_data = None
        
        if analysis_data is None: 
            print(f"DEBUG: AudioAnalyzer - No valid beats cache for {os.path.basename(audio_filepath)}. Performing new beat/bpm analysis.")
            try:
                print("DEBUG: AudioAnalyzer - Loading audio with Essentia's MonoLoader...")
                loader = es.MonoLoader(filename=audio_filepath)
                audio = loader()
                sample_rate = int(loader.paramValue('sampleRate'))

                if sample_rate == 0: print("ERROR: AudioAnalyzer - Sample rate is 0."); return None
                if len(audio) == 0: print("ERROR: AudioAnalyzer - Audio data empty."); return None
                
                print(f"DEBUG: AudioAnalyzer - Audio loaded. SR: {sample_rate}, Duration: {len(audio)/sample_rate:.2f}s")

                print(f"DEBUG: AudioAnalyzer - Performing beat detection ({self.beat_tracker_algo_name})...")
                beat_tracker_class = getattr(es, self.beat_tracker_algo_name)
                beat_tracker = beat_tracker_class()
                beat_timestamps_essentia = beat_tracker(audio)
                print(f"DEBUG: AudioAnalyzer - Beats: {len(beat_timestamps_essentia)}.")

                print(f"DEBUG: AudioAnalyzer - Performing BPM estimation ({self.bpm_estimator_algo_name})...")
                bpm_estimator_class = getattr(es, self.bpm_estimator_algo_name)
                bpm_estimator = bpm_estimator_class()
                
                bpm = 0.0
                if self.bpm_estimator_algo_name == "RhythmExtractor2013":
                    rhythm_results = bpm_estimator(audio) 
                    bpm = rhythm_results[0] 
                elif self.bpm_estimator_algo_name == "PercivalBpmEstimator":
                    bpm = bpm_estimator(audio) 
                else: 
                    bpm_result = bpm_estimator(audio)
                    bpm = float(bpm_result[0] if isinstance(bpm_result, (tuple, list, np.ndarray)) and len(bpm_result)>0 else bpm_result)
                print(f"DEBUG: AudioAnalyzer - BPM: {bpm:.2f}")

                analysis_data = {
                    'file_path_original_for_beats': audio_filepath, 
                    'sample_rate': sample_rate,
                    'beat_timestamps': beat_timestamps_essentia.tolist(), 
                    'bpm': float(bpm) 
                }
                try:
                    if not os.path.exists(self.cache_dir): os.makedirs(self.cache_dir, exist_ok=True)
                    print(f"DEBUG: AudioAnalyzer - Saving analysis to beats cache: {beats_cache_filepath}")
                    with open(beats_cache_filepath, 'w') as f:
                        json.dump(analysis_data, f, indent=4)
                    print("DEBUG: AudioAnalyzer - Beats cache saved.")
                except Exception as e:
                    print(f"ERROR: AudioAnalyzer - Could not save beats cache {beats_cache_filepath}: {e}")
            except Exception as e:
                print(f"ERROR: AudioAnalyzer - Essentia analysis failed for {audio_filepath}: {e}")
                import traceback
                traceback.print_exc() 
                return None

        # 2. Load cue points (always try to load, not part of the .beats cache)
        cue_points_data = self._load_cue_points(audio_filepath)
        analysis_data['cue_points'] = cue_points_data 
        analysis_data['file_path'] = audio_filepath # Ensure final dict has the primary file_path key

        return analysis_data

if __name__ == '__main__':
    print("--- Running AudioAnalyzer Standalone Test (with Cue Point Loading) ---")
    import sys
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR) 
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
    
    try:
        import config as app_config
    except ImportError:
        print("ERROR: Could not import config.py. Make sure it's in the project root.")
        sys.exit(1)

    app_config.ensure_dir_exists(app_config.BEATS_CACHE_DIR)
    # The audio_tracks directory itself for .cue files (AudioAnalyzer doesn't create this one)
    app_config.ensure_dir_exists(app_config.AUDIO_TRACKS_DIR) 


    analyzer = AudioAnalyzer(
        cache_dir=app_config.BEATS_CACHE_DIR,
        beats_cache_file_extension=app_config.BEATS_CACHE_FILE_EXTENSION, 
        beat_tracker_algo_name=app_config.DEFAULT_BEAT_TRACKER_ALGORITHM,
        bpm_estimator_algo_name=app_config.DEFAULT_BPM_ESTIMATOR_ALGORITHM
    )
    
    test_file_name = "starships.mp3" 
    test_audio_path = os.path.join(app_config.AUDIO_TRACKS_DIR, test_file_name)
    
    # --- Create a dummy .cue file for starships.mp3 for testing ---
    # Cue files are stored next to the audio track itself.
    # Path will be like .../audio_tracks/starships.mp3.cue
    dummy_cue_filepath = test_audio_path + ".cue" # Uses the corrected naming principle
    dummy_cue_data = {
        "intro_start": {"start_beat": 1},
        "drop1": {"start_beat": 65, "comment": "First drop"},
        "main_break_loop": {"start_beat": 97, "end_beat": 105}
    }
    try:
        with open(dummy_cue_filepath, 'w') as f:
            json.dump(dummy_cue_data, f, indent=4)
        print(f"DEBUG: Test - Created dummy cue file: {dummy_cue_filepath}")
    except Exception as e:
        print(f"ERROR: Test - Could not create dummy cue file {dummy_cue_filepath}: {e}")

    if not os.path.exists(test_audio_path):
        print(f"WARNING: Test audio file {test_audio_path} not found.")
    else:
        print(f"\n--- Analyzing: {test_audio_path} (will try to load its .cue file) ---")
        analysis_result = analyzer.analyze_track(test_audio_path)
        if analysis_result:
            print("\n--- Analysis Result ---")
            print(f"File Path: {analysis_result.get('file_path')}")
            print(f"Sample Rate: {analysis_result.get('sample_rate')}")
            print(f"BPM: {analysis_result.get('bpm')}")
            print(f"Beat Timestamps (first 5): {analysis_result.get('beat_timestamps', [])[:5]}")
            print(f"Cue Points Loaded: {json.dumps(analysis_result.get('cue_points'), indent=2)}")
        else:
            print("Analysis failed.")

        # Clean up dummy cue file if you want
        if os.path.exists(dummy_cue_filepath):
            # os.remove(dummy_cue_filepath)
            # print(f"DEBUG: Test - Removed dummy cue file: {dummy_cue_filepath}")
            pass # Keep it for now for manual inspection