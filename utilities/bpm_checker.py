#!/usr/bin/env python3
# dj-gemini/utilities/bpm_checker.py

import os
import sys
import logging
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import config as app_config
    from audio_engine.audio_analyzer import AudioAnalyzer
except ImportError as e:
    logger.error(f"Could not import necessary modules: {e}")
    sys.exit(1)

def print_bpm(audio_file_path):
    """Print the BPM of an audio file"""
    
    # Ensure the file exists
    if not os.path.exists(audio_file_path):
        logger.error(f"File not found: {audio_file_path}")
        return False
    
    # Initialize analyzer
    analyzer = AudioAnalyzer(
        cache_dir=app_config.BEATS_CACHE_DIR,
        beats_cache_file_extension=app_config.BEATS_CACHE_FILE_EXTENSION,
        beat_tracker_algo_name=app_config.DEFAULT_BEAT_TRACKER_ALGORITHM,
        bpm_estimator_algo_name=app_config.DEFAULT_BPM_ESTIMATOR_ALGORITHM
    )
    
    # Analyze the track
    logger.info(f"Analyzing: {os.path.basename(audio_file_path)}")
    analysis_result = analyzer.analyze_track(audio_file_path)
    
    if analysis_result:
        bpm = analysis_result.get('bpm', 0)
        beat_count = len(analysis_result.get('beat_timestamps', []))
        duration = len(analysis_result.get('beat_timestamps', [])) / (bpm / 60) if bpm > 0 else 0
        
        logger.info(f"BPM: {bpm:.2f}")
        logger.info(f"Beat count: {beat_count}")
        logger.info(f"Estimated duration: {duration:.1f} seconds")
        
        # Print cue points if available
        cue_points = analysis_result.get('cue_points', {})
        if cue_points:
            logger.info(f"Cue points: {list(cue_points.keys())}")
        
        return True
    else:
        logger.error("Could not analyze the audio file")
        return False

def main():
    if len(sys.argv) != 2:
        logger.info("Usage: python bpm_checker.py <audio_file_path>")
        logger.info("Example: python bpm_checker.py audio_tracks/starships.mp3")
        sys.exit(1)
    
    audio_file_path = sys.argv[1]
    
    # If it's a relative path, try to resolve it from the audio_tracks directory
    if not os.path.isabs(audio_file_path):
        # Try from audio_tracks directory
        audio_tracks_path = os.path.join(app_config.AUDIO_TRACKS_DIR, audio_file_path)
        if os.path.exists(audio_tracks_path):
            audio_file_path = audio_tracks_path
        # If still not found, try as relative to current directory
        elif not os.path.exists(audio_file_path):
            logger.error(f"File not found: {audio_file_path}")
            logger.error(f"Tried: {audio_tracks_path}")
            sys.exit(1)
    
    success = print_bpm(audio_file_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 