# dj-gemini/config.py

import os
import logging
logger = logging.getLogger(__name__)

# --- Project Root Directory ---
# This assumes config.py is in the project's root directory (e.g., dj-gemini/)
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Core Directory Names (relative to project root) ---
AUDIO_TRACKS_DIR_NAME = "audio_tracks"
MIX_CONFIGS_DIR_NAME = "mix_configs"
CACHE_DIR_NAME = "cache" # Main folder for cached data
UTILITIES_DIR_NAME = "utilities"

# --- Full Absolute Paths (derived from above) ---
AUDIO_TRACKS_DIR = os.path.join(PROJECT_ROOT_DIR, AUDIO_TRACKS_DIR_NAME)
MIX_CONFIGS_DIR = os.path.join(PROJECT_ROOT_DIR, MIX_CONFIGS_DIR_NAME)
CACHE_DIR = os.path.join(PROJECT_ROOT_DIR, CACHE_DIR_NAME)
UTILITIES_DIR = os.path.join(PROJECT_ROOT_DIR, UTILITIES_DIR_NAME)

# Legacy compatibility - beats cache is now part of song-based cache structure
BEATS_CACHE_DIR = CACHE_DIR  # For backwards compatibility

# --- Essentia Algorithm Defaults (examples, can be expanded) ---
DEFAULT_BEAT_TRACKER_ALGORITHM = "BeatTrackerDegara"
DEFAULT_BPM_ESTIMATOR_ALGORITHM = "RhythmExtractor2013"

# --- Cache File Extension ---
BEATS_CACHE_FILE_EXTENSION = ".beats" # <<< UPDATED AS PER YOUR REQUEST
# Note: The cue files will still be, for example, "starships.mp3.cue" as per previous discussion.
# This BEATS_CACHE_FILE_EXTENSION is for the JSON file caching beat timestamps, BPM, etc.

# --- Helper Function to Ensure Directory Existence ---
def ensure_dir_exists(dir_path):
    """Checks if a directory exists, and creates it if it doesn't."""
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True) 
            logger.info(f"CONFIG: Created directory: {dir_path}")
        except OSError as e:
            logger.error(f"CONFIG - Could not create directory {dir_path}: {e}")
    # else:
    #     logger.debug(f"DEBUG: CONFIG - Directory already exists: {dir_path}")

# --- New Cache Structure Functions ---
import re
import hashlib

def sanitize_filename(filename):
    """Replace problematic chars with underscores for filesystem safety"""
    return re.sub(r'[<>:"/\\|?*\s]', '_', filename)

def get_song_cache_dir(audio_filepath):
    """Get cache directory for a specific song using hash + sanitized filename"""
    # Get absolute path for consistent hashing
    abs_path = os.path.abspath(audio_filepath)
    
    # Create hash of the full path
    path_hash = hashlib.md5(abs_path.encode('utf-8')).hexdigest()[:12]
    
    # Sanitize basename for safety
    basename = os.path.basename(audio_filepath)
    safe_name = sanitize_filename(basename)
    
    # Combine: hash_sanitized-filename
    cache_dir_name = f"{path_hash}_{safe_name}"
    
    return os.path.join(CACHE_DIR, cache_dir_name)

def get_beats_cache_filepath(audio_filepath):
    """Get filepath for beat analysis cache"""
    song_dir = get_song_cache_dir(audio_filepath)
    return os.path.join(song_dir, "analysis.beats")

def get_tempo_cache_filepath(audio_filepath, target_bpm):
    """Get filepath for tempo-processed audio cache"""
    song_dir = get_song_cache_dir(audio_filepath)
    return os.path.join(song_dir, f"tempo_{target_bpm:.1f}.npy")

def get_pitch_cache_filepath(audio_filepath, semitones):
    """Get filepath for pitch-processed audio cache"""
    song_dir = get_song_cache_dir(audio_filepath)
    return os.path.join(song_dir, f"pitch_{semitones:+.1f}.npy")

# --- Optional: Automatically ensure critical directories exist upon import ---
# print("DEBUG: CONFIG - Ensuring critical directories exist...")
# ensure_dir_exists(AUDIO_TRACKS_DIR)
# ensure_dir_exists(MIX_CONFIGS_DIR)
# ensure_dir_exists(CACHE_DIR) 
# ensure_dir_exists(BEATS_CACHE_DIR)   
# print("DEBUG: CONFIG - Directory check complete.")

# Global setting for EQ smoothing duration (in milliseconds).
# This controls how quickly EQ changes are interpolated to prevent clicks/pops
# when using the set_eq command (i.e., for near-instant EQ changes).
# It does NOT affect the fade_eq command, which uses its own per-action duration.
# Lower values make transitions faster (more 'instant'), but too low may reintroduce artifacts.
# Typical range: 0.5-5.0 ms. Default is 0.5 ms for near-instant, click-free EQ changes.
EQ_SMOOTHING_MS = 0.5

if __name__ == '__main__':
    logger.info(f"Project Root Directory: {PROJECT_ROOT_DIR}")
    logger.info(f"Audio Tracks Directory: {AUDIO_TRACKS_DIR}")
    logger.info(f"Mix Configs Directory: {MIX_CONFIGS_DIR}")
    logger.info(f"Cache Directory: {CACHE_DIR}")
    logger.info(f"Beats Cache Directory: {BEATS_CACHE_DIR} (Files will use {BEATS_CACHE_FILE_EXTENSION})")
    logger.info(f"Utilities Directory: {UTILITIES_DIR}")

    logger.info("\nEnsuring directories exist (example calls):")
    ensure_dir_exists(AUDIO_TRACKS_DIR)
    ensure_dir_exists(MIX_CONFIGS_DIR)
    ensure_dir_exists(CACHE_DIR)