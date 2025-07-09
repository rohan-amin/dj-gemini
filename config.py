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
ANALYSIS_DATA_DIR_NAME = "analysis_data" # Main folder for any analysis output
UTILITIES_DIR_NAME = "utilities"

# --- Subdirectory Names for Analysis Data ---
# User specified: analysis_data/beats_data/ for the cache files
BEATS_DATA_SUBDIR_NAME = "beats_data" 

# --- Full Absolute Paths (derived from above) ---
AUDIO_TRACKS_DIR = os.path.join(PROJECT_ROOT_DIR, AUDIO_TRACKS_DIR_NAME)
MIX_CONFIGS_DIR = os.path.join(PROJECT_ROOT_DIR, MIX_CONFIGS_DIR_NAME)
ANALYSIS_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, ANALYSIS_DATA_DIR_NAME)
UTILITIES_DIR = os.path.join(PROJECT_ROOT_DIR, UTILITIES_DIR_NAME)

# Full path to the directory where beat analysis files will be cached/stored
BEATS_CACHE_DIR = os.path.join(ANALYSIS_DATA_DIR, BEATS_DATA_SUBDIR_NAME)

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

# --- Optional: Automatically ensure critical directories exist upon import ---
# print("DEBUG: CONFIG - Ensuring critical directories exist...")
# ensure_dir_exists(AUDIO_TRACKS_DIR)
# ensure_dir_exists(MIX_CONFIGS_DIR)
# ensure_dir_exists(ANALYSIS_DATA_DIR) 
# ensure_dir_exists(BEATS_CACHE_DIR)   
# print("DEBUG: CONFIG - Directory check complete.")

if __name__ == '__main__':
    logger.info(f"Project Root Directory: {PROJECT_ROOT_DIR}")
    logger.info(f"Audio Tracks Directory: {AUDIO_TRACKS_DIR}")
    logger.info(f"Mix Configs Directory: {MIX_CONFIGS_DIR}")
    logger.info(f"Analysis Data Directory: {ANALYSIS_DATA_DIR}")
    logger.info(f"Beats Cache Directory: {BEATS_CACHE_DIR} (Files will use {BEATS_CACHE_FILE_EXTENSION})")
    logger.info(f"Utilities Directory: {UTILITIES_DIR}")

    logger.info("\nEnsuring directories exist (example calls):")
    ensure_dir_exists(AUDIO_TRACKS_DIR)
    ensure_dir_exists(MIX_CONFIGS_DIR)
    ensure_dir_exists(BEATS_CACHE_DIR)