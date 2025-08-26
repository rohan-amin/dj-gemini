#!/usr/bin/env python3
"""
DJ Gemini Stem Preprocessor

Simplified preprocessor focused on stem separation since we now have real-time tempo/pitch processing.
Can process stems for tracks referenced in JSON files or individual audio files.

Usage:
    python preprocess_stems.py song.mp3                    # Process single file
    python preprocess_stems.py mix_configs/my_mix.json     # Process all tracks in JSON
    python preprocess_stems.py --help                      # Show all options
"""

import argparse
import json
import os
import sys
import logging
import time
from pathlib import Path

# Fix OpenMP issue on macOS before importing any ML libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="Processing"):
        print(f"{desc}...")
        return iterable

# Add project root to path
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

import config as app_config

def setup_logging(log_level='INFO'):
    """Set up logging"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_stems_cache_dir(audio_filepath):
    """Get the stems cache directory for an audio file"""
    # Get the song cache directory (named after the audio file)
    song_cache_dir = app_config.get_song_cache_dir(audio_filepath)
    
    # Stems go in a 'stems' subdirectory
    stems_dir = os.path.join(song_cache_dir, 'stems')
    return stems_dir

def check_stems_exist(audio_filepath):
    """Check if stems already exist for this audio file"""
    stems_dir = get_stems_cache_dir(audio_filepath)
    
    # Check for all 4 expected stem files
    stem_names = ['vocals', 'drums', 'bass', 'other']
    stem_files = [os.path.join(stems_dir, f"{stem}.npy") for stem in stem_names]
    
    return all(os.path.exists(f) for f in stem_files)

def separate_stems_demucs(audio_filepath, output_dir, model='htdemucs', shifts=1, overlap=0.25):
    """
    Use Demucs to separate stems
    Returns dict of stem_name -> numpy array
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Try importing demucs
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        from demucs.audio import convert_audio
        import torchaudio
        
        logger.info(f"Loading Demucs model: {model}")
        
        # Load the model
        separator = get_model(model)
        
        # Check for GPU availability - prioritize Apple Silicon MPS, then CUDA, then CPU
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            
        separator = separator.to(device)
        
        logger.info(f"Using device: {device}")
        logger.info(f"Loading audio: {os.path.basename(audio_filepath)}")
        
        # Load audio
        wav, sr = torchaudio.load(audio_filepath)
        wav = convert_audio(wav, sr, separator.samplerate, separator.audio_channels)
        
        # Move to device
        wav = wav.to(device)
        
        logger.info(f"Separating stems (this may take several minutes)...")
        
        # Apply separation with quality settings
        with torch.no_grad():
            sources = apply_model(
                separator, 
                wav[None], 
                device=device, 
                progress=True,
                shifts=shifts,  # Test-time augmentation
                overlap=overlap  # Overlap between segments
            )[0]
        
        # Convert to numpy and save
        stems = {}
        stem_names = separator.sources  # ['drums', 'bass', 'other', 'vocals']
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        for i, stem_name in enumerate(stem_names):
            # Convert to numpy and keep stereo
            stem_audio = sources[i].cpu().numpy()
            # Ensure consistent stereo format: (channels, samples) -> transpose if needed
            if len(stem_audio.shape) > 1 and stem_audio.shape[0] == 2:
                # Already in (2, samples) format - keep as stereo
                pass
            elif len(stem_audio.shape) == 1:
                # Mono - convert to stereo by duplicating
                stem_audio = np.stack([stem_audio, stem_audio])
            else:
                # Other format - ensure stereo
                if stem_audio.shape[1] == 2:
                    stem_audio = stem_audio.T  # (samples, 2) -> (2, samples)
                else:
                    # Take first channel and duplicate
                    stem_audio = np.stack([stem_audio[0], stem_audio[0]])
            
            # Save as numpy array
            output_path = os.path.join(output_dir, f"{stem_name}.npy")
            import numpy as np
            np.save(output_path, stem_audio.astype(np.float32))
            
            stems[stem_name] = stem_audio
            logger.info(f"  Saved {stem_name}: {output_path}")
        
        logger.info(f"✓ Stem separation completed: {len(stems)} stems")
        return stems
        
    except ImportError as e:
        logger.error(f"Demucs not available: {e}")
        logger.info("Install with: pip install demucs")
        return None
    except Exception as e:
        logger.error(f"Stem separation failed: {e}")
        return None


def process_single_file(audio_filepath, force=False, model='htdemucs', shifts=1, overlap=0.25):
    """Process stem separation for a single audio file"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(audio_filepath):
        logger.error(f"File not found: {audio_filepath}")
        return False
    
    logger.info(f"Processing: {os.path.basename(audio_filepath)}")
    
    # Check if stems already exist
    if not force and check_stems_exist(audio_filepath):
        logger.info("  Stems already exist (use --force to reprocess)")
        return True
    
    # Get output directory
    output_dir = get_stems_cache_dir(audio_filepath)
    
    # Use Demucs for stem separation
    stems = separate_stems_demucs(audio_filepath, output_dir, model, shifts, overlap)
    
    if stems is None:
        logger.error("  ✗ Stem separation failed with Demucs")
        logger.error("  Please check that Demucs is properly installed: pip install demucs")
        return False
    
    logger.info(f"  ✓ Completed: {os.path.basename(audio_filepath)}")
    return True

def extract_tracks_from_json(json_path):
    """Extract all track paths from a JSON mix file"""
    logger = logging.getLogger(__name__)
    
    try:
        with open(json_path, 'r') as f:
            script_data = json.load(f)
        
        track_paths = set()
        actions = script_data.get('actions', [])
        
        for action in actions:
            if action.get('command') == 'load_track':
                parameters = action.get('parameters', {})
                file_path = parameters.get('file_path') or parameters.get('filepath')
                
                if file_path:
                    # Convert relative paths to absolute
                    if not os.path.isabs(file_path):
                        if file_path.startswith(app_config.AUDIO_TRACKS_DIR_NAME + "/"):
                            file_path = os.path.join(app_config.PROJECT_ROOT_DIR, file_path)
                        else:
                            file_path = os.path.join(app_config.AUDIO_TRACKS_DIR, file_path)
                    
                    track_paths.add(file_path)
        
        logger.info(f"Found {len(track_paths)} unique tracks in JSON file")
        return list(track_paths)
        
    except Exception as e:
        logger.error(f"Failed to parse JSON file: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="DJ Gemini Stem Preprocessor")
    parser.add_argument("input", type=str,
                       help="Audio file (.mp3, .wav, etc.) or JSON mix file")
    parser.add_argument("--model", "-m", 
                       choices=['htdemucs', 'htdemucs_ft', 'mdx', 'mdx_extra'],
                       default='htdemucs_ft',  # Use fine-tuned version by default
                       help='Demucs model to use (default: htdemucs_ft)')
    parser.add_argument("--shifts", type=int, default=1,
                       help='Number of random shifts for better separation quality (default: 1, higher=better/slower)')
    parser.add_argument("--overlap", type=float, default=0.25,
                       help='Overlap between segments (default: 0.25, higher=better/slower)')
    parser.add_argument("--force", "-f", action='store_true',
                       help='Force reprocessing even if stems exist')
    parser.add_argument("--log-level", 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Set logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_level)
    
    logger.info("DJ Gemini Stem Preprocessor starting...")
    logger.info(f"Input: {args.input}")
    logger.info(f"Model: {args.model}")
    
    # Ensure cache directory exists
    app_config.ensure_dir_exists(app_config.CACHE_DIR)
    
    # Determine if input is JSON or audio file
    input_path = Path(args.input)
    
    if input_path.suffix.lower() == '.json':
        # Process JSON file
        logger.info("Processing JSON mix file...")
        
        # Find JSON file
        json_path = args.input
        if not os.path.isabs(json_path):
            # Try mix_configs directory first
            mix_configs_path = os.path.join(app_config.MIX_CONFIGS_DIR, json_path)
            if os.path.exists(mix_configs_path):
                json_path = mix_configs_path
            else:
                # Try project root
                project_root_path = os.path.join(app_config.PROJECT_ROOT_DIR, json_path)
                if os.path.exists(project_root_path):
                    json_path = project_root_path
        
        # Extract track paths
        track_paths = extract_tracks_from_json(json_path)
        if not track_paths:
            logger.error("No tracks found in JSON file")
            return 1
        
    else:
        # Process single audio file
        logger.info("Processing single audio file...")
        
        # Handle relative paths
        audio_path = args.input
        if not os.path.isabs(audio_path):
            # Try audio_tracks directory
            tracks_path = os.path.join(app_config.AUDIO_TRACKS_DIR, audio_path)
            if os.path.exists(tracks_path):
                audio_path = tracks_path
            else:
                # Try project root
                project_root_path = os.path.join(app_config.PROJECT_ROOT_DIR, audio_path)
                if os.path.exists(project_root_path):
                    audio_path = project_root_path
        
        track_paths = [audio_path]
    
    # Process all tracks
    start_time = time.time()
    success_count = 0
    
    for track_path in track_paths:
        if process_single_file(track_path, args.force, args.model, args.shifts, args.overlap):
            success_count += 1
    
    # Summary
    elapsed_time = time.time() - start_time
    logger.info(f"\nStem preprocessing complete!")
    logger.info(f"Processed: {success_count}/{len(track_paths)} tracks")
    logger.info(f"Time elapsed: {elapsed_time:.1f} seconds")
    
    if success_count == len(track_paths):
        logger.info("✓ All stems processed successfully!")
        return 0
    else:
        logger.warning(f"⚠ {len(track_paths) - success_count} tracks failed processing")
        return 1

if __name__ == "__main__":
    sys.exit(main())