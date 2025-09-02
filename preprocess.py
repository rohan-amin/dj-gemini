#!/usr/bin/env python3
"""
DJ Gemini Preprocessor

Analyzes JSON mix scripts and pre-processes all required tempo and pitch transformations.
This separates the heavy audio processing from the real-time performance execution.

Usage:
    python preprocess.py mix_configs/my_mix.json
"""

import argparse
import json
import os
import sys
import logging
import time
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc="Processing"):
        print(f"{desc}...")
        return iterable

# Add project root to path
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

import config as app_config
from audio_engine.audio_analyzer import AudioAnalyzer
from audio_engine.stem_separation import create_stem_separator
from audio_engine.extended_commands import validate_extended_command

def setup_logging(log_level='INFO'):
    """Set up logging for preprocessing"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_json_script(script_path):
    """Load and parse JSON mix script"""
    try:
        with open(script_path, 'r') as f:
            script_data = json.load(f)
        return script_data
    except Exception as e:
        logging.error(f"Failed to load script '{script_path}': {e}")
        return None

def analyze_script_requirements(script_data):
    """
    Analyze the script to extract all required audio transformations.
    Enhanced with smart BPM analysis for bpm_match commands and stem separation detection.
    Returns a dict of {track_path: {transformations}}
    """
    logger = logging.getLogger(__name__)
    requirements = {}
    deck_to_filepath = {}  # Map deck_id to file_path
    
    actions = script_data.get('actions', [])
    
    # First pass: build deck to filepath mapping
    for action in actions:
        command = action.get('command')
        deck_id = action.get('deck_id')
        parameters = action.get('parameters', {})
        
        if command == 'load_track':
            file_path = parameters.get('file_path') or parameters.get('filepath')
            if file_path:
                # Convert relative paths to absolute
                if not os.path.isabs(file_path):
                    # Check if path already includes audio_tracks directory
                    if file_path.startswith(app_config.AUDIO_TRACKS_DIR_NAME + "/"):
                        file_path = os.path.join(app_config.PROJECT_ROOT_DIR, file_path)
                    else:
                        file_path = os.path.join(app_config.AUDIO_TRACKS_DIR, file_path)
                
                deck_to_filepath[deck_id] = file_path
                
                if file_path not in requirements:
                    requirements[file_path] = {
                        'tempos': set(),
                        'pitches': set(),
                        'deck_ids': set(),
                        'needs_stems': False
                    }
                requirements[file_path]['deck_ids'].add(deck_id)
    
    # Second pass: analyze tempo transformations including smart BPM analysis
    predicted_bpms = analyze_bpm_flow(actions, deck_to_filepath)
    
    for action in actions:
        command = action.get('command')
        deck_id = action.get('deck_id')
        parameters = action.get('parameters', {})
        
        # Direct tempo transformations
        if command == 'set_tempo':
            target_bpm = parameters.get('target_bpm')
            if target_bpm and deck_id in deck_to_filepath:
                file_path = deck_to_filepath[deck_id]
                requirements[file_path]['tempos'].add(float(target_bpm))
        
        # Pitch transformations
        elif command == 'set_pitch':
            semitones = parameters.get('semitones')
            if semitones and deck_id in deck_to_filepath:
                file_path = deck_to_filepath[deck_id]
                requirements[file_path]['pitches'].add(float(semitones))
        
        # Tempo ramps - extract start/end BPM
        elif command == 'ramp_tempo':
            start_bpm = parameters.get('start_bpm')
            end_bpm = parameters.get('end_bpm')
            if start_bpm and end_bpm and deck_id in deck_to_filepath:
                file_path = deck_to_filepath[deck_id]
                requirements[file_path]['tempos'].add(float(start_bpm))
                requirements[file_path]['tempos'].add(float(end_bpm))
    
    # Add predicted BPMs from smart analysis
    for deck_id, bpm_set in predicted_bpms.items():
        if deck_id in deck_to_filepath:
            file_path = deck_to_filepath[deck_id]
            if file_path in requirements:
                requirements[file_path]['tempos'].update(bpm_set)
                if bpm_set:
                    logger.info(f"Smart analysis predicted BPMs for {deck_id}: {sorted(bpm_set)}")
    
    # Third pass: detect stem-based commands
    stem_commands = [
        'set_stem_volume', 'fade_stem_volume', 'set_stem_eq', 'fade_stem_eq',
        'solo_stem', 'mute_stem', 'isolate_stems', 'scratch_stem',
        'stem_crossfade', 'add_stem_routing', 'stem_filter_sweep', 
        'add_stem_reverb', 'analyze_stem_key', 'detect_stem_energy'
    ]
    
    for action in actions:
        command = action.get('command')
        deck_id = action.get('deck_id')
        
        if command in stem_commands:
            # This track needs stem separation
            if deck_id and deck_id in deck_to_filepath:
                file_path = deck_to_filepath[deck_id]
                if file_path in requirements:
                    requirements[file_path]['needs_stems'] = True
                    logger.info(f"Detected stem command '{command}' - {os.path.basename(file_path)} requires stem separation")
        
        # Multi-deck stem commands (no specific deck_id)
        elif command in ['stem_crossfade', 'add_stem_routing', 'remove_stem_routing', 'harmonic_mix', 'stem_sync']:
            parameters = action.get('parameters', {})
            # Check all deck references in parameters
            deck_refs = []
            for param_name in ['from_deck', 'to_deck', 'source_deck', 'target_deck', 'deck_a', 'deck_b', 'reference_deck']:
                if param_name in parameters:
                    deck_refs.append(parameters[param_name])
            
            for deck_ref in deck_refs:
                if deck_ref in deck_to_filepath:
                    file_path = deck_to_filepath[deck_ref]
                    if file_path in requirements:
                        requirements[file_path]['needs_stems'] = True
                        logger.info(f"Detected multi-deck stem command '{command}' - {os.path.basename(file_path)} requires stem separation")
    
    return requirements

def analyze_bpm_flow(actions, deck_to_filepath):
    """
    Smart analysis to predict BPM values that will be needed for bpm_match commands.
    Traces through script execution to predict possible BPM states.
    """
    logger = logging.getLogger(__name__)
    predicted_bpms = {}
    deck_bpm_timeline = {}  # Track BPM changes over time for each deck
    
    logger.info("Running smart BPM analysis for bpm_match prediction...")
    
    # Initialize with original BPMs (will be loaded from beat analysis)
    for deck_id, file_path in deck_to_filepath.items():
        deck_bpm_timeline[deck_id] = []
        predicted_bpms[deck_id] = set()
    
    # Sort actions by trigger beat number for timeline analysis
    timed_actions = []
    for action in actions:
        trigger = action.get('trigger', {})
        beat_number = 0
        
        if trigger.get('type') == 'script_start':
            beat_number = 0
        elif trigger.get('type') == 'on_deck_beat':
            beat_number = trigger.get('beat_number', 0)
        
        timed_actions.append((beat_number, action))
    
    # Sort by beat number
    timed_actions.sort(key=lambda x: x[0])
    
    # Analyze timeline
    for beat_number, action in timed_actions:
        command = action.get('command')
        deck_id = action.get('deck_id')
        parameters = action.get('parameters', {})
        
        # Track tempo changes
        if command == 'set_tempo' and deck_id:
            target_bpm = parameters.get('target_bpm')
            if target_bpm:
                deck_bpm_timeline[deck_id].append((beat_number, float(target_bpm)))
        
        elif command == 'ramp_tempo' and deck_id:
            start_bpm = parameters.get('start_bpm')
            end_bpm = parameters.get('end_bpm')
            start_beat = parameters.get('start_beat', beat_number)
            end_beat = parameters.get('end_beat', beat_number)
            
            if start_bpm and end_bpm:
                deck_bpm_timeline[deck_id].append((start_beat, float(start_bpm)))
                deck_bpm_timeline[deck_id].append((end_beat, float(end_bpm)))
        
        # Analyze bpm_match commands
        elif command == 'bpm_match':
            reference_deck = parameters.get('reference_deck')
            follow_deck = parameters.get('follow_deck')
            
            if reference_deck and follow_deck:
                # Predict what BPM the reference deck will have at this beat
                reference_bpm = predict_deck_bpm_at_beat(reference_deck, beat_number, deck_bpm_timeline, deck_to_filepath)
                
                if reference_bpm:
                    predicted_bpms[follow_deck].add(reference_bpm)
                    logger.info(f"Predicted bpm_match: {follow_deck} will need {reference_bpm} BPM at beat {beat_number}")
                else:
                    logger.warning(f"Could not predict BPM for {reference_deck} at beat {beat_number}")
                    # Ensure deck exists in predicted_bpms before updating
                    if follow_deck not in predicted_bpms:
                        predicted_bpms[follow_deck] = set()
                    # Add common BPM ranges as fallback
                    predicted_bpms[follow_deck].update([120, 125, 128, 130, 135, 140])
                    logger.info(f"Added fallback BPM range for {follow_deck}")
    
    return predicted_bpms

def predict_deck_bpm_at_beat(deck_id, target_beat, deck_bpm_timeline, deck_to_filepath):
    """
    Predict what BPM a deck will have at a specific beat number.
    """
    logger = logging.getLogger(__name__)
    if deck_id not in deck_bpm_timeline:
        return None
    
    timeline = deck_bpm_timeline[deck_id]
    if not timeline:
        # No tempo changes, try to get original BPM
        if deck_id in deck_to_filepath:
            try:
                # Try to get original BPM from beat analysis
                file_path = deck_to_filepath[deck_id]
                beat_cache_path = app_config.get_beats_cache_filepath(file_path)
                if os.path.exists(beat_cache_path):
                    with open(beat_cache_path, 'r') as f:
                        beat_data = json.load(f)
                    return float(beat_data.get('bpm', 0))
            except:
                pass
        return None
    
    # Find the last tempo change before or at the target beat
    applicable_bpm = None
    for beat, bpm in timeline:
        if beat <= target_beat:
            applicable_bpm = bpm
        else:
            break
    
    return applicable_bpm

def process_audio_file(audio_filepath, requirements, analyzer):
    """Process a single audio file with all required transformations"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing: {os.path.basename(audio_filepath)}")
    
    # Ensure song cache directory exists
    song_cache_dir = app_config.get_song_cache_dir(audio_filepath)
    app_config.ensure_dir_exists(song_cache_dir)
    
    # Step 1: Analyze beats and BPM if not already cached
    beats_cache_path = app_config.get_beats_cache_filepath(audio_filepath)
    if not os.path.exists(beats_cache_path):
        logger.info(f"  Analyzing beats and BPM...")
        analyzer.analyze_track(audio_filepath)
    else:
        logger.info(f"  Beat analysis found in cache")
    
    # Load audio for processing
    try:
        import essentia.standard as es
        import numpy as np
        
        # Load audio file
        loader = es.MonoLoader(filename=audio_filepath)
        audio_data = loader()
        sample_rate = loader.paramValue('sampleRate')
        
        logger.info(f"  Loaded audio: {len(audio_data)} samples at {sample_rate} Hz")
        
    except Exception as e:
        logger.error(f"  Failed to load audio: {e}")
        return False
    
    # Step 2: Process tempo transformations
    tempos = requirements.get('tempos', set())
    if tempos:
        logger.info(f"  Processing {len(tempos)} tempo variations...")
        
        try:
            import pyrubberband as pyrb
            
            # Get original BPM for tempo calculations
            with open(beats_cache_path, 'r') as f:
                beat_data = json.load(f)
            original_bpm = beat_data.get('bpm', 120.0)
            
            for target_bpm in tqdm(tempos, desc="  Tempo processing"):
                cache_path = app_config.get_tempo_cache_filepath(audio_filepath, target_bpm)
                
                if os.path.exists(cache_path):
                    logger.debug(f"    Tempo {target_bpm} BPM already cached")
                    continue
                
                # Calculate tempo ratio
                tempo_ratio = target_bpm / original_bpm
                
                # Process with Rubber Band
                logger.debug(f"    Processing tempo {target_bpm} BPM (ratio: {tempo_ratio:.3f})")
                processed_audio = pyrb.time_stretch(audio_data, int(sample_rate), tempo_ratio)
                
                # Save to cache
                np.save(cache_path, processed_audio.astype(np.float32))
                logger.debug(f"    Saved: {cache_path}")
                
        except ImportError:
            logger.error("  PyRubberBand not available for tempo processing")
            return False
        except Exception as e:
            logger.error(f"  Tempo processing failed: {e}")
            return False
    
    # Step 3: Process pitch transformations
    pitches = requirements.get('pitches', set())
    if pitches:
        logger.info(f"  Processing {len(pitches)} pitch variations...")
        
        try:
            import pyrubberband as pyrb
            
            for semitones in tqdm(pitches, desc="  Pitch processing"):
                cache_path = app_config.get_pitch_cache_filepath(audio_filepath, semitones)
                
                if os.path.exists(cache_path):
                    logger.debug(f"    Pitch {semitones:+.1f} semitones already cached")
                    continue
                
                # Process with Rubber Band
                logger.debug(f"    Processing pitch {semitones:+.1f} semitones")
                processed_audio = pyrb.pitch_shift(audio_data, int(sample_rate), semitones)
                
                # Save to cache
                np.save(cache_path, processed_audio.astype(np.float32))
                logger.debug(f"    Saved: {cache_path}")
                
        except Exception as e:
            logger.error(f"  Pitch processing failed: {e}")
            return False
    
    # Step 4: Process stem separation if needed
    needs_stems = requirements.get('needs_stems', False)
    if needs_stems:
        logger.info(f"  Processing stem separation...")
        
        try:
            # Create stem separator
            stem_separator = create_stem_separator()
            
            # Check if stems already exist
            stem_cache_dir = app_config.get_stems_cache_dir(audio_filepath)
            stems_exist = all(
                os.path.exists(os.path.join(stem_cache_dir, f"{stem}.npy"))
                for stem in ['vocals', 'drums', 'bass', 'other']
            )
            
            if stems_exist:
                logger.info(f"    Stems already cached")
            else:
                logger.info(f"    Separating stems (this may take several minutes)...")
                stem_result = stem_separator.separate_file(audio_filepath)
                
                if stem_result and stem_result.success:
                    logger.info(f"    ✓ Stem separation completed: {len(stem_result.stems)} stems")
                    for stem_name, stem_data in stem_result.stems.items():
                        logger.debug(f"      {stem_name}: {len(stem_data.audio_data)} samples, RMS: {stem_data.rms_level:.4f}")
                else:
                    logger.warning(f"    Stem separation failed or returned no results")
                    # Continue processing - stems are optional for some features
        
        except Exception as e:
            logger.error(f"  Stem separation failed: {e}")
            logger.warning(f"  Continuing without stems - some features may not work")
    
    logger.info(f"  ✓ Completed: {os.path.basename(audio_filepath)}")
    return True

def main():
    parser = argparse.ArgumentParser(description="DJ Gemini Preprocessor")
    parser.add_argument("json_script_path", type=str,
                       help="Path to the JSON mix script file")
    parser.add_argument("--log-level", 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Set logging level (default: INFO)')
    parser.add_argument("--force", "-f", action='store_true',
                       help='Force reprocessing even if cache exists')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_level)
    
    logger.info("DJ Gemini Preprocessor starting...")
    logger.info(f"Script: {args.json_script_path}")
    
    # Ensure directories exist
    app_config.ensure_dir_exists(app_config.CACHE_DIR)
    app_config.ensure_dir_exists(app_config.AUDIO_TRACKS_DIR)
    
    # Load script
    script_path = args.json_script_path
    if not os.path.isabs(script_path):
        # Try mix_configs directory first
        mix_configs_path = os.path.join(app_config.MIX_CONFIGS_DIR, script_path)
        if os.path.exists(mix_configs_path):
            script_path = mix_configs_path
        else:
            # Try project root
            project_root_path = os.path.join(app_config.PROJECT_ROOT_DIR, script_path)
            if os.path.exists(project_root_path):
                script_path = project_root_path
    
    script_data = load_json_script(script_path)
    if not script_data:
        logger.error("Failed to load script. Exiting.")
        return 1
    
    # Validate extended commands in script
    logger.info("Validating extended commands...")
    actions = script_data.get('actions', [])
    validation_errors = []
    
    for i, action in enumerate(actions):
        try:
            # Validate extended commands
            is_valid, error_msg = validate_extended_command(action)
            if not is_valid and error_msg:
                # Only warn for extended commands, don't fail validation
                command = action.get('command', 'unknown')
                logger.warning(f"Action {i+1} ('{command}'): {error_msg}")
        except Exception as e:
            logger.debug(f"Command validation skipped for action {i+1}: {e}")
    
    # Analyze requirements
    logger.info("Analyzing script requirements...")
    requirements = analyze_script_requirements(script_data)
    
    if not requirements:
        logger.warning("No audio transformations found in script.")
        return 0
    
    logger.info(f"Found {len(requirements)} tracks requiring preprocessing:")
    total_tempos = 0
    total_pitches = 0
    total_stems = 0
    for track_path, req in requirements.items():
        track_name = os.path.basename(track_path)
        tempo_count = len(req['tempos'])
        pitch_count = len(req['pitches'])
        needs_stems = req.get('needs_stems', False)
        
        total_tempos += tempo_count
        total_pitches += pitch_count
        if needs_stems:
            total_stems += 1
        
        stem_status = "+ stems" if needs_stems else ""
        logger.info(f"  {track_name}: {tempo_count} tempos, {pitch_count} pitches {stem_status}")
        if tempo_count > 0:
            logger.debug(f"    Tempo BPMs: {sorted(req['tempos'])}")
    
    if total_tempos > 0:
        logger.info(f"Smart analysis + explicit commands: {total_tempos} total tempo variants to process")
    if total_stems > 0:
        logger.info(f"Stem separation required for {total_stems} tracks")
    
    # Initialize analyzer
    analyzer = AudioAnalyzer(
        cache_dir=app_config.BEATS_CACHE_DIR,  # Will be updated by audio_analyzer.py changes
        beats_cache_file_extension=app_config.BEATS_CACHE_FILE_EXTENSION,
        beat_tracker_algo_name=app_config.DEFAULT_BEAT_TRACKER_ALGORITHM,
        bpm_estimator_algo_name=app_config.DEFAULT_BPM_ESTIMATOR_ALGORITHM
    )
    
    # Process each track
    start_time = time.time()
    success_count = 0
    
    for track_path, req in requirements.items():
        if not os.path.exists(track_path):
            logger.error(f"Audio file not found: {track_path}")
            continue
        
        if process_audio_file(track_path, req, analyzer):
            success_count += 1
    
    # Summary
    elapsed_time = time.time() - start_time
    logger.info(f"\nPreprocessing complete!")
    logger.info(f"Processed: {success_count}/{len(requirements)} tracks")
    logger.info(f"Time elapsed: {elapsed_time:.1f} seconds")
    
    if success_count == len(requirements):
        logger.info("✓ All tracks processed successfully. Ready for performance!")
        return 0
    else:
        logger.warning(f"⚠ {len(requirements) - success_count} tracks failed processing")
        return 1

if __name__ == "__main__":
    sys.exit(main())