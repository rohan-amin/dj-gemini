# dj-gemini/main.py

import argparse
import time
import sys
import os
import logging
import json

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

def setup_logging(log_level_str='INFO'):
    """Set up logging with specified level"""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)



def run_dj_gemini():
    parser = argparse.ArgumentParser(description="DJ Gemini - JSON Audio Mixer")
    parser.add_argument("json_script_path", type=str, 
                        help="Path to the JSON mix script file (e.g., mix_configs/my_mix.json or an absolute path).")
    parser.add_argument("--max_wait_after_script", type=int, default=3600, 
                        help="Maximum seconds to wait for audio to finish after script actions complete (default: 3600s = 1hr). 0 for no wait if no audio playing.")
    parser.add_argument("--log-level", 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='INFO',
                       help='Set logging level (default: INFO)')
    parser.add_argument("--verbose", "-v", action='store_true', 
                       help='Enable verbose debug logging (same as --log-level DEBUG)')
    parser.add_argument("--quiet", "-q", action='store_true',
                       help='Only show errors and warnings (same as --log-level WARNING)')
    args = parser.parse_args()

    # Determine log level based on arguments
    if args.quiet:
        log_level = 'WARNING'
    elif args.verbose:
        log_level = 'DEBUG'
    else:
        log_level = args.log_level

    # Set up logging
    logger = setup_logging(log_level)
    
    logger.info(f"DJ Gemini starting with log level: {log_level}")
    logger.info(f"Script: {args.json_script_path}")
    if args.max_wait_after_script > 0:
        logger.info(f"Max wait time for audio after script actions: {args.max_wait_after_script}s")
    else:
        logger.info(f"No extra wait time after script actions. Program will exit once actions dispatched.")

    try:
        import config as app_config 
        from audio_engine.engine import AudioEngine
    except ImportError as e:
        logger.error(f"Main - Could not import necessary modules.")
        logger.error(f"       Ensure 'config.py' is in the project root ({PROJECT_ROOT_DIR})")
        logger.error(f"       and the 'audio_engine' package (with __init__.py and its modules) is present.")
        logger.error(f"       Import error details: {e}")
        sys.exit(1)
    except Exception as e_general: 
        logger.error(f"Main - An unexpected error occurred during initial imports: {e_general}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    logger.info("Ensuring necessary directories exist...")
    try:
        app_config.ensure_dir_exists(app_config.ANALYSIS_DATA_DIR)
        app_config.ensure_dir_exists(app_config.BEATS_CACHE_DIR)
        app_config.ensure_dir_exists(app_config.AUDIO_TRACKS_DIR)
        app_config.ensure_dir_exists(app_config.MIX_CONFIGS_DIR)
        logger.info("Directory check complete.")
    except Exception as e_dir:
        logger.error(f"Could not ensure directories: {e_dir}. Please check config.py and permissions.")
        return

    audio_engine = None 
    try:
        audio_engine = AudioEngine(app_config_module=app_config)
    except Exception as e_engine_init:
        logger.error(f"Failed to initialize AudioEngine: {e_engine_init}")
        import traceback
        traceback.print_exc()
        return 

    script_path_to_load = args.json_script_path
    if not os.path.isabs(script_path_to_load):
        path_in_mix_configs = os.path.join(app_config.MIX_CONFIGS_DIR, script_path_to_load)
        if os.path.exists(path_in_mix_configs):
            script_path_to_load = path_in_mix_configs
        else:
            path_in_project_root = os.path.join(app_config.PROJECT_ROOT_DIR, script_path_to_load)
            if os.path.exists(path_in_project_root):
                script_path_to_load = path_in_project_root

    if not audio_engine.load_script_from_file(script_path_to_load):
        logger.error(f"Failed to load script '{script_path_to_load}'. Exiting.")
        return

    try:
        audio_engine.start_script_processing()

        logger.info("Script action processing initiated.")
        
        wait_start_time = time.time()
        loop_count = 0

        logger.info("Monitoring. Press Ctrl+C to exit early.")
        while True: 
            loop_count += 1
            engine_is_dispatching_actions = audio_engine.is_processing_script_actions 
            decks_are_active = audio_engine.any_deck_active()
            
            # Reduce status logging frequency
            if loop_count % 50 == 0 or not engine_is_dispatching_actions:
                pending_triggers_count = len(getattr(audio_engine, '_pending_on_beat_actions', []))
                logger.debug(f"Main - Status: EngineDispatching={engine_is_dispatching_actions}, DecksActive={decks_are_active}, PendingActions={pending_triggers_count}")

            # Exit condition
            if not engine_is_dispatching_actions and not decks_are_active:
                if not getattr(audio_engine, '_pending_on_beat_actions', True):
                    logger.info("Engine done, no pending actions, and no decks active.")
                    break 
            
            # Timeout check
            if args.max_wait_after_script > 0 and (time.time() - wait_start_time) > args.max_wait_after_script:
                logger.warning(f"Max wait time of {args.max_wait_after_script}s reached.")
                if hasattr(audio_engine, '_pending_on_beat_actions') and audio_engine._pending_on_beat_actions:
                    logger.debug(f"Still {len(audio_engine._pending_on_beat_actions)} pending actions on timeout:")
                    for pa_idx, pa in enumerate(audio_engine._pending_on_beat_actions):
                        logger.debug(f"  Pending {pa_idx+1}: ID='{pa.get('id','N/A')}', CMD='{pa.get('command')}', Deck='{pa.get('deck_id','N/A')}', Trigger={pa.get('trigger')}")
                break
            time.sleep(0.5)
        logger.info("Monitoring loop finished.")

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Initiating shutdown of audio engine and decks...")
        if audio_engine: 
            audio_engine.stop_script_processing() 
        logger.info("DJ Gemini finished.")

if __name__ == "__main__":
    run_dj_gemini()