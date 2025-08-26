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
    
    # Suppress verbose logging from third-party libraries
    if log_level_str.upper() == 'DEBUG':
        # Keep numba at INFO level to avoid bytecode dumps
        logging.getLogger('numba').setLevel(logging.INFO)
        # Suppress other verbose libraries
        logging.getLogger('librosa').setLevel(logging.WARNING)
        logging.getLogger('essentia').setLevel(logging.WARNING)
    
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
        app_config.ensure_dir_exists(app_config.CACHE_DIR)
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
        logger.error(f"Failed to load script '{script_path_to_load}'.")
        logger.error("If you see cache validation errors, run preprocessing first:")
        logger.error(f"  python preprocess.py {args.json_script_path}")
        return

    try:
        # Get initial script status
        if hasattr(audio_engine, '_all_actions_from_script'):
            total_actions = len(audio_engine._all_actions_from_script)
            logger.info(f"📜 Script loaded with {total_actions} actions")
        
        audio_engine.start_script_processing()

        logger.info("🚀 Script action processing initiated.")
        logger.info("📊 Event-driven architecture active - actions will be scheduled and executed automatically")
        
        wait_start_time = time.time()
        loop_count = 0

        logger.info("Monitoring. Press Ctrl+C to exit early.")
        
        # Get initial status for comparison
        last_status_time = time.time()
        last_engine_status = False
        last_deck_status = False
        
        while True: 
            loop_count += 1
            current_time = time.time()
            
            # Get current status
            engine_is_dispatching_actions = audio_engine.is_processing_script_actions 
            decks_are_active = audio_engine.any_deck_active()
            
            # Get event scheduler status
            scheduler_status = "unknown"
            pending_events = 0
            if hasattr(audio_engine, 'event_scheduler') and audio_engine.event_scheduler:
                try:
                    scheduler_stats = audio_engine.event_scheduler.get_stats()
                    total_scheduled = scheduler_stats.get('queue', {}).get('total', {}).get('total_scheduled', 0)
                    total_executed = scheduler_stats.get('queue', {}).get('total', {}).get('total_executed', 0)
                    pending_events = total_scheduled - total_executed
                    
                    if pending_events > 0:
                        scheduler_status = f"{pending_events} pending"
                    else:
                        scheduler_status = "idle"
                except Exception as e:
                    scheduler_status = f"error: {e}"
                    pending_events = 0
            
            # Status change detection and logging
            status_changed = (engine_is_dispatching_actions != last_engine_status or 
                            decks_are_active != last_deck_status or
                            loop_count % 20 == 0)  # Log every 20 iterations (10 seconds)
            
            if status_changed:
                elapsed = current_time - wait_start_time
                status_msg = f"Status: Engine={'🔄' if engine_is_dispatching_actions else '⏸️'} | Decks={'🎵' if decks_are_active else '🔇'} | Scheduler={scheduler_status} | Elapsed={elapsed:.1f}s"
                logger.info(status_msg)
                
                # Update last status for change detection
                last_engine_status = engine_is_dispatching_actions
                last_deck_status = decks_are_active
                last_status_time = current_time

            # Exit condition: Check if everything is complete
            if not engine_is_dispatching_actions and not decks_are_active:
                if pending_events == 0:
                    logger.info("🎉 All actions completed successfully!")
                    logger.info(f"Total execution time: {current_time - wait_start_time:.1f}s")
                    break
                else:
                    # Still have pending events but engine stopped - this might indicate an issue
                    logger.warning(f"Engine stopped but {pending_events} events still pending. This might indicate an issue.")
                    break
            
            # Timeout check
            if args.max_wait_after_script > 0 and (current_time - wait_start_time) > args.max_wait_after_script:
                logger.warning(f"⏰ Max wait time of {args.max_wait_after_script}s reached.")
                logger.info(f"Final status - Engine: {engine_is_dispatching_actions}, Decks: {decks_are_active}, Pending: {pending_events}")
                break
                
            time.sleep(0.5)
        logger.info("Monitoring loop finished.")

    except KeyboardInterrupt:
        logger.info("🛑 KeyboardInterrupt received. Shutting down gracefully...")
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("🔄 Initiating shutdown of audio engine and decks...")
        if audio_engine: 
            try:
                # Get final status before shutdown
                if hasattr(audio_engine, 'event_scheduler') and audio_engine.event_scheduler:
                    final_stats = audio_engine.event_scheduler.get_stats()
                    logger.info(f"Final event scheduler stats: {final_stats}")
                
                audio_engine.stop_script_processing()
                logger.info("✅ Audio engine shutdown completed successfully")
            except Exception as shutdown_error:
                logger.error(f"⚠️ Error during shutdown: {shutdown_error}")
        logger.info("🎯 DJ Gemini finished.")

if __name__ == "__main__":
    run_dj_gemini()