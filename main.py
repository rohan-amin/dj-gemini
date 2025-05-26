# dj-gemini/main.py

import argparse
import time
import sys
import os

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

try:
    import config as app_config 
    from audio_engine.engine import AudioEngine
except ImportError as e:
    print(f"ERROR: Main - Could not import necessary modules.")
    print(f"       Ensure 'config.py' is in the project root ({PROJECT_ROOT_DIR})")
    print(f"       and the 'audio_engine' package (with __init__.py and its modules) is present.")
    print(f"       Import error details: {e}")
    sys.exit(1)
except Exception as e_general: 
    print(f"ERROR: Main - An unexpected error occurred during initial imports: {e_general}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def run_dj_gemini():
    parser = argparse.ArgumentParser(description="DJ Gemini - JSON Audio Mixer")
    parser.add_argument("json_script_path", type=str, 
                        help="Path to the JSON mix script file (e.g., mix_configs/my_mix.json or an absolute path).")
    parser.add_argument("--max_wait_after_script", type=int, default=3600, 
                        help="Maximum seconds to wait for audio to finish after script actions complete (default: 3600s = 1hr). 0 for no wait if no audio playing.")
    args = parser.parse_args()

    print(f"INFO: Main - DJ Gemini starting with script: {args.json_script_path}")
    if args.max_wait_after_script > 0:
        print(f"INFO: Main - Max wait time for audio after script actions (if any audio playing): {args.max_wait_after_script}s")
    else:
        print(f"INFO: Main - No extra wait time after script actions. Program will exit once actions dispatched (if no audio playing).")


    print("INFO: Main - Ensuring necessary directories exist...")
    try:
        app_config.ensure_dir_exists(app_config.ANALYSIS_DATA_DIR)
        app_config.ensure_dir_exists(app_config.BEATS_CACHE_DIR)
        app_config.ensure_dir_exists(app_config.AUDIO_TRACKS_DIR)
        app_config.ensure_dir_exists(app_config.MIX_CONFIGS_DIR)
        print("INFO: Main - Directory check complete.")
    except Exception as e_dir:
        print(f"ERROR: Main - Could not ensure directories: {e_dir}. Please check config.py and permissions.")
        return


    audio_engine = None 
    try:
        audio_engine = AudioEngine(app_config_module=app_config)
    except Exception as e_engine_init:
        print(f"ERROR: Main - Failed to initialize AudioEngine: {e_engine_init}")
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
        print(f"ERROR: Main - Failed to load script '{script_path_to_load}'. Exiting.")
        return

    try:
        audio_engine.start_script_processing()

        print("INFO: Main - Script action processing initiated.")
        
        wait_start_time = time.time() # Initialize before the loop
        loop_count = 0

        # Enter monitoring loop if there were actions, or if user wants to wait regardless
        # (max_wait_after_script > 0 implies waiting even if script was empty but somehow engine started)
        # A more direct check is if the engine thread is expected to be running.
        # Let's simplify: if start_script_processing was called, we monitor.
        
        print("INFO: Main - Monitoring. Press Ctrl+C to exit early.")
        while True: 
            loop_count += 1
            engine_is_dispatching_actions = audio_engine.is_processing_script_actions 
            decks_are_active = audio_engine.any_deck_active()
            
            if loop_count % 20 == 0 or not engine_is_dispatching_actions: # Print status every ~10s or when dispatching done
                # Safely access _pending_on_beat_actions, which is an internal detail of AudioEngine
                pending_triggers_count_for_log = len(getattr(audio_engine, '_pending_on_beat_actions', []))
                print(f"DEBUG: Main - Status: EngineDispatchingActions={engine_is_dispatching_actions}, DecksActive={decks_are_active}, PendingOnBeatActions={pending_triggers_count_for_log} ({int(time.time() - wait_start_time)}s)")

            # Exit condition: Engine has finished dispatching all its script actions 
            # AND no decks are currently playing out audio.
            if not engine_is_dispatching_actions and not decks_are_active:
                # Further check: make sure engine's internal pending list is also empty
                if not getattr(audio_engine, '_pending_on_beat_actions', True): # If attr exists and is empty
                    print("INFO: Main - Engine done, no pending actions, and no decks active.")
                    break 
            
            # Timeout check
            if args.max_wait_after_script > 0 and (time.time() - wait_start_time) > args.max_wait_after_script:
                print(f"WARNING: Main - Max wait time of {args.max_wait_after_script}s reached.")
                if hasattr(audio_engine, '_pending_on_beat_actions') and audio_engine._pending_on_beat_actions:
                    print(f"DEBUG: Main - Still {len(audio_engine._pending_on_beat_actions)} pending actions on timeout:")
                    for pa_idx, pa in enumerate(audio_engine._pending_on_beat_actions):
                        print(f"  Pending {pa_idx+1}: ID='{pa.get('id','N/A')}', CMD='{pa.get('command')}', Deck='{pa.get('deck_id','N/A')}', Trigger={pa.get('trigger')}")
                break
            time.sleep(0.5) 
        print("INFO: Main - Monitoring loop finished.")

    except KeyboardInterrupt:
        print("\nINFO: Main - KeyboardInterrupt received. Shutting down...")
    except Exception as e:
        print(f"ERROR: Main - An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("INFO: Main - Initiating shutdown of audio engine and decks...")
        if audio_engine: 
            audio_engine.stop_script_processing() 
        print("INFO: Main - DJ Gemini finished.")

if __name__ == "__main__":
    run_dj_gemini()