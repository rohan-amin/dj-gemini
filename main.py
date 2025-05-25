# dj-gemini/main.py

import argparse
import time
import sys
import os

# Ensure the project root is in sys.path for consistent imports
# This assumes main.py is in the project's root directory.
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

try:
    import config as app_config # Expects config.py in the project root
    from audio_engine.engine import AudioEngine
    # audio_analyzer and deck are used internally by AudioEngine
except ImportError as e:
    print(f"ERROR: Main - Could not import necessary modules.")
    print(f"       Ensure 'config.py' is in the project root ({PROJECT_ROOT_DIR})")
    print(f"       and the 'audio_engine' package (with __init__.py and its modules) is present.")
    print(f"       Import error details: {e}")
    sys.exit(1)
except Exception as e_general: # Catch any other exception during initial imports
    print(f"ERROR: Main - An unexpected error occurred during initial imports: {e_general}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def run_dj_gemini():
    parser = argparse.ArgumentParser(description="DJ Gemini - JSON Audio Mixer")
    parser.add_argument("json_script_path", type=str, 
                        help="Path to the JSON mix script file (e.g., mix_configs/my_mix.json or an absolute path).")
    parser.add_argument("--max_wait_after_script", type=int, default=3600, 
                        help="Maximum seconds to wait for audio to finish after script actions complete (default: 3600s = 1hr). 0 for no wait.")
    args = parser.parse_args()

    print(f"INFO: Main - DJ Gemini starting with script: {args.json_script_path}")
    if args.max_wait_after_script > 0:
        print(f"INFO: Main - Max wait time for audio after script actions: {args.max_wait_after_script}s")
    else:
        print(f"INFO: Main - No extra wait time after script actions. Program will exit once actions are dispatched.")


    # Ensure critical directories exist (AudioEngine's __init__ also does some checks)
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


    audio_engine = None # Define engine outside try block for finally
    try:
        # Pass the imported app_config module to AudioEngine
        audio_engine = AudioEngine(app_config_module=app_config)
    except Exception as e_engine_init:
        print(f"ERROR: Main - Failed to initialize AudioEngine: {e_engine_init}")
        import traceback
        traceback.print_exc()
        return 

    # Resolve the JSON script path
    script_path_to_load = args.json_script_path
    if not os.path.isabs(script_path_to_load):
        # Try relative to MIX_CONFIGS_DIR first
        path_in_mix_configs = os.path.join(app_config.MIX_CONFIGS_DIR, script_path_to_load)
        if os.path.exists(path_in_mix_configs):
            script_path_to_load = path_in_mix_configs
        else:
            # Fallback: try relative to project root (if it's not the same as MIX_CONFIGS_DIR)
            path_in_project_root = os.path.join(app_config.PROJECT_ROOT_DIR, script_path_to_load)
            if os.path.exists(path_in_project_root):
                script_path_to_load = path_in_project_root
            # If still not found as absolute, os.path.exists in load_script_from_file will handle it.

    if not audio_engine.load_script_from_file(script_path_to_load):
        print(f"ERROR: Main - Failed to load script '{script_path_to_load}'. Exiting.")
        return

    try:
        audio_engine.run_script() # This processes all actions in the JSON

        print("INFO: Main - Script action processing dispatched.")
        
        if not audio_engine.decks and args.max_wait_after_script > 0: # No decks were even created
            print("INFO: Main - No decks were initialized by the script. Exiting.")
        elif args.max_wait_after_script > 0 :
            print("INFO: Main - Monitoring active decks. Press Ctrl+C to exit early.")
            wait_start_time = time.time()
            while True:
                # is_running_script becomes False when engine finishes dispatching actions
                # any_deck_active checks if any deck's internal audio thread is still playing
                if not audio_engine.is_running_script and not audio_engine.any_deck_active():
                    print("INFO: Main - All decks idle and script actions finished. Mix has concluded.")
                    break
                
                if (time.time() - wait_start_time) > args.max_wait_after_script:
                    print(f"WARNING: Main - Max wait time of {args.max_wait_after_script}s reached. Forcing shutdown.")
                    break
                
                if int(time.time() - wait_start_time) > 0 and int(time.time() - wait_start_time) % 10 == 0 : 
                    active_deck_ids = [deck_id for deck_id, deck in audio_engine.decks.items() if deck.is_active()]
                    if active_deck_ids:
                        print(f"DEBUG: Main - Still waiting for active decks: {active_deck_ids} ... ({int(time.time() - wait_start_time)}s elapsed)")
                    elif not audio_engine.is_running_script: # Should have been caught by the main break condition
                        print(f"DEBUG: Main - Waiting, no decks seem active and script done... ({int(time.time() - wait_start_time)}s elapsed)")

                time.sleep(0.5) # Check every half second

    except KeyboardInterrupt:
        print("\nINFO: Main - KeyboardInterrupt received. Shutting down...")
    except Exception as e:
        print(f"ERROR: Main - An unexpected error occurred during script execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("INFO: Main - Initiating shutdown of audio engine and decks...")
        if audio_engine: 
            audio_engine.shutdown_decks()
        print("INFO: Main - DJ Gemini finished.")

if __name__ == "__main__":
    run_dj_gemini()