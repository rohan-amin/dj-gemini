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
except ImportError as e:
    print(f"ERROR: Main - Could not import necessary modules.")
    print(f"       Ensure 'config.py' is in the project root ({PROJECT_ROOT_DIR})")
    print(f"       and the 'audio_engine' package (with __init__.py) is present.")
    print(f"       Import error details: {e}")
    sys.exit(1)
except Exception as e_general: # Catch any other exception during initial imports
    print(f"ERROR: Main - An unexpected error occurred during initial imports: {e_general}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def run_dj_gemini():
    parser = argparse.ArgumentParser(description="DJ Gemini - JSON Audio Mixer")
    parser.add_argument("json_script_path", type=str, help="Path to the JSON mix script file.")
    args = parser.parse_args()

    print(f"INFO: Main - DJ Gemini starting with script: {args.json_script_path}")

    # Create and initialize the audio engine
    # The AudioEngine's __init__ should handle importing its own config dependencies
    # or be passed configuration. Our current AudioEngine attempts to import config.
    try:
        audio_engine = AudioEngine()
    except Exception as e_engine_init:
        print(f"ERROR: Main - Failed to initialize AudioEngine: {e_engine_init}")
        import traceback
        traceback.print_exc()
        return # Exit if engine can't start

    if not audio_engine.load_script_from_file(args.json_script_path):
        print(f"ERROR: Main - Failed to load script '{args.json_script_path}'. Exiting.")
        return

    try:
        audio_engine.run_script() # This processes all actions in the JSON

        # Keep the main thread alive while decks are potentially playing.
        # The AudioEngine's run_script() is currently sequential and will return after
        # all actions (including 'wait' commands) are processed.
        # Decks play audio in their own background threads.
        print("INFO: Main - Script action processing complete. Monitoring active decks...")
        
        active_decks_found_after_script = False
        if audio_engine.decks:
            for deck_id, deck_instance in audio_engine.decks.items():
                if deck_instance.is_active():
                    active_decks_found_after_script = True
                    print(f"INFO: Main - Deck {deck_id} is still active.")
                    break # Found at least one active deck
        
        if not active_decks_found_after_script and not audio_engine.is_running_script:
            print("INFO: Main - No decks initially active after script completion. Mix might be short or finished.")
            # If the script was very short or only contained loads, it might end here.

        # Loop to keep alive if decks are playing, or if the script implies long running audio.
        # This is a simple polling mechanism.
        while True:
            if not audio_engine.is_running_script: # Script actions have been dispatched
                any_deck_truly_active = False
                if audio_engine.decks:
                    for deck in audio_engine.decks.values():
                        if deck.is_active():
                            any_deck_truly_active = True
                            break
                
                if not any_deck_truly_active:
                    print("INFO: Main - All decks appear idle. Assuming mix has concluded.")
                    break 
            
            time.sleep(1) # Check every second

    except KeyboardInterrupt:
        print("\nINFO: Main - KeyboardInterrupt received. Shutting down...")
    except Exception as e:
        print(f"ERROR: Main - An unexpected error occurred during script execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("INFO: Main - Initiating shutdown of audio engine and decks...")
        if 'audio_engine' in locals() and audio_engine: # Ensure engine was created
            audio_engine.shutdown_decks()
        print("INFO: Main - DJ Gemini finished.")

if __name__ == "__main__":
    run_dj_gemini()