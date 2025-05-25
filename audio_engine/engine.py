# dj-gemini/audio_engine/engine.py

import json
import time
import os
import sys # Needed for __main__ block path adjustments

# Attempt to import config. This should work when engine.py is imported by main.py (at project root).
# For standalone testing (if __name__ == '__main__'), the test block will adjust sys.path.
try:
    import config as app_config
except ImportError:
    app_config = None # Will be handled in __init__ or __main__ test block
    print("WARNING: engine.py - Initial 'import config' failed. This is expected if not run via main.py or test block.")

from .audio_analyzer import AudioAnalyzer
from .deck import Deck

class AudioEngine:
    def __init__(self):
        print("DEBUG: AudioEngine - Initializing...")
        
        # Ensure app_config is loaded, especially for standalone testing or if AudioEngine is instantiated first.
        # This is a bit redundant if main.py already ensures config is loaded, but safe.
        global app_config # Allow modification of the module-level app_config if it was None
        if app_config is None:
            try:
                # Try to load config again, assuming project root might now be in path via __main__
                # or because this is part of a larger import chain.
                import config as imported_config
                app_config = imported_config
            except ImportError:
                 # Last resort if still not found - this indicates a problem with how engine is run/imported
                print("CRITICAL ERROR: AudioEngine - app_config module not found. Cannot proceed without configuration.")
                raise ImportError("AudioEngine requires config.py to be accessible in the Python path.")

        # Ensure necessary directories from config exist
        app_config.ensure_dir_exists(app_config.BEATS_CACHE_DIR)
        # AUDIO_TRACKS_DIR and MIX_CONFIGS_DIR are more for user content, but good to check.
        # app_config.ensure_dir_exists(app_config.AUDIO_TRACKS_DIR) 
        # app_config.ensure_dir_exists(app_config.MIX_CONFIGS_DIR)


        self.analyzer = AudioAnalyzer(
            cache_dir=app_config.BEATS_CACHE_DIR,
            beats_cache_file_extension=app_config.BEATS_CACHE_FILE_EXTENSION,
            beat_tracker_algo_name=app_config.DEFAULT_BEAT_TRACKER_ALGORITHM,
            bpm_estimator_algo_name=app_config.DEFAULT_BPM_ESTIMATOR_ALGORITHM
        )
        self.decks = {} # Store Deck instances, keyed by deck_id (e.g., "deckA")
        self.script_name = "Untitled Mix"
        self.script_actions = []
        self.is_running_script = False # Flag to indicate if script processing loop is active
        print("DEBUG: AudioEngine - Initialized.")

    def _get_or_create_deck(self, deck_id):
        """Retrieves a deck by ID, creating it if it doesn't exist."""
        if deck_id not in self.decks:
            print(f"DEBUG: AudioEngine - Creating new Deck: {deck_id}")
            self.decks[deck_id] = Deck(deck_id, self.analyzer)
        return self.decks[deck_id]

    def load_script_from_file(self, json_filepath):
        """Loads and parses a JSON mix script."""
        print(f"DEBUG: AudioEngine - Loading script from: {json_filepath}")
        # Ensure the path is absolute if app_config is available and path is relative
        if app_config and not os.path.isabs(json_filepath):
            abs_json_filepath = os.path.join(app_config.PROJECT_ROOT_DIR, json_filepath)
            if not os.path.exists(abs_json_filepath) and os.path.exists(json_filepath):
                 # If absolute path didn't work but original (potentially relative to CWD) did
                 pass # Use original json_filepath
            else:
                json_filepath = abs_json_filepath


        if not os.path.exists(json_filepath):
            print(f"ERROR: AudioEngine - Script file not found: {json_filepath}")
            self.script_actions = []
            return False
            
        try:
            with open(json_filepath, 'r') as f:
                script_data = json.load(f)
            
            self.script_name = script_data.get("script_name", "Untitled Mix")
            self.script_actions = script_data.get("actions", []) # Expect a list of actions
            if not isinstance(self.script_actions, list):
                print(f"ERROR: AudioEngine - 'actions' in script must be a list. Found: {type(self.script_actions)}")
                self.script_actions = []
                return False

            print(f"DEBUG: AudioEngine - Script '{self.script_name}' loaded with {len(self.script_actions)} actions.")
            return True
        except json.JSONDecodeError:
            print(f"ERROR: AudioEngine - Error decoding JSON from script: {json_filepath}")
        except Exception as e:
            print(f"ERROR: AudioEngine - Failed to load script {json_filepath}: {e}")
        
        self.script_actions = []
        return False

    def run_script(self):
        """Executes the loaded script actions sequentially."""
        if not self.script_actions:
            print("INFO: AudioEngine - No actions in script to run.")
            return

        print(f"INFO: AudioEngine - Running script: '{self.script_name}'")
        self.is_running_script = True
        action_count = 0
        try:
            for action in self.script_actions:
                action_count += 1
                action_command = action.get('command', 'N/A')
                action_deck_id = action.get('deck_id', 'Engine-Level')
                print(f"\nINFO: AudioEngine - Executing action {action_count}/{len(self.script_actions)}: CMD='{action_command}', Deck='{action_deck_id}'")
                
                self._execute_action(action)
                
                if not self.is_running_script: # Check if an action (like future 'stop_engine') stopped it
                    print("INFO: AudioEngine - Script execution halted by an internal signal.")
                    break
            print("INFO: AudioEngine - All script actions have been dispatched.")
        except Exception as e:
            print(f"ERROR: AudioEngine - Error during script execution loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running_script = False 
            # Note: run_script() finishes when all actions are dispatched.
            # Actual audio might continue playing in Deck threads.
            # main.py will be responsible for keeping the program alive.

    def _execute_action(self, action_dict):
        """Executes a single action from the script."""
        command = action_dict.get("command")
        deck_id = action_dict.get("deck_id")
        parameters = action_dict.get("parameters", {})

        if not command:
            print("WARNING: AudioEngine - Action missing 'command'. Skipping.")
            return

        print(f"DEBUG: AudioEngine - Action details: cmd='{command}', deck='{deck_id}', params='{parameters}'")

        try:
            if command == "load_track":
                if not deck_id or "file_path" not in parameters:
                    print(f"WARNING: AudioEngine - 'load_track' missing deck_id or file_path. Skipping.")
                    return
                deck = self._get_or_create_deck(deck_id)
                
                file_path_param = parameters["file_path"]
                track_path = file_path_param
                if app_config and not os.path.isabs(file_path_param):
                    # Assume relative paths in JSON are from AUDIO_TRACKS_DIR
                    constructed_path = os.path.join(app_config.AUDIO_TRACKS_DIR, file_path_param)
                    if os.path.exists(constructed_path):
                        track_path = constructed_path
                    else: # Fallback to try relative to project root if not in AUDIO_TRACKS_DIR
                        constructed_path_alt = os.path.join(app_config.PROJECT_ROOT_DIR, file_path_param)
                        if os.path.exists(constructed_path_alt):
                             track_path = constructed_path_alt
                        else:
                             print(f"WARNING: AudioEngine - Track file '{file_path_param}' not found in default locations. Trying as is.")
                
                deck.load_track(track_path)

            elif command == "play":
                if not deck_id:
                    print("WARNING: AudioEngine - 'play' missing deck_id. Skipping.")
                    return
                deck = self._get_or_create_deck(deck_id)
                start_at_beat = parameters.get("start_at_beat") 
                start_at_cue_name = parameters.get("start_at_cue_name")
                deck.play(start_at_beat=start_at_beat, start_at_cue_name=start_at_cue_name)

            elif command == "pause":
                if not deck_id:
                    print("WARNING: AudioEngine - 'pause' missing deck_id. Skipping.")
                    return
                deck = self._get_or_create_deck(deck_id)
                deck.pause()
                
            elif command == "stop":
                if not deck_id:
                    print("WARNING: AudioEngine - 'stop' missing deck_id. Skipping.")
                    return
                deck = self._get_or_create_deck(deck_id)
                deck.stop()

            elif command == "wait":
                duration = parameters.get("duration_seconds")
                if duration is None or not isinstance(duration, (int, float)) or duration < 0:
                    print("WARNING: AudioEngine - 'wait' command missing valid 'duration_seconds'. Skipping.")
                    return
                print(f"INFO: AudioEngine - Waiting for {duration} seconds...")
                time.sleep(duration)
                print("INFO: AudioEngine - Wait finished.")
            
            else:
                print(f"WARNING: AudioEngine - Unknown command '{command}'. Skipping.")
        except Exception as e_action:
            print(f"ERROR: AudioEngine - Failed to execute action {action_dict}: {e_action}")
            import traceback
            traceback.print_exc()


    def shutdown_decks(self):
        """Shuts down all managed deck threads cleanly."""
        print("INFO: AudioEngine - Shutting down all decks...")
        for deck_id, deck_instance in self.decks.items():
            print(f"INFO: AudioEngine - Requesting shutdown for {deck_id}...")
            deck_instance.shutdown() # This now sends command and waits for join
        # Wait for all deck threads to actually finish joining from their shutdown()
        # This is important to ensure resources are released before program exits.
        # Deck.shutdown() already includes a join.
        print("INFO: AudioEngine - All decks have been requested to shut down.")
        self.decks.clear()
        print("INFO: AudioEngine - Deck shutdown process complete.")

    def any_deck_active(self):
        """Checks if any managed deck is currently active."""
        if not self.decks:
            return False
        for deck in self.decks.values():
            if deck.is_active():
                return True
        return False


if __name__ == '__main__':
    print("--- AudioEngine Standalone Test ---")
    
    # For standalone testing, ensure project root is in path to import config
    # This assumes engine.py is in dj-gemini/audio_engine/
    CURRENT_DIR_OF_ENGINE_PY = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT_FOR_ENGINE_TEST = os.path.dirname(CURRENT_DIR_OF_ENGINE_PY) 
    
    if PROJECT_ROOT_FOR_ENGINE_TEST not in sys.path:
        sys.path.insert(0, PROJECT_ROOT_FOR_ENGINE_TEST) # Add project root to import config
    
    # Re-attempt import of app_config if it failed at module level
    if app_config is None:
        try:
            import config as loaded_config
            app_config = loaded_config
            print("DEBUG: engine.py __main__ - Successfully loaded config.")
        except ImportError:
            print("CRITICAL ERROR: engine.py __main__ - config.py not found in project root. Cannot run test.")
            sys.exit(1)

    engine = AudioEngine() 

    # Create a dummy JSON file for testing in mix_configs directory
    # Ensure mix_configs directory exists using config
    app_config.ensure_dir_exists(app_config.MIX_CONFIGS_DIR)
    dummy_json_path = os.path.join(app_config.MIX_CONFIGS_DIR, "test_engine_mix.json")
    
    test_audio_filename = "starships.mp3" 
    # Path in JSON should be relative to AUDIO_TRACKS_DIR or project root for flexibility
    # For simplicity, let's assume paths in JSON are just the filename,
    # and engine's load_track prepends AUDIO_TRACKS_DIR
    json_audio_path = test_audio_filename # Just filename, engine._execute_action will prepend based on config

    test_json_content = {
        "script_name": "Engine Test Script",
        "actions": [
            {
                "command": "load_track", "deck_id": "deckA",
                "parameters": {"file_path": json_audio_path } 
            },
            {
                "command": "load_track", "deck_id": "deckB", # Load second track
                "parameters": {"file_path": json_audio_path } # Use same track for simplicity
            },
            {"command": "play", "deck_id": "deckA", "parameters": {"start_at_beat": 1}},
            {"command": "wait", "parameters": {"duration_seconds": 2}},
            {"command": "play", "deck_id": "deckB", "parameters": {"start_at_cue_name": "drop1"}}, # Assuming starships.mp3.cue exists
            {"command": "wait", "parameters": {"duration_seconds": 3}}, # Both play
            {"command": "pause", "deck_id": "deckA"},
            {"command": "wait", "parameters": {"duration_seconds": 2}}, # B plays, A paused
            {"command": "play", "deck_id": "deckA"}, # Resume A
            {"command": "wait", "parameters": {"duration_seconds": 3}}, # Both play
            {"command": "stop", "deck_id": "deckA"},
            {"command": "stop", "deck_id": "deckB"}
        ]
    }

    # Ensure the audio file for the test exists
    full_test_audio_path = os.path.join(app_config.AUDIO_TRACKS_DIR, test_audio_filename)
    if not os.path.exists(full_test_audio_path):
        print(f"WARNING: Test audio file '{full_test_audio_path}' not found for engine test.")
        # Create a dummy cue file for starships.mp3 so Deck doesn't fail if audio_analyzer looks for it
        dummy_cue_filepath = os.path.join(app_config.AUDIO_TRACKS_DIR, test_audio_filename + ".cue")
        if not os.path.exists(dummy_cue_filepath):
            try:
                with open(dummy_cue_filepath, 'w') as f:
                    json.dump({"test_cue": {"start_beat": 1}}, f)
                print(f"Created placeholder cue file: {dummy_cue_filepath} for test robustness.")
            except: pass # Ignore if cannot create

    try:
        with open(dummy_json_path, 'w') as f:
            json.dump(test_json_content, f, indent=4)
        print(f"Created dummy JSON script for testing: {dummy_json_path}")
    except Exception as e_json_write:
        print(f"Error creating dummy JSON: {e_json_write}")
        sys.exit(1)

    if engine.load_script_from_file(dummy_json_path): # Pass the path relative to project root
        engine.run_script()
        
        print("INFO: Engine Test - Script processing finished. Waiting for active decks to complete...")
        start_wait_time = time.time()
        max_wait_time = 20 # Max seconds to wait for decks to finish
        while (time.time() - start_wait_time) < max_wait_time:
            if not engine.any_deck_active():
                print("INFO: Engine Test - All decks idle.")
                break
            print(f"DEBUG: Engine Test - Decks still active, waiting... ({int(time.time() - start_wait_time)}s)")
            time.sleep(1)
        else: # Loop finished due to timeout
            print(f"WARNING: Engine Test - Max wait time of {max_wait_time}s reached. Forcing shutdown.")
    
    engine.shutdown_decks() 
    print("--- AudioEngine Standalone Test Finished ---")