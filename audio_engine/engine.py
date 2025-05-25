# dj-gemini/audio_engine/engine.py

import json
import time
import os
import sys # For __main__ path adjustments

# audio_analyzer and deck are imported relative to this engine.py file
from .audio_analyzer import AudioAnalyzer
from .deck import Deck

# app_config will be passed to __init__

class AudioEngine:
    def __init__(self, app_config_module): # Accept config module as an argument
        print("DEBUG: AudioEngine - Initializing...")
        self.app_config = app_config_module # Store the passed config module
        if self.app_config is None:
            # This should ideally not happen if main.py passes it.
            # If it does, it means AudioEngine was instantiated without proper config.
            print("CRITICAL ERROR: AudioEngine - app_config module was not provided during initialization.")
            raise ValueError("AudioEngine requires a valid configuration module.")

        # Ensure necessary directories from config exist (AudioAnalyzer might also do this)
        # Using the passed app_config
        self.app_config.ensure_dir_exists(self.app_config.BEATS_CACHE_DIR)
        # self.app_config.ensure_dir_exists(self.app_config.AUDIO_TRACKS_DIR)
        # self.app_config.ensure_dir_exists(self.app_config.MIX_CONFIGS_DIR)

        self.analyzer = AudioAnalyzer(
            cache_dir=self.app_config.BEATS_CACHE_DIR,
            beats_cache_file_extension=self.app_config.BEATS_CACHE_FILE_EXTENSION,
            beat_tracker_algo_name=self.app_config.DEFAULT_BEAT_TRACKER_ALGORITHM,
            bpm_estimator_algo_name=self.app_config.DEFAULT_BPM_ESTIMATOR_ALGORITHM
        )
        self.decks = {} 
        self.script_name = "Untitled Mix"
        self.script_actions = []
        self.is_running_script = False 
        print("DEBUG: AudioEngine - Initialized.")

    def _get_or_create_deck(self, deck_id):
        """Retrieves a deck by ID, creating it if it doesn't exist."""
        if deck_id not in self.decks:
            print(f"DEBUG: AudioEngine - Creating new Deck: {deck_id}")
            self.decks[deck_id] = Deck(deck_id, self.analyzer) # Pass the shared analyzer
        return self.decks[deck_id]

    def load_script_from_file(self, json_filepath):
        """Loads and parses a JSON mix script."""
        print(f"DEBUG: AudioEngine - Loading script from: {json_filepath}")
        
        # Use self.app_config for path construction
        path_to_load = json_filepath
        if self.app_config and not os.path.isabs(path_to_load):
            # First, try relative to MIX_CONFIGS_DIR
            abs_json_filepath_mix_config = os.path.join(self.app_config.MIX_CONFIGS_DIR, path_to_load)
            # Then, try relative to project root as a fallback
            abs_json_filepath_project_root = os.path.join(self.app_config.PROJECT_ROOT_DIR, path_to_load)
            
            if os.path.exists(abs_json_filepath_mix_config):
                path_to_load = abs_json_filepath_mix_config
            elif os.path.exists(abs_json_filepath_project_root):
                path_to_load = abs_json_filepath_project_root
            # If neither, and original was relative, it might be relative to CWD.
            # os.path.exists below will catch it if truly not found.

        if not os.path.exists(path_to_load):
            print(f"ERROR: AudioEngine - Script file not found at resolved path: {path_to_load} (original: {json_filepath})")
            self.script_actions = []
            return False
            
        try:
            with open(path_to_load, 'r') as f:
                script_data = json.load(f)
            
            self.script_name = script_data.get("script_name", "Untitled Mix")
            self.script_actions = script_data.get("actions", []) 
            if not isinstance(self.script_actions, list):
                print(f"ERROR: AudioEngine - 'actions' in script must be a list. Found: {type(self.script_actions)}")
                self.script_actions = []
                return False

            print(f"DEBUG: AudioEngine - Script '{self.script_name}' loaded from '{path_to_load}' with {len(self.script_actions)} actions.")
            return True
        except json.JSONDecodeError:
            print(f"ERROR: AudioEngine - Error decoding JSON from script: {path_to_load}")
        except Exception as e:
            print(f"ERROR: AudioEngine - Failed to load script {path_to_load}: {e}")
        
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
                
                if not self.is_running_script: 
                    print("INFO: AudioEngine - Script execution halted by an internal signal (e.g., future stop_engine command).")
                    break
            print("INFO: AudioEngine - All script actions have been dispatched.")
        except Exception as e:
            print(f"ERROR: AudioEngine - Error during script execution loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running_script = False 
            print("INFO: AudioEngine - Script action dispatch loop finished.")


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
                track_path = file_path_param # Assume it might be absolute initially
                
                # Use self.app_config (passed during __init__)
                if self.app_config and not os.path.isabs(file_path_param):
                    # Prefer path relative to AUDIO_TRACKS_DIR from config
                    constructed_path = os.path.join(self.app_config.AUDIO_TRACKS_DIR, file_path_param)
                    if os.path.exists(constructed_path):
                        track_path = constructed_path
                    else: 
                        # Fallback to try relative to project root if not in AUDIO_TRACKS_DIR
                        constructed_path_alt = os.path.join(self.app_config.PROJECT_ROOT_DIR, file_path_param)
                        if os.path.exists(constructed_path_alt):
                             track_path = constructed_path_alt
                        else:
                             print(f"WARNING: AudioEngine - Track file '{file_path_param}' not found in default locations ({self.app_config.AUDIO_TRACKS_DIR} or {self.app_config.PROJECT_ROOT_DIR}). Trying path as is.")
                
                print(f"DEBUG: AudioEngine - Attempting to load track from resolved path: {track_path}")
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
        if not self.decks:
            print("INFO: AudioEngine - No decks to shut down.")
            return
            
        for deck_id, deck_instance in list(self.decks.items()): # Iterate over items for safe removal if needed
            print(f"INFO: AudioEngine - Requesting shutdown for {deck_id}...")
            deck_instance.shutdown() 
        
        # Wait for all deck threads to actually finish joining.
        # Deck.shutdown() now includes a join, but we can add a final check or longer wait here if needed.
        print("INFO: AudioEngine - All decks have been requested to shut down.")
        self.decks.clear() # Clear decks dictionary after shutdown
        print("INFO: AudioEngine - Deck shutdown process complete.")

    def any_deck_active(self):
        """Checks if any managed deck is currently active."""
        if not self.decks:
            return False
        for deck in self.decks.values():
            if deck.is_active(): # Relies on Deck.is_active()
                return True
        return False


if __name__ == '__main__':
    print("--- AudioEngine Standalone Test ---")
    
    # For standalone testing, ensure project root is in path to import config
    CURRENT_DIR_OF_ENGINE_PY = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT_FOR_ENGINE_TEST = os.path.dirname(CURRENT_DIR_OF_ENGINE_PY) 
    
    if PROJECT_ROOT_FOR_ENGINE_TEST not in sys.path:
        sys.path.insert(0, PROJECT_ROOT_FOR_ENGINE_TEST)
    
    try:
        import config as main_test_app_config # Import config for the test
    except ImportError:
        print("CRITICAL ERROR: engine.py __main__ - config.py not found in project root. Cannot run test.")
        sys.exit(1)

    # Ensure directories for the test
    main_test_app_config.ensure_dir_exists(main_test_app_config.MIX_CONFIGS_DIR)
    main_test_app_config.ensure_dir_exists(main_test_app_config.AUDIO_TRACKS_DIR)
    main_test_app_config.ensure_dir_exists(main_test_app_config.BEATS_CACHE_DIR) # For Analyzer


    # Instantiate AudioEngine, passing the imported config module
    engine = AudioEngine(app_config_module=main_test_app_config) 

    dummy_json_path = os.path.join(main_test_app_config.MIX_CONFIGS_DIR, "test_engine_mix.json")
    test_audio_filename = "starships.mp3" 
    # Path in JSON should be just the filename if it's in AUDIO_TRACKS_DIR
    # Or a path relative to project root, or absolute.
    # Engine's _execute_action for load_track will try to resolve it.
    json_audio_path = test_audio_filename # Keep it simple, just filename

    # Ensure the audio file for the test exists for a meaningful test
    full_test_audio_path = os.path.join(main_test_app_config.AUDIO_TRACKS_DIR, test_audio_filename)
    if not os.path.exists(full_test_audio_path):
        print(f"WARNING: Test audio file '{full_test_audio_path}' not found for engine test. Creating placeholder.")
        # Create a dummy empty file for path testing if it doesn't exist
        # Note: Essentia will fail to load an empty file, but the path logic will be tested.
        try:
            with open(full_test_audio_path, 'a') as f: pass
            print(f"Created placeholder audio file: {full_test_audio_path}")
        except Exception as e_placeholder:
            print(f"Could not create placeholder audio file: {e_placeholder}")

    # Create a dummy cue file if it doesn't exist for starships.mp3, so test doesn't fail on cue load
    # This should ideally be handled by having actual test assets.
    dummy_cue_filepath = os.path.join(main_test_app_config.AUDIO_TRACKS_DIR, test_audio_filename + ".cue")
    if not os.path.exists(dummy_cue_filepath):
        try:
            with open(dummy_cue_filepath, 'w') as f:
                json.dump({"drop1": {"start_beat": 65}}, f) # Minimal cue for the test JSON
            print(f"Created placeholder cue file: {dummy_cue_filepath}")
        except Exception as e_cue_create:
            print(f"Could not create placeholder cue file: {e_cue_create}")


    test_json_content = {
        "script_name": "Engine Test Script",
        "actions": [
            {
                "command": "load_track", "deck_id": "deckA",
                "parameters": {"file_path": json_audio_path } 
            },
            {
                "command": "load_track", "deck_id": "deckB",
                "parameters": {"file_path": json_audio_path } 
            },
            {"command": "play", "deck_id": "deckA", "parameters": {"start_at_beat": 1}},
            {"command": "wait", "parameters": {"duration_seconds": 2}}, # A plays
            {"command": "play", "deck_id": "deckB", "parameters": {"start_at_cue_name": "drop1"}}, # B starts from cue
            {"command": "wait", "parameters": {"duration_seconds": 3}}, # A & B play
            {"command": "pause", "deck_id": "deckA"},
            {"command": "wait", "parameters": {"duration_seconds": 2}}, # B plays, A paused
            {"command": "play", "deck_id": "deckA"}, # Resume A
            {"command": "wait", "parameters": {"duration_seconds": 3}}, # A & B play
            {"command": "stop", "deck_id": "deckA"},
            {"command": "stop", "deck_id": "deckB"}
        ]
    }
    
    try:
        with open(dummy_json_path, 'w') as f:
            json.dump(test_json_content, f, indent=4)
        print(f"Created dummy JSON script for testing: {dummy_json_path}")
    except Exception as e_json_write:
        print(f"Error creating dummy JSON: {e_json_write}")
        sys.exit(1)

    if engine.load_script_from_file(dummy_json_path): # Pass the full path
        engine.run_script()
        
        print("INFO: Engine Test - Script processing finished. Waiting for active decks to complete...")
        start_wait_time = time.time()
        max_wait_time = 20 
        while (time.time() - start_wait_time) < max_wait_time:
            if not engine.any_deck_active():
                print("INFO: Engine Test - All decks idle.")
                break
            # print(f"DEBUG: Engine Test - Decks still active, waiting... ({int(time.time() - start_wait_time)}s)")
            time.sleep(0.5) # Check more frequently
        else: 
            print(f"WARNING: Engine Test - Max wait time of {max_wait_time}s reached. Forcing shutdown.")
    
    engine.shutdown_decks() 
    print("--- AudioEngine Standalone Test Finished ---")