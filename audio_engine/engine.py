# dj-gemini/audio_engine/engine.py

import json
import time
import os
import sys 
import threading

try:
    import config as app_config
except ImportError:
    app_config = None 
    print("WARNING: engine.py - Initial 'import config' failed. Ensure config.py is in project root or PYTHONPATH.")

from .audio_analyzer import AudioAnalyzer
from .deck import Deck

ENGINE_TICK_INTERVAL = 0.05 

class AudioEngine:
    def __init__(self, app_config_module): 
        print("DEBUG: AudioEngine - Initializing...")
        self.app_config = app_config_module 
        if self.app_config is None:
            raise ValueError("CRITICAL: AudioEngine requires a valid config module.")

        self.app_config.ensure_dir_exists(self.app_config.BEATS_CACHE_DIR)

        self.analyzer = AudioAnalyzer(
            cache_dir=self.app_config.BEATS_CACHE_DIR,
            beats_cache_file_extension=self.app_config.BEATS_CACHE_FILE_EXTENSION,
            beat_tracker_algo_name=self.app_config.DEFAULT_BEAT_TRACKER_ALGORITHM,
            bpm_estimator_algo_name=self.app_config.DEFAULT_BPM_ESTIMATOR_ALGORITHM
        )
        self.decks = {}
        self.script_name = "Untitled Mix"
        
        self._all_actions_from_script = [] 
        self._pending_on_beat_actions = [] 
        
        self._engine_thread = None
        self._engine_stop_event = threading.Event()
        self.is_processing_script_actions = False 
        
        self._script_start_time = 0 
        
        print("DEBUG: AudioEngine - Initialized.")

    def _get_or_create_deck(self, deck_id):
        if deck_id not in self.decks:
            print(f"DEBUG: AudioEngine - Creating new Deck: {deck_id}")
            self.decks[deck_id] = Deck(deck_id, self.analyzer) 
        return self.decks[deck_id]

    def _validate_action(self, action, action_index):
        command = action.get("command")
        action_id_for_log = action.get('id', f"action_idx_{action_index+1}")
        # print(f"DEBUG: Validating action {action_index+1} (ID: {action_id_for_log}): CMD='{command}'") 

        if not command:
            print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): Missing 'command'.")
            return False

        trigger = action.get("trigger")
        
        if not trigger: 
            print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): Trigger object missing entirely (should have been defaulted).")
            return False 
            
        if not isinstance(trigger, dict) or "type" not in trigger:
            print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): Malformed 'trigger' or missing 'type'. Trigger: {trigger}")
            return False
        
        trigger_type = trigger.get("type")
        if trigger_type == "on_deck_beat":
            source_deck_id_val = trigger.get("source_deck_id")
            beat_number_val = trigger.get("beat_number")
            if not source_deck_id_val or not isinstance(beat_number_val, (int, float)):
                print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'on_deck_beat' trigger missing 'source_deck_id' ('{source_deck_id_val}') or valid 'beat_number' ('{beat_number_val}').")
                return False
        elif trigger_type == "on_loop_complete":
            source_deck_id_val = trigger.get("source_deck_id")
            if not source_deck_id_val:
                print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'on_loop_complete' trigger missing 'source_deck_id' ('{source_deck_id_val}').")
                return False
        elif trigger_type == "script_start":
            pass 
        else:
            print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): Unsupported trigger type: '{trigger_type}'. Supported: 'script_start', 'on_deck_beat', 'on_loop_complete'.")
            return False
        
        deck_specific_commands = ["play", "pause", "stop", "activate_loop", "deactivate_loop", "load_track", "stop_at_beat", "set_tempo", "set_volume", "fade_volume", "ramp_tempo"]
        engine_level_commands = ["crossfade", "beatmatch"] 
        
        if command in deck_specific_commands and not action.get("deck_id"):
            print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): Command '{command}' is missing 'deck_id'.")
            return False
        
        if command == "activate_loop":
            params = action.get("parameters", {})
            if params.get("start_at_beat") is None or params.get("length_beats") is None:
                print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'activate_loop' missing 'start_at_beat' or 'length_beats' in parameters. Params: {params}")
                return False
            try: 
                float(params["start_at_beat"])
                float(params["length_beats"])
                if params.get("repetitions") is not None: # Optional, but if present, check if int
                    if not (isinstance(params["repetitions"], str) and params["repetitions"].lower() == "infinite"):
                        int(params["repetitions"])
            except (ValueError, TypeError):
                print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'activate_loop' parameters not valid numbers/type. Params: {params}")
                return False
        elif command == "set_tempo":
            params = action.get("parameters", {})
            if params.get("target_bpm") is None:
                print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'set_tempo' missing 'target_bpm' in parameters. Params: {params}")
                return False
            try:
                target_bpm = float(params["target_bpm"])
                if target_bpm <= 0:
                    print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'target_bpm' must be positive. Value: {target_bpm}")
                    return False
            except (ValueError, TypeError):
                print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'target_bpm' not a valid number. Params: {params}")
                return False
        elif command == "set_volume":
            params = action.get("parameters", {})
            if params.get("volume") is None:
                print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'set_volume' missing 'volume' in parameters. Params: {params}")
                return False
            try:
                volume = float(params["volume"])
                if volume < 0.0 or volume > 1.0:
                    print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'volume' must be between 0.0 and 1.0. Value: {volume}")
                    return False
            except (ValueError, TypeError):
                print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'volume' not a valid number. Params: {params}")
                return False
        elif command == "fade_volume":
            params = action.get("parameters", {})
            if params.get("target_volume") is None or params.get("duration_seconds") is None:
                print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'fade_volume' missing 'target_volume' or 'duration_seconds' in parameters. Params: {params}")
                return False
            try:
                target_volume = float(params["target_volume"])
                duration_seconds = float(params["duration_seconds"])
                if target_volume < 0.0 or target_volume > 1.0:
                    print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'target_volume' must be between 0.0 and 1.0. Value: {target_volume}")
                    return False
                if duration_seconds < 0.0:
                    print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'duration_seconds' must be positive. Value: {duration_seconds}")
                    return False
            except (ValueError, TypeError):
                print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'fade_volume' parameters not valid numbers. Params: {params}")
                return False
        elif command == "crossfade":
            params = action.get("parameters", {})
            if params.get("from_deck") is None or params.get("to_deck") is None or params.get("duration_seconds") is None:
                print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'crossfade' missing required parameters. Params: {params}")
                return False
            try:
                duration_seconds = float(params["duration_seconds"])
                if duration_seconds < 0.0:
                    print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'duration_seconds' must be positive. Value: {duration_seconds}")
                    return False
            except (ValueError, TypeError):
                print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'crossfade' duration not a valid number. Params: {params}")
                return False
        elif command == "beatmatch":
            params = action.get("parameters", {})
            if params.get("reference_deck") is None or params.get("follow_deck") is None:
                print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'beatmatch' missing 'reference_deck' or 'follow_deck' in parameters. Params: {params}")
                return False
            sync_method = params.get("sync_method", "auto")
            if sync_method not in ["auto", "manual", "grid"]:
                print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'sync_method' must be 'auto', 'manual', or 'grid'. Value: {sync_method}")
                return False
            if sync_method == "manual" and params.get("target_beat") is None:
                print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'manual' sync_method requires 'target_beat' parameter. Params: {params}")
                return False
            try:
                if sync_method == "manual" and params.get("target_beat") is not None:
                    target_beat = float(params["target_beat"])
                    if target_beat <= 0:
                        print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'target_beat' must be positive. Value: {target_beat}")
                        return False
            except (ValueError, TypeError):
                print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'target_beat' not a valid number. Params: {params}")
                return False
        elif command == "ramp_tempo":
            params = action.get("parameters", {})
            if (params.get("start_beat") is None or params.get("end_beat") is None or 
                params.get("start_bpm") is None or params.get("end_bpm") is None):
                print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'ramp_tempo' missing required parameters. Params: {params}")
                return False
            try:
                start_beat = float(params["start_beat"])
                end_beat = float(params["end_beat"])
                start_bpm = float(params["start_bpm"])
                end_bpm = float(params["end_bpm"])
                curve = params.get("curve", "linear")
                
                if start_beat >= end_beat:
                    print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'start_beat' must be less than 'end_beat'. Values: {start_beat}, {end_beat}")
                    return False
                if start_bpm <= 0 or end_bpm <= 0:
                    print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): BPM values must be positive. Values: {start_bpm}, {end_bpm}")
                    return False
                if curve not in ["linear", "exponential", "smooth", "step"]:
                    print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'curve' must be 'linear', 'exponential', 'smooth', or 'step'. Value: {curve}")
                    return False
            except (ValueError, TypeError):
                print(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'ramp_tempo' parameters not valid numbers. Params: {params}")
                return False
        # print(f"DEBUG: Validation OK for action {action_index+1} (ID: {action_id_for_log})")
        return True


    def load_script_from_file(self, json_filepath):
        print(f"DEBUG: AudioEngine - Loading script from: {json_filepath}")
        path_to_load = json_filepath
        if self.app_config and not os.path.isabs(path_to_load):
            abs_json_filepath_mix_config = os.path.join(self.app_config.MIX_CONFIGS_DIR, path_to_load)
            abs_json_filepath_project_root = os.path.join(self.app_config.PROJECT_ROOT_DIR, path_to_load)
            if os.path.exists(abs_json_filepath_mix_config): path_to_load = abs_json_filepath_mix_config
            elif os.path.exists(abs_json_filepath_project_root): path_to_load = abs_json_filepath_project_root

        if not os.path.exists(path_to_load):
            print(f"ERROR: AudioEngine - Script file not found: {path_to_load}")
            return False
        try:
            with open(path_to_load, 'r') as f: script_data = json.load(f)
            self.script_name = script_data.get("script_name", "Untitled Mix")
            raw_actions = script_data.get("actions", [])
            if not isinstance(raw_actions, list):
                print("ERROR: AudioEngine - 'actions' in script must be a list."); return False
            
            self._all_actions_from_script = [] 
            for i, action in enumerate(raw_actions):
                if "trigger" not in action:
                    print(f"DEBUG: AudioEngine - Action {i+1} ('{action.get('command')}') missing trigger, defaulting to 'script_start'.")
                    action["trigger"] = {"type": "script_start"}
                
                if not self._validate_action(action, i): 
                    print(f"ERROR: AudioEngine - Invalid action at index {i} (Command: {action.get('command')}). Aborting script load.")
                    self._all_actions_from_script = [] 
                    return False 
                
                self._all_actions_from_script.append(action)
            
            print(f"DEBUG: AudioEngine - Script '{self.script_name}' loaded and validated with {len(self._all_actions_from_script)} actions.")
            return True
        except Exception as e:
            print(f"ERROR: AudioEngine - Failed to load/parse script {path_to_load}: {e}"); return False


    def start_script_processing(self): 
        if self.is_processing_script_actions:
            print("WARNING: AudioEngine - Script processing already running."); return
        if not self._all_actions_from_script : 
            print("INFO: AudioEngine - No actions loaded to process."); return

        print(f"INFO: AudioEngine - Starting script processing: '{self.script_name}'")
        self._engine_stop_event.clear()
        self.is_processing_script_actions = True 
        self._script_start_time = time.time() 
        
        self._pending_on_beat_actions = [] 
        initial_actions_to_execute = []    

        for action in self._all_actions_from_script:
            if action.get("trigger", {}).get("type") == "script_start":
                initial_actions_to_execute.append(action)
            else: 
                self._pending_on_beat_actions.append(action)
        
        print(f"DEBUG: AudioEngine - Executing {len(initial_actions_to_execute)} initial 'script_start' actions...")
        for action in initial_actions_to_execute:
            action_command = action.get('command', 'N/A')
            action_deck_id = action.get('deck_id', 'Engine-Level')
            action_id_for_log = action.get('id', 'N/A_script_start')
            print(f"\nINFO: AudioEngine (Initial) - Executing: CMD='{action_command}', Deck='{action_deck_id}', ActionID='{action_id_for_log}'")
            self._execute_action(action) 
            if self._engine_stop_event.is_set(): 
                print("ERROR: AudioEngine - Stop event set during initial action execution.")
                self.is_processing_script_actions = False
                return
        print(f"DEBUG: AudioEngine - Initial 'script_start' actions dispatched. Pending on_beat: {len(self._pending_on_beat_actions)}")

        if not self._pending_on_beat_actions and not self.any_deck_active():
            print("INFO: AudioEngine - No future-triggered actions and no decks active after initial. Script complete.")
            self.is_processing_script_actions = False
            self.shutdown_decks() 
            return

        self._engine_thread = threading.Thread(target=self._engine_loop, daemon=True)
        self._engine_thread.start()

    def stop_script_processing(self): 
        # (Same as before)
        print("INFO: AudioEngine - Stop script processing requested externally.")
        if not self.is_processing_script_actions and (self._engine_thread is None or not self._engine_thread.is_alive()):
            print("INFO: AudioEngine - Engine loop was not running or already finished.")
            self.is_processing_script_actions = False 
            self.shutdown_decks()
            return

        self._engine_stop_event.set() 
        if self._engine_thread and self._engine_thread.is_alive():
            print("INFO: AudioEngine - Waiting for engine thread to complete...")
            self._engine_thread.join(timeout=1.0) 
            if self._engine_thread.is_alive(): 
                print("WARNING: AudioEngine - Engine thread did not stop cleanly via event.")
        
        self.is_processing_script_actions = False 
        print("INFO: AudioEngine - Engine processing loop signaled to stop/completed.")
        self.shutdown_decks() 


    def _engine_loop(self):
        print(f"DEBUG: AudioEngine - Engine loop started (monitoring {len(self._pending_on_beat_actions)} initial on_deck_beat actions).")

        while not self._engine_stop_event.is_set():
            if not self._pending_on_beat_actions:
                self.is_processing_script_actions = False 
                if not self.any_deck_active():
                    print("INFO: AudioEngine - All actions dispatched and no decks active. Engine loop self-terminating.")
                    break 
                time.sleep(ENGINE_TICK_INTERVAL)
                continue

            # Check for on_loop_complete triggers
            for deck_id, deck in self.decks.items():
                if hasattr(deck, '_loop_just_completed') and deck._loop_just_completed:
                    completed_action_id = getattr(deck, '_completed_loop_action_id', None)
                    # Loop just completed all repetitions
                    for action in self._pending_on_beat_actions[:]:
                        trigger = action.get("trigger", {})
                        if (trigger.get("type") == "on_loop_complete" and 
                            trigger.get("source_deck_id") == deck_id):
                            
                            # Check if we need a specific loop action ID
                            required_action_id = trigger.get("loop_action_id")
                            if required_action_id is None or required_action_id == completed_action_id:
                                print(f"DEBUG: AudioEngine - Trigger MET: on_loop_complete for action '{action.get('id')}' (loop: {completed_action_id})")
                                self._pending_on_beat_actions.remove(action)
                                self._execute_action(action)
                    
                    # Reset the flag
                    deck._loop_just_completed = False
                    deck._completed_loop_action_id = None

            actions_executed_this_tick = []
            next_round_pending_actions = [] 

            for idx, action in enumerate(self._pending_on_beat_actions): 
                trigger = action.get("trigger") 
                trigger_type = trigger.get("type")
                action_id_for_log = action.get('id', f'pending_action_idx_{idx}')
                
                triggered_this_tick = False

                if trigger_type == "on_deck_beat":
                    source_deck_id = trigger.get("source_deck_id")
                    target_beat_str = trigger.get("beat_number") 
                    deck = self.decks.get(source_deck_id)
                    
                    if deck and target_beat_str is not None:
                        if not hasattr(deck, 'get_current_beat_count'):
                            print(f"CRITICAL ERROR: Deck {source_deck_id} missing get_current_beat_count! Action ID: {action_id_for_log}")
                            next_round_pending_actions.append(action) 
                            continue
                        try:
                            target_beat = float(target_beat_str)
                            current_beat = deck.get_current_beat_count()
                            print(f"DEBUG: AudioEngine - LOOP CHECK: Action='{action_id_for_log}', Deck='{source_deck_id}', CurrentBeat={current_beat}, TargetBeat={target_beat}")
                            if current_beat >= target_beat: 
                                print(f"DEBUG: AudioEngine - Trigger MET: on_deck_beat for action '{action_id_for_log}'")
                                triggered_this_tick = True
                        except ValueError:
                             print(f"ERROR: AudioEngine - Invalid 'beat_number' format ('{target_beat_str}') for action '{action_id_for_log}'.")
                        except Exception as e_beat_check:
                             print(f"ERROR: AudioEngine - Error checking beat count for deck {source_deck_id} (action '{action_id_for_log}'): {e_beat_check}")
                    else:
                        print(f"WARNING: AudioEngine - Invalid trigger data for on_deck_beat action '{action_id_for_log}': Deck={source_deck_id}, BeatNum={target_beat_str}")
                elif trigger_type == "on_loop_complete":
                    source_deck_id = trigger.get("source_deck_id")
                    loop_action_id = trigger.get("loop_action_id")  # Optional: specific loop action ID
                    deck = self.decks.get(source_deck_id)
                    
                    if deck and hasattr(deck, '_loop_just_completed') and deck._loop_just_completed:
                        completed_action_id = getattr(deck, '_completed_loop_action_id', None)
                        
                        # Check if we need a specific loop action ID
                        if loop_action_id is None or loop_action_id == completed_action_id:
                            print(f"DEBUG: AudioEngine - Trigger MET: on_loop_complete for action '{action_id_for_log}' (loop: {completed_action_id})")
                            triggered_this_tick = True
                            # Reset the flag
                            deck._loop_just_completed = False
                            deck._completed_loop_action_id = None
                else: 
                    print(f"WARNING: AudioEngine - Action '{action_id_for_log}' in _pending_on_beat_actions has unexpected trigger type: '{trigger_type}'. Removing.")
                    triggered_this_tick = True 
                
                if triggered_this_tick:
                    actions_executed_this_tick.append(action) 
                else:
                    next_round_pending_actions.append(action) 
            
            self._pending_on_beat_actions = next_round_pending_actions 
            
            if actions_executed_this_tick:
                for exec_action in actions_executed_this_tick: 
                    action_command = exec_action.get('command', 'N/A')
                    action_deck_id = exec_action.get('deck_id', 'Engine-Level')
                    print(f"\nINFO: AudioEngine Loop (Triggered) - Executing: CMD='{action_command}', Deck='{action_deck_id}' for action ID '{exec_action.get('id', 'N/A')}'")
                    self._execute_action(exec_action) 
                    if self._engine_stop_event.is_set(): break 
                if self._engine_stop_event.is_set(): break 
            
            time.sleep(ENGINE_TICK_INTERVAL) 
        
        self.is_processing_script_actions = False 
        print(f"DEBUG: AudioEngine - Engine loop finished (stop_event: {self._engine_stop_event.is_set()}). Pending on_deck_beat actions: {len(self._pending_on_beat_actions)}")


    def _execute_action(self, action_dict):
        command = action_dict.get("command")
        deck_id = action_dict.get("deck_id")
        parameters = action_dict.get("parameters", {})

        if not command: print("WARNING: AudioEngine - Action missing 'command'. Skipping."); return
        print(f"DEBUG: AudioEngine Executing - cmd='{command}', deck='{deck_id}', params='{parameters}'")

        try:
            if command == "load_track":
                if not deck_id or "file_path" not in parameters: print(f"WARNING: AudioEngine - 'load_track' missing deck_id or file_path. Skipping."); return
                deck = self._get_or_create_deck(deck_id)
                file_path_param = parameters["file_path"]
                track_path = file_path_param 
                if self.app_config and not os.path.isabs(file_path_param):
                    constructed_path = os.path.join(self.app_config.AUDIO_TRACKS_DIR, file_path_param)
                    if os.path.exists(constructed_path): track_path = constructed_path
                    else: 
                        constructed_path_alt = os.path.join(self.app_config.PROJECT_ROOT_DIR, file_path_param)
                        if os.path.exists(constructed_path_alt): track_path = constructed_path_alt
                        else: print(f"WARNING: AudioEngine - Track file '{file_path_param}' not found. Trying as is.")
                print(f"DEBUG: AudioEngine - Attempting to load track from resolved path: {track_path}")
                deck.load_track(track_path)

            elif command == "play":
                if not deck_id: print("WARNING: AudioEngine - 'play' missing deck_id. Skipping."); return
                deck = self._get_or_create_deck(deck_id)
                deck.play(start_at_beat=parameters.get("start_at_beat"), 
                          start_at_cue_name=parameters.get("start_at_cue_name"))

            elif command == "pause":
                if not deck_id: print("WARNING: AudioEngine - 'pause' missing deck_id. Skipping."); return
                deck = self._get_or_create_deck(deck_id)
                deck.pause()
                
            elif command == "stop":
                if not deck_id: print("WARNING: AudioEngine - 'stop' missing deck_id. Skipping."); return
                deck = self._get_or_create_deck(deck_id)
                deck.stop()
            
            elif command == "stop_at_beat":
                if not deck_id: print("WARNING: AudioEngine - 'stop_at_beat' missing deck_id. Skipping."); return
                if "beat_number" not in parameters: print("WARNING: AudioEngine - 'stop_at_beat' missing 'beat_number' in parameters. Skipping."); return
                
                try:
                    beat_number = float(parameters["beat_number"])
                    if beat_number <= 0:
                        print("WARNING: AudioEngine - 'beat_number' for 'stop_at_beat' must be positive. Skipping.")
                        return
                    
                    deck = self._get_or_create_deck(deck_id)
                    deck.stop_at_beat(beat_number)
                except ValueError:
                    print(f"WARNING: AudioEngine - Invalid 'beat_number' value for 'stop_at_beat': {parameters['beat_number']}. Skipping.")

            elif command == "activate_loop":
                if not deck_id: print("WARNING: AudioEngine - 'activate_loop' missing deck_id. Skipping."); return
                
                # Parameters from JSON (using .get for safety)
                start_beat_from_json = parameters.get("start_at_beat")
                length_beats_from_json = parameters.get("length_beats")
                repetitions_param_from_json = parameters.get("repetitions") 

                if start_beat_from_json is None or length_beats_from_json is None: 
                    print("WARNING: AudioEngine - 'activate_loop' requires 'start_at_beat' and 'length_beats' in parameters. Skipping.")
                    return
                try:
                    start_beat_val = float(start_beat_from_json)
                    length_beats_val = float(length_beats_from_json)
                    
                    repetitions_val = None 
                    if repetitions_param_from_json is not None:
                        if isinstance(repetitions_param_from_json, str) and repetitions_param_from_json.lower() == "infinite": 
                            repetitions_val = None 
                        else:
                            try: 
                                repetitions_val = int(repetitions_param_from_json)
                                if repetitions_val <= 0: 
                                    print("WARNING: AudioEngine - 'repetitions' for 'activate_loop' must be positive int or 'infinite'. Defaulting to infinite.")
                                    repetitions_val = None
                            except ValueError: 
                                print("WARNING: AudioEngine - Invalid 'repetitions' value for 'activate_loop'. Must be int or 'infinite'. Defaulting to infinite.")
                                repetitions_val = None
                    
                    if length_beats_val <= 0:
                        print("WARNING: AudioEngine - 'length_beats' for 'activate_loop' must be positive. Skipping.")
                        return
                        
                    deck = self._get_or_create_deck(deck_id)
                    # Pass the action ID to the deck
                    deck.activate_loop(start_beat=start_beat_val, 
                                       length_beats=length_beats_val, 
                                       repetitions=repetitions_val,
                                       action_id=action_dict.get('id'))
                except ValueError: 
                    print(f"WARNING: AudioEngine - Invalid numeric values for loop parameters: start_beat='{start_beat_from_json}', length_beats='{length_beats_from_json}'. Skipping.")

            elif command == "deactivate_loop":
                if not deck_id: print("WARNING: AudioEngine - 'deactivate_loop' missing deck_id. Skipping."); return
                deck = self._get_or_create_deck(deck_id)
                deck.deactivate_loop()
            
            elif command == "set_tempo":
                deck_id = action_dict.get("deck_id")
                deck = self._get_or_create_deck(deck_id)
                target_bpm = float(action_dict.get("parameters", {}).get("target_bpm"))
                deck.set_tempo(target_bpm)
            
            elif command == "set_volume":
                deck_id = action_dict.get("deck_id")
                deck = self._get_or_create_deck(deck_id)
                volume = float(action_dict.get("parameters", {}).get("volume"))
                deck.set_volume(volume)
            
            elif command == "fade_volume":
                deck_id = action_dict.get("deck_id")
                deck = self._get_or_create_deck(deck_id)
                params = action_dict.get("parameters", {})
                target_volume = float(params.get("target_volume"))
                duration_seconds = float(params.get("duration_seconds"))
                deck.fade_volume(target_volume, duration_seconds)
            
            elif command == "crossfade":
                params = action_dict.get("parameters", {})
                from_deck_id = params.get("from_deck")
                to_deck_id = params.get("to_deck")
                duration_seconds = float(params.get("duration_seconds"))
                
                from_deck = self._get_or_create_deck(from_deck_id)
                to_deck = self._get_or_create_deck(to_deck_id)
                
                # Start crossfade
                from_deck.fade_volume(0.0, duration_seconds)
                to_deck.fade_volume(1.0, duration_seconds)
                
                print(f"DEBUG: AudioEngine - Crossfading from {from_deck_id} to {to_deck_id} over {duration_seconds}s")
            
            elif command == "beatmatch":
                params = action_dict.get("parameters", {})
                reference_deck_id = params.get("reference_deck")
                follow_deck_id = params.get("follow_deck")
                sync_method = params.get("sync_method", "auto")
                target_beat = params.get("target_beat", None)
                
                reference_deck = self._get_or_create_deck(reference_deck_id)
                follow_deck = self._get_or_create_deck(follow_deck_id)
                
                reference_bpm = reference_deck.bpm
                follow_bpm = follow_deck.bpm
                
                print(f"DEBUG: AudioEngine - Beatmatching {follow_deck_id} to reference {reference_deck_id}")
                print(f"DEBUG: AudioEngine - Reference: {reference_bpm} BPM, Follow: {follow_bpm} BPM")
                
                if reference_bpm <= 0 or follow_bpm <= 0:
                    print(f"ERROR: AudioEngine - Invalid BPM for beatmatching")
                    return
                
                # Step 1: Match tempo
                tempo_ratio = reference_bpm / follow_bpm
                follow_deck.set_tempo(reference_bpm)
                print(f"DEBUG: AudioEngine - Tempo matched: ratio {tempo_ratio:.3f}")
                
                # Step 2: Phase alignment
                if sync_method == "auto":
                    reference_beat = reference_deck.get_current_beat_count()
                    follow_beat = follow_deck.get_current_beat_count()
                    beat_difference = reference_beat - follow_beat
                    
                    if abs(beat_difference) > 0.5:
                        frames_per_beat = (60.0 / reference_bpm) * follow_deck.sample_rate
                        offset_frames = int(beat_difference * frames_per_beat)
                        new_frame = follow_deck.get_current_display_frame() + offset_frames
                        
                        if 0 <= new_frame < follow_deck.total_frames:
                            follow_deck.seek(new_frame)
                            print(f"DEBUG: AudioEngine - Auto-aligned: offset {beat_difference:.1f} beats")
                
                elif sync_method == "manual" and target_beat is not None:
                    target_frame = follow_deck.get_frame_from_beat(int(target_beat))
                    if target_frame is not None:
                        follow_deck.seek(target_frame)
                        print(f"DEBUG: AudioEngine - Manual alignment to beat {target_beat}")
                
                elif sync_method == "grid":
                    current_beat = follow_deck.get_current_beat_count()
                    grid_beat = ((current_beat - 1) // 4) * 4 + 1
                    grid_frame = follow_deck.get_frame_from_beat(grid_beat)
                    if grid_frame is not None:
                        follow_deck.seek(grid_frame)
                        print(f"DEBUG: AudioEngine - Grid alignment to beat {grid_beat}")
                
                print(f"DEBUG: AudioEngine - Beatmatch complete: {follow_deck_id} synchronized")
            
            elif command == "ramp_tempo":
                deck_id = action_dict.get("deck_id")
                deck = self._get_or_create_deck(deck_id)
                params = action_dict.get("parameters", {})
                start_beat = float(params.get("start_beat"))
                end_beat = float(params.get("end_beat"))
                start_bpm = float(params.get("start_bpm"))
                end_bpm = float(params.get("end_bpm"))
                curve = params.get("curve", "linear")
                deck.ramp_tempo(start_beat, end_beat, start_bpm, end_bpm, curve)
            
            else: print(f"WARNING: AudioEngine - Unknown command '{command}'. Skipping.")
        except Exception as e_action:
            print(f"ERROR: AudioEngine - Failed to execute action {action_dict}: {e_action}")
            import traceback
            traceback.print_exc()

    def shutdown_decks(self):
        print("INFO: AudioEngine - Shutting down all decks...")
        if not self.decks: print("INFO: AudioEngine - No decks to shut down."); return
        for deck_id, deck_instance in list(self.decks.items()): 
            print(f"INFO: AudioEngine - Requesting shutdown for {deck_id}...")
            deck_instance.shutdown() 
        print("INFO: AudioEngine - All decks have been requested to shut down.")
        self.decks.clear() 
        print("INFO: AudioEngine - Deck shutdown process complete.")

    def any_deck_active(self):
        if not self.decks: return False
        for deck in self.decks.values():
            if deck.is_active(): return True
        return False


if __name__ == '__main__':
    print("--- AudioEngine Standalone Test (v5.1 - Corrected activate_loop call) ---")
    
    CURRENT_DIR_OF_ENGINE_PY = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT_FOR_ENGINE_TEST = os.path.dirname(CURRENT_DIR_OF_ENGINE_PY) 
    
    if PROJECT_ROOT_FOR_ENGINE_TEST not in sys.path:
        sys.path.insert(0, PROJECT_ROOT_FOR_ENGINE_TEST)
    
    try:
        import config as main_test_app_config 
    except ImportError:
        print("CRITICAL ERROR: engine.py __main__ - config.py not found in project root. Cannot run test.")
        sys.exit(1)

    main_test_app_config.ensure_dir_exists(main_test_app_config.MIX_CONFIGS_DIR)
    main_test_app_config.ensure_dir_exists(main_test_app_config.AUDIO_TRACKS_DIR)
    main_test_app_config.ensure_dir_exists(main_test_app_config.BEATS_CACHE_DIR)

    engine = AudioEngine(app_config_module=main_test_app_config) 

    dummy_json_filename = "test_engine_v5_1_mix.json" # New filename for this test
    dummy_json_path = os.path.join(main_test_app_config.MIX_CONFIGS_DIR, dummy_json_filename) 
    
    test_audio_filename = "starships.mp3" 
    json_audio_path = test_audio_filename 

    full_test_audio_path = os.path.join(main_test_app_config.AUDIO_TRACKS_DIR, test_audio_filename)
    if not os.path.exists(full_test_audio_path):
        print(f"WARNING: Test audio file '{full_test_audio_path}' not found for engine test.")
    
    dummy_cue_filepath = os.path.join(main_test_app_config.AUDIO_TRACKS_DIR, test_audio_filename + ".cue")
    test_cues = {"drop1": {"start_beat": 65}, "intro_start": {"start_beat":1}, 
                 "beat_5":{"start_beat":5}, "beat_10": {"start_beat": 10}, # For simplified test
                 "beat_17": {"start_beat": 17}, "beat_20": {"start_beat": 20},
                 "beat_28": {"start_beat":28}, "beat_33":{"start_beat":33},
                 "beat_66":{"start_beat":66}, "beat_97":{"start_beat":97}}
    try:
        with open(dummy_cue_filepath, 'w') as f: json.dump(test_cues, f, indent=4)
        print(f"Created/Updated placeholder cue file: {dummy_cue_filepath}")
    except Exception as e_cue_create: print(f"Could not create placeholder cue file: {e_cue_create}")

    # Using the simplified test JSON that worked for deck.py to isolate engine logic
    test_json_content = {
        "script_name": "Engine Simplified Loop Test v5.1",
        "actions": [
            {"id": "loadA", "command": "load_track", "deck_id": "deckA", "parameters": {"file_path": json_audio_path}},
            {"id": "playA", "command": "play", "deck_id": "deckA", "parameters": {"start_at_beat": 1}},
            { 
                "id": "loopA_at_beat_5", "command": "activate_loop", "deck_id": "deckA",
                "parameters": {"start_at_beat": 5, "length_beats": 2, "repetitions": 3},
                "trigger": {"type": "on_deck_beat", "source_deck_id": "deckA", "beat_number": 5}
            },
            { 
                "id": "stop_A_at_beat_15", "command": "stop", "deck_id": "deckA",
                "trigger": {"type": "on_deck_beat", "source_deck_id": "deckA", "beat_number": 15} 
            }
        ]
    }
    
    try:
        with open(dummy_json_path, 'w') as f: json.dump(test_json_content, f, indent=4)
        print(f"Created dummy JSON script for testing: {dummy_json_path}")
    except Exception as e_json_write: print(f"Error creating dummy JSON: {e_json_write}"); sys.exit(1)

    if engine.load_script_from_file(dummy_json_path): 
        engine.start_script_processing() 
        
        print("INFO: Engine Test - Script processing started. Monitoring. Press Ctrl+C to stop.")
        start_wait_time = time.time()
        max_script_duration_estimate = 25 # Adjusted for the shorter test
        
        try:
            while True: 
                engine_is_processing = engine.is_processing_script_actions 
                decks_are_active = engine.any_deck_active()

                if not engine_is_processing and not decks_are_active:
                    if not engine._pending_on_beat_actions : 
                        print("INFO: Engine Test - Engine done, no pending actions, and no decks active.")
                        break 
                
                if (time.time() - start_wait_time) > max_script_duration_estimate:
                    print(f"WARNING: Engine Test - Max wait time of {max_script_duration_estimate}s reached.")
                    if engine._pending_on_beat_actions:
                        print(f"DEBUG: Engine Test - Still {len(engine._pending_on_beat_actions)} pending actions on timeout:")
                        for pa_idx, pa in enumerate(engine._pending_on_beat_actions):
                            print(f"  Pending {pa_idx+1}: ID='{pa.get('id','N/A')}', CMD='{pa.get('command')}', Deck='{pa.get('deck_id','N/A')}', Trigger={pa.get('trigger')}")
                    break
                time.sleep(0.5)
            print("INFO: Engine Test - Monitoring loop finished or timed out.")
        except KeyboardInterrupt:
            print("\nINFO: Engine Test - KeyboardInterrupt received by test runner.")
        finally:
            engine.stop_script_processing() 
    
    print("--- AudioEngine Standalone Test Finished ---")