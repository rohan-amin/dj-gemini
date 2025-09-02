# dj-gemini/audio_engine/engine.py

import json
import time
import os
import sys 
import threading
import logging
logger = logging.getLogger(__name__)

try:
    import config as app_config
except ImportError:
    app_config = None 
    logger.warning("engine.py - Initial 'import config' failed. Ensure config.py is in project root or PYTHONPATH.")

from .audio_analyzer import AudioAnalyzer
from .deck import Deck
from .audio_clock import AudioClock
from .event_scheduler import EventScheduler
from .loop import (
    ObservabilityIntegrator,
    LoopSystemHealthMonitor
)
# LoopConfigurationManager removed - using existing action-based format

# Event-driven architecture replaces old polling system 

class AudioEngine:
    def __init__(self, app_config_module): 
        logger.debug("AudioEngine - Initializing...")
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
        
        # NEW: Event-driven architecture
        self.audio_clock = AudioClock()
        logger.debug(f"AudioEngine - Created audio clock instance {id(self.audio_clock)}")
        self.event_scheduler = EventScheduler(self.audio_clock)
        
        # Phase 2: BeatManager references for deck-specific beat operations
        self._deck_beat_managers = {}  # deck_id -> BeatManager
        logger.debug("AudioEngine - Initialized deck BeatManager storage for Phase 2")
        
        # Phase 3: Set EventScheduler engine reference for deck-specific beat timing
        self.event_scheduler.set_engine_reference(self)
        logger.debug("AudioEngine - Set EventScheduler engine reference for Phase 3")
        
        # Phase 7: Initialize loop observability only (config management uses existing action format)
        self.loop_observability = None  # Will be created when first deck is added
        logger.info("🔄 AudioEngine - Loop observability ready (using existing action-based format)")
        
        # Debug: log engine instance ID
        logger.debug(f"AudioEngine - Engine instance {id(self)} created with audio clock {id(self.audio_clock)}")
        
        # Event-driven architecture replaces old polling system
        self._all_actions_from_script = []
        self.is_processing_script_actions = False 
        
        self._script_start_time = 0 
        
        logger.debug("AudioEngine - Initialized.")
        
        # Register event handlers for different action types
        self._register_event_handlers()

    def _get_or_create_deck(self, deck_id):
        """Get existing deck or create new one with Phase 7 loop management integration"""
        if deck_id not in self.decks:
            print(f"DEBUG: Creating deck {deck_id}")
            deck = Deck(deck_id, self.analyzer, engine_instance=self)
            
            # Legacy LoopManager reference removed - using frame-accurate system only
            
            # Phase 2: Store BeatManager reference for deck-specific beat operations
            if hasattr(deck, 'beat_manager'):
                self._deck_beat_managers[deck_id] = deck.beat_manager
                logger.debug(f"AudioEngine - Stored BeatManager reference for deck {deck_id}")
            else:
                logger.warning(f"AudioEngine - Deck {deck_id} has no beat_manager attribute")
                
            # Phase 7.1: Initialize completion system and action adapter for this deck
            if hasattr(deck, 'loop_controller'):
                try:
                    # Initialize completion system with this engine as the deck manager
                    deck.loop_controller.initialize_completion_system(deck_manager=self)
                    logger.info(f"🔄 Phase 7 - Initialized completion system for deck {deck_id}")
                    
                    # Phase 2.2: Create ActionLoopAdapter for this deck
                    from audio_engine.loop.action_adapter import create_action_adapter_for_deck
                    deck.action_adapter = create_action_adapter_for_deck(deck, self.event_scheduler)
                    logger.info(f"🔄 Phase 2.2 - Created ActionLoopAdapter for deck {deck_id}")
                    
                    # Create system-wide observability if this is the first deck
                    if self.loop_observability is None:
                        from audio_engine.loop.observability import ObservabilityIntegrator
                        # Initialize with first deck's loop controller
                        self.loop_observability = ObservabilityIntegrator(deck.loop_controller)
                        logger.info("🔄 Phase 7 - Created system-wide observability integrator")
                    
                    # Register this deck's loop controller with observability
                    self.loop_observability.add_loop_controller(deck_id, deck.loop_controller)
                    logger.debug(f"🔄 Phase 7 - Registered deck {deck_id} with observability system")
                    
                except Exception as e:
                    logger.error(f"Phase 7 - Failed to initialize loop management for deck {deck_id}: {e}")
            
            # Add deck to dictionary FIRST
            self.decks[deck_id] = deck
            
            # THEN register beat callback with EventScheduler if it's running
            if hasattr(self, 'event_scheduler') and self.event_scheduler.is_running():
                print(f"DEBUG: Registering callback for deck {deck_id} during creation")
                self.event_scheduler.register_deck_callback(deck_id)
            print(f"DEBUG: Deck {deck_id} created. Total decks: {len(self.decks)}")
        return self.decks[deck_id]
    
    def get_beat_manager_for_deck(self, deck_id: str):
        """
        Get BeatManager for a specific deck.
        Phase 2: Helper method for EventScheduler to access deck-specific beat timing.
        """
        return self._deck_beat_managers.get(deck_id)
    
    def _register_event_handlers(self):
        """Register handlers for different action types with the event scheduler"""
        logger.debug("AudioEngine - Registering event handlers")
        
        # Register handlers for deck-specific commands
        deck_commands = [
            "load_track", "play", "pause", "stop", "seek_and_play", 
            "activate_loop", "deactivate_loop", "stop_at_beat", 
            "set_tempo", "set_pitch", "set_volume", "fade_volume",
            "set_eq", "fade_eq", "ramp_tempo", "play_scratch_sample",
            "set_stem_eq", "enable_stem_eq", "set_stem_volume",
            "set_master_eq", "set_all_stem_eq", "loop_completed", "loop_repetition_complete"
        ]
        
        for command in deck_commands:
            self.event_scheduler.register_handler(command, self._execute_deck_action)
        
        # Register handlers for engine-level commands
        engine_commands = ["crossfade", "bpm_match"]
        for command in engine_commands:
            self.event_scheduler.register_handler(command, self._execute_engine_action)
        
        # Register default handler for unknown commands
        self.event_scheduler.register_default_handler(self._execute_unknown_action)
        
        # Register error callback
        self.event_scheduler.register_error_callback(self._handle_scheduler_error)
        
        logger.debug("AudioEngine - Event handlers registered")
    
    def _execute_deck_action(self, action: dict):
        """Execute a deck-specific action"""
        try:
            deck_id = action.get("deck_id")
            if not deck_id:
                logger.error(f"Deck action missing deck_id: {action}")
                return False
            
            deck = self._get_or_create_deck(deck_id)
            if not deck:
                logger.error(f"Could not get or create deck: {deck_id}")
                return False
            
            # Execute the action using existing deck methods
            return self._execute_action(action)
            
        except Exception as e:
            logger.error(f"Error executing deck action: {e}")
            return False
    
    def _execute_engine_action(self, action: dict):
        """Execute an engine-level action"""
        try:
            # Execute the action using existing engine methods
            return self._execute_action(action)
            
        except Exception as e:
            logger.error(f"Error executing engine action: {e}")
            return False
    
    def _execute_unknown_action(self, action: dict):
        """Handle unknown action types"""
        logger.warning(f"Unknown action type: {action.get('command', 'unknown')}")
        return False
    
    def _handle_scheduler_error(self, error: Exception):
        """Handle errors from the event scheduler"""
        logger.error(f"Event scheduler error: {error}")
        # Could implement error recovery logic here

    def _validate_action(self, action, action_index):
        command = action.get("command")
        action_id_for_log = action.get('action_id', f"action_idx_{action_index+1}")
        # print(f"DEBUG: Validating action {action_index+1} (ID: {action_id_for_log}): CMD='{command}'") 

        if not command:
            logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): Missing 'command'.")
            return False

        trigger = action.get("trigger")
        
        if not trigger: 
            logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): Trigger object missing entirely (should have been defaulted).")
            return False 
            
        if not isinstance(trigger, dict) or "type" not in trigger:
            logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): Malformed 'trigger' or missing 'type'. Trigger: {trigger}")
            return False
        
        trigger_type = trigger.get("type")
        if trigger_type == "on_deck_beat":
            source_deck_id_val = trigger.get("source_deck_id")
            beat_number_val = trigger.get("beat_number")
            if not source_deck_id_val or not isinstance(beat_number_val, (int, float)):
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'on_deck_beat' trigger missing 'source_deck_id' ('{source_deck_id_val}') or valid 'beat_number' ('{beat_number_val}').")
                return False
        elif trigger_type == "on_loop_complete":
            # Phase 2.2: on_loop_complete triggers now handled by ActionLoopAdapter
            source_deck_id = trigger.get("source_deck_id")
            loop_action_id = trigger.get("loop_action_id")
            if not source_deck_id or not loop_action_id:
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'on_loop_complete' trigger missing 'source_deck_id' or 'loop_action_id'.")
                return False
            logger.debug(f"VALIDATION: on_loop_complete trigger found for action {action_id_for_log} - will be handled by ActionLoopAdapter")
        elif trigger_type == "script_start":
            pass 
        else:
            logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): Unsupported trigger type: '{trigger_type}'. Supported: 'script_start', 'on_deck_beat'.")
            return False
        
        deck_specific_commands = ["play", "pause", "stop", "seek_and_play", "activate_loop", "deactivate_loop", "load_track", "stop_at_beat", "set_tempo", "set_pitch", "set_volume", "fade_volume", "set_eq", "fade_eq", "ramp_tempo", "play_scratch_sample", "set_stem_eq", "enable_stem_eq", "set_stem_volume", "set_master_eq", "set_all_stem_eq"]
        engine_level_commands = ["crossfade", "bpm_match"] 
        
        if command in deck_specific_commands and not action.get("deck_id"):
            logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): Command '{command}' is missing 'deck_id'.")
            return False
        
        if command == "activate_loop":
            params = action.get("parameters", {})
            if params.get("start_at_beat") is None or params.get("length_beats") is None:
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'activate_loop' missing 'start_at_beat' or 'length_beats' in parameters. Params: {params}")
                return False
            try: 
                float(params["start_at_beat"])
                float(params["length_beats"])
                if params.get("repetitions") is not None: # Optional, but if present, check if int
                    if not (isinstance(params["repetitions"], str) and params["repetitions"].lower() == "infinite"):
                        int(params["repetitions"])
            except (ValueError, TypeError):
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'activate_loop' parameters not valid numbers/type. Params: {params}")
                return False
        elif command == "loop_completed":
            params = action.get("parameters", {})
            if not action.get("deck_id"):
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'loop_completed' missing 'deck_id'.")
                return False
            if not params.get("action_id"):
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'loop_completed' missing 'action_id' in parameters.")
                return False
        elif command == "set_tempo":
            params = action.get("parameters", {})
            if params.get("target_bpm") is None:
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'set_tempo' missing 'target_bpm' in parameters. Params: {params}")
                return False
            try:
                target_bpm = float(params["target_bpm"])
                if target_bpm <= 0:
                    logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'target_bpm' must be positive. Value: {target_bpm}")
                    return False
            except (ValueError, TypeError):
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'target_bpm' not a valid number. Params: {params}")
                return False
        elif command == "set_volume":
            params = action.get("parameters", {})
            if params.get("volume") is None:
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'set_volume' missing 'volume' in parameters. Params: {params}")
                return False
            try:
                volume = float(params["volume"])
                if volume < 0.0 or volume > 1.0:
                    logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'volume' must be between 0.0 and 1.0. Value: {volume}")
                    return False
            except (ValueError, TypeError):
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'volume' not a valid number. Params: {params}")
                return False
        elif command == "fade_volume":
            params = action.get("parameters", {})
            if params.get("target_volume") is None or params.get("duration_seconds") is None:
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'fade_volume' missing 'target_volume' or 'duration_seconds' in parameters. Params: {params}")
                return False
            try:
                target_volume = float(params["target_volume"])
                duration_seconds = float(params["duration_seconds"])
                if target_volume < 0.0 or target_volume > 1.0:
                    logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'target_volume' must be between 0.0 and 1.0. Value: {target_volume}")
                    return False
                if duration_seconds < 0.0:
                    logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'duration_seconds' must be positive. Value: {duration_seconds}")
                    return False
            except (ValueError, TypeError):
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'fade_volume' parameters not valid numbers. Params: {params}")
                return False
        
        elif command == "set_eq":
            params = action.get("parameters", {})
            # At least one EQ band must be specified
            eq_bands = ["low", "mid", "high"]
            found_bands = [band for band in eq_bands if band in params]
            
            if not found_bands:
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'set_eq' must specify at least one of: 'low', 'mid', 'high'. Params: {params}")
                return False
            
            try:
                for band in found_bands:
                    value = float(params[band])
                    if value < 0.0 or value > 2.0:
                        logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): '{band}' must be between 0.0 and 2.0. Value: {value}")
                        return False
            except (ValueError, TypeError):
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'set_eq' parameters not valid numbers. Params: {params}")
                return False
        
        elif command == "fade_eq":
            params = action.get("parameters", {})
            # At least one EQ band must be specified
            eq_bands = ["target_low", "target_mid", "target_high"]
            found_bands = [band for band in eq_bands if band in params]
            
            if not found_bands:
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'fade_eq' must specify at least one of: 'target_low', 'target_mid', 'target_high'. Params: {params}")
                return False
            
            try:
                for band in found_bands:
                    value = float(params[band])
                    if value < 0.0 or value > 2.0:
                        logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): '{band}' must be between 0.0 and 2.0. Value: {value}")
                        return False
                
                # Validate duration if present
                if "duration_seconds" in params:
                    duration = float(params["duration_seconds"])
                    if duration < 0.0:
                        logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'duration_seconds' must be positive. Value: {duration}")
                        return False
            except (ValueError, TypeError):
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'fade_eq' parameters not valid numbers. Params: {params}")
                return False

        elif command == "set_stem_eq":
            params = action.get("parameters", {})
            
            # Stem name is required
            if "stem" not in params:
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'set_stem_eq' missing required parameter 'stem'. Params: {params}")
                return False
                
            stem = params["stem"]
            valid_stems = ["vocals", "drums", "bass", "other"]
            if stem not in valid_stems:
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): Invalid stem '{stem}'. Must be one of: {valid_stems}")
                return False
            
            # At least one EQ band must be specified (support both legacy and professional formats)
            legacy_bands = ["low", "mid", "high"]
            professional_bands = ["low_db", "mid_db", "high_db"]
            
            found_legacy = [band for band in legacy_bands if band in params]
            found_professional = [band for band in professional_bands if band in params]
            
            if not found_legacy and not found_professional:
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'set_stem_eq' must specify at least one of: 'low', 'mid', 'high' (legacy) or 'low_db', 'mid_db', 'high_db' (professional). Params: {params}")
                return False
            
            try:
                # Validate legacy format (linear gains 0.0-3.0)
                for band in found_legacy:
                    value = float(params[band])
                    if value < 0.0 or value > 3.0:
                        logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): '{band}' must be between 0.0 and 3.0. Value: {value}")
                        return False
                        
                # Validate professional format (dB gains -20.0 to +20.0)
                for band in found_professional:
                    value = float(params[band])
                    if value < -20.0 or value > 20.0:
                        logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): '{band}' must be between -20.0 and +20.0 dB. Value: {value}")
                        return False
                        
            except (ValueError, TypeError):
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'set_stem_eq' parameters not valid numbers. Params: {params}")
                return False

        elif command == "enable_stem_eq":
            params = action.get("parameters", {})
            
            # Stem name is required
            if "stem" not in params:
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'enable_stem_eq' missing required parameter 'stem'. Params: {params}")
                return False
                
            stem = params["stem"]
            valid_stems = ["vocals", "drums", "bass", "other"]
            if stem not in valid_stems:
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): Invalid stem '{stem}'. Must be one of: {valid_stems}")
                return False
                
            # Enabled parameter is required
            if "enabled" not in params:
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'enable_stem_eq' missing required parameter 'enabled'. Params: {params}")
                return False
            
            if not isinstance(params["enabled"], bool):
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'enabled' must be true or false. Value: {params['enabled']}")
                return False

        elif command == "set_stem_volume":
            params = action.get("parameters", {})
            
            # Stem name is required
            if "stem" not in params:
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'set_stem_volume' missing required parameter 'stem'. Params: {params}")
                return False
                
            stem = params["stem"]
            valid_stems = ["vocals", "drums", "bass", "other"]
            if stem not in valid_stems:
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): Invalid stem '{stem}'. Must be one of: {valid_stems}")
                return False
                
            # Volume parameter is required
            if "volume" not in params:
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'set_stem_volume' missing required parameter 'volume'. Params: {params}")
                return False
            
            try:
                volume = float(params["volume"])
                if volume < 0.0 or volume > 2.0:
                    logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'volume' must be between 0.0 and 2.0. Value: {volume}")
                    return False
            except (ValueError, TypeError):
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'volume' parameter not a valid number. Value: {params['volume']}")
                return False

        elif command == "set_master_eq":
            params = action.get("parameters", {})
            
            # Validate gain parameters if specified
            gain_params = ["low_gain", "mid_gain", "high_gain"]
            for gain_param in gain_params:
                if gain_param in params:
                    try:
                        gain_value = float(params[gain_param])
                        if gain_value < 0.0 or gain_value > 2.0:
                            logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): '{gain_param}' must be between 0.0 and 2.0. Value: {gain_value}")
                            return False
                    except (ValueError, TypeError):
                        logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): '{gain_param}' not a valid number. Value: {params[gain_param]}")
                        return False
            
            # Validate enabled parameter if specified
            if "enabled" in params:
                if not isinstance(params["enabled"], bool):
                    logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'enabled' must be true or false. Value: {params['enabled']}")
                    return False

        elif command == "set_all_stem_eq":
            params = action.get("parameters", {})
            valid_stems = ["vocals", "drums", "bass", "other"]
            
            # At least one stem must be specified
            if not any(stem in params for stem in valid_stems):
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'set_all_stem_eq' must specify at least one stem. Valid stems: {valid_stems}")
                return False
            
            # Validate each stem's settings
            for stem_name, stem_settings in params.items():
                if stem_name not in valid_stems:
                    continue  # Skip non-stem parameters
                    
                if not isinstance(stem_settings, dict):
                    logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): Settings for '{stem_name}' must be an object. Value: {stem_settings}")
                    return False
                
                # Validate EQ values if present
                for eq_band in ["low", "mid", "high"]:
                    if eq_band in stem_settings:
                        try:
                            value = float(stem_settings[eq_band])
                            if value < 0.0 or value > 3.0:
                                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): '{stem_name}.{eq_band}' must be between 0.0 and 3.0. Value: {value}")
                                return False
                        except (ValueError, TypeError):
                            logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): '{stem_name}.{eq_band}' not a valid number. Value: {stem_settings[eq_band]}")
                            return False
                
                # Validate enabled if present
                if "enabled" in stem_settings:
                    if not isinstance(stem_settings["enabled"], bool):
                        logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): '{stem_name}.enabled' must be true or false. Value: {stem_settings['enabled']}")
                        return False

        elif command == "crossfade":
            params = action.get("parameters", {})
            if params.get("from_deck") is None or params.get("to_deck") is None or params.get("duration_seconds") is None:
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'crossfade' missing required parameters. Params: {params}")
                return False
            try:
                duration_seconds = float(params["duration_seconds"])
                if duration_seconds < 0.0:
                    logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'duration_seconds' must be positive. Value: {duration_seconds}")
                    return False
            except (ValueError, TypeError):
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'crossfade' duration not a valid number. Params: {params}")
                return False
        elif command == "bpm_match":
            params = action.get("parameters", {})
            if params.get("reference_deck") is None or params.get("follow_deck") is None:
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'bpm_match' missing 'reference_deck' or 'follow_deck' in parameters. Params: {params}")
                return False
            try:
                if params.get("target_beat") is not None:
                    target_beat = float(params["target_beat"])
                    if target_beat <= 0:
                        logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'target_beat' must be positive. Value: {target_beat}")
                        return False
                # Validate phase_offset_beats if present
                if params.get("phase_offset_beats") is not None:
                    phase_offset = float(params["phase_offset_beats"])
                    logger.debug(f"VALIDATION OK: phase_offset_beats = {phase_offset}")
            except (ValueError, TypeError):
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): Invalid numeric parameter. Params: {params}")
                return False
        elif command == "seek_and_play":
            params = action.get("parameters", {})
            # Must have exactly one of the seek targets
            seek_targets = ["start_at_beat", "start_at_cue_name", "start_at_loop"]
            found_targets = [target for target in seek_targets if target in params]
            
            if len(found_targets) != 1:
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'seek_and_play' must have exactly one of: 'start_at_beat', 'start_at_cue_name', or 'start_at_loop'. Found: {found_targets}")
                return False
            
            target = found_targets[0]
            if target == "start_at_beat":
                try:
                    start_beat = float(params["start_at_beat"])
                    if start_beat < 0:
                        logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'start_at_beat' must be non-negative. Value: {start_beat}")
                        return False
                except (ValueError, TypeError):
                    logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'start_at_beat' must be a valid number. Value: {params['start_at_beat']}")
                    return False
            elif target == "start_at_cue_name":
                cue_name = params["start_at_cue_name"]
                if not isinstance(cue_name, str) or not cue_name.strip():
                    logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'start_at_cue_name' must be a non-empty string. Value: {cue_name}")
                    return False
            elif target == "start_at_loop":
                loop_params = params["start_at_loop"]
                if not isinstance(loop_params, dict):
                    logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'start_at_loop' must be an object. Value: {loop_params}")
                    return False
                
                if loop_params.get("start_at_beat") is None or loop_params.get("length_beats") is None:
                    logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'start_at_loop' requires 'start_at_beat' and 'length_beats'. Params: {loop_params}")
                    return False
                
                try:
                    start_beat = float(loop_params["start_at_beat"])
                    length_beats = float(loop_params["length_beats"])
                    if start_beat < 0 or length_beats <= 0:
                        logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'start_at_beat' must be non-negative and 'length_beats' must be positive. Values: {start_beat}, {length_beats}")
                        return False
                    
                    # Validate repetitions if present
                    repetitions = loop_params.get("repetitions")
                    if repetitions is not None:
                        if not (isinstance(repetitions, str) and repetitions.lower() == "infinite"):
                            try:
                                int(repetitions)
                            except (ValueError, TypeError):
                                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'repetitions' must be an integer or 'infinite'. Value: {repetitions}")
                                return False
                except (ValueError, TypeError):
                    logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'start_at_loop' parameters not valid numbers. Params: {loop_params}")
                    return False
        
        elif command == "ramp_tempo":
            params = action.get("parameters", {})
            if (params.get("start_beat") is None or params.get("end_beat") is None or 
                params.get("start_bpm") is None or params.get("end_bpm") is None):
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'ramp_tempo' missing required parameters. Params: {params}")
                return False
            try:
                start_beat = float(params["start_beat"])
                end_beat = float(params["end_beat"])
                start_bpm = float(params["start_bpm"])
                end_bpm = float(params["end_bpm"])
                curve = params.get("curve", "linear")
                
                if start_beat >= end_beat:
                    logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'start_beat' must be less than 'end_beat'. Values: {start_beat}, {end_beat}")
                    return False
                if start_bpm <= 0 or end_bpm <= 0:
                    logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): BPM values must be positive. Values: {start_bpm}, {end_bpm}")
                    return False
                if curve not in ["linear", "exponential", "smooth", "step"]:
                    logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'curve' must be 'linear', 'exponential', 'smooth', or 'step'. Value: {curve}")
                    return False
            except (ValueError, TypeError):
                logger.error(f"VALIDATION FAIL (Action ID: {action_id_for_log}): 'ramp_tempo' parameters not valid numbers. Params: {params}")
                return False
        # print(f"DEBUG: Validation OK for action {action_index+1} (ID: {action_id_for_log})")
        return True


    def load_script_from_file(self, json_filepath):
        logger.debug(f"AudioEngine - Loading script from: {json_filepath}")
        
        path_to_load = json_filepath
        if self.app_config and not os.path.isabs(json_filepath):
            constructed_path = os.path.join(self.app_config.MIX_CONFIGS_DIR, json_filepath)
            if os.path.exists(constructed_path): path_to_load = constructed_path
            else: 
                constructed_path_alt = os.path.join(self.app_config.PROJECT_ROOT_DIR, json_filepath)
                if os.path.exists(constructed_path_alt): path_to_load = constructed_path_alt
                else: logger.warning(f"Script file '{json_filepath}' not found in expected locations. Trying as is.")
        
        if not os.path.exists(path_to_load):
            logger.error(f"AudioEngine - Script file not found: {path_to_load}")
            return False
        
        try:
            with open(path_to_load, 'r') as f:
                script_data = json.load(f)
            
            if not isinstance(script_data.get("actions"), list):
                logger.error("AudioEngine - 'actions' in script must be a list."); return False
            
            self._all_actions_from_script = []
            for i, action in enumerate(script_data["actions"]):
                if not action.get("trigger"):
                    logger.debug(f"AudioEngine - Action {i+1} ('{action.get('command')}') missing trigger, defaulting to 'script_start'.")
                    action["trigger"] = {"type": "script_start"}
                
                if not self._validate_action(action, i):
                    logger.error(f"AudioEngine - Invalid action at index {i} (Command: {action.get('command')}). Aborting script load.")
                    return False
                self._all_actions_from_script.append(action)
            
            self.script_name = script_data.get("mix_name", "Untitled Mix")
            logger.debug(f"AudioEngine - Script '{self.script_name}' loaded and validated with {len(self._all_actions_from_script)} actions.")

            # Pre-process all audio transformations
            if not self._validate_audio_cache():
                logger.error("AudioEngine - Pre-processing of audio transformations failed. Aborting script load.")
                return False

            # === NEW: Load mix configuration with frame-accurate timing ===
            try:
                from .mix_loader import MixConfigLoader
                self.mix_loader = MixConfigLoader(self)
                
                # Load and pre-schedule all actions with frame-accurate timing
                if self.mix_loader.load_mix_config(path_to_load):
                    logger.info("🎯 Mix configuration loaded with frame-accurate timing - bypassing old event scheduler")
                    # Set flag to indicate we're using the new timing system
                    self._using_frame_accurate_timing = True
                else:
                    logger.warning("Frame-accurate mix loader failed - falling back to old event scheduler")
                    self._using_frame_accurate_timing = False
                    
            except Exception as e:
                logger.error(f"Error initializing frame-accurate mix loader: {e}")
                logger.warning("Falling back to old event scheduler")
                self._using_frame_accurate_timing = False

            return True
            
        except Exception as e:
            logger.error(f"AudioEngine - Failed to load/parse script {path_to_load}: {e}"); return False

    def _validate_audio_cache(self):
        """Validate that all required audio transformations are pre-processed and cached"""
        logger.info("AudioEngine - Validating audio cache...")
        
        # Collect all required transformations
        required_transformations = {}
        missing_cache = []
        
        for action in self._all_actions_from_script:
            command = action.get("command")
            deck_id = action.get("deck_id")
            parameters = action.get("parameters", {})
            
            if command == "load_track":
                filepath = parameters.get("filepath")
                if filepath and deck_id:
                    # Convert relative paths to absolute (same logic as preprocessing)
                    if not os.path.isabs(filepath):
                        filepath = os.path.join(self.app_config.PROJECT_ROOT_DIR, filepath)
                    
                    if deck_id not in required_transformations:
                        required_transformations[deck_id] = {
                            "filepath": filepath,
                            "tempo_changes": set(),
                            "pitch_changes": set()
                        }
            
            elif command == "set_tempo" and deck_id:
                target_bpm = parameters.get("target_bpm")
                if target_bpm and deck_id in required_transformations:
                    required_transformations[deck_id]["tempo_changes"].add(float(target_bpm))
            
            elif command == "set_pitch" and deck_id:
                semitones = parameters.get("semitones")
                if semitones and deck_id in required_transformations:
                    required_transformations[deck_id]["pitch_changes"].add(float(semitones))
            
            elif command == "ramp_tempo" and deck_id:
                start_bpm = parameters.get("start_bpm")
                end_bpm = parameters.get("end_bpm")
                if start_bpm and end_bpm and deck_id in required_transformations:
                    required_transformations[deck_id]["tempo_changes"].add(float(start_bpm))
                    required_transformations[deck_id]["tempo_changes"].add(float(end_bpm))
        
        # Validate cache for each deck's transformations
        for deck_id, transformations in required_transformations.items():
            filepath = transformations["filepath"]
            
            # Check tempo cache
            for target_bpm in transformations["tempo_changes"]:
                cache_filepath = self.app_config.get_tempo_cache_filepath(filepath, target_bpm)
                if not os.path.exists(cache_filepath):
                    missing_cache.append(f"Tempo {target_bpm} BPM for {os.path.basename(filepath)}")
            
            # Check pitch cache
            for semitones in transformations["pitch_changes"]:
                cache_filepath = self.app_config.get_pitch_cache_filepath(filepath, semitones)
                if not os.path.exists(cache_filepath):
                    missing_cache.append(f"Pitch {semitones:+.1f} semitones for {os.path.basename(filepath)}")
        
        if missing_cache:
            logger.error("AudioEngine - Missing required audio cache files:")
            for missing in missing_cache:
                logger.error(f"  - {missing}")
            logger.error("Run preprocessing first: python preprocess.py <script.json>")
            return False
        
        logger.info(f"AudioEngine - Cache validation complete. Found cache for {len(required_transformations)} tracks.")
        return True


    def _preprocess_deck_transformations(self, deck_id, transformations):
        """Pre-process all transformations for a specific deck"""
        try:
            # Resolve filepath
            filepath = transformations["filepath"]
            if not os.path.isabs(filepath):
                filepath = os.path.join(self.app_config.PROJECT_ROOT_DIR, filepath)
            
            # Create deck and load track
            deck = Deck(deck_id, self.analyzer, engine_instance=self)
            deck.load_track(filepath)
            
            # Pre-process tempo changes
            for target_bpm in transformations["tempo_changes"]:
                if not deck._preprocess_tempo_change(target_bpm):
                    logger.error(f"AudioEngine - Failed to pre-process tempo change to {target_bpm} BPM for {deck_id}")
                    return False
            
            # Pre-process pitch changes
            for semitones in transformations["pitch_changes"]:
                if not deck._preprocess_pitch_change(semitones):
                    logger.error(f"AudioEngine - Failed to pre-process pitch change to {semitones} semitones for {deck_id}")
                    return False
            

            
            # Store the deck
            self.decks[deck_id] = deck
            
            logger.info(f"AudioEngine - Successfully pre-processed all transformations for {deck_id}")
            return True
            
        except Exception as e:
            logger.error(f"AudioEngine - Error pre-processing deck {deck_id}: {e}")
            return False

    def _preprocess_global_sample(self, sample_filepath, sample_id, pitch_semitones=None):
        """Pre-process a global sample with optional pitch adjustment"""
        try:
            import essentia.standard as es
            import librosa
            import numpy as np
            
            # Load the sample audio
            loader = es.MonoLoader(filename=sample_filepath)
            sample_audio = loader()
            sample_sr = int(loader.paramValue('sampleRate'))
            
            # Resample if needed to match standard sample rate (44100)
            if sample_sr != 44100:
                sample_audio = librosa.resample(
                    sample_audio,
                    orig_sr=sample_sr,
                    target_sr=44100
                )
            
            # Apply pitch adjustment if specified
            if pitch_semitones is not None and pitch_semitones != 0:
                sample_audio = librosa.effects.pitch_shift(sample_audio, sr=44100, n_steps=pitch_semitones)
            
            # Apply a gentle fade-in to the entire sample to prevent clicks
            fade_samples = min(128, len(sample_audio) // 16)  # 128 samples or 1/16 of sample length
            if fade_samples > 0:
                fade_in = np.linspace(0, 1, fade_samples)
                # Ensure sample_audio is 1D for the fade operation
                if sample_audio.ndim == 1:
                    sample_audio[:fade_samples] *= fade_in
                else:
                    sample_audio[:fade_samples] *= fade_in.reshape(-1, 1)
            
            # Apply a gentle fade-out to the entire sample
            if fade_samples > 0:
                fade_out = np.linspace(1, 0, fade_samples)
                # Ensure sample_audio is 1D for the fade operation
                if sample_audio.ndim == 1:
                    sample_audio[-fade_samples:] *= fade_out
                else:
                    sample_audio[-fade_samples:] *= fade_out.reshape(-1, 1)
            
            # Store for global playback
            if not hasattr(self, '_global_sample_cache'):
                self._global_sample_cache = {}
            
            self._global_sample_cache[sample_id] = sample_audio
            
            logger.debug(f"AudioEngine - Pre-processed global sample: {sample_id} (pitch: {pitch_semitones or 0} semitones)")
            return True
            
        except Exception as e:
            logger.error(f"AudioEngine - Error pre-processing global sample {sample_id}: {e}")
            return False
            
            # Apply a gentle fade-in to the entire sample to prevent clicks
            fade_samples = min(128, len(sample_audio) // 16)  # 128 samples or 1/16 of sample length
            if fade_samples > 0:
                fade_in = np.linspace(0, 1, fade_samples)
                # Ensure sample_audio is 1D for the fade operation
                if sample_audio.ndim == 1:
                    sample_audio[:fade_samples] *= fade_in
                else:
                    sample_audio[:fade_samples] *= fade_in.reshape(-1, 1)
            
            # Apply a gentle fade-out to the entire sample
            if fade_samples > 0:
                fade_out = np.linspace(1, 0, fade_samples)
                # Ensure sample_audio is 1D for the fade operation
                if sample_audio.ndim == 1:
                    sample_audio[-fade_samples:] *= fade_out
                else:
                    sample_audio[-fade_samples:] *= fade_out.reshape(-1, 1)
            
            # Store for global playback
            self._global_scratch_sample_audio = sample_audio
            self._global_scratch_sample_position = 0
            self._global_scratch_sample_active = True
            
            logger.debug(f"AudioEngine - Pre-processed global scratch sample: {os.path.basename(sample_filepath)}")
            return True
            
        except Exception as e:
            logger.error(f"AudioEngine - Error pre-processing global scratch sample: {e}")
            return False


    def start_script_processing(self): 
        print("DEBUG: start_script_processing() called")  # Basic print to bypass logging
        if self.is_processing_script_actions:
            logger.warning("Script processing already running.")
            return
        if not self._all_actions_from_script : 
            logger.info("No actions loaded to process.")
            return

        print(f"DEBUG: Starting script processing with {len(self._all_actions_from_script)} actions")
        
        # Check if we're using the new frame-accurate timing system
        if getattr(self, '_using_frame_accurate_timing', False):
            logger.info(f"🎯 Starting frame-accurate script processing: '{self.script_name}' - OLD EVENT SCHEDULER BYPASSED")
            self.is_processing_script_actions = True 
            self._script_start_time = time.time() 
            
            # Start the audio clock
            self.audio_clock.start()
            
            # The mix loader has already pre-scheduled all actions with frame-accurate timing
            # No need to start the old event scheduler or register beat callbacks
            logger.info("🚀 All actions pre-scheduled with sample-accurate timing")
            
        else:
            # Fallback to old event scheduler system
            logger.info(f"Starting event-driven script processing: '{self.script_name}'")
            self.is_processing_script_actions = True 
            self._script_start_time = time.time() 
            
            # Start the audio clock
            self.audio_clock.start()
            
            # Start the event scheduler FIRST so it's running when decks are created
            self.event_scheduler.start()
            
            # Schedule all actions using the event scheduler (this creates decks)
            self._schedule_all_actions()
            
            # FORCE register beat callbacks with all existing decks
            for deck_id, deck in self.decks.items():
                if hasattr(deck, 'beat_manager'):
                    deck.beat_manager.add_beat_callback(self.event_scheduler._on_beat_boundary)
                    print(f"FORCE REGISTERED: Beat callback for deck {deck_id}")
        
        # Also ensure EventScheduler has registered callbacks
        if hasattr(self, 'event_scheduler') and self.event_scheduler.is_running():
            self.event_scheduler._register_beat_callbacks()
            print("DEBUG: Re-registered all beat callbacks with EventScheduler")
        
        print(f"TOTAL CALLBACKS REGISTERED: {sum(len(deck.beat_manager._beat_callbacks) for deck in self.decks.values() if hasattr(deck, 'beat_manager'))}")
        
        logger.info("AudioEngine - Event-driven script processing started")
        
        # Debug: Check if callbacks were actually registered
        total_callbacks = 0
        for deck_id, deck in self.decks.items():
            if hasattr(deck, 'beat_manager'):
                callback_count = len(deck.beat_manager._beat_callbacks)
                total_callbacks += callback_count
                logger.info(f"AudioEngine: Deck {deck_id} has {callback_count} beat callbacks")
        logger.info(f"AudioEngine: Total beat callbacks registered: {total_callbacks}")
    
    def _register_all_beat_callbacks(self):
        """Register beat callbacks with all existing decks after EventScheduler starts"""
        logger.info(f"AudioEngine: _register_all_beat_callbacks called")
        logger.info(f"AudioEngine: Has event_scheduler: {hasattr(self, 'event_scheduler')}")
        logger.info(f"AudioEngine: EventScheduler running: {hasattr(self, 'event_scheduler') and self.event_scheduler.is_running()}")
        logger.info(f"AudioEngine: Number of decks: {len(self.decks)}")
        
        if hasattr(self, 'event_scheduler') and self.event_scheduler.is_running():
            registered_count = 0
            for deck_id, deck in self.decks.items():
                logger.info(f"AudioEngine: Processing deck {deck_id}")
                if hasattr(deck, 'beat_manager'):
                    deck.beat_manager.add_beat_callback(self.event_scheduler._on_beat_boundary)
                    registered_count += 1
                    logger.info(f"AudioEngine: Registered beat callback with deck {deck_id}")
                else:
                    logger.warning(f"AudioEngine: Deck {deck_id} has no beat_manager")
            logger.info(f"AudioEngine: Registered beat callbacks with {registered_count} decks")
        else:
            logger.warning("AudioEngine: EventScheduler not running, cannot register beat callbacks")
    
    def _schedule_all_actions(self):
        """Schedule all script actions using the event scheduler"""
        print(f"DEBUG: _schedule_all_actions called with {len(self._all_actions_from_script)} actions")
        logger.debug("AudioEngine - Scheduling all script actions")
        
        for i, action in enumerate(self._all_actions_from_script):
            try:
                print(f"DEBUG: Scheduling action {i}: {action.get('command', 'NO_COMMAND')} for deck {action.get('deck_id', 'NO_DECK')}")
                self._schedule_action(action)
            except Exception as e:
                logger.error(f"Failed to schedule action {action.get('action_id', 'unknown')}: {e}")
        
        logger.debug("AudioEngine - All actions scheduled")
    
    def _schedule_action(self, action: dict):
        """Schedule a single action based on its trigger type"""
        trigger = action.get("trigger", {})
        trigger_type = trigger.get("type")
        
        if trigger_type == "script_start":
            # Phase 5B: Execute on next beat boundary for musical timing
            # Ensure deck_id is set for beat-based scheduling
            if 'deck_id' not in action:
                # For script_start actions without deck_id, use the primary deck
                primary_deck = list(self.decks.keys())[0] if self.decks else None
                if primary_deck:
                    action = action.copy()  # Don't modify original
                    action['deck_id'] = primary_deck
                    logger.debug(f"AudioEngine - Added deck_id {primary_deck} to script_start action")
                else:
                    logger.error(f"AudioEngine - No decks available for script_start action: {action}")
                    return
            
            event_id = self.event_scheduler.schedule_immediate_action(
                action, priority=100  # High priority for immediate actions
            )
            logger.debug(f"AudioEngine - Scheduled immediate beat action: {event_id}")
            
        elif trigger_type == "on_deck_beat":
            # Beat-triggered actions - store for later execution when deck reaches the beat
            source_deck_id = trigger.get("source_deck_id")
            target_beat = trigger.get("beat_number")
            
            if source_deck_id and target_beat is not None:
                # Phase 4: Always use BeatManager system for deck-specific timing
                event_id = self.event_scheduler.schedule_beat_action(
                    action, target_beat, deck_id=source_deck_id, priority=50
                )
                logger.debug(f"AudioEngine - Scheduled beat action {event_id} for deck {source_deck_id} beat {target_beat} (BeatManager)")
            else:
                logger.error(f"Invalid on_deck_beat trigger: {trigger}")
                
        elif trigger_type == "on_loop_complete":
            # Phase 2.2: Register this action with ActionLoopAdapter for completion handling
            source_deck_id = trigger.get("source_deck_id")
            loop_action_id = trigger.get("loop_action_id")
            
            if source_deck_id and loop_action_id:
                deck = self._get_or_create_deck(source_deck_id)
                if hasattr(deck, 'action_adapter') and deck.action_adapter:
                    deck.action_adapter.register_completion_trigger(action, action)
                    logger.info(f"Phase 2.2: Registered on_loop_complete trigger for loop {loop_action_id} -> action {action.get('action_id')}")
                    return False  # Don't execute now, will be triggered by completion
                else:
                    logger.warning(f"Phase 2.2: Deck {source_deck_id} has no ActionLoopAdapter - on_loop_complete trigger ignored")
            else:
                logger.error(f"Phase 2.2: Invalid on_loop_complete trigger for action {action.get('action_id')}")
            
            return False
            
        else:
            logger.warning(f"Unknown trigger type: {trigger_type}")



    def stop_script_processing(self): 
        logger.info("Stop event-driven script processing requested externally.")
        
        if not self.is_processing_script_actions:
            logger.info("Script processing was not running.")
            return

        # Stop the event scheduler
        if self.event_scheduler:
            self.event_scheduler.stop()
            logger.info("Event scheduler stopped.")
        
        # Stop the audio clock
        if self.audio_clock:
            self.audio_clock.stop()
            logger.info("Audio clock stopped.")
        
        self.is_processing_script_actions = False 
        logger.info("Event-driven script processing stopped.")
        self.shutdown_decks() 

    # Legacy loop completion method removed - now handled by musical timing system

    # OLD: Engine loop removed - replaced by event-driven scheduler
    # def _engine_loop(self):
    #     # This method has been replaced by the EventScheduler._execution_loop
    #     pass 
        logger.debug("Engine loop finished - replaced by event-driven scheduler")

    def _print_beat_indicator(self):
        """Print a clear beat indicator showing current status of all decks"""
        try:
            # Get current time
            current_time = time.time()
            time_str = time.strftime("%M:%S", time.gmtime(current_time)) + f".{int((current_time % 1) * 1000):03d}"
            
            # Build deck status strings
            deck_statuses = []
            current_beat_changed = False
            
            for deck_id, deck in self.decks.items():
                if deck.is_active():
                    current_beat = deck.get_current_beat_count()
                    total_beats = len(deck.beat_timestamps) if hasattr(deck, 'beat_timestamps') and deck.beat_timestamps is not None else 0
                    bpm = deck.bpm if hasattr(deck, 'bpm') else 0
                    
                    # Check if beat changed since last print
                    last_beat = getattr(self, f'_last_beat_{deck_id}', -1)
                    if int(current_beat) != int(last_beat):
                        current_beat_changed = True
                        setattr(self, f'_last_beat_{deck_id}', int(current_beat))
                    
                    # Build deck status string
                    deck_status = f"{deck_id}: {current_beat:.0f}/{total_beats} (BPM: {bpm:.2f})"
                    
                    # Add loop status if active (frame-accurate system)
                    if hasattr(deck, '_frame_accurate_loop') and deck._frame_accurate_loop and deck._frame_accurate_loop.get('active'):
                        loop_info = deck._frame_accurate_loop
                        current_rep = loop_info.get('current_repetition', 0)
                        total_reps = loop_info.get('repetitions', 1)
                        if total_reps is not None:
                            deck_status += f" [Loop: {current_rep}/{total_reps}]"
                        else:
                            deck_status += " [Loop: ∞]"
                    
                    deck_statuses.append(deck_status)
                else:
                    deck_statuses.append(f"{deck_id}: inactive")
            
            # Only print if a beat changed
            if current_beat_changed:
                # Get pending actions count from event scheduler
                if hasattr(self, 'event_scheduler') and self.event_scheduler is not None:
                    scheduler_stats = self.event_scheduler.get_stats()
                    pending_actions = scheduler_stats.get('queue', {}).get('total', {}).get('total_scheduled', 0) - scheduler_stats.get('queue', {}).get('total', {}).get('total_executed', 0)
                else:
                    pending_actions = 0
                
                # Print the beat indicator
                deck_info = " | ".join(deck_statuses)
                print(f"[BEAT] {deck_info} | Time: {time_str} | Actions: {pending_actions} pending")
            
        except Exception as e:
            logger.debug(f"Error printing beat indicator: {e}")
            
    def _execute_action(self, action_dict):
        command = action_dict.get("command")
        deck_id = action_dict.get("deck_id")
        parameters = action_dict.get("parameters", {})

        if not command: logger.warning("Action missing 'command'. Skipping."); return False
        logger.debug(f"Executing action: {action_dict}")

        try:
            if command == "load_track":
                filepath = parameters.get("filepath")
                if not deck_id or not filepath:
                    logger.warning("'load_track' missing deck_id or filepath in parameters. Skipping."); return False
                
                # Validate file exists
                if not os.path.exists(filepath):
                    logger.error(f"Track file not found: {filepath}. Exiting due to invalid load_track filepath.")
                    sys.exit(1)
                
                deck = self._get_or_create_deck(deck_id)
                deck.load_track(filepath)
                
                # Handle tempo matching after track is loaded
                match_tempo_to = parameters.get("match_tempo_to")
                tempo_offset_bpm = parameters.get("tempo_offset_bpm", 0.0)
                
                if match_tempo_to:
                    reference_deck = self._get_or_create_deck(match_tempo_to)
                    reference_bpm = reference_deck.beat_manager.get_bpm()
                    if reference_bpm > 0:
                        target_bpm = reference_bpm + tempo_offset_bpm
                        deck.beat_manager.handle_tempo_change(target_bpm)
                        logger.info(f"Tempo matched {deck_id} to {match_tempo_to}: {reference_bpm} + {tempo_offset_bpm} = {target_bpm} BPM via BeatManager")
                    else:
                        logger.warning(f"Cannot match tempo: reference deck {match_tempo_to} has invalid BPM ({reference_bpm})")
                
                return False  # Engine convention: False = success

            elif command == "play":
                if not deck_id: logger.warning("'play' missing deck_id. Skipping."); return False
                deck = self._get_or_create_deck(deck_id)
                
                # Set volume if specified
                if "volume" in parameters:
                    volume = float(parameters["volume"])
                    volume = max(0.0, min(1.0, volume))  # Clamp to 0.0-1.0 range
                    deck.set_volume(volume)
                    logger.debug(f"Set volume for deck {deck_id} to {volume}")
                
                deck.play(start_at_beat=parameters.get("start_at_beat"), 
                          start_at_cue_name=parameters.get("start_at_cue_name"))
                
                return False  # Engine convention: False = success

            elif command == "pause":
                if not deck_id: logger.warning("'pause' missing deck_id. Skipping."); return False
                deck = self._get_or_create_deck(deck_id)
                deck.pause()
                return False  # Engine convention: False = success
                
            elif command == "stop":
                if not deck_id: logger.warning("'stop' missing deck_id. Skipping."); return False
                deck = self._get_or_create_deck(deck_id)
                deck.stop()
                return False  # Engine convention: False = success
            
            elif command == "seek_and_play":
                if not deck_id: logger.warning("'seek_and_play' missing deck_id. Skipping."); return
                deck = self._get_or_create_deck(deck_id)
                
                # Check for different seek targets
                if "start_at_beat" in parameters:
                    start_beat = float(parameters["start_at_beat"])
                    deck.play(start_at_beat=start_beat)
                    logger.info(f"Seeking and playing deck {deck_id} from beat {start_beat}")
                elif "start_at_cue_name" in parameters:
                    cue_name = parameters["start_at_cue_name"]
                    deck.play(start_at_cue_name=cue_name)
                    logger.info(f"Seeking and playing deck {deck_id} from cue '{cue_name}'")
                elif "start_at_loop" in parameters:
                    loop_params = parameters["start_at_loop"]
                    start_beat = float(loop_params.get("start_at_beat"))
                    length_beats = float(loop_params.get("length_beats"))
                    repetitions = loop_params.get("repetitions")
                    
                    # Convert repetitions to proper format
                    if repetitions is not None:
                        if isinstance(repetitions, str) and repetitions.lower() == "infinite":
                            repetitions = None
                        else:
                            try:
                                repetitions = int(repetitions)
                            except ValueError:
                                logger.warning("Invalid repetitions value, defaulting to infinite")
                                repetitions = None
                    
                    deck.activate_loop(start_beat=start_beat, length_beats=length_beats, repetitions=repetitions)
                    logger.info(f"Seeking and playing deck {deck_id} with loop: beat {start_beat}, length {length_beats}, reps {repetitions}")
                else:
                    logger.warning("'seek_and_play' requires one of: 'start_at_beat', 'start_at_cue_name', or 'start_at_loop' in parameters. Skipping.")
                    return
            
            elif command == "stop_at_beat":
                if not deck_id: logger.warning("'stop_at_beat' missing deck_id. Skipping."); return
                if "beat_number" not in parameters: logger.warning("'stop_at_beat' missing 'beat_number' in parameters. Skipping."); return
                
                try:
                    beat_number = float(parameters["beat_number"])
                    if beat_number <= 0:
                        logger.warning("'beat_number' for 'stop_at_beat' must be positive. Skipping.")
                        return
                    
                    deck = self._get_or_create_deck(deck_id)
                    deck.stop_at_beat(beat_number)
                except ValueError:
                    logger.warning(f"Invalid 'beat_number' value for 'stop_at_beat': {parameters['beat_number']}. Skipping.")

            elif command == "activate_loop":
                if not deck_id: 
                    logger.warning("'activate_loop' missing deck_id. Skipping.")
                    return
                
                # Phase 2.2: Use ActionLoopAdapter instead of calling deck directly
                deck = self._get_or_create_deck(deck_id)
                
                if not hasattr(deck, 'action_adapter') or not deck.action_adapter:
                    logger.error(f"Deck {deck_id} has no ActionLoopAdapter - falling back to legacy method")
                    # Fall back to old method if adapter not available
                    return self._handle_legacy_activate_loop(action_dict)
                
                # Use ActionLoopAdapter to handle the activate_loop action
                success = deck.action_adapter.handle_activate_loop_action(action_dict)
                
                if success:
                    logger.info(f"Phase 2.2: Successfully processed activate_loop action via ActionLoopAdapter for deck {deck_id}")
                else:
                    logger.error(f"Phase 2.2: Failed to process activate_loop action via ActionLoopAdapter for deck {deck_id}")
                
                return False  # Engine convention: False = success

            elif command == "deactivate_loop":
                if not deck_id: 
                    logger.warning("'deactivate_loop' missing deck_id. Skipping.")
                    return False
                
                # Phase 2.2: Use ActionLoopAdapter instead of calling deck directly  
                deck = self._get_or_create_deck(deck_id)
                
                if not hasattr(deck, 'action_adapter') or not deck.action_adapter:
                    logger.warning(f"Deck {deck_id} has no ActionLoopAdapter - falling back to legacy method")
                    deck.deactivate_loop()
                    return False
                
                # Use ActionLoopAdapter to handle the deactivate_loop action
                success = deck.action_adapter.handle_deactivate_loop_action(action_dict)
                
                if success:
                    logger.info(f"Phase 2.2: Successfully processed deactivate_loop action via ActionLoopAdapter for deck {deck_id}")
                else:
                    logger.error(f"Phase 2.2: Failed to process deactivate_loop action via ActionLoopAdapter for deck {deck_id}")
                
                return False  # Engine convention: False = success
                
            elif command == "loop_completed":
                # Legacy loop completion system removed - now handled by musical timing system
                logger.warning(f"Legacy loop_completed command received for deck {deck_id} - ignoring")
                return False  # Engine convention: False = success
                
            elif command == "loop_repetition_complete":
                # Handle loop repetition completion events (legacy LoopManager removed)
                if not deck_id: logger.warning("'loop_repetition_complete' missing deck_id. Skipping."); return False
                action_id = parameters.get("action_id")
                repetition = parameters.get("repetition")
                total_repetitions = parameters.get("total_repetitions")
                
                if not action_id: logger.warning("'loop_repetition_complete' missing action_id. Skipping."); return False
                
                logger.info(f"AudioEngine - Processing loop repetition completion: deck {deck_id}, action {action_id}, repetition {repetition}/{total_repetitions}")
                
                # Legacy LoopManager removed - loop completion handled by frame-accurate system
                deck = self._get_or_create_deck(deck_id)
                logger.info(f"AudioEngine - Loop repetition completion handled by frame-accurate system for deck {deck_id}")
                # Note: Actual completion logic is now handled directly in the deck's audio thread
                        
                return False  # Engine convention: False = success
            
            elif command == "set_tempo":
                deck_id = action_dict.get("deck_id")
                deck = self._get_or_create_deck(deck_id)
                target_bpm = float(action_dict.get("parameters", {}).get("target_bpm"))
                deck.beat_manager.handle_tempo_change(target_bpm)
                logger.info(f"Set tempo for deck {deck_id} to {target_bpm} BPM via BeatManager")
            elif command == "set_pitch":
                deck_id = action_dict.get("deck_id")
                deck = self._get_or_create_deck(deck_id)
                semitones = float(action_dict.get("parameters", {}).get("semitones"))
                deck.set_pitch(semitones)
            
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
            
            elif command == "set_eq":
                deck_id = action_dict.get("deck_id")
                deck = self._get_or_create_deck(deck_id)
                params = action_dict.get("parameters", {})
                low = params.get("low")
                mid = params.get("mid")
                high = params.get("high")
                deck.set_eq(low=low, mid=mid, high=high)
            
            elif command == "fade_eq":
                deck_id = action_dict.get("deck_id")
                deck = self._get_or_create_deck(deck_id)
                params = action_dict.get("parameters", {})
                target_low = params.get("target_low")
                target_mid = params.get("target_mid")
                target_high = params.get("target_high")
                duration_seconds = float(params.get("duration_seconds", 1.0))
                deck.fade_eq(target_low=target_low, target_mid=target_mid, target_high=target_high, duration_seconds=duration_seconds)

            elif command == "set_stem_eq":
                deck_id = action_dict.get("deck_id")
                deck = self._get_or_create_deck(deck_id)
                params = action_dict.get("parameters", {})
                stem = params.get("stem")
                
                # Support both formats: new professional API (low_db, mid_db, high_db) and legacy (low, mid, high)
                if 'low_db' in params or 'mid_db' in params or 'high_db' in params:
                    # Professional dB format
                    low_db = params.get("low_db", 0.0)
                    mid_db = params.get("mid_db", 0.0) 
                    high_db = params.get("high_db", 0.0)
                    enabled = params.get("enabled", None)
                    result = deck.set_stem_eq(stem, low_db=low_db, mid_db=mid_db, high_db=high_db, enabled=enabled)
                    if result:
                        logger.info(f"Set {stem} professional EQ: L={low_db:+.1f}dB M={mid_db:+.1f}dB H={high_db:+.1f}dB")
                    else:
                        logger.warning(f"Failed to set {stem} professional EQ")
                else:
                    # Legacy linear gain format - convert to dB
                    low = params.get("low", 1.0)
                    mid = params.get("mid", 1.0) 
                    high = params.get("high", 1.0)
                    
                    # Convert linear gains to dB (1.0 = 0dB, 2.0 = +6dB, etc.)
                    import math
                    low_db = 20 * math.log10(max(0.01, low)) if low > 0 else -40.0
                    mid_db = 20 * math.log10(max(0.01, mid)) if mid > 0 else -40.0
                    high_db = 20 * math.log10(max(0.01, high)) if high > 0 else -40.0
                    
                    result = deck.set_stem_eq(stem, low_db=low_db, mid_db=mid_db, high_db=high_db, enabled=True)
                    if result:
                        logger.info(f"Set {stem} EQ (converted): L={low}→{low_db:+.1f}dB M={mid}→{mid_db:+.1f}dB H={high}→{high_db:+.1f}dB")
                    else:
                        logger.warning(f"Failed to set {stem} EQ")

            elif command == "enable_stem_eq":
                deck_id = action_dict.get("deck_id")
                deck = self._get_or_create_deck(deck_id)
                params = action_dict.get("parameters", {})
                stem = params.get("stem")
                enabled = params.get("enabled")
                result = deck.enable_stem_eq(stem, enabled)
                if result:
                    logger.info(f"{stem} EQ {'enabled' if enabled else 'disabled'}")
                else:
                    logger.warning(f"Failed to {'enable' if enabled else 'disable'} {stem} EQ")

            elif command == "set_stem_volume":
                deck_id = action_dict.get("deck_id")
                deck = self._get_or_create_deck(deck_id)
                params = action_dict.get("parameters", {})
                stem = params.get("stem")
                volume = float(params.get("volume"))
                result = deck.set_stem_volume(stem, volume)
                if result:
                    logger.info(f"Set {stem} volume: {volume}")
                else:
                    logger.warning(f"Failed to set {stem} volume")

            elif command == "set_master_eq":
                deck_id = action_dict.get("deck_id")
                deck = self._get_or_create_deck(deck_id)
                params = action_dict.get("parameters", {})
                low_gain = params.get("low_gain", 1.0)
                mid_gain = params.get("mid_gain", 1.0) 
                high_gain = params.get("high_gain", 1.0)
                enabled = params.get("enabled", None)
                result = deck.set_master_eq(low_gain=low_gain, mid_gain=mid_gain, high_gain=high_gain, enabled=enabled)
                if result:
                    logger.info(f"Set master EQ: L={low_gain:.2f}, M={mid_gain:.2f}, H={high_gain:.2f}, enabled={enabled}")
                else:
                    logger.warning("Failed to set master EQ")

            elif command == "set_all_stem_eq":
                deck_id = action_dict.get("deck_id")
                deck = self._get_or_create_deck(deck_id)
                params = action_dict.get("parameters", {})
                
                # Filter out only stem settings
                valid_stems = ["vocals", "drums", "bass", "other"]
                stem_settings = {k: v for k, v in params.items() if k in valid_stems}
                
                results = deck.set_all_stem_eq(stem_settings)
                success_count = sum(1 for success in results.values() if success)
                total_count = len(results)
                logger.info(f"Set all stem EQ: {success_count}/{total_count} stems configured successfully")
                return False  # Engine convention: False = success
            
            elif command == "play_sample":
                params = action_dict.get("parameters", {})
                sample_id = params.get("sample_id")
                volume = float(params.get("volume", 1.0))
                key_match_deck = params.get("key_match_deck")
                repetitions = params.get("repetitions")  # New parameter for looping
                
                if not sample_id:
                    logger.warning("'play_sample' missing 'sample_id'. Skipping.")
                    return False
                
                # Global playback with optional key matching and looping
                logger.debug(f"Playing sample globally: {sample_id} (volume: {volume}, key_match: {key_match_deck}, repetitions: {repetitions})")
                success = self._play_global_sample(sample_id=sample_id, volume=volume, key_match_deck=key_match_deck, repetitions=repetitions)
                if not success:
                    logger.warning(f"Failed to play global sample: {sample_id}")
                
                return False  # Engine convention: False = success
            
            elif command == "stop_sample":
                # Stop any currently playing global samples
                if hasattr(self, '_global_scratch_sample_active'):
                    self._global_scratch_sample_active = False
                    self._global_sample_loop_active = False
                    self._global_sample_loop_repetitions = None
                    self._global_sample_loop_repetitions_done = 0
                logger.debug("Stopped global sample playback")
                return False  # Engine convention: False = success
            
            elif command == "load_sample":
                sample_filepath = parameters.get("sample_filepath")
                sample_id = parameters.get("sample_id")
                pitch_semitones = parameters.get("pitch_semitones")
                
                if not sample_filepath:
                    logger.error("load_sample: Missing sample_filepath parameter")
                    return False
                
                if not sample_id:
                    logger.error("load_sample: Missing sample_id parameter")
                    return False
                
                # Global sample - store in engine for pre-processing
                if not hasattr(self, '_global_samples'):
                    self._global_samples = {}
                self._global_samples[sample_id] = {
                    'filepath': sample_filepath,
                    'pitch_semitones': pitch_semitones
                }
                return False  # Engine convention: False = success
            
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
                
                # CRITICAL FIX: Don't stop the source deck after crossfade
                # Just leave it playing at 0 volume so it can be faded back up later if needed
                logger.info(f"🎚️ CROSSFADE: {from_deck_id}→{to_deck_id} over {duration_seconds}s ({from_deck_id}: 100%→0%, {to_deck_id}: 0%→100%)")
                return False  # Engine convention: False = success
            
            elif command == "bpm_match":
                params = action_dict.get("parameters", {})
                reference_deck_id = params.get("reference_deck")
                follow_deck_id = params.get("follow_deck")
                
                if not reference_deck_id or not follow_deck_id:
                    logger.error(f"bpm_match missing reference_deck or follow_deck")
                    return False
                
                reference_deck = self._get_or_create_deck(reference_deck_id)
                follow_deck = self._get_or_create_deck(follow_deck_id)
                
                # Use BeatManager for consistent BPM access
                reference_bpm = reference_deck.beat_manager.get_bpm()
                follow_bpm = follow_deck.beat_manager.get_bpm()
                
                logger.info(f"BPM-matching {follow_deck_id} to reference {reference_deck_id}")
                logger.info(f"Reference: {reference_bpm} BPM, Follow: {follow_bpm} BPM")
                
                if reference_bpm <= 0 or follow_bpm <= 0:
                    logger.error(f"Invalid BPM for bpm_match")
                    return False
                
                # Use BeatManager for tempo changes (replaces deprecated set_tempo)
                follow_deck.beat_manager.handle_tempo_change(reference_bpm)
                logger.info(f"BPM match complete: {follow_deck_id} synchronized to {reference_bpm} BPM via BeatManager")
                return False  # Engine convention: False = success
            
            elif command == "ramp_tempo":
                deck_id = action_dict.get("deck_id")
                deck = self._get_or_create_deck(deck_id)
                params = action_dict.get("parameters", {})
                start_beat = float(params.get("start_beat"))
                end_beat = float(params.get("end_beat"))
                start_bpm = float(params.get("start_bpm"))
                end_bpm = float(params.get("end_bpm"))
                curve = params.get("curve", "linear")
                
                # Calculate tempo change over the beat range
                beat_duration = end_beat - start_beat
                if beat_duration <= 0:
                    logger.error("Invalid beat range for tempo ramp")
                    return False
                
                # Use BeatManager for tempo ramp (replaces deprecated set_tempo)
                # Calculate ramp duration in beats
                ramp_duration_beats = end_beat - start_beat
                
                # Use BeatManager's built-in tempo ramp functionality
                deck.beat_manager.handle_tempo_change(end_bpm, ramp_duration_beats)
                
                logger.info(f"Tempo ramp scheduled via BeatManager: {start_bpm}→{end_bpm} BPM over {ramp_duration_beats} beats")
                return False  # Engine convention: False = success
            
            # Add default return for unhandled commands
            else:
                logger.warning(f"Unhandled command: {command}")
                return False  # Engine convention: False = success
                
        except Exception as e:
            logger.error(f"Error executing action {command}: {e}")
            return False  # Engine convention: False = success (even on error, we don't want to fail the script)

    def shutdown_decks(self):
        logger.info("AudioEngine - Shutting down all decks...")
        if not self.decks: logger.info("AudioEngine - No decks to shut down."); return
        
        for deck_id in list(self.decks.keys()):
            logger.info(f"AudioEngine - Requesting shutdown for {deck_id}...")
            self.decks[deck_id].shutdown()
        
        logger.info("AudioEngine - All decks have been requested to shut down.")
        self.decks.clear()
        logger.info("AudioEngine - Deck shutdown process complete.")

    def any_deck_active(self):
        if not self.decks: return False
        for deck in self.decks.values():
            if deck.is_active(): return True
        return False
    
    def get_event_scheduler_stats(self):
        """Get statistics from the event scheduler"""
        if hasattr(self, 'event_scheduler'):
            return self.event_scheduler.get_stats()
        return {}
    
    def get_audio_clock_state(self):
        """Get current audio clock state"""
        if hasattr(self, 'audio_clock'):
            return self.audio_clock.get_state()
        return {}
        
    def get_global_timing_info(self):
        """
        Get global timing information from all decks using BeatManager.
        This provides a centralized view of timing across the entire system.
        
        Returns:
            Dict with global timing information
        """
        timing_info = {
            "total_decks": len(self.decks),
            "active_decks": 0,
            "global_bpm": None,
            "deck_timing": {},
            "synchronization_status": "unknown"
        }
        
        if not self.decks:
            timing_info["synchronization_status"] = "no_decks"
            return timing_info
            
        # Collect timing information from all decks
        active_decks = 0
        bpms = []
        
        for deck_id, deck in self.decks.items():
            if hasattr(deck, 'beat_manager'):
                bpm = deck.beat_manager.get_bpm()
                current_beat = deck.beat_manager.get_current_beat()
                current_frame = deck.beat_manager.get_current_frame()
                
                deck_info = {
                    "bpm": bpm,
                    "current_beat": current_beat,
                    "current_frame": current_frame,
                    "has_tempo_ramp": deck.beat_manager.is_tempo_ramp_active(),
                    "is_active": deck.is_active() if hasattr(deck, 'is_active') else False
                }
                
                timing_info["deck_timing"][deck_id] = deck_info
                
                if deck_info["is_active"]:
                    active_decks += 1
                    if bpm > 0:
                        bpms.append(bpm)
            else:
                # Fallback for decks without BeatManager
                timing_info["deck_timing"][deck_id] = {
                    "bpm": getattr(deck, 'bpm', 0),
                    "current_beat": 0,
                    "current_frame": 0,
                    "has_tempo_ramp": False,
                    "is_active": deck.is_active() if hasattr(deck, 'is_active') else False
                }
        
        timing_info["active_decks"] = active_decks
        
        # Calculate global BPM if we have active decks with valid BPMs
        if bpms:
            # Use the most common BPM or average if they're close
            bpms.sort()
            if len(bpms) == 1 or (bpms[-1] - bpms[0]) < 5.0:  # Within 5 BPM
                timing_info["global_bpm"] = sum(bpms) / len(bpms)
                timing_info["synchronization_status"] = "synchronized"
            else:
                timing_info["global_bpm"] = bpms[0]  # Use first BPM as reference
                timing_info["synchronization_status"] = "desynchronized"
        else:
            timing_info["synchronization_status"] = "no_active_decks"
            
        return timing_info
        
    def synchronize_all_decks_to_bpm(self, target_bpm: float, ramp_duration_beats: float = 0.0):
        """
        Synchronize all active decks to a common BPM using BeatManager.
        This ensures consistent timing across the entire system.
        
        Args:
            target_bpm: Target BPM for all decks
            ramp_duration_beats: Duration of tempo ramp in beats (0 = instant change)
            
        Returns:
            Dict with synchronization results
        """
        if target_bpm <= 0:
            logger.error(f"Invalid target BPM: {target_bpm}")
            return {"success": False, "error": "Invalid target BPM"}
            
        results = {
            "target_bpm": target_bpm,
            "ramp_duration_beats": ramp_duration_beats,
            "decks_synchronized": 0,
            "decks_failed": 0,
            "details": {}
        }
        
        for deck_id, deck in self.decks.items():
            try:
                if hasattr(deck, 'beat_manager'):
                    # Use BeatManager for tempo synchronization
                    deck.beat_manager.handle_tempo_change(target_bpm, ramp_duration_beats)
                    results["decks_synchronized"] += 1
                    results["details"][deck_id] = {
                        "success": True,
                        "method": "BeatManager",
                        "previous_bpm": deck.beat_manager.get_bpm()
                    }
                    logger.info(f"Deck {deck_id} synchronized to {target_bpm} BPM via BeatManager")
                else:
                    # Fallback for decks without BeatManager
                    if hasattr(deck, 'set_tempo'):
                        deck.set_tempo(target_bpm)
                        results["decks_synchronized"] += 1
                        results["details"][deck_id] = {
                            "success": True,
                            "method": "legacy_set_tempo",
                            "previous_bpm": getattr(deck, 'bpm', 0)
                        }
                        logger.info(f"Deck {deck_id} synchronized to {target_bpm} BPM via legacy method")
                    else:
                        results["decks_failed"] += 1
                        results["details"][deck_id] = {
                            "success": False,
                            "error": "No tempo control method available"
                        }
                        logger.warning(f"Deck {deck_id} has no tempo control method")
                        
            except Exception as e:
                results["decks_failed"] += 1
                results["details"][deck_id] = {
                    "success": False,
                    "error": str(e)
                }
                logger.error(f"Failed to synchronize deck {deck_id}: {e}")
                
        logger.info(f"Global synchronization complete: {results['decks_synchronized']} decks synchronized, {results['decks_failed']} failed")
        return results

    def _play_global_sample(self, sample_id, volume=1.0, key_match_deck=None, repetitions=None):
        """Play a pre-processed global sample with optional key matching and looping"""
        try:
            # Check if sample exists in cache
            if not hasattr(self, '_global_sample_cache'):
                logger.error(f"AudioEngine - Global sample cache not initialized")
                return False
            
            if sample_id not in self._global_sample_cache:
                logger.error(f"AudioEngine - Sample '{sample_id}' not found in cache. Available samples: {list(self._global_sample_cache.keys())}")
                return False
            
            # Get the cached sample
            sample_audio = self._global_sample_cache[sample_id].copy()
            
            # Apply key matching if requested
            if key_match_deck:
                if key_match_deck not in self.decks:
                    logger.error(f"AudioEngine - Key match deck '{key_match_deck}' not found")
                    return False
                
                deck = self.decks[key_match_deck]
                track_key_info = deck.get_track_key()
                track_key = track_key_info['key']
                
                # Calculate pitch shift needed for key matching
                # This would require analyzing the sample's key first
                # For now, we'll implement a simple key matching system
                logger.info(f"AudioEngine - Key matching sample '{sample_id}' to deck '{key_match_deck}' (track key: {track_key})")
                
                # TODO: Implement key matching logic here
                # For now, just log that key matching was requested
                logger.warning(f"AudioEngine - Key matching not yet implemented for sample '{sample_id}'")
            
            # Apply volume (no hardcoded reduction)
            sample_audio *= volume
            
            # Apply looping if specified
            if repetitions is not None:
                if repetitions == "infinite":
                    repetitions = None # essentia's loop_samples expects None for infinite
                else:
                    try:
                        repetitions = int(repetitions)
                        if repetitions <= 0:
                            repetitions = None # essentia's loop_samples expects None for non-positive
                    except ValueError:
                        logger.warning(f"Invalid 'repetitions' value for 'play_sample': {repetitions}. Defaulting to infinite.")
                        repetitions = None

            if repetitions is not None:
                # Store for global playback
                self._global_scratch_sample_audio = sample_audio
                self._global_scratch_sample_position = 0
                self._global_scratch_sample_active = True
                self._global_sample_loop_active = True
                self._global_sample_loop_repetitions = repetitions
                self._global_sample_loop_repetitions_done = 0
                
                logger.debug(f"AudioEngine - Playing global sample with loop: {sample_id} (volume: {volume}, key_match: {key_match_deck}, repetitions: {repetitions})")
                return True
            else:
                # Store for global playback
                self._global_scratch_sample_audio = sample_audio
                self._global_scratch_sample_position = 0
                self._global_scratch_sample_active = True
                
                logger.debug(f"AudioEngine - Playing global sample: {sample_id} (volume: {volume}, key_match: {key_match_deck})")
                return True
            
        except Exception as e:
            logger.error(f"AudioEngine - Error playing global sample '{sample_id}': {e}")
            return False

    def _load_global_sample(self, sample_filepath):
        """Load a sample globally for playback"""
        try:
            import essentia.standard as es
            import librosa
            import numpy as np
            
            # Load the sample audio
            loader = es.MonoLoader(filename=sample_filepath)
            sample_audio = loader()
            sample_sr = int(loader.paramValue('sampleRate'))
            
            # Resample if needed to match standard sample rate (44100)
            if sample_sr != 44100:
                sample_audio = librosa.resample(
                    sample_audio,
                    orig_sr=sample_sr,
                    target_sr=44100
                )
            
            # Store for global playback
            self._global_scratch_sample_audio = sample_audio
            self._global_scratch_sample_position = 0
            self._global_scratch_sample_active = True
            
            logger.debug(f"AudioEngine - Loaded global scratch sample: {os.path.basename(sample_filepath)}")
            return True
            
        except Exception as e:
            logger.error(f"AudioEngine - Error loading global scratch sample: {e}")
            return False
    
    def _handle_legacy_activate_loop(self, action_dict: dict) -> bool:
        """
        Legacy fallback method for activate_loop when ActionLoopAdapter is not available.
        This preserves the old behavior for compatibility.
        """
        try:
            deck_id = action_dict.get("deck_id")
            parameters = action_dict.get("parameters", {})
            
            # Parameters from JSON (using .get for safety)
            start_beat_from_json = parameters.get("start_at_beat")
            length_beats_from_json = parameters.get("length_beats")
            repetitions_param_from_json = parameters.get("repetitions") 

            if start_beat_from_json is None or length_beats_from_json is None: 
                logger.warning("'activate_loop' requires 'start_at_beat' and 'length_beats' in parameters. Skipping.")
                return False
                
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
                            logger.warning("'repetitions' for 'activate_loop' must be positive int or 'infinite'. Defaulting to infinite.")
                            repetitions_val = None
                    except ValueError: 
                        logger.warning("Invalid 'repetitions' value for 'activate_loop'. Must be int or 'infinite'. Defaulting to infinite.")
                        repetitions_val = None
            
            if length_beats_val <= 0:
                logger.warning("'length_beats' for 'activate_loop' must be positive. Skipping.")
                return False
                
            deck = self._get_or_create_deck(deck_id)
            # Pass the action ID to the deck (legacy method)
            deck.activate_loop(start_beat=start_beat_val, 
                               length_beats=length_beats_val, 
                               repetitions=repetitions_val,
                               action_id=action_dict.get('action_id'))
            return True  # Success
            
        except Exception as e: 
            logger.error(f"Legacy activate_loop failed: {e}")
            return False


if __name__ == '__main__':
    logger.info("--- AudioEngine Standalone Test (v5.1 - Corrected activate_loop call) ---")
    
    CURRENT_DIR_OF_ENGINE_PY = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT_FOR_ENGINE_TEST = os.path.dirname(CURRENT_DIR_OF_ENGINE_PY) 
    
    if PROJECT_ROOT_FOR_ENGINE_TEST not in sys.path:
        sys.path.insert(0, PROJECT_ROOT_FOR_ENGINE_TEST)
    
    try:
        import config as main_test_app_config 
    except ImportError:
        logger.critical("engine.py __main__ - config.py not found in project root. Cannot run test.")
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
        logger.warning(f"Test audio file '{full_test_audio_path}' not found for engine test.")
    
    dummy_cue_filepath = os.path.join(main_test_app_config.AUDIO_TRACKS_DIR, test_audio_filename + ".cue")
    test_cues = {"drop1": {"start_beat": 65}, "intro_start": {"start_beat":1}, 
                 "beat_5":{"start_beat":5}, "beat_10": {"start_beat": 10}, # For simplified test
                 "beat_17": {"start_beat": 17}, "beat_20": {"start_beat": 20},
                 "beat_28": {"start_beat":28}, "beat_33":{"start_beat":33},
                 "beat_66":{"start_beat":66}, "beat_97":{"start_beat":97}}
    try:
        with open(dummy_cue_filepath, 'w') as f: json.dump(test_cues, f, indent=4)
        logger.info(f"Created/Updated placeholder cue file: {dummy_cue_filepath}")
    except Exception as e_cue_create: logger.error(f"Could not create placeholder cue file: {e_cue_create}")

    # Using the simplified test JSON that worked for deck.py to isolate engine logic
    test_json_content = {
        "script_name": "Engine Simplified Loop Test v5.1",
        "actions": [
            {"id": "loadA", "command": "load_track", "deck_id": "deckA", "track_id": "track1"},
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
        ],
        "tracks": [
            {"track_id": "track1", "filepath": full_test_audio_path}
        ]
    }
    
    try:
        with open(dummy_json_path, 'w') as f: json.dump(test_json_content, f, indent=4)
        logger.info(f"Created dummy JSON script for testing: {dummy_json_path}")
    except Exception as e_json_write: logger.error(f"Error creating dummy JSON: {e_json_write}"); sys.exit(1)

    if engine.load_script_from_file(dummy_json_path): 
        engine.start_script_processing() 
        
        logger.info("Engine Test - Script processing started. Monitoring. Press Ctrl+C to stop.")
        start_wait_time = time.time()
        max_script_duration_estimate = 25 # Adjusted for the shorter test
        
        try:
            while True: 
                engine_is_processing = engine.is_processing_script_actions 
                decks_are_active = engine.any_deck_active()

                if not engine_is_processing and not decks_are_active:
                    # Check if event scheduler has any pending events
                    if hasattr(engine, 'event_scheduler') and engine.event_scheduler:
                        scheduler_stats = engine.event_scheduler.get_stats()
                        pending_events = scheduler_stats.get('queue', {}).get('total', {}).get('total_scheduled', 0) - scheduler_stats.get('queue', {}).get('total', {}).get('total_executed', 0)
                        if pending_events == 0:
                            logger.info("Engine Test - Engine done, no pending events, and no decks active.")
                            break
                    else:
                        logger.info("Engine Test - Engine done, no event scheduler, and no decks active.")
                        break 
                
                if (time.time() - start_wait_time) > max_script_duration_estimate:
                    logger.warning(f"Engine Test - Max wait time of {max_script_duration_estimate}s reached.")
                    # Log event scheduler status on timeout
                    if hasattr(engine, 'event_scheduler') and engine.event_scheduler:
                        scheduler_stats = engine.event_scheduler.get_stats()
                        pending_events = scheduler_stats.get('queue', {}).get('total', {}).get('total_scheduled', 0) - scheduler_stats.get('queue', {}).get('total', {}).get('total_executed', 0)
                        if pending_events > 0:
                            logger.debug(f"Engine Test - Still {pending_events} pending events on timeout")
                            logger.debug(f"Event scheduler stats: {scheduler_stats}")
                    break
                time.sleep(0.5)
            logger.info("Engine Test - Monitoring loop finished or timed out.")
        except KeyboardInterrupt:
            logger.info("Engine Test - KeyboardInterrupt received by test runner.")
        finally:
            engine.stop_script_processing() 
    
    logger.info("--- AudioEngine Standalone Test Finished ---")