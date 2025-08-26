# dj-gemini/audio_engine/engine_fixed.py

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
        self.event_scheduler = EventScheduler(self.audio_clock)
        
        # Event-driven architecture replaces old polling system
        self._all_actions_from_script = []
        self.is_processing_script_actions = False 
        
        self._script_start_time = 0 
        
        # NEW: Beat-triggered actions storage
        self._beat_triggered_actions = {}
        
        logger.debug("AudioEngine - Initialized.")
        
        # Register event handlers for different action types
        self._register_event_handlers()

    def _schedule_action(self, action: dict):
        """Schedule a single action based on its trigger type"""
        trigger = action.get("trigger", {})
        trigger_type = trigger.get("type")
        
        if trigger_type == "script_start":
            # Execute immediately
            event_id = self.event_scheduler.schedule_immediate_action(
                action, priority=100  # High priority for immediate actions
            )
            logger.debug(f"AudioEngine - Scheduled immediate action: {event_id}")
            
        elif trigger_type == "on_deck_beat":
            # Beat-triggered actions - store for later execution when deck reaches the beat
            source_deck_id = trigger.get("source_deck_id")
            target_beat = trigger.get("beat_number")
            
            if source_deck_id and target_beat is not None:
                # Store the action to be executed when the deck reaches this beat
                if source_deck_id not in self._beat_triggered_actions:
                    self._beat_triggered_actions[source_deck_id] = {}
                
                if target_beat not in self._beat_triggered_actions[source_deck_id]:
                    self._beat_triggered_actions[source_deck_id][target_beat] = []
                
                self._beat_triggered_actions[source_deck_id][target_beat].append(action)
                logger.debug(f"AudioEngine - Stored beat-triggered action for deck {source_deck_id} at beat {target_beat}")
            else:
                logger.error(f"Invalid on_deck_beat trigger: {trigger}")
                return
                
        elif trigger_type == "on_loop_complete":
            # This will be handled dynamically when loops complete
            # For now, schedule with a placeholder time
            event_id = self.event_scheduler.schedule_action(
                action, float('inf'), priority=30
            )
            logger.debug(f"AudioEngine - Scheduled loop completion action: {event_id} (placeholder)")
            
        else:
            logger.warning(f"Unknown trigger type: {trigger_type}")

    def _check_beat_triggers(self, deck_id: str, current_beat: float):
        """Check if any beat-triggered actions should execute for the current beat"""
        if deck_id not in self._beat_triggered_actions:
            return
        
        # Check if we've reached any target beats
        target_beats = list(self._beat_triggered_actions[deck_id].keys())
        for target_beat in target_beats:
            if current_beat >= target_beat:
                # Execute all actions for this beat
                actions = self._beat_triggered_actions[deck_id][target_beat]
                logger.info(f"🎯 Beat trigger MET: deck {deck_id} reached beat {target_beat}, executing {len(actions)} actions")
                
                for action in actions:
                    try:
                        logger.info(f"🎯 Executing beat-triggered action: {action.get('action_id', 'unknown')}")
                        self._execute_action(action)
                    except Exception as e:
                        logger.error(f"Error executing beat-triggered action: {e}")
                
                # Remove the executed actions
                del self._beat_triggered_actions[deck_id][target_beat]
                
                # Clean up empty deck entries
                if not self._beat_triggered_actions[deck_id]:
                    del self._beat_triggered_actions[deck_id]

    def _execute_action(self, action_dict):
        """Execute a single action - this is the core action execution logic"""
        # Implementation would go here - same as original engine
        pass

    # ... rest of the engine implementation would go here
