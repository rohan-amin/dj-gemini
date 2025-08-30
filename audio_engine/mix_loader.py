"""
Mix Configuration Loader for Frame-Accurate Musical Timing System

This module loads mix configuration JSON files and pre-schedules all actions
into the musical timing system for frame-accurate execution, bypassing the
old event scheduler entirely.
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class MixConfigLoader:
    """
    Loads mix configurations and schedules actions with frame-accurate timing.
    
    This replaces the old event scheduler for mix configurations by pre-scheduling
    all actions into the musical timing system at startup.
    """
    
    def __init__(self, engine):
        """
        Initialize mix loader with engine reference.
        
        Args:
            engine: Audio engine instance with access to decks
        """
        self.engine = engine
        self.loaded_mix = None
        self.scheduled_actions = {}  # Track scheduled action IDs
        
    def load_mix_config(self, json_filepath: str) -> bool:
        """
        Load mix configuration from JSON file and pre-schedule all actions.
        
        Args:
            json_filepath: Path to JSON mix configuration file
            
        Returns:
            True if successfully loaded and scheduled
        """
        try:
            # Load JSON configuration
            if not os.path.exists(json_filepath):
                logger.error(f"Mix config file not found: {json_filepath}")
                return False
            
            with open(json_filepath, 'r') as f:
                mix_config = json.load(f)
            
            logger.info(f"Loading mix configuration: {mix_config.get('mix_name', 'Unknown')}")
            
            self.loaded_mix = mix_config
            
            # Process all actions in the mix
            actions = mix_config.get('actions', [])
            
            # Separate actions by trigger type
            immediate_actions = [action for action in actions if self._is_immediate_action(action)]
            beat_actions = [action for action in actions if self._is_beat_triggered_action(action)]
            loop_complete_actions = [action for action in actions if self._is_loop_complete_action(action)]
            other_actions = [action for action in actions if not self._is_immediate_action(action) and 
                           not self._is_beat_triggered_action(action) and not self._is_loop_complete_action(action)]
            
            logger.info(f"Found {len(immediate_actions)} immediate actions, {len(beat_actions)} beat-triggered actions, "
                       f"{len(loop_complete_actions)} loop-complete actions, and {len(other_actions)} other actions")
            
            # DEBUG: Show exactly which actions are in each category
            logger.info(f"Immediate actions: {[action.get('action_id') for action in immediate_actions]}")
            logger.info(f"Beat actions: {[action.get('action_id') for action in beat_actions]}")
            logger.info(f"Loop-complete actions: {[action.get('action_id') for action in loop_complete_actions]}")
            logger.info(f"Other actions: {[action.get('action_id') for action in other_actions]}")
            
            # Execute immediate actions in proper order: load_track first, then play actions
            load_actions = [action for action in immediate_actions if action.get('command') == 'load_track']
            play_actions = [action for action in immediate_actions if action.get('command') == 'play']
            other_immediate = [action for action in immediate_actions if action.get('command') not in ['load_track', 'play']]
            
            # First, load all tracks
            for action in load_actions:
                self._execute_immediate_action(action)
            
            # Wait for beat data to be available
            if load_actions:
                logger.info("Waiting for beat data to be available after track loading...")
                import time
                time.sleep(0.3)  # Give more time for beat analysis
            
            # Then execute play actions (with beat data now available)
            logger.info(f"🎵 Executing {len(play_actions)} play actions...")
            for action in play_actions:
                logger.info(f"🎵 About to execute play action: {action.get('action_id')}")
                self._execute_immediate_action(action)
            
            # Finally, other immediate actions
            for action in other_immediate:
                self._execute_immediate_action(action)
            
            # Pre-schedule all beat-triggered actions 
            self._schedule_beat_actions(beat_actions)
            
            # Handle loop completion actions (for now, just log them)
            for action in loop_complete_actions:
                trigger = action.get('trigger', {})
                logger.warning(f"Loop completion trigger not yet implemented for action {action.get('action_id')}: "
                             f"depends on loop {trigger.get('loop_action_id')} from deck {trigger.get('source_deck_id')}")
            
            # Handle other trigger types (for now, just log them)
            for action in other_actions:
                trigger = action.get('trigger', {})
                logger.warning(f"Unsupported trigger type for action {action.get('action_id')}: {trigger.get('type')}")
            
            logger.info(f"Mix configuration loaded successfully: {len(self.scheduled_actions)} actions scheduled")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load mix configuration from {json_filepath}: {e}")
            return False
    
    def _is_immediate_action(self, action: dict) -> bool:
        """Check if action should be executed immediately at startup"""
        trigger = action.get('trigger', {})
        return trigger.get('type') == 'script_start'
    
    def _is_beat_triggered_action(self, action: dict) -> bool:
        """Check if action is triggered by a beat event"""
        trigger = action.get('trigger', {})
        return trigger.get('type') == 'on_deck_beat'
    
    def _is_loop_complete_action(self, action: dict) -> bool:
        """Check if action is triggered by loop completion"""
        trigger = action.get('trigger', {})
        return trigger.get('type') == 'on_loop_complete'
    
    def _execute_immediate_action(self, action: dict) -> None:
        """Execute actions that should happen immediately at startup"""
        try:
            command = action.get('command')
            deck_id = action.get('deck_id')
            parameters = action.get('parameters', {})
            action_id = action.get('action_id')
            
            logger.info(f"🚀 _execute_immediate_action called for: {action_id} ({command}) on deck {deck_id}")
            logger.info(f"🚀 Parameters: {parameters}")
            
            if command == 'load_track':
                filepath = parameters.get('filepath')
                if deck_id and filepath:
                    deck = self.engine._get_or_create_deck(deck_id)
                    if deck:
                        success = deck.load_track(filepath)
                        if success:
                            logger.info(f"Loaded track on {deck_id}: {filepath}")
                        else:
                            logger.error(f"Failed to load track on {deck_id}: {filepath}")
                    else:
                        logger.error(f"Deck not found: {deck_id}")
            
            elif command == 'play':
                logger.info(f"🎵 EXECUTING PLAY ACTION: {action_id} on {deck_id}")
                if deck_id:
                    deck = self.engine._get_or_create_deck(deck_id)
                    if deck:
                        start_at_beat = parameters.get('start_at_beat')
                        logger.info(f"🎵 Play parameters: start_at_beat={start_at_beat}")
                        
                        if start_at_beat:
                            # Use the deck's native start_at_beat parameter - much simpler!
                            try:
                                logger.info(f"🎵 Calling deck.play(start_at_beat={start_at_beat})")
                                result = deck.play(start_at_beat=start_at_beat)
                                logger.info(f"🎵 deck.play() result: {result}")
                                logger.info(f"✅ Started playback on {deck_id} at beat {start_at_beat}")
                                
                            except Exception as e:
                                logger.error(f"❌ Error starting playback at beat {start_at_beat}: {e}")
                                import traceback
                                traceback.print_exc()
                                logger.info(f"🔄 Falling back to normal play for {deck_id}")
                                deck.play()  # Fallback to normal play
                        else:
                            logger.info(f"🎵 No start_at_beat specified, using normal play")
                            deck.play()
                    else:
                        logger.error(f"❌ Failed to get/create deck: {deck_id}")
                else:
                    logger.error(f"❌ No deck_id specified for play action: {action_id}")
            
            elif command == 'bpm_match':
                # Handle BPM matching immediately 
                ref_deck_id = parameters.get('reference_deck')
                follow_deck_id = parameters.get('follow_deck')
                if ref_deck_id and follow_deck_id:
                    ref_deck = self.engine._get_or_create_deck(ref_deck_id)
                    follow_deck = self.engine._get_or_create_deck(follow_deck_id)
                    if ref_deck and follow_deck:
                        # This would need to be implemented in the engine
                        logger.info(f"BPM match: {follow_deck_id} -> {ref_deck_id}")
            
        except Exception as e:
            logger.error(f"Error executing immediate action {action.get('action_id')}: {e}")
    
    def _schedule_beat_actions(self, beat_actions: List[dict]) -> None:
        """Pre-schedule all beat-triggered actions into the musical timing system"""
        
        for action in beat_actions:
            try:
                trigger = action.get('trigger', {})
                beat_number = trigger.get('beat_number')
                source_deck_id = trigger.get('source_deck_id')
                
                if not beat_number or not source_deck_id:
                    logger.warning(f"Skipping action {action.get('action_id')} - missing beat_number or source_deck_id")
                    continue
                
                # Get the deck that will execute this action
                target_deck_id = action.get('deck_id', source_deck_id)
                deck = self.engine._get_or_create_deck(target_deck_id)
                
                if not deck:
                    logger.error(f"Target deck not found: {target_deck_id}")
                    continue
                
                if not hasattr(deck, 'musical_timing_system') or not deck.musical_timing_system:
                    logger.error(f"Deck {target_deck_id} does not have musical timing system")
                    continue
                
                # Convert action to musical timing system format
                action_type = action.get('command')
                parameters = action.get('parameters', {})
                action_id = action.get('action_id')
                
                # Schedule the action with frame-accurate timing
                scheduled_id = deck.musical_timing_system.schedule_beat_action(
                    beat_number=beat_number,
                    action_type=action_type,
                    parameters=parameters,
                    action_id=action_id,
                    priority=0
                )
                
                if scheduled_id:
                    self.scheduled_actions[action_id] = {
                        'scheduled_id': scheduled_id,
                        'deck_id': target_deck_id,
                        'beat_number': beat_number,
                        'action_type': action_type
                    }
                    logger.info(f"Pre-scheduled action: {action_id} -> {action_type} at beat {beat_number} on {target_deck_id}")
                else:
                    logger.error(f"Failed to schedule action: {action_id}")
                
            except Exception as e:
                logger.error(f"Error scheduling beat action {action.get('action_id')}: {e}")
    
    def get_scheduled_actions(self) -> Dict[str, Any]:
        """Get information about all scheduled actions"""
        return self.scheduled_actions.copy()
    
    def cancel_action(self, action_id: str) -> bool:
        """Cancel a scheduled action"""
        if action_id not in self.scheduled_actions:
            return False
        
        try:
            action_info = self.scheduled_actions[action_id]
            deck_id = action_info['deck_id']
            scheduled_id = action_info['scheduled_id']
            
            deck = self.engine._get_or_create_deck(deck_id)
            if deck and hasattr(deck, 'musical_timing_system') and deck.musical_timing_system:
                success = deck.musical_timing_system.cancel_action(scheduled_id)
                if success:
                    del self.scheduled_actions[action_id]
                    logger.info(f"Cancelled action: {action_id}")
                return success
            
        except Exception as e:
            logger.error(f"Error cancelling action {action_id}: {e}")
        
        return False
    
    def get_mix_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the loaded mix"""
        if not self.loaded_mix:
            return None
        
        return {
            'mix_name': self.loaded_mix.get('mix_name'),
            'total_actions': len(self.loaded_mix.get('actions', [])),
            'scheduled_actions': len(self.scheduled_actions),
            'actions': self.scheduled_actions
        }