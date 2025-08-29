"""
Adapter to integrate ActionExecutors with existing Deck implementations.
Provides specialized executors that work with the current DJ Gemini deck system.
"""

import logging
from typing import Dict, Any
from ..interfaces.timing_interfaces import ActionExecutor

logger = logging.getLogger(__name__)

class DeckExecutorAdapter(ActionExecutor):
    """
    Adapter that provides ActionExecutor interface for existing Deck methods.
    
    This adapter wraps a Deck instance and provides action execution
    capabilities compatible with the new musical timing system.
    """
    
    def __init__(self, deck):
        """
        Initialize adapter with existing Deck.
        
        Args:
            deck: Existing Deck instance to wrap
        """
        self._deck = deck
        self._deck_id = getattr(deck, 'deck_id', 'unknown')
        
        # Map action types to deck methods
        self._action_methods = {
            'play': self._execute_play,
            'pause': self._execute_pause, 
            'stop': self._execute_stop,
            'seek': self._execute_seek,
            'activate_loop': self._execute_activate_loop,
            'deactivate_loop': self._execute_deactivate_loop,
            'set_volume': self._execute_set_volume,
            'set_tempo': self._execute_set_tempo,
            'bpm_match': self._execute_bpm_match
        }
        
        # Validate deck has basic required methods
        required_methods = ['play', 'pause', 'stop']
        missing_methods = [method for method in required_methods if not hasattr(deck, method)]
        
        if missing_methods:
            logger.warning(f"Deck {self._deck_id} missing some methods: {missing_methods}")
        
        logger.debug(f"DeckExecutorAdapter initialized for deck {self._deck_id}")
    
    def execute_action(self, action_type: str, parameters: Dict[str, Any], 
                      execution_context: Dict[str, Any]) -> bool:
        """
        Execute action on the wrapped deck.
        
        Args:
            action_type: Type of action to execute
            parameters: Action-specific parameters  
            execution_context: Context information
            
        Returns:
            True if execution was successful
        """
        if action_type not in self._action_methods:
            logger.error(f"Deck {self._deck_id}: Unsupported action type: {action_type}")
            return False
        
        try:
            action_method = self._action_methods[action_type]
            success = action_method(parameters, execution_context)
            
            if success:
                logger.info(f"Deck {self._deck_id}: Successfully executed {action_type}")
            else:
                logger.warning(f"Deck {self._deck_id}: Failed to execute {action_type}")
            
            return success
            
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Exception executing {action_type}: {e}")
            return False
    
    def can_execute(self, action_type: str) -> bool:
        """
        Check if this adapter can execute the given action type.
        
        Args:
            action_type: Type of action to check
            
        Returns:
            True if this adapter can handle the action type
        """
        return action_type in self._action_methods
    
    def get_supported_actions(self) -> list[str]:
        """
        Get list of supported action types.
        
        Returns:
            List of action types this adapter can handle
        """
        return list(self._action_methods.keys())
    
    def _execute_play(self, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Execute play action"""
        try:
            # Extract play parameters
            start_at_frame = params.get('start_at_frame')
            start_at_beat = params.get('start_at_beat')
            start_at_cue_name = params.get('start_at_cue_name')
            
            # Call appropriate deck method based on parameters
            if start_at_frame is not None:
                success = self._deck.play(start_at_frame=start_at_frame)
            elif start_at_beat is not None:
                # Convert beat to frame using beat manager
                if hasattr(self._deck, 'beat_manager'):
                    target_frame = self._deck.beat_manager.get_frame_for_beat(start_at_beat)
                    success = self._deck.play(start_at_frame=target_frame)
                else:
                    logger.error(f"Deck {self._deck_id}: No beat manager for beat-based play")
                    return False
            elif start_at_cue_name is not None:
                success = self._deck.play(start_at_cue_name=start_at_cue_name)
            else:
                # Resume from current position
                success = self._deck.play()
            
            return success if success is not None else True
            
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error in play action: {e}")
            return False
    
    def _execute_pause(self, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Execute pause action"""
        try:
            self._deck.pause()
            return True
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error in pause action: {e}")
            return False
    
    def _execute_stop(self, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Execute stop action"""
        try:
            self._deck.stop()
            return True
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error in stop action: {e}")
            return False
    
    def _execute_seek(self, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Execute seek action"""
        try:
            target_frame = params.get('target_frame')
            target_beat = params.get('target_beat')
            target_cue = params.get('target_cue')
            
            if target_frame is not None:
                # Frame-accurate seek
                if hasattr(self._deck, 'seek_to_frame'):
                    self._deck.seek_to_frame(target_frame)
                    return True
                else:
                    logger.error(f"Deck {self._deck_id}: No seek_to_frame method")
                    return False
                    
            elif target_beat is not None:
                # Beat-accurate seek
                if hasattr(self._deck, 'beat_manager'):
                    frame = self._deck.beat_manager.get_frame_for_beat(target_beat)
                    if hasattr(self._deck, 'seek_to_frame'):
                        self._deck.seek_to_frame(frame)
                        return True
                logger.error(f"Deck {self._deck_id}: Cannot perform beat-based seek")
                return False
                
            elif target_cue is not None:
                # Cue-based seek
                if hasattr(self._deck, 'seek_to_cue'):
                    self._deck.seek_to_cue(target_cue)
                    return True
                else:
                    logger.error(f"Deck {self._deck_id}: No cue seeking support")
                    return False
            
            logger.error(f"Deck {self._deck_id}: No seek target specified")
            return False
            
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error in seek action: {e}")
            return False
    
    def _execute_activate_loop(self, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Execute seamless loop activation without audio artifacts"""
        try:
            # Extract loop parameters
            start_at_beat = params.get('start_at_beat')
            length_beats = params.get('length_beats')
            repetitions = params.get('repetitions', 1)
            action_id = context.get('action_id', 'unknown_loop')
            target_frame = context.get('target_frame')
            
            # Validate parameters
            if start_at_beat is None or length_beats is None:
                logger.error(f"Deck {self._deck_id}: Missing loop parameters")
                return False
            
            logger.info(f"🔄 Deck {self._deck_id}: Frame-accurate loop activation - {action_id} at beat {start_at_beat}")
            
            # === NEW: Frame-accurate seamless loop activation ===
            # Instead of using the old loop_manager that causes artifacts,
            # we'll implement seamless looping by modifying the deck's playback position
            
            if hasattr(self._deck, 'beat_manager') and self._deck.beat_manager:
                try:
                    # Calculate loop boundaries in frames
                    loop_start_frame = self._deck.beat_manager.get_frame_for_beat(start_at_beat)
                    loop_end_frame = self._deck.beat_manager.get_frame_for_beat(start_at_beat + length_beats)
                    
                    logger.info(f"🔄 Loop bounds: beat {start_at_beat}-{start_at_beat + length_beats} = "
                               f"frames {loop_start_frame}-{loop_end_frame}")
                    
                    # Store loop info in the deck for the producer thread to use
                    self._deck._frame_accurate_loop = {
                        'start_frame': loop_start_frame,
                        'end_frame': loop_end_frame,
                        'repetitions': repetitions,
                        'current_repetition': 0,
                        'action_id': action_id,
                        'active': True
                    }
                    
                    logger.info(f"✅ Deck {self._deck_id}: Seamless loop activated - {action_id}")
                    return True
                    
                except Exception as e:
                    logger.error(f"Deck {self._deck_id}: Error calculating loop frames: {e}")
                    return False
            else:
                logger.error(f"Deck {self._deck_id}: No beat manager for frame calculation")
                return False
                
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error in seamless loop activation: {e}")
            return False
    
    def _execute_deactivate_loop(self, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Execute loop deactivation"""
        try:
            if hasattr(self._deck, 'loop_manager'):
                self._deck.loop_manager.deactivate_loop()
                return True
            else:
                logger.error(f"Deck {self._deck_id}: No loop manager available")
                return False
                
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error in loop deactivation: {e}")
            return False
    
    def _execute_set_volume(self, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Execute volume change"""
        try:
            volume = params.get('volume')
            if volume is None:
                logger.error(f"Deck {self._deck_id}: No volume specified")
                return False
            
            # Try different volume setting methods
            if hasattr(self._deck, 'set_volume'):
                self._deck.set_volume(volume)
                return True
            elif hasattr(self._deck, 'volume'):
                self._deck.volume = volume
                return True
            else:
                logger.error(f"Deck {self._deck_id}: No volume control available")
                return False
                
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error setting volume: {e}")
            return False
    
    def _execute_set_tempo(self, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Execute tempo change"""
        try:
            new_bpm = params.get('bpm')
            ramp_duration = params.get('ramp_duration_beats', 0)
            
            if new_bpm is None:
                logger.error(f"Deck {self._deck_id}: No BPM specified")
                return False
            
            # Use beat manager for tempo changes
            if hasattr(self._deck, 'beat_manager'):
                if hasattr(self._deck.beat_manager, 'handle_tempo_change'):
                    self._deck.beat_manager.handle_tempo_change(new_bpm, ramp_duration)
                    return True
                else:
                    logger.error(f"Deck {self._deck_id}: Beat manager doesn't support tempo changes")
                    return False
            else:
                logger.error(f"Deck {self._deck_id}: No beat manager for tempo changes")
                return False
                
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error setting tempo: {e}")
            return False
    
    def _execute_bpm_match(self, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Execute BPM matching between decks"""
        try:
            reference_deck_id = params.get('reference_deck')
            follow_deck_id = params.get('follow_deck')
            
            if not reference_deck_id or not follow_deck_id:
                logger.error(f"Deck {self._deck_id}: Missing reference_deck or follow_deck for BPM match")
                return False
            
            # For now, just log the BPM match request
            # This would need engine-level implementation to access other decks
            logger.info(f"🎛️ BPM Match requested: {follow_deck_id} should match {reference_deck_id}")
            logger.warning(f"Deck {self._deck_id}: BPM matching not yet implemented - requires engine-level coordination")
            
            # TODO: Implement actual BPM matching via engine reference
            # This would require:
            # 1. Get reference deck's current BPM
            # 2. Set follow deck's BPM to match
            # 3. Potentially handle tempo ramping for smooth transitions
            
            return True  # Return true for now to avoid errors
            
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error in BPM match: {e}")
            return False
    
    def get_deck_info(self) -> dict:
        """
        Get information about the wrapped deck.
        
        Returns:
            Dictionary with deck information and capabilities
        """
        info = {
            'adapter_type': 'DeckExecutorAdapter',
            'deck_id': self._deck_id,
            'supported_actions': self.get_supported_actions()
        }
        
        try:
            # Check deck capabilities
            capabilities = []
            
            if hasattr(self._deck, 'play'):
                capabilities.append('playback')
            if hasattr(self._deck, 'loop_manager'):
                capabilities.append('loops')
            if hasattr(self._deck, 'beat_manager'):
                capabilities.append('beat_tracking')
            if hasattr(self._deck, 'seek_to_frame'):
                capabilities.append('frame_seeking')
            if hasattr(self._deck, 'set_volume'):
                capabilities.append('volume_control')
            
            info['capabilities'] = capabilities
            
            # Get current deck state if possible
            deck_state = {}
            
            if hasattr(self._deck, 'is_playing'):
                deck_state['is_playing'] = getattr(self._deck, 'is_playing', False)
            
            if hasattr(self._deck, 'beat_manager'):
                try:
                    deck_state['current_beat'] = self._deck.beat_manager.get_current_beat()
                    deck_state['current_frame'] = self._deck.beat_manager.get_current_frame()
                    deck_state['current_bpm'] = self._deck.beat_manager.get_bpm()
                except:
                    pass
            
            if hasattr(self._deck, 'loop_manager'):
                try:
                    loop_info = self._deck.loop_manager.get_current_loop_info()
                    if loop_info:
                        deck_state['active_loop'] = loop_info
                except:
                    pass
            
            info['deck_state'] = deck_state
            
        except Exception as e:
            logger.debug(f"Error gathering deck info: {e}")
            info['info_error'] = str(e)
        
        return info
    
    def __repr__(self) -> str:
        """String representation of the adapter."""
        return f"DeckExecutorAdapter(deck_id={self._deck_id}, actions={len(self._action_methods)})"