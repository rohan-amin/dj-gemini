"""
Adapter to make BeatManager compatible with BeatFrameConverter interface.
Provides a clean interface for the musical timing system.
"""

import logging
from typing import Optional
from ..interfaces.timing_interfaces import BeatFrameConverter

logger = logging.getLogger(__name__)

class BeatManagerAdapter(BeatFrameConverter):
    """
    Adapter to make BeatManager compatible with BeatFrameConverter interface.
    
    This adapter wraps the existing BeatManager and provides the clean
    interface required by the musical timing system components.
    """
    
    def __init__(self, beat_manager):
        """
        Initialize adapter with existing BeatManager.
        
        Args:
            beat_manager: Existing BeatManager instance to wrap
        """
        self._beat_manager = beat_manager
        # Try to get deck_id from beat_manager.deck.deck_id or fallback to 'unknown'
        if hasattr(beat_manager, 'deck') and hasattr(beat_manager.deck, 'deck_id'):
            self._deck_id = beat_manager.deck.deck_id
        else:
            self._deck_id = 'unknown'
        
        # Validate beat manager has required methods
        required_methods = ['get_frame_for_beat', 'get_beat_from_frame', 'get_current_beat', 'get_current_frame']
        missing_methods = [method for method in required_methods if not hasattr(beat_manager, method)]
        
        if missing_methods:
            logger.error(f"BeatManager missing required methods: {missing_methods}")
            raise ValueError(f"BeatManager missing required methods: {missing_methods}")
        
        logger.debug(f"BeatManagerAdapter initialized for deck {self._deck_id}")
    
    def get_frame_for_beat(self, beat_number: float) -> int:
        """
        Convert beat number to frame position.
        
        Args:
            beat_number: Musical beat number (1-based, can be fractional)
            
        Returns:
            Frame position corresponding to the beat
            
        Raises:
            RuntimeError: If beat manager fails to convert
        """
        try:
            frame = self._beat_manager.get_frame_for_beat(beat_number)
            
            # Validate result
            if not isinstance(frame, int) or frame < 0:
                logger.warning(f"Invalid frame result from BeatManager: {frame} for beat {beat_number}")
                return 0
            
            logger.debug(f"Deck {self._deck_id}: Beat {beat_number} → Frame {frame}")
            return frame
            
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error converting beat {beat_number} to frame: {e}")
            raise RuntimeError(f"Failed to convert beat {beat_number} to frame: {e}")
    
    def get_beat_from_frame(self, frame: int) -> float:
        """
        Convert frame position to beat number.
        
        Args:
            frame: Audio frame position
            
        Returns:
            Musical beat number (1-based, can be fractional)
            
        Raises:
            RuntimeError: If beat manager fails to convert
        """
        try:
            beat = self._beat_manager.get_beat_from_frame(frame)
            
            # Validate result
            if not isinstance(beat, (int, float)) or beat < 0:
                logger.warning(f"Invalid beat result from BeatManager: {beat} for frame {frame}")
                return 1.0  # Default to beat 1
            
            logger.debug(f"Deck {self._deck_id}: Frame {frame} → Beat {beat:.3f}")
            return float(beat)
            
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error converting frame {frame} to beat: {e}")
            raise RuntimeError(f"Failed to convert frame {frame} to beat: {e}")
    
    def get_current_beat(self) -> float:
        """
        Get current beat position.
        
        Returns:
            Current musical beat number
            
        Raises:
            RuntimeError: If beat manager fails to provide current beat
        """
        try:
            beat = self._beat_manager.get_current_beat()
            
            # Validate result
            if not isinstance(beat, (int, float)):
                logger.warning(f"Invalid current beat from BeatManager: {beat}")
                return 1.0  # Default to beat 1
            
            return float(beat)
            
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error getting current beat: {e}")
            raise RuntimeError(f"Failed to get current beat: {e}")
    
    def get_current_frame(self) -> int:
        """
        Get current frame position.
        
        Returns:
            Current audio frame position
            
        Raises:
            RuntimeError: If beat manager fails to provide current frame
        """
        try:
            frame = self._beat_manager.get_current_frame()
            
            # Validate result
            if not isinstance(frame, int):
                logger.warning(f"Invalid current frame from BeatManager: {frame}")
                return 0  # Default to frame 0
            
            return frame
            
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error getting current frame: {e}")
            raise RuntimeError(f"Failed to get current frame: {e}")
    
    def get_bpm(self) -> float:
        """
        Get current BPM (additional method for tempo controller).
        
        Returns:
            Current beats per minute
            
        Raises:
            RuntimeError: If beat manager fails to provide BPM
        """
        try:
            # Try different methods that BeatManager might have
            if hasattr(self._beat_manager, 'get_bpm'):
                bpm = self._beat_manager.get_bpm()
            elif hasattr(self._beat_manager, 'get_current_bpm'):
                bpm = self._beat_manager.get_current_bpm()
            elif hasattr(self._beat_manager, '_bpm'):
                bpm = self._beat_manager._bpm
            else:
                logger.warning(f"Deck {self._deck_id}: BeatManager has no BPM access method")
                return 120.0  # Reasonable default
            
            # Validate result
            if not isinstance(bpm, (int, float)) or bpm <= 0:
                logger.warning(f"Invalid BPM from BeatManager: {bpm}")
                return 120.0  # Reasonable default
            
            return float(bpm)
            
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error getting BPM: {e}")
            raise RuntimeError(f"Failed to get BPM: {e}")
    
    def is_beat_manager_valid(self) -> bool:
        """
        Check if the wrapped beat manager is in a valid state.
        
        Returns:
            True if beat manager appears to be working correctly
        """
        try:
            # Test basic functionality
            current_beat = self.get_current_beat()
            current_frame = self.get_current_frame()
            bpm = self.get_bpm()
            
            # Basic sanity checks
            if current_beat < 0 or current_frame < 0 or bpm <= 0:
                return False
            
            # Test round-trip conversion
            test_beat = max(1.0, current_beat)
            frame = self.get_frame_for_beat(test_beat)
            recovered_beat = self.get_beat_from_frame(frame)
            
            # Allow some tolerance for floating point precision
            beat_difference = abs(recovered_beat - test_beat)
            if beat_difference > 0.1:  # Allow 0.1 beat tolerance
                logger.warning(f"Beat conversion round-trip failed: {test_beat} → {frame} → {recovered_beat}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Beat manager validation failed: {e}")
            return False
    
    def get_adapter_info(self) -> dict:
        """
        Get information about this adapter and the wrapped beat manager.
        
        Returns:
            Dictionary with adapter and beat manager information
        """
        info = {
            'adapter_type': 'BeatManagerAdapter',
            'deck_id': self._deck_id,
            'is_valid': self.is_beat_manager_valid()
        }
        
        try:
            # Get beat manager information if available
            if hasattr(self._beat_manager, 'get_debug_info'):
                info['beat_manager_debug'] = self._beat_manager.get_debug_info()
            
            # Add current state
            info['current_state'] = {
                'current_beat': self.get_current_beat(),
                'current_frame': self.get_current_frame(),
                'current_bpm': self.get_bpm()
            }
            
            # Check for additional capabilities
            capabilities = []
            if hasattr(self._beat_manager, 'handle_tempo_change'):
                capabilities.append('tempo_change')
            if hasattr(self._beat_manager, 'is_tempo_ramp_active'):
                capabilities.append('tempo_ramp')
            if hasattr(self._beat_manager, 'get_beat_timestamps'):
                capabilities.append('beat_timestamps')
            
            info['capabilities'] = capabilities
            
        except Exception as e:
            logger.debug(f"Error gathering adapter info: {e}")
            info['info_error'] = str(e)
        
        return info
    
    def __repr__(self) -> str:
        """String representation of the adapter."""
        try:
            current_beat = self.get_current_beat()
            current_bpm = self.get_bpm()
            return f"BeatManagerAdapter(deck={self._deck_id}, beat={current_beat:.2f}, bpm={current_bpm:.1f})"
        except:
            return f"BeatManagerAdapter(deck={self._deck_id}, status=error)"