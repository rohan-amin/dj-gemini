"""
Beat Manager Adapter for the loop system.

This module provides an adapter that allows the loop management system to
integrate with the existing beat manager for frame/beat conversions and
musical timing information.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BeatManagerAdapter:
    """
    Adapter for integrating with the beat manager system.
    
    This class provides a clean interface for the loop controller to interact
    with the beat manager without tight coupling to the specific beat manager
    implementation.
    """
    
    def __init__(self, beat_manager):
        """
        Initialize the beat manager adapter.
        
        Args:
            beat_manager: The beat manager instance to adapt
        """
        self._beat_manager = beat_manager
        self._cached_bpm: Optional[float] = None
        self._cached_tempo_ratio: Optional[float] = None
        
        logger.debug("🔄 BeatManagerAdapter initialized")
    
    def get_frame_for_beat(self, beat: float) -> int:
        """
        Convert beat position to frame position.
        
        Args:
            beat: Beat position (1-based)
            
        Returns:
            Frame position corresponding to the beat
        """
        try:
            return self._beat_manager.get_frame_for_beat(beat)
        except Exception as e:
            logger.error(f"🔄 Error converting beat {beat} to frame: {e}")
            # Fallback calculation if beat manager fails
            return self._fallback_beat_to_frame(beat)
    
    def get_beat_from_frame(self, frame: int) -> float:
        """
        Convert frame position to beat position.
        
        Args:
            frame: Frame position
            
        Returns:
            Beat position corresponding to the frame
        """
        try:
            return self._beat_manager.get_beat_from_frame(frame)
        except Exception as e:
            logger.error(f"🔄 Error converting frame {frame} to beat: {e}")
            # Fallback calculation if beat manager fails
            return self._fallback_frame_to_beat(frame)
    
    def get_current_beat(self) -> float:
        """
        Get the current beat position.
        
        Returns:
            Current beat position
        """
        try:
            return self._beat_manager.get_current_beat()
        except Exception as e:
            logger.error(f"🔄 Error getting current beat: {e}")
            return 1.0  # Default fallback
    
    def get_current_frame(self) -> int:
        """
        Get the current frame position.
        
        Returns:
            Current frame position
        """
        try:
            return self._beat_manager.get_current_frame()
        except Exception as e:
            logger.error(f"🔄 Error getting current frame: {e}")
            return 0  # Default fallback
    
    def get_bpm(self) -> float:
        """
        Get the current BPM (beats per minute).
        
        Returns:
            Current BPM value
        """
        try:
            bpm = self._beat_manager.get_bpm()
            self._cached_bpm = bpm
            return bpm
        except Exception as e:
            logger.error(f"🔄 Error getting BPM: {e}")
            return self._cached_bpm or 120.0  # Default fallback
    
    def get_tempo_ratio(self) -> float:
        """
        Get the current tempo ratio (for pitch-shifted playback).
        
        Returns:
            Current tempo ratio (1.0 = normal speed)
        """
        try:
            ratio = self._beat_manager.get_tempo_ratio()
            self._cached_tempo_ratio = ratio
            return ratio
        except Exception as e:
            logger.error(f"🔄 Error getting tempo ratio: {e}")
            return self._cached_tempo_ratio or 1.0  # Default fallback
    
    def get_sample_rate(self) -> int:
        """
        Get the audio sample rate.
        
        Returns:
            Sample rate in Hz
        """
        try:
            if hasattr(self._beat_manager, 'get_sample_rate'):
                return self._beat_manager.get_sample_rate()
            elif hasattr(self._beat_manager, 'sample_rate'):
                return self._beat_manager.sample_rate
            else:
                return 44100  # Default fallback
        except Exception as e:
            logger.error(f"🔄 Error getting sample rate: {e}")
            return 44100
    
    def is_playing(self) -> bool:
        """
        Check if the track is currently playing.
        
        Returns:
            True if playing, False otherwise
        """
        try:
            if hasattr(self._beat_manager, 'is_playing'):
                return self._beat_manager.is_playing()
            else:
                # Fallback: assume playing if we can get current beat/frame
                current_beat = self.get_current_beat()
                return current_beat > 0
        except Exception as e:
            logger.error(f"🔄 Error checking play status: {e}")
            return False
    
    def _fallback_beat_to_frame(self, beat: float) -> int:
        """
        Fallback beat to frame conversion using cached BPM.
        
        Args:
            beat: Beat position
            
        Returns:
            Estimated frame position
        """
        try:
            bpm = self._cached_bpm or 120.0
            sample_rate = self.get_sample_rate()
            frames_per_beat = (60.0 / bpm) * sample_rate
            return int((beat - 1.0) * frames_per_beat)
        except Exception as e:
            logger.error(f"🔄 Error in fallback beat to frame conversion: {e}")
            return 0
    
    def _fallback_frame_to_beat(self, frame: int) -> float:
        """
        Fallback frame to beat conversion using cached BPM.
        
        Args:
            frame: Frame position
            
        Returns:
            Estimated beat position
        """
        try:
            bpm = self._cached_bpm or 120.0
            sample_rate = self.get_sample_rate()
            frames_per_beat = (60.0 / bpm) * sample_rate
            return (frame / frames_per_beat) + 1.0
        except Exception as e:
            logger.error(f"🔄 Error in fallback frame to beat conversion: {e}")
            return 1.0
    
    def validate_beat_range(self, start_beat: float, end_beat: float, track_length_frames: int) -> bool:
        """
        Validate that a beat range is within track bounds.
        
        Args:
            start_beat: Start beat position
            end_beat: End beat position  
            track_length_frames: Total track length in frames
            
        Returns:
            True if beat range is valid
        """
        try:
            start_frame = self.get_frame_for_beat(start_beat)
            end_frame = self.get_frame_for_beat(end_beat)
            
            return (start_frame >= 0 and 
                   end_frame > start_frame and 
                   end_frame <= track_length_frames)
        except Exception as e:
            logger.error(f"🔄 Error validating beat range: {e}")
            return False
    
    def get_beat_duration_frames(self, start_beat: float, end_beat: float) -> int:
        """
        Get the frame duration for a beat range.
        
        Args:
            start_beat: Start beat position
            end_beat: End beat position
            
        Returns:
            Duration in frames
        """
        try:
            start_frame = self.get_frame_for_beat(start_beat)
            end_frame = self.get_frame_for_beat(end_beat)
            return end_frame - start_frame
        except Exception as e:
            logger.error(f"🔄 Error calculating beat duration: {e}")
            return 0
    
    def get_stats(self) -> dict:
        """
        Get adapter statistics and current state.
        
        Returns:
            Dictionary with adapter statistics
        """
        try:
            return {
                'current_beat': self.get_current_beat(),
                'current_frame': self.get_current_frame(),
                'bpm': self.get_bpm(),
                'tempo_ratio': self.get_tempo_ratio(),
                'sample_rate': self.get_sample_rate(),
                'is_playing': self.is_playing(),
                'cached_bpm': self._cached_bpm,
                'cached_tempo_ratio': self._cached_tempo_ratio,
                'beat_manager_type': type(self._beat_manager).__name__
            }
        except Exception as e:
            logger.error(f"🔄 Error getting adapter stats: {e}")
            return {'error': str(e)}