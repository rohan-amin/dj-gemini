#!/usr/bin/env python3
"""
Modular audio clock system for precise timing and synchronization.
Provides consistent timing for event scheduling across the audio engine.

Phase 4: Beat data removed - now handled by BeatManager per deck.
AudioClock now focuses purely on master timing.
"""

import time
import threading
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class BeatInfo:
    """Information about a specific beat"""
    beat_number: int
    timestamp: float
    confidence: float = 1.0

@dataclass
class ClockState:
    """Current state of the audio clock"""
    is_running: bool = False
    start_time: float = 0.0
    current_time: float = 0.0
    current_beat: float = 0.0  # Kept for backwards compatibility, but not used
    current_bpm: float = 120.0  # Default BPM
    total_frames: int = 0
    sample_rate: int = 44100

class AudioClock:
    """
    High-precision audio clock synchronized with audio device.
    Phase 4: Simplified to focus on pure timing - beat operations moved to BeatManager.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._state = ClockState(sample_rate=sample_rate)
        # Phase 4: Beat data removed - now handled by BeatManager per deck
        self._lock = threading.RLock()
        self._observers: List[Callable] = []
        
        logger.debug(f"AudioClock initialized with sample rate: {sample_rate} (Phase 4: beat data removed)")
    
    def start(self) -> None:
        """Start the audio clock"""
        with self._lock:
            if self._state.is_running:
                logger.warning("AudioClock already running")
                return
            
            self._state.start_time = time.time()
            self._state.is_running = True
            self._state.current_time = 0.0
            self._state.current_beat = 0.0
            self._state.total_frames = 0
            
            logger.info("AudioClock started")
            self._notify_observers("started")
    
    def advance_time_for_testing(self, seconds: float) -> None:
        """Advance time for testing purposes (not for production use)"""
        with self._lock:
            if not self._state.is_running:
                return
            
            # Simulate frame advancement
            frames = int(seconds * self.sample_rate)
            self._state.total_frames += frames
            self._state.current_time = self._state.total_frames / self.sample_rate
    
    def stop(self) -> None:
        """Stop the audio clock"""
        with self._lock:
            if not self._state.is_running:
                logger.warning("AudioClock not running")
                return
            
            self._state.is_running = False
            logger.info("AudioClock stopped")
            self._notify_observers("stopped")
    
    def reset(self) -> None:
        """Reset the audio clock to initial state"""
        with self._lock:
            self._state = ClockState(sample_rate=self.sample_rate)
            # Phase 4: Beat data removal - no longer stored in AudioClock
            logger.info("AudioClock reset (Phase 4: beat data now managed by BeatManager)")
            self._notify_observers("reset")
    
    def update_frame_count(self, frames: int) -> None:
        """Update frame count from audio callback"""
        with self._lock:
            if not self._state.is_running:
                return
            
            old_time = self._state.current_time
            
            self._state.total_frames += frames
            self._state.current_time = self._state.total_frames / self.sample_rate
            
            # Debug logging for significant changes
            if abs(self._state.current_time - old_time) > 0.1:  # Time changed by more than 0.1s
                logger.debug(f"AudioClock: Frame update: +{frames} frames, time: {old_time:.3f}s → {self._state.current_time:.3f}s")
    
    # Phase 4: Beat-related methods removed - now handled by BeatManager per deck
    # The following methods have been moved to BeatManager:
    # - set_beat_timestamps() -> BeatManager.set_beat_data()
    # - get_beat_at_time() -> BeatManager.get_beat_at_time() 
    # - get_time_at_beat() -> BeatManager.get_time_for_beat()
    # - get_time_to_next_beat() (removed - deck-specific functionality)
    # - set_beat_positions() (removed - internal BeatManager concern)
    # - get_current_beat() (removed - deck-specific)
    
    def get_current_time(self) -> float:
        """Get current time in seconds since start"""
        with self._lock:
            if not self._state.is_running:
                return 0.0
            return self._state.current_time
    
    def get_current_bpm(self) -> float:
        """
        Get current BPM - Phase 4: Returns default BPM.
        Note: Actual BPM now managed by BeatManager per deck.
        """
        with self._lock:
            return self._state.current_bpm
    
    def add_observer(self, callback: Callable) -> None:
        """Add an observer callback for clock state changes"""
        with self._lock:
            if callback not in self._observers:
                self._observers.append(callback)
    
    def remove_observer(self, callback: Callable) -> None:
        """Remove an observer callback"""
        with self._lock:
            if callback in self._observers:
                self._observers.remove(callback)
    
    def _notify_observers(self, event: str) -> None:
        """Notify all observers of a clock event"""
        for callback in self._observers:
            try:
                callback(event, self._state)
            except Exception as e:
                logger.error(f"Error in clock observer callback: {e}")
    
    def get_state(self) -> ClockState:
        """Get current clock state (thread-safe copy)"""
        with self._lock:
            return ClockState(
                is_running=self._state.is_running,
                start_time=self._state.start_time,
                current_time=self._state.current_time,
                current_beat=self._state.current_beat,  # Kept for compatibility
                current_bpm=self._state.current_bpm,
                total_frames=self._state.total_frames,
                sample_rate=self._state.sample_rate
            )
    
    def is_running(self) -> bool:
        """Check if clock is running"""
        with self._lock:
            return self._state.is_running