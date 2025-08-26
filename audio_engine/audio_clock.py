#!/usr/bin/env python3
"""
Modular audio clock system for precise timing and synchronization.
Provides consistent timing for event scheduling across the audio engine.
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
    current_beat: float = 0.0
    current_bpm: float = 120.0
    total_frames: int = 0
    sample_rate: int = 44100

class AudioClock:
    """
    High-precision audio clock synchronized with audio device.
    Provides consistent timing for event scheduling.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._state = ClockState(sample_rate=sample_rate)
        self._beat_timestamps: List[float] = []
        self._beat_positions: Dict[int, float] = {}  # beat_number -> frame_position
        self._lock = threading.RLock()
        self._observers: List[Callable] = []
        
        logger.debug(f"AudioClock initialized with sample rate: {sample_rate}")
    
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
            
            # Update current beat based on new time
            self._update_current_beat()
    
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
            self._beat_timestamps.clear()
            self._beat_positions.clear()
            logger.info("AudioClock reset")
            self._notify_observers("reset")
    
    def update_frame_count(self, frames: int) -> None:
        """Update frame count from audio callback"""
        with self._lock:
            if not self._state.is_running:
                return
            
            self._state.total_frames += frames
            self._state.current_time = self._state.total_frames / self.sample_rate
            
            # Update current beat based on new time
            self._update_current_beat()
    
    def set_beat_timestamps(self, timestamps: List[float], bpm: float) -> None:
        """Set beat timing information"""
        with self._lock:
            self._beat_timestamps = timestamps.copy()
            self._state.current_bpm = bpm
            
            # Convert timestamps to beat positions
            self._beat_positions.clear()
            for i, timestamp in enumerate(timestamps):
                self._beat_positions[i] = timestamp
            
            logger.debug(f"AudioClock: Set {len(timestamps)} beat timestamps, BPM: {bpm}")
            self._notify_observers("beats_updated")
    
    def set_beat_positions(self, positions: Dict[int, float]) -> None:
        """Set beat positions in frames"""
        with self._lock:
            self._beat_positions = positions.copy()
            logger.debug(f"AudioClock: Set {len(positions)} beat positions")
            self._notify_observers("beats_updated")
    
    def get_current_time(self) -> float:
        """Get current time in seconds since start"""
        with self._lock:
            if not self._state.is_running:
                return 0.0
            return self._state.current_time
    
    def get_current_beat(self) -> float:
        """Get current beat number (can be fractional)"""
        with self._lock:
            if not self._state.is_running:
                return 0.0
            return self._state.current_beat
    
    def get_current_bpm(self) -> float:
        """Get current BPM"""
        with self._lock:
            return self._state.current_bpm
    
    def get_time_to_next_beat(self) -> Optional[float]:
        """Get time until next beat boundary"""
        with self._lock:
            if not self._state.is_running or not self._beat_timestamps:
                return None
            
            current_time = self._state.current_time
            
            # Find next beat timestamp
            for timestamp in self._beat_timestamps:
                if timestamp > current_time:
                    return timestamp - current_time
            
            return None
    
    def get_beat_at_time(self, target_time: float) -> Optional[float]:
        """Get beat number at a specific time"""
        with self._lock:
            if not self._beat_timestamps:
                return None
            
            # Find the beat at or before the target time
            for i, timestamp in enumerate(self._beat_timestamps):
                if timestamp > target_time:
                    if i == 0:
                        return 0.0
                    # Interpolate between beats
                    prev_timestamp = self._beat_timestamps[i - 1]
                    beat_progress = (target_time - prev_timestamp) / (timestamp - prev_timestamp)
                    return float(i - 1) + beat_progress
            
            # Target time is after all beats
            return float(len(self._beat_timestamps) - 1)
    
    def get_time_at_beat(self, beat_number: float) -> Optional[float]:
        """Get time at a specific beat number"""
        with self._lock:
            if not self._beat_timestamps:
                return None
            
            beat_int = int(beat_number)
            beat_frac = beat_number - beat_int
            
            if beat_int >= len(self._beat_timestamps):
                return None
            
            if beat_int == len(self._beat_timestamps) - 1:
                # Last beat
                return self._beat_timestamps[beat_int]
            
            # Interpolate between beats
            current_beat_time = self._beat_timestamps[beat_int]
            next_beat_time = self._beat_timestamps[beat_int + 1]
            interpolated_time = current_beat_time + (next_beat_time - current_beat_time) * beat_frac
            
            return interpolated_time
    
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
    
    def _update_current_beat(self) -> None:
        """Update current beat based on current time"""
        if not self._beat_timestamps:
            # Fallback to BPM calculation
            self._state.current_beat = (self._state.current_time * self._state.current_bpm) / 60.0
            return
        
        # Find current beat using beat timestamps
        current_time = self._state.current_time
        beat_index = 0
        
        for i, timestamp in enumerate(self._beat_timestamps):
            if current_time >= timestamp:
                beat_index = i
            else:
                break
        
        # Interpolate fractional beats
        if beat_index < len(self._beat_timestamps) - 1:
            current_beat_time = self._beat_timestamps[beat_index]
            next_beat_time = self._beat_timestamps[beat_index + 1]
            
            if next_beat_time > current_beat_time:
                beat_progress = (current_time - current_beat_time) / (next_beat_time - current_beat_time)
                self._state.current_beat = beat_index + beat_progress
            else:
                self._state.current_beat = float(beat_index)
        else:
            self._state.current_beat = float(beat_index)
    
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
                current_beat=self._state.current_beat,
                current_bpm=self._state.current_bpm,
                total_frames=self._state.total_frames,
                sample_rate=self._state.sample_rate
            )
    
    def is_running(self) -> bool:
        """Check if clock is running"""
        with self._lock:
            return self._state.is_running
