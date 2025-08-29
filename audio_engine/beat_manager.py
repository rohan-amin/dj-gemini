# dj-gemini/audio_engine/beat_manager.py
# Centralized Beat Tracking and Management

import threading
import logging
import numpy as np
import time
from typing import Optional, Dict, Any, List, Callable

logger = logging.getLogger(__name__)

class BeatManager:
    """
    Centralized beat tracking and management for a single deck.
    
    This class provides a single source of truth for all beat-related calculations,
    eliminating race conditions between audio and main threads.
    
    Key Features:
    - Thread-safe beat position updates from audio thread
    - Consistent beat-to-frame and frame-to-beat conversions
    - Proper tempo change handling with ramp support
    - Optimized performance with caching
    """
    
    def __init__(self, deck):
        self.deck = deck
        self._beat_lock = threading.RLock()  # Reentrant lock for nested calls
        
        # Core beat state
        self._current_beat = 0.0
        self._current_frame = 0
        self._bpm = 120.0
        self._original_bpm = 120.0
        
        # Beat timing data
        self._beat_timestamps = []
        self._original_beat_positions = {}
        
        # Tempo change handling
        self._tempo_ramp_active = False
        self._ramp_beat_timestamps = None
        self._current_tempo_ratio = 1.0
        
        # Thread-safe state updates
        self._last_update_time = 0.0
        self._update_sequence = 0
        
        # Performance optimization
        self._beat_cache = {}  # Cache frequently accessed beat positions
        self._cache_valid = False
        
        # Beat boundary callback system
        self._beat_callbacks = []  # List of callbacks to fire on beat boundaries
        self._previous_beat_floor = None  # Track when we cross beat boundaries
        
        # Initialize with deck's current state
        self._sync_with_deck()
    
    def _sync_with_deck(self):
        """Synchronize BeatManager state with deck's current state"""
        with self._beat_lock:
            try:
                # Get current deck state
                if hasattr(self.deck, 'bpm') and self.deck.bpm > 0:
                    self._bpm = self.deck.bpm
                    self._original_bpm = self.deck.bpm
                
                if hasattr(self.deck, 'beat_timestamps') and self.deck.beat_timestamps is not None:
                    self._beat_timestamps = self.deck.beat_timestamps.copy()
                
                if hasattr(self.deck, 'original_beat_positions'):
                    self._original_beat_positions = self.deck.original_beat_positions.copy()
                
                if hasattr(self.deck, 'sample_rate') and self.deck.sample_rate > 0:
                    self._sample_rate = self.deck.sample_rate
                else:
                    self._sample_rate = 44100  # Default fallback
                
                logger.debug(f"BeatManager {self.deck.deck_id} - Synced with deck: BPM={self._bpm}, beats={len(self._beat_timestamps)}")
                
            except Exception as e:
                logger.warning(f"BeatManager {self.deck.deck_id} - Error syncing with deck: {e}")
    
    def update_from_frame(self, frame: int, sample_rate: Optional[int] = None) -> None:
        """
        Update beat position from current frame - called from audio thread.
        This is the single source of truth for beat position updates.
        
        Args:
            frame: Current playback frame position
            sample_rate: Optional sample rate override
        """
        with self._beat_lock:
            try:
                self._current_frame = frame
                self._last_update_time = time.time()
                self._update_sequence += 1
                
                # Use provided sample rate or fall back to stored value
                sr = sample_rate if sample_rate is not None else getattr(self, '_sample_rate', 44100)
                
                if sr <= 0:
                    logger.warning(f"BeatManager {self.deck.deck_id} - Invalid sample rate: {sr}")
                    return
                
                # Calculate current beat position
                if self._beat_timestamps is not None and len(self._beat_timestamps) > 0:
                    current_time_seconds = frame / float(sr)
                    
                    # Use ramp-adjusted timestamps if in tempo ramp
                    if self._tempo_ramp_active and self._ramp_beat_timestamps is not None:
                        beat_timestamps_to_use = self._ramp_beat_timestamps
                    else:
                        beat_timestamps_to_use = self._beat_timestamps
                    
                    # Calculate beat count using searchsorted for accuracy
                    try:
                        # Use side='left' to get the current beat (more accurate for beat boundaries)
                        # This ensures we get the correct beat for the current time position
                        beat_count = np.searchsorted(beat_timestamps_to_use, current_time_seconds, side='left')
                        
                        # If we're exactly at a beat timestamp, use that beat
                        # If we're between beats, interpolate for fractional beats
                        if beat_count < len(beat_timestamps_to_use):
                            if beat_count > 0 and abs(current_time_seconds - beat_timestamps_to_use[beat_count - 1]) < 0.001:
                                # We're exactly at beat boundary
                                self._current_beat = float(beat_count - 1)
                            else:
                                # We're between beats, interpolate
                                if beat_count > 0:
                                    prev_beat_time = beat_timestamps_to_use[beat_count - 1]
                                    if beat_count < len(beat_timestamps_to_use):
                                        next_beat_time = beat_timestamps_to_use[beat_count]
                                        # Linear interpolation
                                        time_diff = next_beat_time - prev_beat_time
                                        if time_diff > 0:
                                            progress = (current_time_seconds - prev_beat_time) / time_diff
                                            self._current_beat = float(beat_count - 1 + progress)
                                        else:
                                            self._current_beat = float(beat_count - 1)
                                    else:
                                        self._current_beat = float(beat_count - 1)
                                else:
                                    self._current_beat = 0.0
                        else:
                            self._current_beat = float(len(beat_timestamps_to_use) - 1)
                        
                        # Debug logging for tempo ramp test
                        if hasattr(self, '_tempo_ramp_active') and self._tempo_ramp_active:
                            logger.debug(f"BeatManager {self.deck.deck_id} - Frame {frame} → Time {current_time_seconds:.3f}s → Beat {self._current_beat}")
                        
                    except Exception as e:
                        logger.debug(f"BeatManager {self.deck.deck_id} - Error calculating beat from frame: {e}")
                        # Fallback to simple calculation
                        if self._bpm > 0:
                            self._current_beat = (current_time_seconds * self._bpm) / 60.0
                
                # Invalidate cache since position changed
                self._cache_valid = False
                
                # Check for beat boundary crossings and fire callbacks
                self._check_beat_boundaries()
                
                logger.debug(f"BeatManager {self.deck.deck_id} - Updated: frame={frame}, beat={self._current_beat:.3f}")
                
            except Exception as e:
                logger.error(f"BeatManager {self.deck.deck_id} - Error in update_from_frame: {e}")
    
    def get_current_beat(self) -> float:
        """
        Get current beat position - thread-safe access from any thread.
        
        Returns:
            Current beat position as float
        """
        with self._beat_lock:
            return self._current_beat
    
    def get_current_frame(self) -> int:
        """
        Get current frame position - thread-safe access from any thread.
        
        Returns:
            Current frame position
        """
        with self._beat_lock:
            return self._current_frame
    
    def get_frame_for_beat(self, beat_number: float) -> int:
        """
        Get frame number for a specific beat, accounting for tempo changes.
        
        Args:
            beat_number: Beat number (can be fractional)
            
        Returns:
            Frame number corresponding to the beat
        """
        with self._beat_lock:
            try:
                # CRITICAL FIX: Use beat timestamps for more accurate conversion
                if self._beat_timestamps is not None and len(self._beat_timestamps) > 0:
                    beat_int = int(beat_number)
                    beat_frac = beat_number - beat_int
                    
                    # If we have the exact beat timestamp, use it
                    if 0 <= beat_int < len(self._beat_timestamps):
                        if beat_frac == 0:
                            # Integer beat - use exact timestamp
                            time_seconds = self._beat_timestamps[beat_int]
                            frame = int(time_seconds * self._sample_rate)
                        else:
                            # Fractional beat - interpolate between timestamps
                            if beat_int + 1 < len(self._beat_timestamps):
                                prev_time = self._beat_timestamps[beat_int]
                                next_time = self._beat_timestamps[beat_int + 1]
                                # Linear interpolation
                                interpolated_time = prev_time + (next_time - prev_time) * beat_frac
                                frame = int(interpolated_time * self._sample_rate)
                            else:
                                # Beyond last beat - extrapolate using BPM
                                last_beat_time = self._beat_timestamps[beat_int]
                                extra_beats = beat_frac
                                extra_time = (extra_beats * 60.0) / self._bpm
                                frame = int((last_beat_time + extra_time) * self._sample_rate)
                        
                        # Apply tempo scaling if needed
                        if self._tempo_ramp_active and self._current_tempo_ratio != 1.0:
                            tempo_ratio = self._current_tempo_ratio
                            scaled_frame = int(frame / tempo_ratio)
                            return scaled_frame
                        elif self._original_bpm > 0 and abs(self._bpm - self._original_bpm) > 0.001:
                            tempo_ratio = self._bpm / self._original_bpm
                            scaled_frame = int(frame / tempo_ratio)
                            return scaled_frame
                        
                        return frame
                
                # Fallback to original beat positions method
                beat_int = int(beat_number)
                beat_frac = beat_number - beat_int
                
                # Check if we have the integer beat position
                if beat_int in self._original_beat_positions:
                    original_frame = self._original_beat_positions[beat_int]
                    # If we have fractional beats and the next beat exists, interpolate
                    if beat_frac > 0 and (beat_int + 1) in self._original_beat_positions:
                        next_frame = self._original_beat_positions[beat_int + 1]
                        # Linear interpolation between beats
                        interpolated_frame = original_frame + (next_frame - original_frame) * beat_frac
                        original_frame = int(interpolated_frame)
                    elif beat_frac > 0:
                        # We have fractional beats but no next beat - use calculated position
                        if self._bpm > 0 and hasattr(self, '_sample_rate'):
                            frames_per_beat = (60.0 / self._bpm) * self._sample_rate
                            # Calculate the exact frame for the fractional beat
                            exact_frame = beat_number * frames_per_beat
                            original_frame = int(exact_frame)
                    
                    # Apply tempo scaling if needed
                    if self._tempo_ramp_active and self._current_tempo_ratio != 1.0:
                        # Use ramp tempo ratio
                        tempo_ratio = self._current_tempo_ratio
                        scaled_frame = int(original_frame / tempo_ratio)
                        return scaled_frame
                    elif self._original_bpm > 0 and abs(self._bpm - self._original_bpm) > 0.001:
                        # Use regular tempo ratio
                        tempo_ratio = self._bpm / self._original_bpm
                        scaled_frame = int(original_frame / tempo_ratio)
                        return scaled_frame
                    
                    return original_frame
                else:
                    # Beat not found in original positions - use calculated position
                    if self._bpm > 0 and hasattr(self, '_sample_rate'):
                        frames_per_beat = (60.0 / self._bpm) * self._sample_rate
                        return int(beat_number * frames_per_beat)
                    return 0
                    
            except Exception as e:
                logger.error(f"BeatManager {self.deck.deck_id} - Error in get_frame_for_beat: {e}")
                return 0
    
    def get_beat_from_frame(self, frame: int) -> float:
        """
        Get beat number for a specific frame position.
        
        Args:
            frame: Frame position
            
        Returns:
            Beat number corresponding to the frame
        """
        with self._beat_lock:
            try:
                if not hasattr(self, '_sample_rate') or self._sample_rate <= 0:
                    return 0.0
                
                current_time_seconds = frame / float(self._sample_rate)
                
                if self._beat_timestamps is not None and len(self._beat_timestamps) > 0:
                    # Use ramp-adjusted timestamps if in tempo ramp
                    if self._tempo_ramp_active and self._ramp_beat_timestamps is not None:
                        beat_timestamps_to_use = self._ramp_beat_timestamps
                    else:
                        beat_timestamps_to_use = self._beat_timestamps
                    
                    try:
                        # Use side='left' for consistency with update_from_frame
                        beat_count = np.searchsorted(beat_timestamps_to_use, current_time_seconds, side='left')
                        
                        # If we're exactly at a beat timestamp, use that beat
                        # If we're between beats, interpolate for fractional beats
                        if beat_count < len(beat_timestamps_to_use):
                            if beat_count > 0 and abs(current_time_seconds - beat_timestamps_to_use[beat_count - 1]) < 0.001:
                                # We're exactly at beat boundary
                                return float(beat_count - 1)
                            else:
                                # We're between beats, interpolate
                                if beat_count > 0:
                                    prev_beat_time = beat_timestamps_to_use[beat_count - 1]
                                    if beat_count < len(beat_timestamps_to_use):
                                        next_beat_time = beat_timestamps_to_use[beat_count]
                                        # Linear interpolation
                                        time_diff = next_beat_time - prev_beat_time
                                        if time_diff > 0:
                                            progress = (current_time_seconds - prev_beat_time) / time_diff
                                            return float(beat_count - 1 + progress)
                                        else:
                                            return float(beat_count - 1)
                                    else:
                                        return float(beat_count - 1)
                                else:
                                    return 0.0
                        else:
                            return float(len(beat_timestamps_to_use) - 1)
                        
                    except Exception as e:
                        logger.debug(f"BeatManager {self.deck.deck_id} - Error calculating beat from frame: {e}")
                
                # Fallback to simple calculation
                if self._bpm > 0:
                    return (current_time_seconds * self._bpm) / 60.0
                
                return 0.0
                
            except Exception as e:
                logger.error(f"BeatManager {self.deck.deck_id} - Error in get_beat_from_frame: {e}")
                return 0.0
    
    def handle_tempo_change(self, new_bpm: float, ramp_duration_beats: float = 0.0) -> None:
        """
        Handle tempo changes with optional ramp support.
        
        Args:
            new_bpm: New BPM value
            ramp_duration_beats: Duration of tempo ramp in beats (0 = instant change)
        """
        with self._beat_lock:
            try:
                old_bpm = self._bpm
                self._bpm = new_bpm
                
                if ramp_duration_beats > 0:
                    # Start tempo ramp
                    self._tempo_ramp_active = True
                    self._ramp_start_beat = self._current_beat
                    # Ramp ends AFTER the duration (e.g., 4-beat ramp from beat 1 ends after beat 5)
                    self._ramp_end_beat = self._current_beat + ramp_duration_beats + 0.001
                    self._ramp_start_bpm = old_bpm
                    self._ramp_end_bpm = new_bpm
                    
                    # During ramp, BPM starts at old_bpm and gradually changes
                    # Don't change self._bpm yet - it will be updated by update_tempo_ramp()
                    
                    # Create ramp-adjusted beat timestamps
                    if self._beat_timestamps is not None and len(self._beat_timestamps) > 0:
                        self._ramp_beat_timestamps = self._beat_timestamps.copy()
                        
                        # Apply tempo scaling to timestamps after ramp start
                        for i, timestamp in enumerate(self._ramp_beat_timestamps):
                            if timestamp >= (self._ramp_start_beat * 60.0 / old_bpm):
                                # Scale timestamp based on tempo change
                                beat_time = timestamp - (self._ramp_start_beat * 60.0 / old_bpm)
                                new_beat_time = beat_time * (old_bpm / new_bpm)
                                self._ramp_beat_timestamps[i] = (self._ramp_start_beat * 60.0 / old_bpm) + new_beat_time
                    
                    logger.info(f"BeatManager {self.deck.deck_id} - Tempo ramp started: {old_bpm:.2f} → {new_bpm:.2f} over {ramp_duration_beats} beats")
                else:
                    # Instant tempo change
                    self._tempo_ramp_active = False
                    self._ramp_beat_timestamps = None
                    
                    # Update tempo ratio for scaling calculations
                    if self._original_bpm > 0:
                        self._current_tempo_ratio = new_bpm / self._original_bpm
                    else:
                        self._current_tempo_ratio = 1.0
                    
                    logger.info(f"BeatManager {self.deck.deck_id} - Tempo changed instantly: {old_bpm:.2f} → {new_bpm:.2f}")
                
                # Invalidate cache
                self._cache_valid = False
                
            except Exception as e:
                logger.error(f"BeatManager {self.deck.deck_id} - Error handling tempo change: {e}")
    
    def update_tempo_ramp(self, current_beat: float) -> None:
        """
        Update tempo ramp state based on current beat position.
        Called from audio thread during playback.
        
        Args:
            current_beat: Current beat position
        """
        with self._beat_lock:
            if not self._tempo_ramp_active:
                return
            
            try:
                if current_beat >= self._ramp_end_beat:
                    # Ramp complete
                    self._tempo_ramp_active = False
                    self._ramp_beat_timestamps = None
                    self._current_tempo_ratio = 1.0
                    logger.info(f"BeatManager {self.deck.deck_id} - Tempo ramp completed")
                else:
                    # Calculate current tempo ratio during ramp
                    ramp_progress = (current_beat - self._ramp_start_beat) / (self._ramp_end_beat - self._ramp_start_beat)
                    ramp_progress = max(0.0, min(1.0, ramp_progress))  # Clamp to [0, 1]
                    
                    # Linear interpolation between start and end BPM
                    current_bpm = self._ramp_start_bpm + (self._ramp_end_bpm - self._ramp_start_bpm) * ramp_progress
                    self._current_tempo_ratio = current_bpm / self._original_bpm if self._original_bpm > 0 else 1.0
                    
                    # Update the main BPM field during ramp
                    self._bpm = current_bpm
                    
                    logger.debug(f"BeatManager {self.deck.deck_id} - Ramp progress: {ramp_progress:.3f}, current BPM: {current_bpm:.2f}")
                
            except Exception as e:
                logger.error(f"BeatManager {self.deck.deck_id} - Error updating tempo ramp: {e}")
    
    def get_beat_timestamps(self) -> List[float]:
        """
        Get current beat timestamps - thread-safe access.
        
        Returns:
            List of beat timestamps in seconds
        """
        with self._beat_lock:
            if self._tempo_ramp_active and self._ramp_beat_timestamps is not None:
                return self._ramp_beat_timestamps.copy()
            else:
                return self._beat_timestamps.copy()
    
    def get_bpm(self) -> float:
        """
        Get current BPM - thread-safe access.
        
        Returns:
            Current BPM value (interpolated during tempo ramps)
        """
        with self._beat_lock:
            if self._tempo_ramp_active:
                # During ramp, interpolate between start and end BPM
                ramp_progress = (self._current_beat - self._ramp_start_beat) / (self._ramp_end_beat - self._ramp_start_beat)
                ramp_progress = max(0.0, min(1.0, ramp_progress))  # Clamp to [0, 1]
                current_bpm = self._ramp_start_bpm + (self._ramp_end_bpm - self._ramp_start_bpm) * ramp_progress
                return current_bpm
            else:
                return self._bpm
    
    def get_original_bpm(self) -> float:
        """
        Get original BPM - thread-safe access.
        
        Returns:
            Original BPM value
        """
        with self._beat_lock:
            return self._original_bpm
    
    def is_tempo_ramp_active(self) -> bool:
        """
        Check if tempo ramp is currently active.
        
        Returns:
            True if tempo ramp is active
        """
        with self._beat_lock:
            return self._tempo_ramp_active
    
    def get_tempo_ratio(self) -> float:
        """
        Get current tempo ratio - thread-safe access.
        
        Returns:
            Current tempo ratio (1.0 = original speed)
        """
        with self._beat_lock:
            return self._current_tempo_ratio
    
    def reset(self) -> None:
        """Reset BeatManager to initial state"""
        with self._beat_lock:
            self._current_beat = 0.0
            self._current_frame = 0
            self._bpm = self._original_bpm
            self._tempo_ramp_active = False
            self._ramp_beat_timestamps = None
            self._current_tempo_ratio = 1.0
            self._cache_valid = False
            self._beat_cache.clear()
            logger.info(f"BeatManager {self.deck.deck_id} - Reset to initial state")
    
    def sync_with_deck(self) -> None:
        """Force synchronization with deck's current state"""
        self._sync_with_deck()
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information for troubleshooting.
        
        Returns:
            Dictionary with debug information
        """
        with self._beat_lock:
            return {
                'current_beat': self._current_beat,
                'current_frame': self._current_frame,
                'bpm': self._bpm,
                'original_bpm': self._original_bpm,
                'tempo_ramp_active': self._tempo_ramp_active,
                'tempo_ratio': self._current_tempo_ratio,
                'beat_timestamps_count': len(self._beat_timestamps),
                'original_beat_positions_count': len(self._original_beat_positions),
                'last_update_time': self._last_update_time,
                'update_sequence': self._update_sequence
            }
    
    # === NEW METHODS FOR PHASE 1: BEAT-TIME CONVERSION ===
    
    def get_time_for_beat(self, beat_number: float) -> Optional[float]:
        """
        Convert beat number to time in seconds.
        This method will replace AudioClock.get_time_at_beat() in Phase 4.
        
        Args:
            beat_number: Beat number (can be fractional)
            
        Returns:
            Time in seconds, or None if conversion not possible
        """
        with self._beat_lock:
            if len(self._beat_timestamps) == 0:
                # Fallback to BPM calculation
                if self._bpm <= 0:
                    logger.warning(f"BeatManager {self.deck.deck_id} - Cannot convert beat to time: invalid BPM {self._bpm}")
                    return None
                return (beat_number * 60.0) / self._bpm
            
            # Use actual beat timestamps with interpolation
            beat_int = int(beat_number)
            beat_frac = beat_number - beat_int
            
            if beat_int >= len(self._beat_timestamps):
                logger.debug(f"BeatManager {self.deck.deck_id} - Beat {beat_number} beyond available timestamps ({len(self._beat_timestamps)})")
                return None
            
            if beat_int < 0:
                logger.debug(f"BeatManager {self.deck.deck_id} - Negative beat number: {beat_number}")
                return None
            
            if beat_int == len(self._beat_timestamps) - 1:
                # Last beat, no interpolation needed
                return self._beat_timestamps[beat_int]
            
            # Interpolate between current and next beat
            current_time = self._beat_timestamps[beat_int]
            next_time = self._beat_timestamps[beat_int + 1]
            interpolated_time = current_time + (next_time - current_time) * beat_frac
            
            logger.debug(f"BeatManager {self.deck.deck_id} - Beat {beat_number} → Time {interpolated_time:.3f}s")
            return interpolated_time

    def get_beat_at_time(self, time_seconds: float) -> Optional[float]:
        """
        Convert time in seconds to beat number.
        This method will replace AudioClock.get_beat_at_time() in Phase 4.
        
        Args:
            time_seconds: Time in seconds
            
        Returns:
            Beat number (fractional), or None if conversion not possible
        """
        with self._beat_lock:
            if time_seconds < 0:
                logger.debug(f"BeatManager {self.deck.deck_id} - Negative time: {time_seconds}")
                return 0.0
            
            if len(self._beat_timestamps) == 0:
                # Fallback to BPM calculation
                if self._bpm <= 0:
                    logger.warning(f"BeatManager {self.deck.deck_id} - Cannot convert time to beat: invalid BPM {self._bpm}")
                    return None
                return (time_seconds * self._bpm) / 60.0
            
            # Find beat using timestamps with interpolation
            for i, timestamp in enumerate(self._beat_timestamps):
                if time_seconds <= timestamp:
                    if i == 0:
                        # Before first beat
                        return 0.0
                    
                    # Interpolate between previous and current beat
                    prev_time = self._beat_timestamps[i - 1]
                    progress = (time_seconds - prev_time) / (timestamp - prev_time)
                    beat_number = float(i - 1) + progress
                    
                    logger.debug(f"BeatManager {self.deck.deck_id} - Time {time_seconds:.3f}s → Beat {beat_number:.3f}")
                    return beat_number
            
            # Time is after all beats
            last_beat = float(len(self._beat_timestamps) - 1)
            logger.debug(f"BeatManager {self.deck.deck_id} - Time {time_seconds:.3f}s beyond beats → Beat {last_beat}")
            return last_beat

    def set_beat_data(self, timestamps, bpm: float) -> None:
        """
        Set beat timing data for this deck.
        This method will replace AudioClock.set_beat_timestamps() in Phase 4.
        
        Args:
            timestamps: Beat timestamps (List, np.ndarray, etc.)
            bpm: Beats per minute
        """
        with self._beat_lock:
            # Handle different input types (numpy array, list, etc.)
            if hasattr(timestamps, 'copy'):
                # NumPy array
                self._beat_timestamps = timestamps.copy()
            elif hasattr(timestamps, '__iter__'):
                # List or other iterable
                self._beat_timestamps = list(timestamps)
            else:
                logger.error(f"BeatManager {self.deck.deck_id} - Invalid timestamps type: {type(timestamps)}")
                return
            
            self._bpm = float(bpm)
            self._original_bpm = float(bpm)
            self._current_tempo_ratio = 1.0
            self._cache_valid = False
            self._beat_cache.clear()
            
            logger.info(f"BeatManager {self.deck.deck_id} - Set {len(self._beat_timestamps)} beat timestamps, BPM: {bpm}")
            
            # Update original beat positions for tempo scaling
            self._original_beat_positions = {}
            for i, timestamp in enumerate(self._beat_timestamps):
                self._original_beat_positions[i] = timestamp

    def add_beat_callback(self, callback: Callable[[str, float], None]) -> None:
        """
        Add a callback to be fired when beat boundaries are crossed.
        
        Args:
            callback: Function that takes (deck_id, current_beat) and returns None
        """
        with self._beat_lock:
            if callback not in self._beat_callbacks:
                self._beat_callbacks.append(callback)
                logger.debug(f"BeatManager {self.deck.deck_id} - Added beat callback")

    def remove_beat_callback(self, callback: Callable[[str, float], None]) -> None:
        """
        Remove a beat callback.
        
        Args:
            callback: The callback function to remove
        """
        with self._beat_lock:
            if callback in self._beat_callbacks:
                self._beat_callbacks.remove(callback)
                logger.debug(f"BeatManager {self.deck.deck_id} - Removed beat callback")

    def _check_beat_boundaries(self) -> None:
        """
        Check if we've crossed beat boundaries and fire callbacks.
        Called internally from update_from_frame().
        """
        try:
            # Initialize previous beat on first call and fire initial callback
            if self._previous_beat_floor is None:
                self._previous_beat_floor = self._current_beat
                # Fire initial callback so EventScheduler knows starting beat position
                for callback in self._beat_callbacks:
                    try:
                        callback(self.deck.deck_id, self._current_beat, None)
                    except Exception as e:
                        logger.error(f"BeatManager {self.deck.deck_id} - Error in initial beat callback: {e}")
                logger.info(f"BeatManager {self.deck.deck_id} - Initial beat position: {self._current_beat:.3f}, callbacks fired: {len(self._beat_callbacks)}")
                return
            
            # Check if we've moved forward in beat position
            if self._current_beat > self._previous_beat_floor:
                # Fire callback with BOTH previous and current beat for proper range checking
                for callback in self._beat_callbacks:
                    try:
                        # Pass both previous and current beat so EventScheduler can check the range
                        callback(self.deck.deck_id, self._current_beat, self._previous_beat_floor)
                    except Exception as e:
                        logger.error(f"BeatManager {self.deck.deck_id} - Error in beat callback: {e}")
                
                logger.debug(f"BeatManager {self.deck.deck_id} - Beat position advanced: {self._previous_beat_floor:.3f} → {self._current_beat:.3f}")
            
            self._previous_beat_floor = self._current_beat
            
        except Exception as e:
            logger.error(f"BeatManager {self.deck.deck_id} - Error checking beat boundaries: {e}")
