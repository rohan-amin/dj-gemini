"""
Tempo controller for detecting and managing tempo changes.
Provides efficient tempo change detection with listener notifications.
"""

import time
import threading
import logging
from typing import List, Callable, Optional
from ..interfaces.timing_interfaces import TempoChangeNotifier, BeatFrameConverter

logger = logging.getLogger(__name__)

class TempoController(TempoChangeNotifier):
    """
    Manages tempo change detection and notification.
    
    This class monitors tempo changes from a BeatFrameConverter and
    efficiently notifies listeners when significant changes occur.
    Designed to be called from the audio thread for real-time detection.
    """
    
    def __init__(self, beat_converter: BeatFrameConverter, 
                 check_interval_frames: int = 1024,
                 tempo_change_threshold: float = 0.1):
        """
        Initialize tempo controller.
        
        Args:
            beat_converter: Source of current tempo information
            check_interval_frames: How often to check for changes (in frames)
            tempo_change_threshold: Minimum BPM change to trigger notification
        """
        self._beat_converter = beat_converter
        self._listeners: List[Callable[[float, float], None]] = []
        self._lock = threading.RLock()
        
        # Tempo change detection settings
        self._check_interval_frames = check_interval_frames
        self._tempo_change_threshold = tempo_change_threshold
        
        # State tracking
        self._last_bpm = self._get_initial_bpm()
        self._last_check_frame = 0
        self._last_check_time = time.time()
        
        # Statistics
        self._stats = {
            'tempo_changes_detected': 0,
            'checks_performed': 0,
            'last_change_time': 0.0,
            'last_change_from': 0.0,
            'last_change_to': 0.0,
            'listeners_count': 0
        }
        
        logger.info(f"TempoController initialized: initial_bpm={self._last_bpm:.1f}, "
                   f"check_interval={check_interval_frames}, threshold={tempo_change_threshold}")
    
    def add_tempo_listener(self, listener: Callable[[float, float], None]) -> None:
        """
        Add tempo change listener.
        
        Args:
            listener: Function that takes (old_bpm, new_bpm) and returns None
        """
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)
                self._stats['listeners_count'] = len(self._listeners)
                logger.debug(f"Added tempo listener: {len(self._listeners)} total")
            else:
                logger.debug("Tempo listener already registered")
    
    def remove_tempo_listener(self, listener: Callable[[float, float], None]) -> None:
        """
        Remove tempo change listener.
        
        Args:
            listener: The callback function to remove
        """
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)
                self._stats['listeners_count'] = len(self._listeners)
                logger.debug(f"Removed tempo listener: {len(self._listeners)} total")
            else:
                logger.debug("Tempo listener not found for removal")
    
    def get_current_bpm(self) -> float:
        """
        Get current BPM.
        
        Returns:
            Current beats per minute
        """
        try:
            if hasattr(self._beat_converter, 'get_bpm'):
                return self._beat_converter.get_bpm()
            elif hasattr(self._beat_converter, 'get_current_bpm'):
                return self._beat_converter.get_current_bpm()
            else:
                logger.warning("BeatFrameConverter doesn't provide BPM access")
                return self._last_bpm
        except Exception as e:
            logger.error(f"Error getting current BPM: {e}")
            return self._last_bpm
    
    def check_for_tempo_changes(self, current_frame: int) -> bool:
        """
        Check for tempo changes - designed to be called from audio thread.
        
        This method is optimized for real-time use:
        - Only checks periodically to avoid overhead
        - Uses frame-based intervals for consistent timing
        - Minimal processing when no change detected
        
        Args:
            current_frame: Current audio frame position
            
        Returns:
            True if tempo change was detected and notifications sent
        """
        # Only check periodically to avoid overhead
        if current_frame - self._last_check_frame < self._check_interval_frames:
            return False
        
        self._last_check_frame = current_frame
        self._stats['checks_performed'] += 1
        
        try:
            current_bpm = self.get_current_bpm()
            
            # Check if significant change occurred
            bpm_difference = abs(current_bpm - self._last_bpm)
            if bpm_difference >= self._tempo_change_threshold:
                
                old_bpm = self._last_bpm
                self._last_bpm = current_bpm
                self._last_check_time = time.time()
                
                # Update statistics
                self._stats['tempo_changes_detected'] += 1
                self._stats['last_change_time'] = self._last_check_time
                self._stats['last_change_from'] = old_bpm
                self._stats['last_change_to'] = current_bpm
                
                # Notify listeners
                self._notify_listeners(old_bpm, current_bpm)
                
                logger.info(f"Tempo change detected: {old_bpm:.1f} → {current_bpm:.1f} BPM "
                           f"(difference: {bpm_difference:.1f})")
                
                return True
            
            return False
                
        except Exception as e:
            logger.error(f"Error checking tempo changes at frame {current_frame}: {e}")
            return False
    
    def force_tempo_check(self) -> bool:
        """
        Force immediate tempo check regardless of interval.
        
        Useful for testing or when tempo changes are expected.
        
        Returns:
            True if tempo change was detected
        """
        self._last_check_frame = 0  # Reset to force check
        current_frame = self._get_current_frame_safe()
        return self.check_for_tempo_changes(current_frame)
    
    def set_check_interval(self, frames: int) -> None:
        """
        Set the tempo check interval.
        
        Args:
            frames: Number of frames between tempo checks
        """
        if frames <= 0:
            raise ValueError("Check interval must be positive")
        
        old_interval = self._check_interval_frames
        self._check_interval_frames = frames
        
        logger.debug(f"Tempo check interval changed: {old_interval} → {frames} frames")
    
    def set_tempo_threshold(self, threshold: float) -> None:
        """
        Set the minimum tempo change threshold.
        
        Args:
            threshold: Minimum BPM change to trigger notifications
        """
        if threshold < 0:
            raise ValueError("Tempo threshold must be non-negative")
        
        old_threshold = self._tempo_change_threshold
        self._tempo_change_threshold = threshold
        
        logger.debug(f"Tempo change threshold: {old_threshold:.1f} → {threshold:.1f} BPM")
    
    def get_stats(self) -> dict:
        """
        Get tempo controller statistics.
        
        Returns:
            Dictionary with controller statistics and state
        """
        with self._lock:
            return {
                'controller_stats': self._stats.copy(),
                'current_bpm': self._last_bpm,
                'check_interval_frames': self._check_interval_frames,
                'tempo_threshold': self._tempo_change_threshold,
                'last_check_frame': self._last_check_frame,
                'listeners_registered': len(self._listeners),
                'time_since_last_check': time.time() - self._last_check_time
            }
    
    def reset_stats(self) -> None:
        """Reset tempo controller statistics."""
        with self._lock:
            self._stats = {
                'tempo_changes_detected': 0,
                'checks_performed': 0,
                'last_change_time': 0.0,
                'last_change_from': 0.0,
                'last_change_to': 0.0,
                'listeners_count': len(self._listeners)
            }
            logger.debug("Reset tempo controller statistics")
    
    def cleanup(self) -> None:
        """
        Clean up tempo controller resources.
        
        Clears all listeners and resets state. Should be called
        when the controller is no longer needed.
        """
        with self._lock:
            listener_count = len(self._listeners)
            self._listeners.clear()
            self._stats['listeners_count'] = 0
            
            logger.info(f"TempoController cleanup: removed {listener_count} listeners")
    
    def _get_initial_bpm(self) -> float:
        """
        Get initial BPM value safely.
        
        Returns:
            Initial BPM or default if unable to retrieve
        """
        try:
            bpm = self.get_current_bpm()
            if bpm > 0:
                return bpm
            else:
                logger.warning(f"Invalid initial BPM: {bpm}, using default 120.0")
                return 120.0
        except Exception as e:
            logger.warning(f"Unable to get initial BPM: {e}, using default 120.0")
            return 120.0
    
    def _get_current_frame_safe(self) -> int:
        """
        Get current frame position safely.
        
        Returns:
            Current frame or 0 if unable to retrieve
        """
        try:
            if hasattr(self._beat_converter, 'get_current_frame'):
                return self._beat_converter.get_current_frame()
            else:
                return 0
        except Exception as e:
            logger.error(f"Error getting current frame: {e}")
            return 0
    
    def _notify_listeners(self, old_bpm: float, new_bpm: float) -> None:
        """
        Notify all listeners of tempo change.
        
        This method is designed to be exception-safe - if one listener
        fails, others will still be notified.
        
        Args:
            old_bpm: Previous BPM value
            new_bpm: New BPM value
        """
        with self._lock:
            listeners_to_notify = self._listeners.copy()  # Copy to avoid modification during iteration
        
        successful_notifications = 0
        failed_notifications = 0
        
        for listener in listeners_to_notify:
            try:
                listener(old_bpm, new_bpm)
                successful_notifications += 1
            except Exception as e:
                failed_notifications += 1
                logger.error(f"Error in tempo change listener: {e}")
        
        logger.debug(f"Tempo change notifications: {successful_notifications} successful, "
                    f"{failed_notifications} failed")

class TempoControllerManager:
    """
    Manager for multiple tempo controllers.
    
    This class can manage tempo controllers for multiple decks,
    providing centralized tempo change coordination.
    """
    
    def __init__(self):
        self._controllers: dict[str, TempoController] = {}
        self._global_listeners: List[Callable[[str, float, float], None]] = []
        self._lock = threading.RLock()
        
        logger.info("TempoControllerManager initialized")
    
    def register_controller(self, deck_id: str, controller: TempoController) -> None:
        """
        Register tempo controller for a deck.
        
        Args:
            deck_id: Unique deck identifier
            controller: TempoController instance
        """
        with self._lock:
            if deck_id in self._controllers:
                logger.warning(f"Replacing existing tempo controller for deck {deck_id}")
            
            self._controllers[deck_id] = controller
            
            # Add global listener wrapper
            def global_listener_wrapper(old_bpm: float, new_bpm: float):
                self._notify_global_listeners(deck_id, old_bpm, new_bpm)
            
            controller.add_tempo_listener(global_listener_wrapper)
            
            logger.info(f"Registered tempo controller for deck {deck_id}")
    
    def unregister_controller(self, deck_id: str) -> bool:
        """
        Unregister tempo controller for a deck.
        
        Args:
            deck_id: Deck identifier
            
        Returns:
            True if controller was found and removed
        """
        with self._lock:
            if deck_id in self._controllers:
                controller = self._controllers[deck_id]
                controller.cleanup()
                del self._controllers[deck_id]
                
                logger.info(f"Unregistered tempo controller for deck {deck_id}")
                return True
            
            return False
    
    def get_controller(self, deck_id: str) -> Optional[TempoController]:
        """
        Get tempo controller for a deck.
        
        Args:
            deck_id: Deck identifier
            
        Returns:
            TempoController instance or None if not found
        """
        return self._controllers.get(deck_id)
    
    def add_global_listener(self, listener: Callable[[str, float, float], None]) -> None:
        """
        Add global tempo change listener.
        
        Global listeners receive notifications from all registered controllers.
        
        Args:
            listener: Function that takes (deck_id, old_bpm, new_bpm)
        """
        with self._lock:
            if listener not in self._global_listeners:
                self._global_listeners.append(listener)
                logger.debug(f"Added global tempo listener: {len(self._global_listeners)} total")
    
    def remove_global_listener(self, listener: Callable[[str, float, float], None]) -> None:
        """
        Remove global tempo change listener.
        
        Args:
            listener: Function to remove
        """
        with self._lock:
            if listener in self._global_listeners:
                self._global_listeners.remove(listener)
                logger.debug(f"Removed global tempo listener: {len(self._global_listeners)} total")
    
    def check_all_controllers(self, deck_frames: dict[str, int]) -> dict[str, bool]:
        """
        Check all controllers for tempo changes.
        
        Args:
            deck_frames: Dictionary of deck_id -> current_frame
            
        Returns:
            Dictionary of deck_id -> tempo_change_detected
        """
        results = {}
        
        for deck_id, controller in self._controllers.items():
            current_frame = deck_frames.get(deck_id, 0)
            try:
                change_detected = controller.check_for_tempo_changes(current_frame)
                results[deck_id] = change_detected
            except Exception as e:
                logger.error(f"Error checking tempo for deck {deck_id}: {e}")
                results[deck_id] = False
        
        return results
    
    def get_all_stats(self) -> dict[str, dict]:
        """
        Get statistics for all controllers.
        
        Returns:
            Dictionary of deck_id -> controller_stats
        """
        with self._lock:
            return {deck_id: controller.get_stats() 
                   for deck_id, controller in self._controllers.items()}
    
    def cleanup(self) -> None:
        """Clean up all controllers."""
        with self._lock:
            for deck_id, controller in self._controllers.items():
                controller.cleanup()
            
            controller_count = len(self._controllers)
            self._controllers.clear()
            self._global_listeners.clear()
            
            logger.info(f"TempoControllerManager cleanup: removed {controller_count} controllers")
    
    def _notify_global_listeners(self, deck_id: str, old_bpm: float, new_bpm: float) -> None:
        """
        Notify global listeners of tempo change.
        
        Args:
            deck_id: Deck that had tempo change
            old_bpm: Previous BPM
            new_bpm: New BPM
        """
        with self._lock:
            listeners = self._global_listeners.copy()
        
        for listener in listeners:
            try:
                listener(deck_id, old_bpm, new_bpm)
            except Exception as e:
                logger.error(f"Error in global tempo listener: {e}")