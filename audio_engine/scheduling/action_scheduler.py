"""
Musical action scheduler with dependency injection for precise timing.
Coordinates frame-based scheduling while maintaining musical beat references.
"""

import time
import logging
from typing import Dict, List, Optional, Callable, Tuple
from ..interfaces.scheduling_interfaces import FrameQueue, ScheduledAction
from ..interfaces.timing_interfaces import BeatFrameConverter, TempoChangeNotifier

logger = logging.getLogger(__name__)

class MusicalActionScheduler:
    """
    Pure scheduling logic with injected dependencies.
    
    This class separates scheduling concerns from execution and timing,
    making it highly testable and modular. All dependencies are injected,
    allowing for easy mocking and different implementations.
    """
    
    def __init__(self, 
                 beat_converter: BeatFrameConverter,
                 tempo_notifier: TempoChangeNotifier,
                 frame_queue: FrameQueue):
        """
        Initialize scheduler with dependency injection.
        
        Args:
            beat_converter: Interface for beat/frame conversions
            tempo_notifier: Interface for tempo change notifications
            frame_queue: Interface for frame-based action queue
        """
        # Injected dependencies
        self._beat_converter = beat_converter
        self._tempo_notifier = tempo_notifier
        self._frame_queue = frame_queue
        
        # Musical action storage (tempo-independent)
        # Key: action_id, Value: ScheduledAction
        self._musical_actions: Dict[str, ScheduledAction] = {}
        
        # Tempo change handling
        self._current_bpm = tempo_notifier.get_current_bpm()
        self._rebuild_pending = False
        self._rebuild_requested_at = 0.0
        
        # Statistics and monitoring
        self._stats = {
            'actions_scheduled': 0,
            'actions_cancelled': 0,
            'queue_rebuilds': 0,
            'last_rebuild_time': 0.0,
            'tempo_changes_handled': 0
        }
        
        # Register for tempo changes
        self._tempo_notifier.add_tempo_listener(self._on_tempo_change)
        
        logger.info(f"MusicalActionScheduler initialized with BPM: {self._current_bpm}")
    
    def schedule_musical_action(self, action: ScheduledAction) -> str:
        """
        Schedule action by musical beat number.
        
        Args:
            action: Musical action to schedule
            
        Returns:
            Action ID for cancellation/tracking
            
        Raises:
            ValueError: If beat_converter fails to convert beat to frame
        """
        try:
            # Store musical action for tempo change handling
            self._musical_actions[action.action_id] = action
            
            # Convert musical beat to current frame position
            target_frame = self._beat_converter.get_frame_for_beat(action.beat_number)
            
            # Schedule in frame queue
            self._frame_queue.schedule_action(target_frame, action)
            
            # Update statistics
            self._stats['actions_scheduled'] += 1
            
            logger.debug(f"Scheduled musical action {action.action_id} "
                        f"at beat {action.beat_number} (frame {target_frame})")
            
            return action.action_id
            
        except Exception as e:
            logger.error(f"Failed to schedule action {action.action_id} at beat {action.beat_number}: {e}")
            # Clean up partial state
            if action.action_id in self._musical_actions:
                del self._musical_actions[action.action_id]
            raise ValueError(f"Failed to schedule action: {e}")
    
    def cancel_action(self, action_id: str) -> bool:
        """
        Cancel scheduled action.
        
        Args:
            action_id: ID of action to cancel
            
        Returns:
            True if action was found and cancelled, False otherwise
        """
        if action_id not in self._musical_actions:
            logger.debug(f"Cannot cancel unknown action: {action_id}")
            return False
        
        # Remove from musical actions storage
        del self._musical_actions[action_id]
        
        # Note: We can't efficiently remove from the frame queue,
        # but get_actions_for_buffer will filter out cancelled actions
        
        self._stats['actions_cancelled'] += 1
        logger.debug(f"Cancelled action: {action_id}")
        
        return True
    
    def get_actions_for_buffer(self, start_frame: int, end_frame: int) -> List[Tuple[int, ScheduledAction]]:
        """
        Get actions for audio buffer processing.
        
        This method handles tempo change rebuilding and filters out
        cancelled actions before returning.
        
        Args:
            start_frame: Start of audio buffer frame range
            end_frame: End of audio buffer frame range
            
        Returns:
            List of (frame, action) tuples for execution
        """
        # Handle pending queue rebuilds
        if self._rebuild_pending:
            self._rebuild_queue_safe()
        
        # Get actions from frame queue
        raw_actions = self._frame_queue.get_actions_in_range(start_frame, end_frame)
        
        # Filter out cancelled actions
        valid_actions = []
        cancelled_count = 0
        
        for frame, action in raw_actions:
            if action.action_id in self._musical_actions:
                valid_actions.append((frame, action))
            else:
                cancelled_count += 1
                logger.debug(f"Filtered out cancelled action: {action.action_id}")
        
        if cancelled_count > 0:
            logger.debug(f"Filtered out {cancelled_count} cancelled actions")
        
        logger.debug(f"Returning {len(valid_actions)} valid actions for buffer "
                    f"range {start_frame}-{end_frame}")
        
        return valid_actions
    
    def force_rebuild_queue(self) -> None:
        """
        Force immediate queue rebuild.
        
        This can be called externally when tempo changes are detected
        outside of the normal notification system.
        """
        logger.info("Forcing queue rebuild")
        self._rebuild_pending = True
        self._rebuild_requested_at = time.time()
    
    def get_scheduled_actions(self) -> Dict[str, ScheduledAction]:
        """
        Get all currently scheduled musical actions.
        
        Returns:
            Dictionary of action_id -> ScheduledAction
        """
        return self._musical_actions.copy()
    
    def get_next_action_info(self) -> Optional[Dict[str, any]]:
        """
        Get information about the next scheduled action.
        
        Returns:
            Dictionary with next action info, or None if no actions scheduled
        """
        next_action_tuple = self._frame_queue.peek_next_action()
        if not next_action_tuple:
            return None
        
        frame, action = next_action_tuple
        
        # Check if action is still valid (not cancelled)
        if action.action_id not in self._musical_actions:
            return None
        
        try:
            current_frame = self._beat_converter.get_current_frame()
            frames_until_execution = frame - current_frame
            
            return {
                'action_id': action.action_id,
                'beat_number': action.beat_number,
                'action_type': action.action_type,
                'target_frame': frame,
                'current_frame': current_frame,
                'frames_until_execution': frames_until_execution,
                'priority': action.priority
            }
        except Exception as e:
            logger.error(f"Error getting next action info: {e}")
            return None
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get scheduler statistics and state information.
        
        Returns:
            Dictionary with scheduler statistics
        """
        try:
            current_frame = self._beat_converter.get_current_frame()
            current_beat = self._beat_converter.get_beat_from_frame(current_frame)
        except Exception:
            current_frame = 0
            current_beat = 0.0
        
        frame_queue_info = self._frame_queue.get_debug_info() if hasattr(self._frame_queue, 'get_debug_info') else {}
        
        return {
            'scheduler_stats': self._stats.copy(),
            'musical_actions_count': len(self._musical_actions),
            'frame_queue_size': self._frame_queue.size(),
            'current_bpm': self._current_bpm,
            'current_frame': current_frame,
            'current_beat': current_beat,
            'rebuild_pending': self._rebuild_pending,
            'frame_queue_info': frame_queue_info
        }
    
    def cleanup(self) -> None:
        """
        Clean up scheduler resources.
        
        This should be called when the scheduler is no longer needed
        to properly unregister from tempo change notifications.
        """
        try:
            # Unregister from tempo changes
            self._tempo_notifier.remove_tempo_listener(self._on_tempo_change)
            
            # Clear all scheduled actions
            self._musical_actions.clear()
            self._frame_queue.clear()
            
            logger.info("MusicalActionScheduler cleaned up")
            
        except Exception as e:
            logger.error(f"Error during scheduler cleanup: {e}")
    
    def _on_tempo_change(self, old_bpm: float, new_bpm: float) -> None:
        """
        Handle tempo change notification.
        
        Args:
            old_bmp: Previous BPM value
            new_bpm: New BPM value
        """
        logger.info(f"Tempo change detected: {old_bpm:.1f} → {new_bpm:.1f} BPM")
        
        self._current_bpm = new_bpm
        self._rebuild_pending = True
        self._rebuild_requested_at = time.time()
        self._stats['tempo_changes_handled'] += 1
        
        # Log impact
        actions_count = len(self._musical_actions)
        if actions_count > 0:
            logger.info(f"Will rebuild queue for {actions_count} scheduled actions")
    
    def _rebuild_queue_safe(self) -> None:
        """
        Safely rebuild frame queue with current tempo.
        
        This method ensures the rebuild happens efficiently and handles
        any errors that might occur during the rebuild process.
        """
        if not self._rebuild_pending:
            return
        
        rebuild_start_time = time.time()
        
        try:
            logger.info(f"Rebuilding frame queue for {len(self._musical_actions)} actions")
            
            # Get all musical actions that haven't been cancelled
            active_actions = list(self._musical_actions.values())
            
            # Rebuild queue with new frame calculations
            self._frame_queue.rebuild_from_actions(active_actions, self._beat_converter)
            
            # Update statistics
            self._stats['queue_rebuilds'] += 1
            self._stats['last_rebuild_time'] = time.time()
            
            rebuild_duration = time.time() - rebuild_start_time
            logger.info(f"Queue rebuild completed in {rebuild_duration*1000:.1f}ms")
            
        except Exception as e:
            logger.error(f"Error during queue rebuild: {e}")
            # Don't clear rebuild_pending flag so we'll try again
            return
        
        # Mark rebuild as complete
        self._rebuild_pending = False