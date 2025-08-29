"""
Frame-based priority queue implementation for precise musical timing.
Provides sample-accurate scheduling of actions using audio frame positions.
"""

import heapq
import logging
from typing import List, Tuple, Optional
from ..interfaces.scheduling_interfaces import FrameQueue, ScheduledAction

logger = logging.getLogger(__name__)

class PriorityFrameQueue(FrameQueue):
    """
    Concrete implementation of frame-based priority queue.
    
    Uses Python's heapq for efficient priority queue operations.
    Actions are ordered by frame position, with priority as a tiebreaker.
    A counter ensures stable sorting for actions at the same frame/priority.
    """
    
    def __init__(self):
        # Heap structure: (frame, priority, counter, action)
        # - frame: Primary sort key (earlier frames first)
        # - priority: Secondary sort key (higher priority = lower number)
        # - counter: Tertiary sort key for stable sorting
        # - action: The actual ScheduledAction
        self._queue: List[Tuple[int, int, int, ScheduledAction]] = []
        self._counter = 0  # For stable sorting when frame/priority are equal
        self._size = 0  # Track size explicitly for O(1) access
    
    def schedule_action(self, frame: int, action: ScheduledAction) -> None:
        """
        Schedule action at specific frame.
        
        Args:
            frame: Target frame for action execution
            action: Action to schedule
        """
        if frame < 0:
            logger.warning(f"Scheduling action {action.action_id} at negative frame {frame}")
        
        # Use negative priority so that higher priority (lower number) comes first
        heap_entry = (frame, -action.priority, self._counter, action)
        heapq.heappush(self._queue, heap_entry)
        
        self._counter += 1
        self._size += 1
        
        logger.debug(f"Scheduled action {action.action_id} at frame {frame} "
                    f"(priority {action.priority}, queue size: {self._size})")
    
    def get_actions_in_range(self, start_frame: int, end_frame: int) -> List[Tuple[int, ScheduledAction]]:
        """
        Get all actions in frame range, removing them from queue.
        
        This method efficiently extracts all actions that should execute
        within the specified frame range, maintaining the queue structure
        for remaining actions.
        
        Args:
            start_frame: Start of frame range (inclusive)
            end_frame: End of frame range (inclusive)
            
        Returns:
            List of (frame, action) tuples sorted by frame
        """
        if start_frame > end_frame:
            logger.warning(f"Invalid frame range: {start_frame} > {end_frame}")
            return []
        
        actions_to_execute = []
        remaining_actions = []
        
        # Allow a small tolerance for slightly late actions (e.g. 100ms at 44.1kHz = ~4410 frames)
        late_tolerance_frames = 4410  # ~100ms tolerance for late actions
        
        # Process all items in queue
        while self._queue:
            frame, neg_priority, counter, action = heapq.heappop(self._queue)
            self._size -= 1
            
            if start_frame <= frame <= end_frame:
                # Action should execute in this buffer
                actions_to_execute.append((frame, action))
                logger.debug(f"Action {action.action_id} scheduled for execution at frame {frame}")
            elif frame > end_frame:
                # Action is in the future, keep it
                remaining_actions.append((frame, neg_priority, counter, action))
            elif frame >= (start_frame - late_tolerance_frames):
                # Action is slightly late but within tolerance - execute it now
                logger.info(f"Executing slightly late action {action.action_id} at frame {frame} "
                           f"(current range: {start_frame}-{end_frame}, tolerance: {late_tolerance_frames})")
                actions_to_execute.append((start_frame, action))  # Execute at start of current buffer
            else:
                # Action is too far in the past
                logger.warning(f"Dropping past action {action.action_id} at frame {frame} "
                             f"(current range: {start_frame}-{end_frame})")
        
        # Rebuild queue with remaining actions
        self._queue = remaining_actions
        self._size = len(remaining_actions)
        heapq.heapify(self._queue)
        
        # Sort actions by frame first, then by priority (lower number = higher priority)
        # We need to extract the original priority from the stored data since we negated it
        actions_with_priority = []
        for frame, action in actions_to_execute:
            # Find the original priority from the action
            original_priority = action.priority
            actions_with_priority.append((frame, original_priority, action))
        
        # Sort by frame first, then by priority (ascending = higher priority first)
        actions_with_priority.sort(key=lambda x: (x[0], x[1]))
        
        # Convert back to (frame, action) tuples
        sorted_actions = [(frame, action) for frame, _, action in actions_with_priority]
        
        logger.debug(f"Returning {len(sorted_actions)} actions for execution, "
                    f"{self._size} actions remaining in queue")
        
        return sorted_actions
    
    def rebuild_from_actions(self, actions: List[ScheduledAction], beat_converter) -> None:
        """
        Rebuild queue from musical actions using current tempo.
        
        This is called when tempo changes to recalculate frame positions
        for all scheduled actions based on their musical beat numbers.
        
        Args:
            actions: List of musical actions to reschedule
            beat_converter: BeatFrameConverter for current tempo calculations
        """
        logger.info(f"Rebuilding frame queue for {len(actions)} actions")
        
        # Clear current queue
        self.clear()
        
        # Get current position to filter out past actions
        try:
            current_frame = beat_converter.get_current_frame()
            current_beat = beat_converter.get_beat_from_frame(current_frame)
        except Exception as e:
            logger.error(f"Error getting current position during rebuild: {e}")
            current_frame = 0
            current_beat = 0.0
        
        # Reschedule future actions
        scheduled_count = 0
        skipped_count = 0
        
        for action in actions:
            if action.beat_number > current_beat:
                try:
                    # Calculate new frame position with current tempo
                    target_frame = beat_converter.get_frame_for_beat(action.beat_number)
                    
                    if target_frame > current_frame:
                        self.schedule_action(target_frame, action)
                        scheduled_count += 1
                    else:
                        logger.debug(f"Skipping action {action.action_id} - "
                                   f"calculated frame {target_frame} <= current {current_frame}")
                        skipped_count += 1
                except Exception as e:
                    logger.error(f"Error calculating frame for action {action.action_id} "
                               f"at beat {action.beat_number}: {e}")
                    skipped_count += 1
            else:
                logger.debug(f"Skipping past action {action.action_id} - "
                           f"beat {action.beat_number} <= current {current_beat}")
                skipped_count += 1
        
        logger.info(f"Queue rebuild complete: {scheduled_count} actions scheduled, "
                   f"{skipped_count} actions skipped")
    
    def clear(self) -> None:
        """Clear all scheduled actions."""
        old_size = self._size
        self._queue.clear()
        self._counter = 0
        self._size = 0
        
        if old_size > 0:
            logger.debug(f"Cleared {old_size} actions from queue")
    
    def size(self) -> int:
        """
        Get number of scheduled actions.
        
        Returns:
            Number of actions currently in the queue
        """
        return self._size
    
    def peek_next_action(self) -> Optional[Tuple[int, ScheduledAction]]:
        """
        Peek at the next action without removing it.
        
        Returns:
            (frame, action) tuple for the next action, or None if queue is empty
        """
        if not self._queue:
            return None
        
        frame, neg_priority, counter, action = self._queue[0]
        return (frame, action)
    
    def get_actions_at_frame(self, target_frame: int) -> List[ScheduledAction]:
        """
        Get all actions scheduled at a specific frame without removing them.
        
        This is useful for debugging and inspection.
        
        Args:
            target_frame: Frame to search for
            
        Returns:
            List of actions scheduled at the target frame
        """
        actions = []
        for frame, neg_priority, counter, action in self._queue:
            if frame == target_frame:
                actions.append(action)
            elif frame > target_frame:
                break  # Heap is sorted by frame, so we can stop early
        
        return actions
    
    def get_frame_range(self) -> Optional[Tuple[int, int]]:
        """
        Get the frame range of all scheduled actions.
        
        Returns:
            (min_frame, max_frame) tuple, or None if queue is empty
        """
        if not self._queue:
            return None
        
        min_frame = self._queue[0][0]  # First item has minimum frame
        max_frame = max(item[0] for item in self._queue)
        
        return (min_frame, max_frame)
    
    def get_debug_info(self) -> dict:
        """
        Get debug information about the queue state.
        
        Returns:
            Dictionary with queue statistics and state information
        """
        frame_range = self.get_frame_range()
        
        info = {
            'size': self._size,
            'counter': self._counter,
            'frame_range': frame_range,
            'is_empty': self._size == 0
        }
        
        if self._queue:
            # Get action types distribution
            action_types = {}
            for _, _, _, action in self._queue:
                action_type = action.action_type
                action_types[action_type] = action_types.get(action_type, 0) + 1
            info['action_types'] = action_types
            
            # Get next few actions for debugging
            next_actions = []
            for i in range(min(5, len(self._queue))):
                frame, _, _, action = self._queue[i]
                next_actions.append({
                    'frame': frame,
                    'beat': action.beat_number,
                    'type': action.action_type,
                    'id': action.action_id
                })
            info['next_actions'] = next_actions
        
        return info