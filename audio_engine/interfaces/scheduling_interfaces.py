"""
Scheduling-related interfaces for the DJ Gemini audio engine.
Defines contracts for frame queues and scheduled actions.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

@dataclass(frozen=True)
class ScheduledAction:
    """
    Immutable scheduled action data.
    
    This represents a musical action that should be executed at a specific beat.
    The frozen dataclass ensures immutability, preventing accidental modification
    after scheduling.
    """
    action_id: str
    beat_number: float
    action_type: str
    parameters: Dict[str, Any]
    priority: int = 0
    
    def __post_init__(self):
        """
        Ensure parameters are immutable.
        Convert dict to frozenset of items to prevent modification.
        """
        if not isinstance(self.parameters, (frozenset, tuple)):
            # Convert mutable dict to immutable representation
            if isinstance(self.parameters, dict):
                frozen_params = frozenset(self.parameters.items())
            else:
                frozen_params = tuple(self.parameters) if hasattr(self.parameters, '__iter__') else self.parameters
            object.__setattr__(self, 'parameters', frozen_params)
    
    @property
    def parameters_dict(self) -> Dict[str, Any]:
        """
        Get parameters as a regular dictionary.
        
        Returns:
            Dictionary representation of parameters
        """
        if isinstance(self.parameters, frozenset):
            return dict(self.parameters)
        elif isinstance(self.parameters, dict):
            return self.parameters.copy()
        else:
            return {}

class FrameQueue(ABC):
    """
    Interface for frame-based action queue.
    
    This abstraction allows different queue implementations (priority queue,
    time-based queue, etc.) while providing a consistent interface for
    the scheduling system.
    """
    
    @abstractmethod
    def schedule_action(self, frame: int, action: ScheduledAction) -> None:
        """
        Schedule action at specific frame.
        
        Args:
            frame: Target frame for action execution
            action: Action to schedule
        """
        pass
    
    @abstractmethod
    def get_actions_in_range(self, start_frame: int, end_frame: int) -> List[Tuple[int, ScheduledAction]]:
        """
        Get all actions in frame range, removing them from queue.
        
        Actions are returned in frame order and removed from the queue
        to prevent duplicate execution.
        
        Args:
            start_frame: Start of frame range (inclusive)
            end_frame: End of frame range (inclusive)
            
        Returns:
            List of (frame, action) tuples sorted by frame
        """
        pass
    
    @abstractmethod
    def rebuild_from_actions(self, actions: List[ScheduledAction], 
                            beat_converter) -> None:
        """
        Rebuild queue from musical actions using current tempo.
        
        This is called when tempo changes to recalculate frame positions
        for all scheduled actions.
        
        Args:
            actions: List of musical actions to reschedule
            beat_converter: BeatFrameConverter for current tempo calculations
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all scheduled actions."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """
        Get number of scheduled actions.
        
        Returns:
            Number of actions currently in the queue
        """
        pass
    
    @abstractmethod
    def peek_next_action(self) -> Tuple[int, ScheduledAction]:
        """
        Peek at the next action without removing it.
        
        Returns:
            (frame, action) tuple for the next action, or None if queue is empty
        """
        pass