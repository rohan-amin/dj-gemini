"""
Timing-related interfaces for the DJ Gemini audio engine.
Defines contracts for beat/frame conversion, tempo changes, and action execution.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable

class BeatFrameConverter(ABC):
    """
    Interface for converting between musical beats and audio frames.
    
    This abstraction allows the scheduling system to work with any timing source
    without being tightly coupled to a specific BeatManager implementation.
    """
    
    @abstractmethod
    def get_frame_for_beat(self, beat_number: float) -> int:
        """
        Convert beat number to frame position.
        
        Args:
            beat_number: Musical beat number (1-based, can be fractional)
            
        Returns:
            Frame position corresponding to the beat
        """
        pass
    
    @abstractmethod
    def get_beat_from_frame(self, frame: int) -> float:
        """
        Convert frame position to beat number.
        
        Args:
            frame: Audio frame position
            
        Returns:
            Musical beat number (1-based, can be fractional)
        """
        pass
    
    @abstractmethod
    def get_current_beat(self) -> float:
        """
        Get current beat position.
        
        Returns:
            Current musical beat number
        """
        pass
    
    @abstractmethod
    def get_current_frame(self) -> int:
        """
        Get current frame position.
        
        Returns:
            Current audio frame position
        """
        pass

class TempoChangeNotifier(ABC):
    """
    Interface for tempo change notifications.
    
    This allows components to be notified when tempo changes occur,
    enabling them to rebuild their frame-based calculations.
    """
    
    @abstractmethod
    def add_tempo_listener(self, listener: Callable[[float, float], None]) -> None:
        """
        Add tempo change listener.
        
        Args:
            listener: Callback function that receives (old_bpm, new_bpm)
        """
        pass
    
    @abstractmethod
    def remove_tempo_listener(self, listener: Callable[[float, float], None]) -> None:
        """
        Remove tempo change listener.
        
        Args:
            listener: Callback function to remove
        """
        pass
    
    @abstractmethod
    def get_current_bpm(self) -> float:
        """
        Get current BPM.
        
        Returns:
            Current beats per minute
        """
        pass

class ActionExecutor(ABC):
    """
    Interface for executing scheduled actions.
    
    This allows different types of actions to have specialized executors
    while maintaining a common interface for the scheduling system.
    """
    
    @abstractmethod
    def execute_action(self, action_type: str, parameters: Dict[str, Any], 
                      execution_context: Dict[str, Any]) -> bool:
        """
        Execute action with given parameters.
        
        Args:
            action_type: Type of action to execute (e.g., 'activate_loop')
            parameters: Action-specific parameters
            execution_context: Context information (frame, beat, action_id, etc.)
            
        Returns:
            True if execution was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def can_execute(self, action_type: str) -> bool:
        """
        Check if this executor can handle the given action type.
        
        Args:
            action_type: Type of action to check
            
        Returns:
            True if this executor can handle the action type
        """
        pass