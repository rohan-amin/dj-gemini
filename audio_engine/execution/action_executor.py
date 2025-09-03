"""
Action execution system using the command pattern.
Provides pluggable executors for different types of musical actions.
"""

import logging
from typing import Dict, Any, Optional
from ..interfaces.timing_interfaces import ActionExecutor

logger = logging.getLogger(__name__)

class CompositeActionExecutor(ActionExecutor):
    """
    Composite executor that delegates to specific action handlers.
    
    This implements the command pattern by maintaining a registry of
    specialized executors for different action types.
    """
    
    def __init__(self):
        self._executors: Dict[str, ActionExecutor] = {}
        self._fallback_executor: Optional[ActionExecutor] = None
        self._execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'executions_by_type': {}
        }
    
    def register_executor(self, action_type: str, executor: ActionExecutor) -> None:
        """
        Register executor for specific action type.
        
        Args:
            action_type: Type of action (e.g., 'play', 'stop')
            executor: Executor instance to handle this action type
        """
        self._executors[action_type] = executor
        logger.debug(f"Registered executor for action type: {action_type}")
    
    def unregister_executor(self, action_type: str) -> bool:
        """
        Unregister executor for specific action type.
        
        Args:
            action_type: Type of action to unregister
            
        Returns:
            True if executor was found and removed, False otherwise
        """
        if action_type in self._executors:
            del self._executors[action_type]
            logger.debug(f"Unregistered executor for action type: {action_type}")
            return True
        return False
    
    def set_fallback_executor(self, executor: ActionExecutor) -> None:
        """
        Set fallback executor for unhandled action types.
        
        Args:
            executor: Executor to use when no specific executor is found
        """
        self._fallback_executor = executor
        logger.debug("Set fallback executor")
    
    def execute_action(self, action_type: str, parameters: Dict[str, Any], 
                      execution_context: Dict[str, Any]) -> bool:
        """
        Execute action using appropriate executor.
        
        Args:
            action_type: Type of action to execute
            parameters: Action-specific parameters
            execution_context: Context information (frame, beat, action_id, etc.)
            
        Returns:
            True if execution was successful, False otherwise
        """
        # Update statistics
        self._execution_stats['total_executions'] += 1
        if action_type not in self._execution_stats['executions_by_type']:
            self._execution_stats['executions_by_type'][action_type] = 0
        self._execution_stats['executions_by_type'][action_type] += 1
        
        # Find appropriate executor
        executor = self._executors.get(action_type, self._fallback_executor)
        
        if executor is None:
            logger.error(f"No executor found for action type: {action_type}")
            self._execution_stats['failed_executions'] += 1
            return False
        
        try:
            # Execute the action
            success = executor.execute_action(action_type, parameters, execution_context)
            
            if success:
                self._execution_stats['successful_executions'] += 1
                logger.debug(f"Successfully executed {action_type} action: {execution_context.get('action_id', 'unknown')}")
            else:
                self._execution_stats['failed_executions'] += 1
                logger.warning(f"Failed to execute {action_type} action: {execution_context.get('action_id', 'unknown')}")
            
            return success
            
        except Exception as e:
            self._execution_stats['failed_executions'] += 1
            logger.error(f"Exception executing {action_type} action {execution_context.get('action_id', 'unknown')}: {e}")
            return False
    
    def can_execute(self, action_type: str) -> bool:
        """
        Check if action type can be executed.
        
        Args:
            action_type: Type of action to check
            
        Returns:
            True if this executor can handle the action type
        """
        if action_type in self._executors:
            return True
        
        if self._fallback_executor and self._fallback_executor.can_execute(action_type):
            return True
        
        return False
    
    def get_supported_action_types(self) -> list[str]:
        """
        Get list of supported action types.
        
        Returns:
            List of action types this executor can handle
        """
        return list(self._executors.keys())
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        return self._execution_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'executions_by_type': {}
        }
        logger.debug("Reset execution statistics")

class PlaybackActionExecutor(ActionExecutor):
    """
    Executor for playback-related actions.
    
    Handles play, pause, stop, and seek operations with frame accuracy.
    """
    
    def __init__(self, deck):
        """
        Initialize playback executor.
        
        Args:
            deck: Deck instance for playback operations
        """
        self._deck = deck
        self._supported_actions = {'play', 'pause', 'stop', 'seek'}
    
    def execute_action(self, action_type: str, parameters: Dict[str, Any], 
                      execution_context: Dict[str, Any]) -> bool:
        """
        Execute playback action.
        
        Args:
            action_type: Type of playback action
            parameters: Action parameters
            execution_context: Execution context
            
        Returns:
            True if execution was successful
        """
        if action_type == 'play':
            return self._execute_play(parameters, execution_context)
        elif action_type == 'pause':
            return self._execute_pause(parameters, execution_context)
        elif action_type == 'stop':
            return self._execute_stop(parameters, execution_context)
        elif action_type == 'seek':
            return self._execute_seek(parameters, execution_context)
        else:
            logger.error(f"Unsupported playback action type: {action_type}")
            return False
    
    def can_execute(self, action_type: str) -> bool:
        """Check if can execute playback actions."""
        return action_type in self._supported_actions
    
    def _execute_play(self, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Execute play command.
        
        Args:
            params: Play parameters (start_at_frame, start_at_beat, start_at_cue_name)
            context: Execution context
            
        Returns:
            True if play was successful
        """
        try:
            action_id = context.get('action_id', 'unknown_play')
            
            # Determine start position
            start_frame = params.get('start_at_frame')
            start_beat = params.get('start_at_beat')
            start_cue = params.get('start_at_cue_name')
            
            if start_frame is not None:
                # Frame-accurate start
                success = self._deck.play(start_at_frame=start_frame)
            elif start_beat is not None:
                # Beat-accurate start
                target_frame = self._deck.beat_manager.get_frame_for_beat(start_beat)
                success = self._deck.play(start_at_frame=target_frame)
            elif start_cue is not None:
                # Cue-based start
                success = self._deck.play(start_at_cue_name=start_cue)
            else:
                # Resume from current position
                success = self._deck.play()
            
            if success:
                logger.info(f"Started playback via action {action_id}")
            else:
                logger.error(f"Failed to start playback via action {action_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Exception during play action {context.get('action_id', 'unknown')}: {e}")
            return False
    
    def _execute_pause(self, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Execute pause command.
        
        Args:
            params: Pause parameters
            context: Execution context
            
        Returns:
            True if pause was successful
        """
        try:
            action_id = context.get('action_id', 'unknown_pause')
            
            self._deck.pause()
            logger.info(f"Paused playback via action {action_id}")
            return True
            
        except Exception as e:
            logger.error(f"Exception during pause action {context.get('action_id', 'unknown')}: {e}")
            return False
    
    def _execute_stop(self, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Execute stop command.
        
        Args:
            params: Stop parameters
            context: Execution context
            
        Returns:
            True if stop was successful
        """
        try:
            action_id = context.get('action_id', 'unknown_stop')
            
            self._deck.stop()
            logger.info(f"Stopped playback via action {action_id}")
            return True
            
        except Exception as e:
            logger.error(f"Exception during stop action {context.get('action_id', 'unknown')}: {e}")
            return False
    
    def _execute_seek(self, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Execute seek command.
        
        Args:
            params: Seek parameters (target_frame, target_beat, target_cue)
            context: Execution context
            
        Returns:
            True if seek was successful
        """
        try:
            action_id = context.get('action_id', 'unknown_seek')
            
            # Determine seek target
            target_frame = params.get('target_frame')
            target_beat = params.get('target_beat')
            target_cue = params.get('target_cue')
            
            if target_frame is not None:
                # Frame-accurate seek
                self._deck.seek_to_frame(target_frame)
                logger.info(f"Seeked to frame {target_frame} via action {action_id}")
            elif target_beat is not None:
                # Beat-accurate seek
                frame = self._deck.beat_manager.get_frame_for_beat(target_beat)
                self._deck.seek_to_frame(frame)
                logger.info(f"Seeked to beat {target_beat} (frame {frame}) via action {action_id}")
            elif target_cue is not None:
                # Cue-based seek
                if hasattr(self._deck, 'seek_to_cue'):
                    self._deck.seek_to_cue(target_cue)
                    logger.info(f"Seeked to cue {target_cue} via action {action_id}")
                else:
                    logger.error(f"Cue seeking not supported on deck {self._deck.deck_id}")
                    return False
            else:
                logger.error(f"No seek target specified in action {action_id}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Exception during seek action {context.get('action_id', 'unknown')}: {e}")
            return False

class LoggingActionExecutor(ActionExecutor):
    """
    Debug executor that logs all actions without executing them.
    
    Useful for testing and debugging the scheduling system.
    """
    
    def __init__(self):
        self._logged_actions = []
    
    def execute_action(self, action_type: str, parameters: Dict[str, Any], 
                      execution_context: Dict[str, Any]) -> bool:
        """
        Log action without executing it.
        
        Args:
            action_type: Type of action
            parameters: Action parameters
            execution_context: Execution context
            
        Returns:
            Always True (logging never fails)
        """
        log_entry = {
            'action_type': action_type,
            'parameters': parameters.copy(),
            'context': execution_context.copy(),
            'timestamp': execution_context.get('target_frame', 0)
        }
        
        self._logged_actions.append(log_entry)
        
        logger.info(f"LOGGED ACTION: {action_type} - {execution_context.get('action_id', 'unknown')} "
                   f"at frame {execution_context.get('target_frame', 'unknown')}")
        
        return True
    
    def can_execute(self, action_type: str) -> bool:
        """Can log any action type."""
        return True
    
    def get_logged_actions(self) -> list[Dict[str, Any]]:
        """Get all logged actions."""
        return self._logged_actions.copy()
    
    def clear_log(self) -> None:
        """Clear logged actions."""
        self._logged_actions.clear()