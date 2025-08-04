# Script execution state machine for dj-gemini
# Manages script lifecycle with pause/resume, error recovery, and state persistence

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import threading
import time
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ScriptState(Enum):
    """Script execution states"""
    UNLOADED = auto()       # No script loaded
    LOADING = auto()        # Loading/parsing script
    LOADED = auto()         # Script loaded and validated
    RUNNING = auto()        # Script executing
    PAUSED = auto()         # Script paused (can resume)
    COMPLETED = auto()      # Script finished successfully
    ERROR = auto()          # Script error (can recover)
    STOPPED = auto()        # Script stopped by user

class ScriptEvent(Enum):
    """Events that trigger state transitions"""
    LOAD_SCRIPT = auto()
    LOAD_COMPLETE = auto()
    LOAD_FAILED = auto()
    START = auto()
    PAUSE = auto()
    RESUME = auto()
    STOP = auto()
    COMPLETE = auto()
    ERROR = auto()
    RESET = auto()

class ActionState(Enum):
    """Individual action states"""
    PENDING = auto()        # Not yet executed
    READY = auto()          # Ready to execute (trigger conditions met)
    EXECUTING = auto()      # Currently executing
    COMPLETED = auto()      # Successfully completed
    FAILED = auto()         # Execution failed
    SKIPPED = auto()        # Skipped due to conditions

@dataclass
class ActionInstance:
    """Individual action with execution state"""
    action_id: str
    command: str
    deck_id: Optional[str]
    parameters: Dict[str, Any]
    trigger: Dict[str, Any]
    
    # Execution state
    state: ActionState = ActionState.PENDING
    execution_count: int = 0
    last_execution_time: Optional[float] = None
    execution_duration: Optional[float] = None
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)

@dataclass
class ScriptExecutionState:
    """Complete script execution state"""
    # Script information
    script_path: Optional[str] = None
    script_name: str = ""
    script_data: Dict[str, Any] = field(default_factory=dict)
    
    # Execution state
    current_state: ScriptState = ScriptState.UNLOADED
    previous_state: ScriptState = ScriptState.UNLOADED
    
    # Actions
    actions: Dict[str, ActionInstance] = field(default_factory=dict)
    action_order: List[str] = field(default_factory=list)
    
    # Progress tracking
    total_actions: int = 0
    completed_actions: int = 0
    failed_actions: int = 0
    current_action_index: int = 0
    
    # Timing
    script_start_time: Optional[float] = None
    script_end_time: Optional[float] = None
    pause_time: Optional[float] = None
    total_pause_duration: float = 0.0
    
    # Error handling
    error_count: int = 0
    last_error: Optional[str] = None
    
    # State persistence
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)
    auto_save_interval: float = 30.0  # seconds
    last_auto_save: float = 0.0

class ScriptStateMachine:
    """Manages script execution lifecycle"""
    
    VALID_TRANSITIONS = {
        ScriptState.UNLOADED: [ScriptState.LOADING, ScriptState.ERROR],
        ScriptState.LOADING: [ScriptState.LOADED, ScriptState.ERROR, ScriptState.UNLOADED],
        ScriptState.LOADED: [ScriptState.RUNNING, ScriptState.UNLOADED, ScriptState.ERROR],
        ScriptState.RUNNING: [ScriptState.PAUSED, ScriptState.COMPLETED, ScriptState.STOPPED, ScriptState.ERROR],
        ScriptState.PAUSED: [ScriptState.RUNNING, ScriptState.STOPPED, ScriptState.ERROR],
        ScriptState.COMPLETED: [ScriptState.LOADED, ScriptState.UNLOADED],
        ScriptState.STOPPED: [ScriptState.LOADED, ScriptState.UNLOADED],
        ScriptState.ERROR: [ScriptState.LOADED, ScriptState.UNLOADED, ScriptState.RUNNING]  # Recovery
    }
    
    def __init__(self, script_id: str = "main"):
        self.script_id = script_id
        self.state = ScriptExecutionState()
        self._state_lock = threading.RLock()
        
        # Callbacks
        self._state_change_callbacks: List[Callable] = []
        self._action_callbacks: List[Callable] = []
        self._error_callbacks: List[Callable] = []
        
        # Persistence
        self._checkpoint_dir = Path("script_checkpoints")
        self._checkpoint_dir.mkdir(exist_ok=True)
        
        logger.info(f"Script state machine initialized: {script_id}")
    
    def load_script(self, script_path: str, script_data: Dict[str, Any]) -> bool:
        """Load and parse script"""
        with self._state_lock:
            if not self._can_transition_to(ScriptState.LOADING):
                logger.error(f"Cannot load script from state {self.state.current_state}")
                return False
            
            self._transition_to_state(ScriptState.LOADING)
            
            try:
                # Store script information
                self.state.script_path = script_path
                self.state.script_name = script_data.get("mix_name", Path(script_path).stem)
                self.state.script_data = script_data.copy()
                
                # Parse actions
                actions_data = script_data.get("actions", [])
                self._parse_actions(actions_data)
                
                # Validate script
                if self._validate_script():
                    self._transition_to_state(ScriptState.LOADED)
                    logger.info(f"Script loaded successfully: {self.state.script_name} ({len(self.state.actions)} actions)")
                    return True
                else:
                    self._transition_to_state(ScriptState.ERROR, error="Script validation failed")
                    return False
                    
            except Exception as e:
                self._transition_to_state(ScriptState.ERROR, error=str(e))
                logger.error(f"Failed to load script: {e}")
                return False
    
    def start_script(self) -> bool:
        """Start script execution"""
        with self._state_lock:
            if not self._can_transition_to(ScriptState.RUNNING):
                logger.error(f"Cannot start script from state {self.state.current_state}")
                return False
            
            self.state.script_start_time = time.time()
            self.state.current_action_index = 0
            self._transition_to_state(ScriptState.RUNNING)
            
            logger.info(f"Script execution started: {self.state.script_name}")
            return True
    
    def pause_script(self) -> bool:
        """Pause script execution"""
        with self._state_lock:
            if not self._can_transition_to(ScriptState.PAUSED):
                logger.error(f"Cannot pause script from state {self.state.current_state}")
                return False
            
            self.state.pause_time = time.time()
            self._transition_to_state(ScriptState.PAUSED)
            
            # Create checkpoint
            self._create_checkpoint()
            
            logger.info(f"Script execution paused: {self.state.script_name}")
            return True
    
    def resume_script(self) -> bool:
        """Resume script execution"""
        with self._state_lock:
            if not self._can_transition_to(ScriptState.RUNNING):
                logger.error(f"Cannot resume script from state {self.state.current_state}")
                return False
            
            # Calculate pause duration
            if self.state.pause_time:
                pause_duration = time.time() - self.state.pause_time
                self.state.total_pause_duration += pause_duration
                self.state.pause_time = None
            
            self._transition_to_state(ScriptState.RUNNING)
            
            logger.info(f"Script execution resumed: {self.state.script_name}")
            return True
    
    def stop_script(self) -> bool:
        """Stop script execution"""
        with self._state_lock:
            if self.state.current_state not in [ScriptState.RUNNING, ScriptState.PAUSED]:
                logger.warning(f"Cannot stop script from state {self.state.current_state}")
                return False
            
            self.state.script_end_time = time.time()
            self._transition_to_state(ScriptState.STOPPED)
            
            logger.info(f"Script execution stopped: {self.state.script_name}")
            return True
    
    def reset_script(self) -> bool:
        """Reset script to initial state"""
        with self._state_lock:
            # Reset all action states
            for action in self.state.actions.values():
                action.state = ActionState.PENDING
                action.execution_count = 0
                action.last_execution_time = None
                action.error_message = None
                action.retry_count = 0
            
            # Reset execution state
            self.state.current_action_index = 0
            self.state.completed_actions = 0
            self.state.failed_actions = 0
            self.state.script_start_time = None
            self.state.script_end_time = None
            self.state.pause_time = None
            self.state.total_pause_duration = 0.0
            
            self._transition_to_state(ScriptState.LOADED)
            
            logger.info(f"Script reset: {self.state.script_name}")
            return True
    
    def execute_action(self, action_id: str, executor_callback: Callable) -> bool:
        """Execute a specific action"""
        with self._state_lock:
            if action_id not in self.state.actions:
                logger.error(f"Action not found: {action_id}")
                return False
            
            action = self.state.actions[action_id]
            
            if action.state != ActionState.READY:
                logger.warning(f"Action {action_id} not ready for execution (state: {action.state})")
                return False
            
            # Mark as executing
            action.state = ActionState.EXECUTING
            action.execution_count += 1
            action.last_execution_time = time.time()
            
            try:
                # Execute via callback
                success = executor_callback(action)
                
                if success:
                    action.state = ActionState.COMPLETED
                    self.state.completed_actions += 1
                    
                    # Update dependencies
                    self._update_dependent_actions(action_id)
                    
                    logger.debug(f"Action completed: {action_id}")
                else:
                    self._handle_action_failure(action, "Execution returned False")
                
                return success
                
            except Exception as e:
                self._handle_action_failure(action, str(e))
                return False
            
            finally:
                if action.last_execution_time:
                    action.execution_duration = time.time() - action.last_execution_time
    
    def get_ready_actions(self) -> List[ActionInstance]:
        """Get actions ready for execution"""
        with self._state_lock:
            ready_actions = []
            for action in self.state.actions.values():
                if action.state == ActionState.READY:
                    ready_actions.append(action)
            return ready_actions
    
    def get_script_progress(self) -> Dict[str, Any]:
        """Get script execution progress"""
        with self._state_lock:
            total_actions = len(self.state.actions)
            completed = self.state.completed_actions
            failed = self.state.failed_actions
            
            progress_percent = (completed / total_actions * 100) if total_actions > 0 else 0
            
            # Calculate execution time
            execution_time = 0.0
            if self.state.script_start_time:
                if self.state.script_end_time:
                    execution_time = self.state.script_end_time - self.state.script_start_time
                else:
                    execution_time = time.time() - self.state.script_start_time
                execution_time -= self.state.total_pause_duration
            
            return {
                'state': self.state.current_state.name,
                'script_name': self.state.script_name,
                'total_actions': total_actions,
                'completed_actions': completed,
                'failed_actions': failed,
                'progress_percent': progress_percent,
                'execution_time_seconds': execution_time,
                'current_action_index': self.state.current_action_index,
                'error_count': self.state.error_count,
                'last_error': self.state.last_error
            }
    
    def save_checkpoint(self, checkpoint_name: str = None) -> bool:
        """Save current state to checkpoint"""
        with self._state_lock:
            try:
                if not checkpoint_name:
                    checkpoint_name = f"auto_save_{int(time.time())}"
                
                checkpoint_path = self._checkpoint_dir / f"{self.script_id}_{checkpoint_name}.json"
                
                checkpoint_data = {
                    'script_id': self.script_id,
                    'timestamp': time.time(),
                    'state': self.state.current_state.name,
                    'script_path': self.state.script_path,
                    'script_name': self.state.script_name,
                    'current_action_index': self.state.current_action_index,
                    'completed_actions': self.state.completed_actions,
                    'failed_actions': self.state.failed_actions,
                    'total_pause_duration': self.state.total_pause_duration,
                    'action_states': {
                        action_id: {
                            'state': action.state.name,
                            'execution_count': action.execution_count,
                            'retry_count': action.retry_count,
                            'error_message': action.error_message
                        }
                        for action_id, action in self.state.actions.items()
                    }
                }
                
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                
                self.state.last_auto_save = time.time()
                logger.info(f"Checkpoint saved: {checkpoint_path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
                return False
    
    def load_checkpoint(self, checkpoint_name: str) -> bool:
        """Load state from checkpoint"""
        with self._state_lock:
            try:
                checkpoint_path = self._checkpoint_dir / f"{self.script_id}_{checkpoint_name}.json"
                
                if not checkpoint_path.exists():
                    logger.error(f"Checkpoint not found: {checkpoint_path}")
                    return False
                
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                
                # Restore state
                self.state.current_action_index = checkpoint_data['current_action_index']
                self.state.completed_actions = checkpoint_data['completed_actions']
                self.state.failed_actions = checkpoint_data['failed_actions']
                self.state.total_pause_duration = checkpoint_data['total_pause_duration']
                
                # Restore action states
                action_states = checkpoint_data.get('action_states', {})
                for action_id, action_state_data in action_states.items():
                    if action_id in self.state.actions:
                        action = self.state.actions[action_id]
                        action.state = ActionState[action_state_data['state']]
                        action.execution_count = action_state_data['execution_count']
                        action.retry_count = action_state_data['retry_count']
                        action.error_message = action_state_data.get('error_message')
                
                logger.info(f"Checkpoint loaded: {checkpoint_path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return False
    
    def add_state_change_callback(self, callback: Callable):
        """Add callback for state changes"""
        self._state_change_callbacks.append(callback)
    
    def add_action_callback(self, callback: Callable):
        """Add callback for action state changes"""
        self._action_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Add callback for errors"""
        self._error_callbacks.append(callback)
    
    def _parse_actions(self, actions_data: List[Dict[str, Any]]):
        """Parse actions from script data"""
        self.state.actions.clear()
        self.state.action_order.clear()
        
        for i, action_data in enumerate(actions_data):
            action_id = action_data.get("action_id", f"action_{i}")
            
            action = ActionInstance(
                action_id=action_id,
                command=action_data.get("command", ""),
                deck_id=action_data.get("deck_id"),
                parameters=action_data.get("parameters", {}),
                trigger=action_data.get("trigger", {})
            )
            
            # Check for script_start trigger
            if action.trigger.get("type") == "script_start":
                action.state = ActionState.READY
            
            self.state.actions[action_id] = action
            self.state.action_order.append(action_id)
        
        self.state.total_actions = len(self.state.actions)
    
    def _validate_script(self) -> bool:
        """Validate script structure and dependencies"""
        try:
            # Check for required fields
            if not self.state.actions:
                logger.error("Script has no actions")
                return False
            
            # Validate each action
            for action_id, action in self.state.actions.items():
                if not action.command:
                    logger.error(f"Action {action_id} has no command")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Script validation error: {e}")
            return False
    
    def _can_transition_to(self, target_state: ScriptState) -> bool:
        """Check if transition to target state is valid"""
        current_state = self.state.current_state
        return target_state in self.VALID_TRANSITIONS.get(current_state, [])
    
    def _transition_to_state(self, new_state: ScriptState, **kwargs):
        """Transition to new state"""
        old_state = self.state.current_state
        self.state.previous_state = old_state
        self.state.current_state = new_state
        
        # Handle state-specific actions
        if new_state == ScriptState.ERROR:
            error_msg = kwargs.get('error', 'Unknown error')
            self.state.last_error = error_msg
            self.state.error_count += 1
        
        # Notify callbacks
        for callback in self._state_change_callbacks:
            try:
                callback(old_state, new_state, self.state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
        
        logger.debug(f"Script state transition: {old_state} -> {new_state}")
    
    def _handle_action_failure(self, action: ActionInstance, error_message: str):
        """Handle action execution failure"""
        action.error_message = error_message
        action.retry_count += 1
        
        if action.retry_count <= action.max_retries:
            # Retry
            action.state = ActionState.READY
            logger.warning(f"Action {action.action_id} failed, will retry ({action.retry_count}/{action.max_retries}): {error_message}")
        else:
            # Give up
            action.state = ActionState.FAILED
            self.state.failed_actions += 1
            logger.error(f"Action {action.action_id} failed permanently: {error_message}")
            
            # Notify error callbacks
            for callback in self._error_callbacks:
                try:
                    callback(action.action_id, error_message)
                except Exception as e:
                    logger.error(f"Error in error callback: {e}")
    
    def _update_dependent_actions(self, completed_action_id: str):
        """Update actions that depend on completed action"""
        # Simple implementation - could be more sophisticated
        pass
    
    def _create_checkpoint(self):
        """Create automatic checkpoint"""
        current_time = time.time()
        if current_time - self.state.last_auto_save >= self.state.auto_save_interval:
            self.save_checkpoint("auto_pause")