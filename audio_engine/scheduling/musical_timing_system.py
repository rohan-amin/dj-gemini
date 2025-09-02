"""
Musical timing system facade that orchestrates all timing components.
Provides a unified interface for sample-accurate musical scheduling.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from ..interfaces.scheduling_interfaces import ScheduledAction
from .frame_queue import PriorityFrameQueue
from .action_scheduler import MusicalActionScheduler
from .tempo_controller import TempoController
from ..execution.action_executor import CompositeActionExecutor, LoopActionExecutor, PlaybackActionExecutor
from ..adapters.beat_manager_adapter import BeatManagerAdapter
from ..adapters.deck_executor_adapter import DeckExecutorAdapter

logger = logging.getLogger(__name__)

class MusicalTimingSystem:
    """
    Facade that orchestrates all musical timing components.
    
    This system provides sample-accurate scheduling of musical actions
    while maintaining beat-based references that adapt to tempo changes.
    It's designed to be a drop-in replacement for the existing event
    scheduling system with much better timing accuracy.
    """
    
    def __init__(self, deck):
        """
        Initialize musical timing system for a deck.
        
        Args:
            deck: Existing Deck instance to enhance with musical timing
        """
        self._deck = deck
        self._deck_id = getattr(deck, 'deck_id', 'unknown')
        
        # Create adapters to bridge with existing system
        self._beat_converter = BeatManagerAdapter(deck.beat_manager)
        
        # Create core components with dependency injection
        self._frame_queue = PriorityFrameQueue()
        self._tempo_controller = TempoController(self._beat_converter)
        self._scheduler = MusicalActionScheduler(
            self._beat_converter,
            self._tempo_controller,
            self._frame_queue
        )
        
        # Create action execution system
        self._executor = CompositeActionExecutor()
        self._deck_executor = DeckExecutorAdapter(deck)
        
        # Register action executors
        self._register_action_executors()
        
        # System state
        self._initialized = True
        self._processing_enabled = True
        
        # Loop completion system
        self._loop_completion_actions = {}  # loop_action_id -> [actions_to_trigger]
        
        logger.info(f"MusicalTimingSystem initialized for deck {self._deck_id}")
    
    def schedule_beat_action(self, beat_number: float, action_type: str, 
                           parameters: Dict[str, Any], action_id: Optional[str] = None,
                           priority: int = 0) -> str:
        """
        Schedule an action to execute at a specific musical beat.
        
        This is the main public API for scheduling musical actions.
        
        Args:
            beat_number: Musical beat number where action should execute (1-based)
            action_type: Type of action ('activate_loop', 'play', 'stop', etc.)
            parameters: Action-specific parameters
            action_id: Optional unique identifier (auto-generated if None)
            priority: Action priority (lower number = higher priority)
            
        Returns:
            Action ID for cancellation/tracking
            
        Raises:
            ValueError: If parameters are invalid or system is not initialized
        """
        if not self._initialized:
            raise ValueError("Musical timing system not initialized")
        
        if not self._processing_enabled:
            logger.warning(f"Deck {self._deck_id}: Scheduling disabled, ignoring action {action_type}")
            return ""
        
        # Generate action ID if not provided
        if action_id is None:
            action_id = f"{action_type}_{beat_number}_{int(time.time() * 1000)}"
        
        # Create scheduled action
        action = ScheduledAction(
            action_id=action_id,
            beat_number=beat_number,
            action_type=action_type,
            parameters=parameters,
            priority=priority
        )
        
        try:
            # Schedule using the musical action scheduler
            scheduled_id = self._scheduler.schedule_musical_action(action)
            
            logger.info(f"Deck {self._deck_id}: Scheduled {action_type} at beat {beat_number} "
                       f"(action_id: {scheduled_id})")
            
            return scheduled_id
            
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Failed to schedule {action_type} at beat {beat_number}: {e}")
            raise
    
    def cancel_action(self, action_id: str) -> bool:
        """
        Cancel a scheduled action.
        
        Args:
            action_id: ID of action to cancel
            
        Returns:
            True if action was found and cancelled
        """
        try:
            success = self._scheduler.cancel_action(action_id)
            if success:
                logger.info(f"Deck {self._deck_id}: Cancelled action {action_id}")
            else:
                logger.warning(f"Deck {self._deck_id}: Action {action_id} not found for cancellation")
            return success
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error cancelling action {action_id}: {e}")
            return False
    
    def register_loop_completion_action(self, loop_action_id: str, action_type: str, 
                                      parameters: Dict[str, Any], action_id: str, 
                                      target_deck_id: Optional[str] = None, priority: int = 0) -> None:
        """
        Register an action to be executed when a specific loop completes.
        
        Args:
            loop_action_id: ID of the loop action to wait for completion
            action_type: Type of action to execute when loop completes
            parameters: Action-specific parameters
            action_id: Unique identifier for this dependent action
            target_deck_id: Deck to execute action on (defaults to this deck)
            priority: Action priority
        """
        if not self._initialized:
            logger.error(f"Deck {self._deck_id}: Cannot register loop completion action - system not initialized")
            return
        
        if loop_action_id not in self._loop_completion_actions:
            self._loop_completion_actions[loop_action_id] = []
        
        completion_action = {
            'action_id': action_id,
            'action_type': action_type,
            'parameters': parameters,
            'target_deck_id': target_deck_id or self._deck_id,
            'priority': priority
        }
        
        self._loop_completion_actions[loop_action_id].append(completion_action)
        
        logger.info(f"🔗 Deck {self._deck_id}: Registered loop completion action {action_id} for loop {loop_action_id}")
        logger.info(f"🔗 Action details: {action_type} on {target_deck_id or self._deck_id} with params {parameters}")
        logger.info(f"🔗 Total completion actions registered: {len(self._loop_completion_actions)}")
    
    def handle_loop_completion(self, loop_action_id: str) -> int:
        """
        Handle loop completion by triggering all dependent actions.
        
        Args:
            loop_action_id: ID of the loop that just completed
            
        Returns:
            Number of actions triggered
        """
        logger.info(f"🔄 Deck {self._deck_id}: handle_loop_completion called for loop {loop_action_id}")
        logger.info(f"🔄 System state: initialized={self._initialized}, processing_enabled={self._processing_enabled}")
        logger.info(f"🔄 Registered completion loops: {list(self._loop_completion_actions.keys())}")
        
        if not self._initialized or not self._processing_enabled:
            logger.warning(f"Deck {self._deck_id}: Loop completion handling disabled")
            return 0
        
        if loop_action_id not in self._loop_completion_actions:
            logger.warning(f"Deck {self._deck_id}: No completion actions registered for loop {loop_action_id}")
            logger.warning(f"Available loop completion actions: {list(self._loop_completion_actions.keys())}")
            return 0
        
        actions = self._loop_completion_actions[loop_action_id]
        triggered_count = 0
        
        logger.info(f"🔄 Deck {self._deck_id}: Processing {len(actions)} completion actions for {loop_action_id} simultaneously")
        
        for completion_action in actions:
            try:
                action_id = completion_action['action_id']
                action_type = completion_action['action_type'] 
                parameters = completion_action['parameters']
                target_deck_id = completion_action['target_deck_id']
                priority = completion_action['priority']
                
                logger.info(f"🔄 Deck {self._deck_id}: Loop {loop_action_id} completed - triggering action {action_id}")
                
                # Create execution context for both same-deck and cross-deck actions
                execution_context = {
                    'action_id': action_id,
                    'trigger_type': 'loop_completion',
                    'source_loop_id': loop_action_id,
                    'target_frame': 0  # Immediate execution
                }
                
                # If action targets this deck, queue command via command queue system
                if target_deck_id == self._deck_id:
                    # PROPER QUEUE-BASED APPROACH: All loop actions go through command queue
                    success = self._queue_completion_command(action_type, parameters, action_id)
                    if success:
                        triggered_count += 1
                        logger.info(f"✅ Deck {self._deck_id}: Queued completion command {action_id}")
                    else:
                        logger.error(f"❌ Deck {self._deck_id}: Failed to queue completion command {action_id}")
                else:
                    # Action targets different deck - use clean command queue approach
                    try:
                        engine = None
                        if hasattr(self._deck, 'engine'):
                            engine = self._deck.engine
                        
                        if engine:
                            target_deck = engine._get_or_create_deck(target_deck_id)
                            if target_deck and hasattr(target_deck, 'musical_timing_system') and target_deck.musical_timing_system:
                                # Queue command on target deck's command queue
                                target_success = target_deck.musical_timing_system._queue_completion_command(
                                    action_type, parameters, action_id
                                )
                                if target_success:
                                    triggered_count += 1
                                    logger.info(f"✅ Deck {self._deck_id}: Cross-deck completion command {action_id} queued on {target_deck_id}")
                                else:
                                    logger.error(f"❌ Deck {self._deck_id}: Failed to queue cross-deck completion command {action_id} on {target_deck_id}")
                            else:
                                logger.error(f"Deck {self._deck_id}: Target deck {target_deck_id} not found or no musical timing system")
                        else:
                            logger.error(f"Deck {self._deck_id}: No engine reference for cross-deck action {action_id}")
                    except Exception as cross_deck_error:
                        logger.error(f"Deck {self._deck_id}: Error in cross-deck command queuing for {action_id}: {cross_deck_error}")
                    
            except Exception as e:
                logger.error(f"Deck {self._deck_id}: Error executing completion action {completion_action['action_id']}: {e}")
        
        # Remove the completed loop's actions since they've been triggered
        del self._loop_completion_actions[loop_action_id]
        
        logger.info(f"Deck {self._deck_id}: Loop completion processed - triggered {triggered_count} actions for loop {loop_action_id}")
        return triggered_count
    
    def process_audio_buffer(self, start_frame: int, end_frame: int) -> Dict[str, Any]:
        """
        Process musical actions for an audio buffer.
        
        This should be called from the deck's audio callback to execute
        sample-accurate musical actions.
        
        Args:
            start_frame: Start of audio buffer frame range
            end_frame: End of audio buffer frame range
            
        Returns:
            Dictionary with processing results and statistics
        """
        if not self._processing_enabled:
            return {'actions_executed': 0, 'processing_disabled': True}
        
        processing_start = time.time()
        actions_executed = 0
        actions_failed = 0
        
        try:
            # Check for tempo changes
            tempo_changed = self._tempo_controller.check_for_tempo_changes(start_frame)
            
            # Get actions for this buffer
            actions = self._scheduler.get_actions_for_buffer(start_frame, end_frame)
            
            # Execute actions in frame order
            for frame, action in actions:
                try:
                    execution_context = {
                        'action_id': action.action_id,
                        'target_frame': frame,
                        'beat_number': action.beat_number,
                        'buffer_start': start_frame,
                        'buffer_end': end_frame
                    }
                    
                    # Convert parameters back to dict if needed
                    if hasattr(action.parameters, 'items'):
                        params = dict(action.parameters)
                    else:
                        params = action.parameters_dict
                    
                    # Execute the action
                    success = self._executor.execute_action(
                        action.action_type,
                        params,
                        execution_context
                    )
                    
                    if success:
                        actions_executed += 1
                        logger.debug(f"Deck {self._deck_id}: Executed {action.action_type} "
                                   f"at frame {frame} (beat {action.beat_number})")
                    else:
                        actions_failed += 1
                        logger.warning(f"Deck {self._deck_id}: Failed to execute {action.action_type} "
                                     f"at frame {frame}")
                    
                except Exception as e:
                    actions_failed += 1
                    logger.error(f"Deck {self._deck_id}: Exception executing {action.action_id}: {e}")
            
            processing_time = time.time() - processing_start
            
            return {
                'actions_executed': actions_executed,
                'actions_failed': actions_failed,
                'tempo_changed': tempo_changed,
                'processing_time_ms': processing_time * 1000,
                'buffer_frames': end_frame - start_frame
            }
            
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error processing audio buffer {start_frame}-{end_frame}: {e}")
            return {
                'actions_executed': 0,
                'actions_failed': 0,
                'error': str(e)
            }
    
    def enable_processing(self, enabled: bool = True) -> None:
        """
        Enable or disable action processing.
        
        When disabled, actions are not executed but scheduling continues.
        Useful for testing or temporary suspension.
        
        Args:
            enabled: Whether to enable action processing
        """
        self._processing_enabled = enabled
        status = "enabled" if enabled else "disabled"
        logger.info(f"Deck {self._deck_id}: Action processing {status}")
    
    def get_scheduled_actions(self) -> Dict[str, ScheduledAction]:
        """
        Get all currently scheduled actions.
        
        Returns:
            Dictionary mapping action_id to ScheduledAction
        """
        return self._scheduler.get_scheduled_actions()
    
    def get_next_action_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the next scheduled action.
        
        Returns:
            Dictionary with next action info, or None if no actions
        """
        return self._scheduler.get_next_action_info()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dictionary with timing system statistics
        """
        try:
            scheduler_stats = self._scheduler.get_stats()
            tempo_stats = self._tempo_controller.get_stats()
            executor_stats = self._executor.get_execution_stats()
            
            # Add beat converter info
            beat_converter_info = {}
            if hasattr(self._beat_converter, 'get_adapter_info'):
                beat_converter_info = self._beat_converter.get_adapter_info()
            
            return {
                'deck_id': self._deck_id,
                'system_initialized': self._initialized,
                'processing_enabled': self._processing_enabled,
                'scheduler': scheduler_stats,
                'tempo_controller': tempo_stats,
                'action_executor': executor_stats,
                'beat_converter': beat_converter_info,
                'loop_completion_actions': len(self._loop_completion_actions),
                'pending_loop_completions': list(self._loop_completion_actions.keys()),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error getting system stats: {e}")
            return {
                'deck_id': self._deck_id,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def force_tempo_check(self) -> bool:
        """
        Force immediate tempo check and queue rebuild.
        
        Returns:
            True if tempo change was detected
        """
        try:
            return self._tempo_controller.force_tempo_check()
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error in forced tempo check: {e}")
            return False
    
    def cleanup(self) -> None:
        """
        Clean up system resources.
        
        Should be called when the timing system is no longer needed.
        """
        try:
            logger.info(f"Deck {self._deck_id}: Cleaning up musical timing system")
            
            # Clean up components
            self._scheduler.cleanup()
            self._tempo_controller.cleanup()
            
            # Clear state
            self._initialized = False
            self._processing_enabled = False
            
            logger.info(f"Deck {self._deck_id}: Musical timing system cleanup complete")
            
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error during cleanup: {e}")
    
    def _queue_completion_command(self, action_type: str, parameters: Dict[str, Any], action_id: str) -> bool:
        """
        Queue a completion command using the deck's existing command queue system.
        
        Args:
            action_type: Type of action (stop, activate_loop, etc.)
            parameters: Action parameters
            action_id: Action identifier
            
        Returns:
            True if command was queued successfully
        """
        try:
            # Import deck command constants
            from ..deck import DECK_CMD_STOP, DECK_CMD_PLAY, DECK_CMD_PAUSE, DECK_CMD_SEEK
            
            # Map action types to deck command constants (loop commands removed)
            command_mapping = {
                'stop': DECK_CMD_STOP,
                'play': DECK_CMD_PLAY,
                'pause': DECK_CMD_PAUSE,
                'seek': DECK_CMD_SEEK
                # activate_loop removed - will be reimplemented with new architecture
            }
            
            if action_type not in command_mapping:
                logger.error(f"Deck {self._deck_id}: Unsupported completion action type: {action_type}")
                return False
            
            deck_command = command_mapping[action_type]
            
            # Queue the command using deck's existing command queue
            if hasattr(self._deck, 'command_queue'):
                # Format parameters for specific commands
                if action_type == 'stop':
                    self._deck.command_queue.put((deck_command, None))
                elif action_type == 'activate_loop':
                    # Convert to format expected by deck command handler
                    command_data = {
                        'start_beat': parameters.get('start_at_beat'),
                        'length_beats': parameters.get('length_beats'),
                        'repetitions': parameters.get('repetitions'),
                        'action_id': action_id
                    }
                    self._deck.command_queue.put((deck_command, command_data))
                else:
                    # Generic parameter passing
                    self._deck.command_queue.put((deck_command, parameters))
                
                logger.info(f"Deck {self._deck_id}: Queued {action_type} command for completion action {action_id}")
                return True
            else:
                logger.error(f"Deck {self._deck_id}: No command queue available")
                return False
                
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error queuing completion command {action_id}: {e}")
            return False
    
    def _register_action_executors(self) -> None:
        """Register action executors with the composite executor."""
        try:
            # Register deck-based executor for all actions it supports
            supported_actions = self._deck_executor.get_supported_actions()
            
            for action_type in supported_actions:
                if self._deck_executor.can_execute(action_type):
                    self._executor.register_executor(action_type, self._deck_executor)
            
            # Could register specialized executors here if needed
            # self._executor.register_executor('custom_action', CustomExecutor())
            
            registered_types = self._executor.get_supported_action_types()
            logger.debug(f"Deck {self._deck_id}: Registered executors for: {registered_types}")
            
        except Exception as e:
            logger.error(f"Deck {self._deck_id}: Error registering action executors: {e}")
            raise
    
    def __repr__(self) -> str:
        """String representation of the timing system."""
        try:
            scheduled_count = len(self.get_scheduled_actions())
            next_action = self.get_next_action_info()
            next_info = f"next at beat {next_action['beat_number']}" if next_action else "no actions"
            
            return (f"MusicalTimingSystem(deck={self._deck_id}, "
                   f"scheduled={scheduled_count}, {next_info})")
        except:
            return f"MusicalTimingSystem(deck={self._deck_id})"

class MusicalTimingSystemManager:
    """
    Manager for multiple musical timing systems.
    
    This can coordinate timing across multiple decks and provide
    global timing system management.
    """
    
    def __init__(self):
        self._timing_systems: Dict[str, MusicalTimingSystem] = {}
        self._global_stats = {
            'systems_created': 0,
            'systems_active': 0,
            'total_actions_scheduled': 0
        }
        
        logger.info("MusicalTimingSystemManager initialized")
    
    def create_timing_system(self, deck) -> MusicalTimingSystem:
        """
        Create musical timing system for a deck.
        
        Args:
            deck: Deck instance to enhance with timing system
            
        Returns:
            MusicalTimingSystem instance
        """
        deck_id = getattr(deck, 'deck_id', f'deck_{len(self._timing_systems)}')
        
        if deck_id in self._timing_systems:
            logger.warning(f"Replacing existing timing system for deck {deck_id}")
            self._timing_systems[deck_id].cleanup()
        
        timing_system = MusicalTimingSystem(deck)
        self._timing_systems[deck_id] = timing_system
        
        self._global_stats['systems_created'] += 1
        self._global_stats['systems_active'] = len(self._timing_systems)
        
        logger.info(f"Created musical timing system for deck {deck_id}")
        return timing_system
    
    def get_timing_system(self, deck_id: str) -> Optional[MusicalTimingSystem]:
        """
        Get timing system for a deck.
        
        Args:
            deck_id: Deck identifier
            
        Returns:
            MusicalTimingSystem or None if not found
        """
        return self._timing_systems.get(deck_id)
    
    def remove_timing_system(self, deck_id: str) -> bool:
        """
        Remove and cleanup timing system for a deck.
        
        Args:
            deck_id: Deck identifier
            
        Returns:
            True if system was found and removed
        """
        if deck_id in self._timing_systems:
            self._timing_systems[deck_id].cleanup()
            del self._timing_systems[deck_id]
            
            self._global_stats['systems_active'] = len(self._timing_systems)
            logger.info(f"Removed timing system for deck {deck_id}")
            return True
        
        return False
    
    def process_all_audio_buffers(self, deck_frames: Dict[str, tuple[int, int]]) -> Dict[str, Dict[str, Any]]:
        """
        Process audio buffers for all timing systems.
        
        Args:
            deck_frames: Dictionary of deck_id -> (start_frame, end_frame)
            
        Returns:
            Dictionary of deck_id -> processing_results
        """
        results = {}
        
        for deck_id, timing_system in self._timing_systems.items():
            if deck_id in deck_frames:
                start_frame, end_frame = deck_frames[deck_id]
                results[deck_id] = timing_system.process_audio_buffer(start_frame, end_frame)
            else:
                results[deck_id] = {'skipped': 'no frame data'}
        
        return results
    
    def get_global_stats(self) -> Dict[str, Any]:
        """
        Get global timing system statistics.
        
        Returns:
            Dictionary with manager and all system stats
        """
        system_stats = {}
        total_scheduled = 0
        
        for deck_id, timing_system in self._timing_systems.items():
            try:
                stats = timing_system.get_system_stats()
                system_stats[deck_id] = stats
                
                # Count scheduled actions
                if 'scheduler' in stats and 'musical_actions_count' in stats['scheduler']:
                    total_scheduled += stats['scheduler']['musical_actions_count']
            except Exception as e:
                system_stats[deck_id] = {'error': str(e)}
        
        self._global_stats['total_actions_scheduled'] = total_scheduled
        
        return {
            'manager_stats': self._global_stats.copy(),
            'timing_systems': system_stats,
            'timestamp': time.time()
        }
    
    def cleanup_all(self) -> None:
        """Clean up all timing systems."""
        deck_ids = list(self._timing_systems.keys())
        
        for deck_id in deck_ids:
            self.remove_timing_system(deck_id)
        
        logger.info("MusicalTimingSystemManager: Cleaned up all timing systems")