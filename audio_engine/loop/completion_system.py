"""
Event-driven completion system for DJ loop management.

This module implements a centralized completion system that handles loop completion
events and triggers associated actions. The system supports multiple completion
actions per loop and ensures reliable event delivery without fragmentation.
"""

from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import RLock, Thread
from queue import Queue, Empty
import logging
import time
import json

from .loop_events import LoopEventType, LoopEventPublisher

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Supported completion action types."""
    STOP = auto()
    PLAY = auto() 
    ACTIVATE_LOOP = auto()
    DEACTIVATE_LOOP = auto()
    VOLUME_CHANGE = auto()
    SEEK = auto()
    TRIGGER_EVENT = auto()
    CUSTOM = auto()


@dataclass
class CompletionAction:
    """
    Immutable completion action configuration.
    
    Represents a single action that should be triggered when a loop completes.
    """
    action_type: ActionType
    target_deck: Optional[str] = None
    target_loop: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    delay_ms: int = 0  # Delay before executing action
    priority: int = 0  # Higher priority actions execute first
    
    def __post_init__(self):
        """Validate action configuration after initialization."""
        if self.action_type in [ActionType.ACTIVATE_LOOP, ActionType.DEACTIVATE_LOOP]:
            if not self.target_loop:
                raise ValueError(f"Action type {self.action_type} requires target_loop")
        
        if self.action_type in [ActionType.STOP, ActionType.PLAY, ActionType.VOLUME_CHANGE]:
            if not self.target_deck:
                raise ValueError(f"Action type {self.action_type} requires target_deck")


@dataclass
class CompletionConfiguration:
    """
    Configuration for loop completion behavior.
    
    Defines what actions should be triggered when a specific loop completes.
    """
    loop_id: str
    actions: List[CompletionAction] = field(default_factory=list)
    enabled: bool = True
    max_retries: int = 3
    timeout_ms: int = 5000
    
    def add_action(self, action: CompletionAction) -> None:
        """Add a completion action to this configuration."""
        self.actions.append(action)
        # Sort by priority (higher priority first)
        self.actions.sort(key=lambda a: -a.priority)
    
    def remove_action(self, action_type: ActionType, target: Optional[str] = None) -> bool:
        """Remove matching completion actions."""
        removed_count = 0
        self.actions = [
            action for action in self.actions
            if not (action.action_type == action_type and 
                   (target is None or action.target_deck == target or action.target_loop == target))
        ]
        return removed_count > 0


@dataclass
class CompletionEvent:
    """
    Internal completion event for processing.
    """
    loop_id: str
    deck_id: str
    timestamp: float
    configuration: CompletionConfiguration
    attempt_count: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if this event has expired based on timeout."""
        return (time.time() - self.timestamp) * 1000 > self.configuration.timeout_ms


class CompletionActionHandler:
    """
    Handles execution of specific completion actions.
    
    This class provides the interface between the completion system and the
    actual deck/loop operations that need to be performed.
    """
    
    def __init__(self):
        self._custom_handlers: Dict[str, Callable] = {}
        logger.debug("🔄 Completion action handler initialized")
    
    def register_custom_handler(self, name: str, handler: Callable) -> None:
        """Register a custom action handler function."""
        self._custom_handlers[name] = handler
        logger.debug(f"🔄 Custom completion handler registered: {name}")
    
    def execute_action(self, action: CompletionAction, context: Dict[str, Any]) -> bool:
        """
        Execute a single completion action.
        
        Args:
            action: The action to execute
            context: Execution context including deck references
            
        Returns:
            True if action was executed successfully
        """
        try:
            action_type = action.action_type
            deck_manager = context.get('deck_manager')
            
            if action_type == ActionType.STOP:
                return self._execute_stop_action(action, deck_manager)
            elif action_type == ActionType.PLAY:
                return self._execute_play_action(action, deck_manager)
            elif action_type == ActionType.ACTIVATE_LOOP:
                return self._execute_activate_loop_action(action, deck_manager)
            elif action_type == ActionType.DEACTIVATE_LOOP:
                return self._execute_deactivate_loop_action(action, deck_manager)
            elif action_type == ActionType.VOLUME_CHANGE:
                return self._execute_volume_change_action(action, deck_manager)
            elif action_type == ActionType.SEEK:
                return self._execute_seek_action(action, deck_manager)
            elif action_type == ActionType.TRIGGER_EVENT:
                return self._execute_trigger_event_action(action, context)
            elif action_type == ActionType.CUSTOM:
                return self._execute_custom_action(action, context)
            else:
                logger.error(f"🔄 Unknown action type: {action_type}")
                return False
                
        except Exception as e:
            logger.error(f"🔄 Error executing completion action {action.action_type}: {e}")
            return False
    
    def _execute_stop_action(self, action: CompletionAction, deck_manager) -> bool:
        """Execute stop action on target deck."""
        if not deck_manager or not action.target_deck:
            return False
        
        deck = deck_manager.get_deck(action.target_deck)
        if not deck:
            logger.error(f"🔄 Target deck not found: {action.target_deck}")
            return False
        
        deck.stop()
        logger.info(f"🔄 COMPLETION ACTION: Stopped deck {action.target_deck}")
        return True
    
    def _execute_play_action(self, action: CompletionAction, deck_manager) -> bool:
        """Execute play action on target deck."""
        if not deck_manager or not action.target_deck:
            return False
        
        deck = deck_manager.get_deck(action.target_deck)
        if not deck:
            logger.error(f"🔄 Target deck not found: {action.target_deck}")
            return False
        
        # Check for play parameters
        position = action.parameters.get('position')
        if position is not None:
            deck.seek(position)
        
        deck.play()
        logger.info(f"🔄 COMPLETION ACTION: Started deck {action.target_deck}")
        return True
    
    def _execute_activate_loop_action(self, action: CompletionAction, deck_manager) -> bool:
        """Execute loop activation action."""
        if not deck_manager or not action.target_loop:
            return False
        
        # Find deck containing the target loop
        target_deck = None
        if action.target_deck:
            target_deck = deck_manager.get_deck(action.target_deck)
        else:
            # Search all decks for the loop
            for deck_id, deck in deck_manager.get_all_decks().items():
                if hasattr(deck, 'loop_controller') and deck.loop_controller.has_loop(action.target_loop):
                    target_deck = deck
                    break
        
        if not target_deck:
            logger.error(f"🔄 Target deck not found for loop: {action.target_loop}")
            return False
        
        if not hasattr(target_deck, 'loop_controller'):
            logger.error(f"🔄 Deck {target_deck.deck_id} has no loop controller")
            return False
        
        success = target_deck.loop_controller.activate_loop(action.target_loop)
        if success:
            logger.info(f"🔄 COMPLETION ACTION: Activated loop {action.target_loop}")
        else:
            logger.error(f"🔄 Failed to activate loop {action.target_loop}")
        return success
    
    def _execute_deactivate_loop_action(self, action: CompletionAction, deck_manager) -> bool:
        """Execute loop deactivation action."""
        if not deck_manager or not action.target_loop:
            return False
        
        # Find deck containing the target loop
        target_deck = None
        if action.target_deck:
            target_deck = deck_manager.get_deck(action.target_deck)
        else:
            # Search all decks for the loop
            for deck_id, deck in deck_manager.get_all_decks().items():
                if hasattr(deck, 'loop_controller') and deck.loop_controller.has_loop(action.target_loop):
                    target_deck = deck
                    break
        
        if not target_deck:
            logger.error(f"🔄 Target deck not found for loop: {action.target_loop}")
            return False
        
        if not hasattr(target_deck, 'loop_controller'):
            logger.error(f"🔄 Deck {target_deck.deck_id} has no loop controller")
            return False
        
        success = target_deck.loop_controller.deactivate_loop(action.target_loop)
        if success:
            logger.info(f"🔄 COMPLETION ACTION: Deactivated loop {action.target_loop}")
        else:
            logger.error(f"🔄 Failed to deactivate loop {action.target_loop}")
        return success
    
    def _execute_volume_change_action(self, action: CompletionAction, deck_manager) -> bool:
        """Execute volume change action."""
        if not deck_manager or not action.target_deck:
            return False
        
        volume = action.parameters.get('volume')
        if volume is None:
            logger.error("🔄 Volume change action missing volume parameter")
            return False
        
        deck = deck_manager.get_deck(action.target_deck)
        if not deck:
            logger.error(f"🔄 Target deck not found: {action.target_deck}")
            return False
        
        # Assuming deck has volume control
        if hasattr(deck, 'set_volume'):
            deck.set_volume(volume)
            logger.info(f"🔄 COMPLETION ACTION: Set deck {action.target_deck} volume to {volume}")
            return True
        else:
            logger.warning(f"🔄 Deck {action.target_deck} does not support volume control")
            return False
    
    def _execute_seek_action(self, action: CompletionAction, deck_manager) -> bool:
        """Execute seek action."""
        if not deck_manager or not action.target_deck:
            return False
        
        position = action.parameters.get('position')
        if position is None:
            logger.error("🔄 Seek action missing position parameter")
            return False
        
        deck = deck_manager.get_deck(action.target_deck)
        if not deck:
            logger.error(f"🔄 Target deck not found: {action.target_deck}")
            return False
        
        success = deck.seek(position)
        if success:
            logger.info(f"🔄 COMPLETION ACTION: Seeked deck {action.target_deck} to position {position}")
        return success
    
    def _execute_trigger_event_action(self, action: CompletionAction, context: Dict[str, Any]) -> bool:
        """Execute event trigger action."""
        event_publisher = context.get('event_publisher')
        if not event_publisher:
            logger.error("🔄 No event publisher available for trigger event action")
            return False
        
        event_type = action.parameters.get('event_type', 'custom_completion_event')
        event_data = action.parameters.get('event_data', {})
        
        event_publisher.publish_event({
            'type': event_type,
            'source': 'completion_system',
            'data': event_data,
            'timestamp': time.time()
        })
        
        logger.info(f"🔄 COMPLETION ACTION: Triggered event {event_type}")
        return True
    
    def _execute_custom_action(self, action: CompletionAction, context: Dict[str, Any]) -> bool:
        """Execute custom action using registered handler."""
        handler_name = action.parameters.get('handler')
        if not handler_name:
            logger.error("🔄 Custom action missing handler parameter")
            return False
        
        handler = self._custom_handlers.get(handler_name)
        if not handler:
            logger.error(f"🔄 Custom handler not found: {handler_name}")
            return False
        
        try:
            result = handler(action, context)
            logger.info(f"🔄 COMPLETION ACTION: Executed custom handler {handler_name}")
            return bool(result)
        except Exception as e:
            logger.error(f"🔄 Error in custom handler {handler_name}: {e}")
            return False


class LoopCompletionSystem:
    """
    Centralized system for handling loop completion events and triggering actions.
    
    This system provides a unified approach to loop completion handling, eliminating
    the fragmented completion mechanisms that were causing audio interruptions.
    """
    
    def __init__(self, deck_manager=None, event_publisher: Optional[LoopEventPublisher] = None):
        self._configurations: Dict[str, CompletionConfiguration] = {}
        self._pending_events: Queue = Queue()
        self._processing_thread: Optional[Thread] = None
        self._running = False
        self._lock = RLock()
        
        # System components
        self._deck_manager = deck_manager
        self._event_publisher = event_publisher or LoopEventPublisher()
        self._action_handler = CompletionActionHandler()
        
        # Statistics and monitoring
        self._stats = {
            'total_completions': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'retry_attempts': 0,
            'timeout_events': 0
        }
        
        logger.info("🔄 Loop completion system initialized")
    
    def start(self) -> None:
        """Start the completion processing thread."""
        if self._running:
            logger.warning("🔄 Completion system already running")
            return
        
        self._running = True
        self._processing_thread = Thread(
            target=self._process_completion_events,
            name="LoopCompletionProcessor",
            daemon=True
        )
        self._processing_thread.start()
        logger.info("🔄 Loop completion system started")
    
    def stop(self) -> None:
        """Stop the completion processing system."""
        if not self._running:
            return
        
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=1.0)
            if self._processing_thread.is_alive():
                logger.warning("🔄 Completion processing thread did not stop gracefully")
        
        logger.info("🔄 Loop completion system stopped")
    
    def register_completion_configuration(self, configuration: CompletionConfiguration) -> None:
        """Register completion configuration for a loop."""
        with self._lock:
            self._configurations[configuration.loop_id] = configuration
            logger.debug(f"🔄 Completion configuration registered for loop {configuration.loop_id}")
    
    def unregister_completion_configuration(self, loop_id: str) -> None:
        """Remove completion configuration for a loop."""
        with self._lock:
            removed = self._configurations.pop(loop_id, None)
            if removed:
                logger.debug(f"🔄 Completion configuration removed for loop {loop_id}")
    
    def handle_loop_completion(self, loop_id: str, deck_id: str) -> None:
        """
        Handle a loop completion event.
        
        This is the main entry point called when a loop completes. It queues
        the completion for processing by the dedicated processing thread.
        """
        with self._lock:
            configuration = self._configurations.get(loop_id)
            if not configuration or not configuration.enabled:
                logger.debug(f"🔄 No completion configuration for loop {loop_id}")
                return
            
            # Create completion event
            completion_event = CompletionEvent(
                loop_id=loop_id,
                deck_id=deck_id,
                timestamp=time.time(),
                configuration=configuration
            )
            
            # Queue for processing
            self._pending_events.put(completion_event)
            self._stats['total_completions'] += 1
            
            logger.info(f"🔄 LOOP COMPLETION: {loop_id} on {deck_id} - {len(configuration.actions)} actions queued")
    
    def _process_completion_events(self) -> None:
        """Main processing loop for completion events (runs in dedicated thread)."""
        logger.debug("🔄 Completion event processing thread started")
        
        while self._running:
            try:
                # Get next completion event with timeout
                try:
                    event = self._pending_events.get(timeout=0.1)
                except Empty:
                    continue
                
                # Check if event has expired
                if event.is_expired:
                    logger.warning(f"🔄 Completion event expired for loop {event.loop_id}")
                    self._stats['timeout_events'] += 1
                    continue
                
                # Process the completion event
                success = self._process_single_completion(event)
                
                # Handle retry logic
                if not success and event.attempt_count < event.configuration.max_retries:
                    event.attempt_count += 1
                    self._pending_events.put(event)
                    self._stats['retry_attempts'] += 1
                    logger.debug(f"🔄 Retrying completion for loop {event.loop_id} (attempt {event.attempt_count})")
                
            except Exception as e:
                logger.error(f"🔄 Error in completion processing thread: {e}")
        
        logger.debug("🔄 Completion event processing thread stopped")
    
    def _process_single_completion(self, event: CompletionEvent) -> bool:
        """
        Process a single completion event by executing all associated actions.
        
        Args:
            event: The completion event to process
            
        Returns:
            True if all actions were executed successfully
        """
        try:
            configuration = event.configuration
            all_success = True
            
            # Prepare execution context
            context = {
                'deck_manager': self._deck_manager,
                'event_publisher': self._event_publisher,
                'loop_id': event.loop_id,
                'deck_id': event.deck_id,
                'completion_time': event.timestamp
            }
            
            # Execute each completion action
            for action in configuration.actions:
                try:
                    # Handle action delay
                    if action.delay_ms > 0:
                        time.sleep(action.delay_ms / 1000.0)
                    
                    # Execute the action
                    success = self._action_handler.execute_action(action, context)
                    
                    if success:
                        self._stats['successful_actions'] += 1
                    else:
                        self._stats['failed_actions'] += 1
                        all_success = False
                        logger.warning(f"🔄 Completion action failed: {action.action_type} for loop {event.loop_id}")
                        
                except Exception as e:
                    logger.error(f"🔄 Error executing completion action {action.action_type}: {e}")
                    self._stats['failed_actions'] += 1
                    all_success = False
            
            # Publish completion processed event
            self._event_publisher.publish_event({
                'type': LoopEventType.COMPLETION_PROCESSED,
                'loop_id': event.loop_id,
                'deck_id': event.deck_id,
                'actions_count': len(configuration.actions),
                'all_successful': all_success,
                'timestamp': time.time()
            })
            
            return all_success
            
        except Exception as e:
            logger.error(f"🔄 Error processing completion for loop {event.loop_id}: {e}")
            return False
    
    def get_completion_configuration(self, loop_id: str) -> Optional[CompletionConfiguration]:
        """Get completion configuration for a specific loop."""
        with self._lock:
            return self._configurations.get(loop_id)
    
    def get_all_configurations(self) -> Dict[str, CompletionConfiguration]:
        """Get all completion configurations."""
        with self._lock:
            return self._configurations.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get completion system statistics."""
        with self._lock:
            stats = self._stats.copy()
            stats['active_configurations'] = len(self._configurations)
            stats['pending_events'] = self._pending_events.qsize()
            stats['running'] = self._running
            return stats
    
    def register_custom_action_handler(self, name: str, handler: Callable) -> None:
        """Register a custom completion action handler."""
        self._action_handler.register_custom_handler(name, handler)
    
    def clear_pending_events(self) -> int:
        """Clear all pending completion events and return the count cleared."""
        count = 0
        try:
            while True:
                self._pending_events.get_nowait()
                count += 1
        except Empty:
            pass
        
        if count > 0:
            logger.info(f"🔄 Cleared {count} pending completion events")
        
        return count


def create_completion_configuration_from_json(config_data: Dict[str, Any]) -> CompletionConfiguration:
    """
    Create a CompletionConfiguration from JSON configuration data.
    
    Args:
        config_data: Dictionary containing completion configuration
        
    Returns:
        CompletionConfiguration object
        
    Example config_data:
    {
        "loop_id": "intro_loop",
        "enabled": true,
        "actions": [
            {
                "action_type": "ACTIVATE_LOOP",
                "target_loop": "main_loop",
                "priority": 1
            },
            {
                "action_type": "VOLUME_CHANGE",
                "target_deck": "deck_b",
                "parameters": {"volume": 0.8},
                "delay_ms": 500
            }
        ]
    }
    """
    loop_id = config_data['loop_id']
    enabled = config_data.get('enabled', True)
    max_retries = config_data.get('max_retries', 3)
    timeout_ms = config_data.get('timeout_ms', 5000)
    
    configuration = CompletionConfiguration(
        loop_id=loop_id,
        enabled=enabled,
        max_retries=max_retries,
        timeout_ms=timeout_ms
    )
    
    # Parse actions
    for action_data in config_data.get('actions', []):
        action_type = ActionType[action_data['action_type']]
        target_deck = action_data.get('target_deck')
        target_loop = action_data.get('target_loop')
        parameters = action_data.get('parameters', {})
        delay_ms = action_data.get('delay_ms', 0)
        priority = action_data.get('priority', 0)
        
        action = CompletionAction(
            action_type=action_type,
            target_deck=target_deck,
            target_loop=target_loop,
            parameters=parameters,
            delay_ms=delay_ms,
            priority=priority
        )
        
        configuration.add_action(action)
    
    return configuration