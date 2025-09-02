"""
ActionLoopAdapter: Bridge between action-based JSON format and LoopController

This module provides the integration layer that allows the existing action-based
configuration format to work with the centralized loop management system.

Key Features:
- Converts activate_loop actions to LoopController calls
- Handles on_loop_complete triggers by registering callbacks
- Maps action parameters to loop controller format
- Maintains backward compatibility with existing JSON configs
- Integrates with EventScheduler for completion action execution
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class PendingCompletionAction:
    """Represents an action waiting to be executed when a loop completes."""
    loop_action_id: str
    target_action: Dict[str, Any]
    registered_at: float


class ActionLoopAdapter:
    """
    Bridges the existing action-based JSON format with LoopController.
    
    This adapter allows DJs to continue using the familiar action format while
    benefiting from the robust centralized loop management system.
    """
    
    def __init__(self, loop_controller, event_scheduler=None):
        """
        Initialize the ActionLoopAdapter.
        
        Args:
            loop_controller: LoopController instance for this deck
            event_scheduler: Optional EventScheduler for triggering completion actions
        """
        self.loop_controller = loop_controller
        self.event_scheduler = event_scheduler
        
        # Track completion callbacks for on_loop_complete triggers
        self.completion_callbacks: Dict[str, List[PendingCompletionAction]] = {}
        
        # Track active action-based loops (action_id -> loop info)
        self.active_action_loops: Dict[str, Dict[str, Any]] = {}
        
        # Lock for thread-safe callback management
        self._callback_lock = threading.RLock()
        
        # Subscribe to loop events from controller
        self._setup_loop_event_handling()
        
        logger.info("ActionLoopAdapter initialized for action-based loop integration")
    
    def _setup_loop_event_handling(self):
        """Setup event handling for loop completions."""
        if hasattr(self.loop_controller, 'event_publisher'):
            # Subscribe to loop completion events
            self.loop_controller.event_publisher.subscribe('loop_completed', self._handle_loop_completion_event)
            logger.debug("ActionLoopAdapter subscribed to loop completion events")
    
    def handle_activate_loop_action(self, action: Dict[str, Any]) -> bool:
        """
        Convert activate_loop action to LoopController calls.
        
        Args:
            action: Action dictionary from JSON config
            
        Returns:
            True if loop was successfully created and activated
            
        Example action format:
        {
            "action_id": "loop_A_at_33",
            "command": "activate_loop",
            "deck_id": "deckA", 
            "parameters": {
                "start_at_beat": 33,
                "length_beats": 8,
                "repetitions": 4
            }
        }
        """
        try:
            # Extract action parameters
            action_id = action.get('action_id') or action.get('id')
            if not action_id:
                logger.error("ActionLoopAdapter: activate_loop action missing action_id")
                return False
            
            deck_id = action.get('deck_id')
            params = action.get('parameters', {})
            
            # Extract loop parameters from action format
            start_beat = params.get('start_at_beat')
            length_beats = params.get('length_beats')
            repetitions = params.get('repetitions', 1)
            
            if start_beat is None or length_beats is None:
                logger.error(f"ActionLoopAdapter: Missing required parameters for loop {action_id}")
                return False
            
            # Convert to loop controller format
            end_beat = start_beat + length_beats
            
            logger.info(f"ActionLoopAdapter: Converting action {action_id} to loop: beats {start_beat}-{end_beat}, {repetitions} iterations")
            
            # Create loop via controller
            success = self.loop_controller.create_loop(
                action_id, 
                float(start_beat), 
                float(end_beat), 
                int(repetitions)
            )
            
            if not success:
                logger.error(f"ActionLoopAdapter: Failed to create loop {action_id}")
                return False
            
            # Activate the loop
            activate_success = self.loop_controller.activate_loop(action_id)
            
            if activate_success:
                # Track this action-based loop
                self.active_action_loops[action_id] = {
                    'deck_id': deck_id,
                    'action': action,
                    'activated_at': time.time()
                }
                logger.info(f"ActionLoopAdapter: Successfully activated loop {action_id}")
                return True
            else:
                logger.error(f"ActionLoopAdapter: Failed to activate loop {action_id}")
                return False
                
        except Exception as e:
            logger.error(f"ActionLoopAdapter: Error processing activate_loop action: {e}")
            return False
    
    def handle_deactivate_loop_action(self, action: Dict[str, Any]) -> bool:
        """
        Handle deactivate_loop action.
        
        Args:
            action: Deactivate loop action dictionary
            
        Returns:
            True if loop was successfully deactivated
        """
        try:
            action_id = action.get('action_id') or action.get('id')
            if not action_id:
                logger.error("ActionLoopAdapter: deactivate_loop action missing action_id")
                return False
            
            success = self.loop_controller.cancel_loop(action_id)
            
            if success:
                # Remove from active tracking
                self.active_action_loops.pop(action_id, None)
                logger.info(f"ActionLoopAdapter: Successfully deactivated loop {action_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"ActionLoopAdapter: Error processing deactivate_loop action: {e}")
            return False
    
    def register_completion_trigger(self, trigger_action: Dict[str, Any], target_action: Dict[str, Any]) -> None:
        """
        Register action to execute when loop completes.
        
        This handles "on_loop_complete" triggers from the existing action format.
        
        Args:
            trigger_action: Action with on_loop_complete trigger
            target_action: Action to execute when trigger loop completes
            
        Example:
        trigger_action = {
            "trigger": {
                "type": "on_loop_complete",
                "source_deck_id": "deckA",
                "loop_action_id": "first_loop"
            }
        }
        """
        try:
            trigger_info = trigger_action.get('trigger', {})
            loop_action_id = trigger_info.get('loop_action_id')
            
            if not loop_action_id:
                logger.error("ActionLoopAdapter: on_loop_complete trigger missing loop_action_id")
                return
            
            with self._callback_lock:
                if loop_action_id not in self.completion_callbacks:
                    self.completion_callbacks[loop_action_id] = []
                
                pending_action = PendingCompletionAction(
                    loop_action_id=loop_action_id,
                    target_action=target_action,
                    registered_at=time.time()
                )
                
                self.completion_callbacks[loop_action_id].append(pending_action)
                
                logger.info(f"ActionLoopAdapter: Registered completion callback for loop {loop_action_id} -> action {target_action.get('action_id', 'unknown')}")
                
        except Exception as e:
            logger.error(f"ActionLoopAdapter: Error registering completion trigger: {e}")
    
    def _handle_loop_completion_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle loop completion events from LoopController.
        
        Args:
            event_data: Event data containing loop_id and completion info
        """
        try:
            loop_id = event_data.get('loop_id')
            if not loop_id:
                return
            
            logger.info(f"ActionLoopAdapter: Processing completion for loop {loop_id}")
            
            # Execute any registered completion actions
            self._execute_completion_actions(loop_id)
            
            # Clean up tracking
            self.active_action_loops.pop(loop_id, None)
            
        except Exception as e:
            logger.error(f"ActionLoopAdapter: Error handling loop completion event: {e}")
    
    def _execute_completion_actions(self, loop_id: str) -> None:
        """
        Execute all actions registered for this loop's completion.
        
        Args:
            loop_id: ID of the completed loop
        """
        with self._callback_lock:
            pending_actions = self.completion_callbacks.get(loop_id, [])
            
            if not pending_actions:
                logger.debug(f"ActionLoopAdapter: No completion actions for loop {loop_id}")
                return
            
            logger.info(f"ActionLoopAdapter: Executing {len(pending_actions)} completion actions for loop {loop_id}")
            
            for pending_action in pending_actions:
                try:
                    self._trigger_completion_action(pending_action.target_action)
                except Exception as e:
                    logger.error(f"ActionLoopAdapter: Error executing completion action: {e}")
            
            # Clear the callbacks for this loop
            del self.completion_callbacks[loop_id]
    
    def _trigger_completion_action(self, action: Dict[str, Any]) -> None:
        """
        Send completion action to EventScheduler for processing.
        
        Args:
            action: Action to execute
        """
        if not self.event_scheduler:
            logger.warning("ActionLoopAdapter: No EventScheduler available for completion action")
            return
        
        try:
            action_id = action.get('action_id', 'completion_action')
            logger.info(f"ActionLoopAdapter: Triggering completion action {action_id}")
            
            # Send action to EventScheduler for immediate execution
            # This integrates with the existing event processing system
            if hasattr(self.event_scheduler, 'execute_action_immediately'):
                self.event_scheduler.execute_action_immediately(action)
            elif hasattr(self.event_scheduler, 'add_action'):
                # Alternative: add action to scheduler queue
                self.event_scheduler.add_action(action)
                logger.info(f"ActionLoopAdapter: Added completion action {action_id} to scheduler queue")
            else:
                logger.warning("ActionLoopAdapter: EventScheduler doesn't support action execution - completion action skipped")
                
        except Exception as e:
            logger.error(f"ActionLoopAdapter: Error triggering completion action: {e}")
    
    def get_adapter_status(self) -> Dict[str, Any]:
        """
        Get status of the action adapter.
        
        Returns:
            Dictionary with adapter status information
        """
        with self._callback_lock:
            return {
                'active_action_loops': len(self.active_action_loops),
                'pending_completion_callbacks': {
                    loop_id: len(callbacks) 
                    for loop_id, callbacks in self.completion_callbacks.items()
                },
                'loop_controller_available': self.loop_controller is not None,
                'event_scheduler_available': self.event_scheduler is not None,
                'active_loops': list(self.active_action_loops.keys())
            }
    
    def cleanup(self) -> None:
        """Clean up adapter resources."""
        try:
            with self._callback_lock:
                # Clear all pending callbacks
                self.completion_callbacks.clear()
                self.active_action_loops.clear()
                
            logger.info("ActionLoopAdapter: Cleanup completed")
            
        except Exception as e:
            logger.error(f"ActionLoopAdapter: Error during cleanup: {e}")


def create_action_adapter_for_deck(deck, event_scheduler=None) -> ActionLoopAdapter:
    """
    Convenience function to create ActionLoopAdapter for a deck.
    
    Args:
        deck: Deck instance with loop_controller
        event_scheduler: EventScheduler instance
        
    Returns:
        ActionLoopAdapter instance or None if deck has no loop_controller
    """
    if not hasattr(deck, 'loop_controller') or not deck.loop_controller:
        logger.error(f"Cannot create ActionLoopAdapter: deck {getattr(deck, 'deck_id', 'unknown')} has no loop_controller")
        return None
    
    adapter = ActionLoopAdapter(deck.loop_controller, event_scheduler)
    logger.info(f"Created ActionLoopAdapter for deck {getattr(deck, 'deck_id', 'unknown')}")
    return adapter