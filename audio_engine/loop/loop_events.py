"""
Event system for loop lifecycle communication.

This module provides an event-driven architecture for loop state changes.
It implements a publisher-subscriber pattern to decouple loop management
from completion action handling, enabling clean separation of concerns.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from threading import Lock
import queue

logger = logging.getLogger(__name__)


class LoopEventType(Enum):
    """Types of loop events that can occur."""
    START = "start"           # Loop activated and started
    ITERATION = "iteration"   # Loop completed one iteration
    COMPLETE = "complete"     # Loop completed all iterations
    COMPLETION_PROCESSED = "completion_processed"  # Loop completion actions processed
    ERROR = "error"          # Loop encountered an error


@dataclass(frozen=True)
class LoopEvent:
    """
    Immutable loop event object.
    
    This dataclass represents a loop lifecycle event that can be published
    to subscribers for processing completion actions or other responses.
    """
    
    event_type: LoopEventType
    loop_id: str
    deck_id: str
    current_iteration: int
    total_iterations: int
    timestamp: float
    error_message: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None
    
    @property
    def is_completion_event(self) -> bool:
        """Check if this is a loop completion event."""
        return self.event_type == LoopEventType.COMPLETE
    
    @property
    def progress_ratio(self) -> float:
        """Get completion progress as a ratio (0.0 to 1.0)."""
        if self.total_iterations <= 0:
            return 1.0
        return min(1.0, self.current_iteration / self.total_iterations)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            'event_type': self.event_type.value,
            'loop_id': self.loop_id,
            'deck_id': self.deck_id,
            'current_iteration': self.current_iteration,
            'total_iterations': self.total_iterations,
            'timestamp': self.timestamp,
            'error_message': self.error_message,
            'additional_data': self.additional_data or {},
            'is_completion': self.is_completion_event,
            'progress_ratio': self.progress_ratio
        }


class LoopEventPublisher:
    """
    Publisher for loop events using subscriber pattern.
    
    This class manages event publication and subscription for loop lifecycle
    events. It maintains a list of subscribers and notifies them when events
    occur, enabling loose coupling between loop management and event handling.
    """
    
    def __init__(self):
        """Initialize the event publisher."""
        self._subscribers: List[Callable[[LoopEvent], None]] = []
        self._lock = Lock()
        self._event_queue = queue.Queue()
        self._processing_enabled = True
        
        # Statistics
        self._stats = {
            'events_published': 0,
            'events_processed': 0,
            'subscribers_count': 0,
            'processing_errors': 0,
            'last_event_time': None
        }
        
        logger.debug("🔄 LoopEventPublisher initialized")
    
    def subscribe(self, callback: Callable[[LoopEvent], None]) -> bool:
        """
        Subscribe to loop events.
        
        Args:
            callback: Function to call when events occur. Should accept LoopEvent parameter.
            
        Returns:
            True if subscription was successful
        """
        try:
            with self._lock:
                if callback not in self._subscribers:
                    self._subscribers.append(callback)
                    self._stats['subscribers_count'] = len(self._subscribers)
                    logger.debug(f"🔄 New subscriber registered, total: {len(self._subscribers)}")
                    return True
                else:
                    logger.warning("🔄 Callback already subscribed")
                    return False
                    
        except Exception as e:
            logger.error(f"🔄 Error subscribing to events: {e}")
            return False
    
    def unsubscribe(self, callback: Callable[[LoopEvent], None]) -> bool:
        """
        Unsubscribe from loop events.
        
        Args:
            callback: Function to remove from subscribers
            
        Returns:
            True if unsubscription was successful
        """
        try:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)
                    self._stats['subscribers_count'] = len(self._subscribers)
                    logger.debug(f"🔄 Subscriber removed, remaining: {len(self._subscribers)}")
                    return True
                else:
                    logger.warning("🔄 Callback not found for unsubscription")
                    return False
                    
        except Exception as e:
            logger.error(f"🔄 Error unsubscribing from events: {e}")
            return False
    
    def publish(self, event: LoopEvent) -> bool:
        """
        Publish a loop event to all subscribers.
        
        Args:
            event: LoopEvent to publish
            
        Returns:
            True if event was published successfully
        """
        if not self._processing_enabled:
            logger.debug(f"🔄 Event publishing disabled, skipping event {event.event_type.value}")
            return False
        
        try:
            # Update statistics
            self._stats['events_published'] += 1
            self._stats['last_event_time'] = event.timestamp
            
            logger.debug(f"🔄 Publishing {event.event_type.value} event for loop {event.loop_id}")
            
            # Get current subscribers (thread-safe copy)
            with self._lock:
                current_subscribers = self._subscribers.copy()
            
            if not current_subscribers:
                logger.debug(f"🔄 No subscribers for event {event.event_type.value}")
                return True
            
            # Notify all subscribers
            successful_notifications = 0
            for subscriber in current_subscribers:
                try:
                    subscriber(event)
                    successful_notifications += 1
                except Exception as e:
                    logger.error(f"🔄 Error notifying subscriber: {e}")
                    self._stats['processing_errors'] += 1
            
            self._stats['events_processed'] += 1
            
            logger.debug(f"🔄 Event {event.event_type.value} delivered to "
                        f"{successful_notifications}/{len(current_subscribers)} subscribers")
            
            return successful_notifications > 0
            
        except Exception as e:
            logger.error(f"🔄 Error publishing event: {e}")
            self._stats['processing_errors'] += 1
            return False
    
    def publish_start_event(self, loop_id: str, deck_id: str, total_iterations: int) -> bool:
        """
        Convenience method to publish a loop start event.
        
        Args:
            loop_id: ID of the loop that started
            deck_id: Deck the loop is on
            total_iterations: Total number of iterations planned
            
        Returns:
            True if event was published successfully
        """
        event = LoopEvent(
            event_type=LoopEventType.START,
            loop_id=loop_id,
            deck_id=deck_id,
            current_iteration=0,
            total_iterations=total_iterations,
            timestamp=time.time()
        )
        return self.publish(event)
    
    def publish_iteration_event(self, loop_id: str, deck_id: str, 
                              current_iteration: int, total_iterations: int) -> bool:
        """
        Convenience method to publish a loop iteration event.
        
        Args:
            loop_id: ID of the loop that completed an iteration
            deck_id: Deck the loop is on
            current_iteration: Current iteration count (1-based)
            total_iterations: Total number of iterations planned
            
        Returns:
            True if event was published successfully
        """
        event = LoopEvent(
            event_type=LoopEventType.ITERATION,
            loop_id=loop_id,
            deck_id=deck_id,
            current_iteration=current_iteration,
            total_iterations=total_iterations,
            timestamp=time.time()
        )
        return self.publish(event)
    
    def publish_completion_event(self, loop_id: str, deck_id: str, 
                               total_iterations: int) -> bool:
        """
        Convenience method to publish a loop completion event.
        
        Args:
            loop_id: ID of the loop that completed
            deck_id: Deck the loop is on
            total_iterations: Total number of iterations completed
            
        Returns:
            True if event was published successfully
        """
        event = LoopEvent(
            event_type=LoopEventType.COMPLETE,
            loop_id=loop_id,
            deck_id=deck_id,
            current_iteration=total_iterations,
            total_iterations=total_iterations,
            timestamp=time.time()
        )
        return self.publish(event)
    
    def publish_error_event(self, loop_id: str, deck_id: str, error_message: str,
                          current_iteration: int = 0, total_iterations: int = 0) -> bool:
        """
        Convenience method to publish a loop error event.
        
        Args:
            loop_id: ID of the loop that encountered an error
            deck_id: Deck the loop is on
            error_message: Description of the error
            current_iteration: Current iteration when error occurred
            total_iterations: Total iterations planned
            
        Returns:
            True if event was published successfully
        """
        event = LoopEvent(
            event_type=LoopEventType.ERROR,
            loop_id=loop_id,
            deck_id=deck_id,
            current_iteration=current_iteration,
            total_iterations=total_iterations,
            timestamp=time.time(),
            error_message=error_message
        )
        return self.publish(event)
    
    def enable_processing(self, enabled: bool = True) -> None:
        """
        Enable or disable event processing.
        
        Args:
            enabled: Whether to enable event processing
        """
        self._processing_enabled = enabled
        status = "enabled" if enabled else "disabled"
        logger.info(f"🔄 Loop event processing {status}")
    
    def get_subscriber_count(self) -> int:
        """Get the number of current subscribers."""
        with self._lock:
            return len(self._subscribers)
    
    def clear_subscribers(self) -> int:
        """
        Clear all subscribers.
        
        Returns:
            Number of subscribers that were removed
        """
        with self._lock:
            count = len(self._subscribers)
            self._subscribers.clear()
            self._stats['subscribers_count'] = 0
            logger.info(f"🔄 Cleared {count} event subscribers")
            return count
    
    def publish_event(self, event_data: Dict[str, Any]) -> bool:
        """
        Publish a generic event with custom data.
        
        This method provides a more flexible way to publish events that don't
        fit the standard LoopEvent structure.
        
        Args:
            event_data: Dictionary containing event data
            
        Returns:
            True if event was published successfully
        """
        try:
            # Log the event
            event_type = event_data.get('type', 'unknown')
            loop_id = event_data.get('loop_id', 'unknown')
            logger.debug(f"🔄 Publishing generic event: {event_type} for {loop_id}")
            
            # Update statistics
            self._stats['events_published'] += 1
            self._stats['last_event_time'] = event_data.get('timestamp', time.time())
            
            # Notify all subscribers with raw event data
            with self._lock:
                current_subscribers = self._subscribers.copy()
            
            if not current_subscribers:
                logger.debug(f"🔄 No subscribers for generic event {event_type}")
                return True
            
            successful_notifications = 0
            for subscriber in current_subscribers:
                try:
                    subscriber(event_data)
                    successful_notifications += 1
                except Exception as e:
                    logger.error(f"🔄 Error notifying subscriber of generic event: {e}")
                    self._stats['processing_errors'] += 1
            
            self._stats['events_processed'] += 1
            
            logger.debug(f"🔄 Generic event {event_type} delivered to "
                        f"{successful_notifications}/{len(current_subscribers)} subscribers")
            
            return successful_notifications > 0
            
        except Exception as e:
            logger.error(f"🔄 Error publishing generic event: {e}")
            self._stats['processing_errors'] += 1
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event publisher statistics."""
        stats = self._stats.copy()
        stats['processing_enabled'] = self._processing_enabled
        stats['current_subscribers'] = self.get_subscriber_count()
        return stats
    
    def cleanup(self) -> None:
        """Clean up event publisher resources."""
        try:
            logger.debug("🔄 Cleaning up LoopEventPublisher")
            
            # Disable processing
            self.enable_processing(False)
            
            # Clear subscribers
            subscriber_count = self.clear_subscribers()
            
            # Clear event queue
            try:
                while not self._event_queue.empty():
                    self._event_queue.get_nowait()
            except queue.Empty:
                pass
            
            logger.debug(f"🔄 LoopEventPublisher cleanup complete - {subscriber_count} subscribers removed")
            
        except Exception as e:
            logger.error(f"🔄 Error during event publisher cleanup: {e}")


class LoopEventHandler:
    """
    Basic event handler for loop completion actions.
    
    This class provides a foundation for handling loop events and can be
    subclassed or used as a template for more complex event handling logic.
    """
    
    def __init__(self, name: str = "LoopEventHandler"):
        """
        Initialize the event handler.
        
        Args:
            name: Name identifier for this handler
        """
        self.name = name
        self._completion_callbacks: Dict[str, List[Callable]] = {}
        self._stats = {
            'events_handled': 0,
            'completions_processed': 0,
            'errors_encountered': 0
        }
        
        logger.debug(f"🔄 {self.name} initialized")
    
    def handle_event(self, event: LoopEvent) -> bool:
        """
        Handle a loop event.
        
        Args:
            event: LoopEvent to handle
            
        Returns:
            True if event was handled successfully
        """
        try:
            self._stats['events_handled'] += 1
            
            logger.debug(f"🔄 {self.name}: Handling {event.event_type.value} event for {event.loop_id}")
            
            if event.event_type == LoopEventType.COMPLETE:
                return self._handle_completion(event)
            elif event.event_type == LoopEventType.START:
                return self._handle_start(event)
            elif event.event_type == LoopEventType.ITERATION:
                return self._handle_iteration(event)
            elif event.event_type == LoopEventType.ERROR:
                return self._handle_error(event)
            else:
                logger.warning(f"🔄 {self.name}: Unknown event type {event.event_type}")
                return False
                
        except Exception as e:
            logger.error(f"🔄 {self.name}: Error handling event: {e}")
            self._stats['errors_encountered'] += 1
            return False
    
    def _handle_completion(self, event: LoopEvent) -> bool:
        """Handle loop completion event."""
        logger.info(f"🔄 {self.name}: Loop {event.loop_id} completed on {event.deck_id}")
        
        # Execute any registered completion callbacks
        if event.loop_id in self._completion_callbacks:
            callbacks = self._completion_callbacks[event.loop_id]
            for callback in callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"🔄 {self.name}: Error in completion callback: {e}")
            
            # Remove callbacks after execution (one-time use)
            del self._completion_callbacks[event.loop_id]
            self._stats['completions_processed'] += 1
        
        return True
    
    def _handle_start(self, event: LoopEvent) -> bool:
        """Handle loop start event."""
        logger.debug(f"🔄 {self.name}: Loop {event.loop_id} started on {event.deck_id}")
        return True
    
    def _handle_iteration(self, event: LoopEvent) -> bool:
        """Handle loop iteration event."""
        logger.debug(f"🔄 {self.name}: Loop {event.loop_id} iteration {event.current_iteration}")
        return True
    
    def _handle_error(self, event: LoopEvent) -> bool:
        """Handle loop error event."""
        logger.error(f"🔄 {self.name}: Loop {event.loop_id} error - {event.error_message}")
        self._stats['errors_encountered'] += 1
        return True
    
    def register_completion_callback(self, loop_id: str, callback: Callable[[LoopEvent], None]) -> None:
        """
        Register a callback to be executed when a specific loop completes.
        
        Args:
            loop_id: ID of the loop to watch for completion
            callback: Function to call when loop completes
        """
        if loop_id not in self._completion_callbacks:
            self._completion_callbacks[loop_id] = []
        
        self._completion_callbacks[loop_id].append(callback)
        logger.debug(f"🔄 {self.name}: Registered completion callback for {loop_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event handler statistics."""
        stats = self._stats.copy()
        stats['name'] = self.name
        stats['pending_completions'] = len(self._completion_callbacks)
        return stats