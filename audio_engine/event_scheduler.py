#!/usr/bin/env python3
"""
Event scheduler that coordinates event execution and timing.
Replaces the busy-wait engine loop with efficient event-driven execution.
"""

import time
import threading
from typing import Optional, Dict, Any, Callable, List
import logging

from .event_types import ScheduledEvent, EventResult, EventPriority
from .event_queue import HybridEventQueue
from .audio_clock import AudioClock

logger = logging.getLogger(__name__)

class EventScheduler:
    """
    Coordinates event scheduling and execution.
    Provides the interface between the audio engine and event system.
    """
    
    def __init__(self, audio_clock: AudioClock):
        self.audio_clock = audio_clock
        self.event_queue = HybridEventQueue(audio_clock)
        
        # Execution state
        self._running = False
        self._execution_thread = None
        self._stop_event = threading.Event()
        
        # Event handlers
        self._event_handlers: Dict[str, Callable] = {}
        self._default_handler: Optional[Callable] = None
        
        # Statistics
        self._stats = {
            "events_scheduled": 0,
            "events_executed": 0,
            "events_failed": 0,
            "execution_time_total": 0.0
        }
        
        # Error handling
        self._error_callbacks: List[Callable] = []
        
        logger.info("EventScheduler initialized")
    
    def start(self) -> None:
        """Start the event scheduler"""
        if self._running:
            logger.warning("EventScheduler already running")
            return
        
        self._running = True
        self._stop_event.clear()
        
        # Start execution thread
        self._execution_thread = threading.Thread(
            target=self._execution_loop,
            daemon=True,
            name="EventScheduler-Execution"
        )
        self._execution_thread.start()
        
        logger.info("EventScheduler started")
    
    def stop(self) -> None:
        """Stop the event scheduler"""
        if not self._running:
            logger.warning("EventScheduler not running")
            return
        
        self._running = False
        self._stop_event.set()
        
        # Wait for execution thread to finish
        if self._execution_thread and self._execution_thread.is_alive():
            self._execution_thread.join(timeout=2.0)
            if self._execution_thread.is_alive():
                logger.warning("EventScheduler execution thread did not stop cleanly")
        
        logger.info("EventScheduler stopped")
    
    def schedule_action(self, action: Dict[str, Any], trigger_time: float, 
                       priority: int = EventPriority.NORMAL.value) -> str:
        """Schedule an action for execution at a specific time"""
        try:
            # Create scheduled event
            event = ScheduledEvent(
                action_id=action.get('action_id', f"action_{self._stats['events_scheduled']}"),
                trigger_time=trigger_time,
                priority=priority,
                action=action
            )
            
            # Schedule the event
            event_id = self.event_queue.schedule(event)
            self._stats["events_scheduled"] += 1
            
            logger.debug(f"EventScheduler: Scheduled action {event_id} for time {trigger_time}")
            return event_id
            
        except Exception as e:
            logger.error(f"EventScheduler: Failed to schedule action: {e}")
            raise
    
    def schedule_beat_action(self, action: Dict[str, Any], beat_number: float,
                           priority: int = EventPriority.NORMAL.value) -> str:
        """Schedule an action to execute at a specific beat"""
        # Convert beat number to time using audio clock
        trigger_time = self.audio_clock.get_time_at_beat(beat_number)
        if trigger_time is None:
            # Fallback to BPM calculation
            bpm = self.audio_clock.get_current_bpm()
            trigger_time = (beat_number * 60.0) / bpm
        
        return self.schedule_action(action, trigger_time, priority)
    
    def schedule_immediate_action(self, action: Dict[str, Any],
                                priority: int = EventPriority.HIGH.value) -> str:
        """Schedule an action to execute immediately"""
        current_time = self.audio_clock.get_current_time()
        return self.schedule_action(action, current_time, priority)
    
    def cancel_action(self, action_id: str) -> bool:
        """Cancel a scheduled action"""
        success = self.event_queue.cancel_event(action_id)
        if success:
            logger.debug(f"EventScheduler: Cancelled action {action_id}")
        return success
    
    def register_handler(self, action_type: str, handler: Callable) -> None:
        """Register a handler for a specific action type"""
        self._event_handlers[action_type] = handler
        logger.debug(f"EventScheduler: Registered handler for action type '{action_type}'")
    
    def register_default_handler(self, handler: Callable) -> None:
        """Register a default handler for unhandled action types"""
        self._default_handler = handler
        logger.debug("EventScheduler: Registered default handler")
    
    def register_error_callback(self, callback: Callable) -> None:
        """Register a callback for error handling"""
        self._error_callbacks.append(callback)
    
    def clear_all_events(self) -> None:
        """Clear all scheduled events"""
        self.event_queue.clear()
        logger.info("EventScheduler: Cleared all scheduled events")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        queue_stats = self.event_queue.get_stats()
        return {
            "scheduler": self._stats.copy(),
            "queue": queue_stats
        }
    
    def is_running(self) -> bool:
        """Check if scheduler is running"""
        return self._running
    
    def _execution_loop(self) -> None:
        """Main execution loop - replaces the old busy-wait engine loop"""
        logger.info("EventScheduler execution loop started")
        
        while not self._stop_event.is_set() and self._running:
            try:
                # Get next event that should execute now
                next_event = self.event_queue.get_next_event()
                
                if next_event:
                    # Execute the event
                    self._execute_event(next_event)
                else:
                    # No events to execute - sleep until next event
                    next_event_time = self.event_queue.get_next_event_time()
                    
                    if next_event_time is not None and next_event_time != float('inf'):
                        # Calculate sleep time
                        current_time = self.audio_clock.get_current_time()
                        sleep_time = max(0.001, next_event_time - current_time)  # Min 1ms
                        
                        # Sleep until next event (much more efficient than 10ms polling)
                        if self._stop_event.wait(sleep_time):
                            break  # Stop event was set
                    else:
                        # No events scheduled - sleep briefly
                        if self._stop_event.wait(0.1):
                            break  # Stop event was set
                        
            except Exception as e:
                logger.error(f"EventScheduler execution loop error: {e}")
                self._notify_error_callbacks(e)
                time.sleep(0.01)  # Brief pause on error
        
        logger.info("EventScheduler execution loop stopped")
    
    def _execute_event(self, event: ScheduledEvent) -> None:
        """Execute a scheduled event"""
        start_time = time.time()
        
        try:
            # Find appropriate handler
            action_type = event.action.get('command', 'unknown')
            handler = self._event_handlers.get(action_type, self._default_handler)
            
            if handler:
                # Execute the action
                result = handler(event.action)
                
                # Update statistics
                execution_time = time.time() - start_time
                self._stats["events_executed"] += 1
                self._stats["execution_time_total"] += execution_time
                
                logger.debug(f"EventScheduler: Executed action {event.action_id} "
                           f"({action_type}) in {execution_time:.3f}s")
                
                # Handle result - engine handlers return False to indicate success
                # This is the opposite of typical boolean logic, so we need to handle it specially
                if result is False:
                    # Engine handlers return False to indicate success
                    logger.debug(f"EventScheduler: Action {event.action_id} completed successfully (engine returned False)")
                elif result is True or (hasattr(result, 'success') and result.success):
                    # Standard success case
                    logger.debug(f"EventScheduler: Action {event.action_id} completed successfully")
                else:
                    # Actual failure case
                    self._stats["events_failed"] += 1
                    logger.warning(f"EventScheduler: Action {event.action_id} failed")
                    
            else:
                # No handler found
                logger.warning(f"EventScheduler: No handler found for action type '{action_type}'")
                self._stats["events_failed"] += 1
                
        except Exception as e:
            # Handle execution errors
            execution_time = time.time() - start_time
            self._stats["events_failed"] += 1
            self._stats["execution_time_total"] += execution_time
            
            logger.error(f"EventScheduler: Error executing action {event.action_id}: {e}")
            self._notify_error_callbacks(e)
    
    def _notify_error_callbacks(self, error: Exception) -> None:
        """Notify all error callbacks"""
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all events to complete or timeout"""
        start_time = time.time()
        
        while self._running:
            # Check if any events are pending
            queue_info = self.event_queue.get_queue_info()
            total_pending = (queue_info["immediate"]["pending_count"] + 
                           queue_info["beat_aligned"]["pending_count"] + 
                           queue_info["future"]["pending_count"])
            
            if total_pending == 0:
                return True  # All events completed
            
            # Check timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                logger.warning(f"EventScheduler: Wait timeout after {timeout}s")
                return False
            
            time.sleep(0.01)  # Brief pause
        
        return False  # Scheduler stopped before completion
