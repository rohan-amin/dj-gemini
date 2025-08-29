#!/usr/bin/env python3
"""
Event scheduler that coordinates event execution and timing.
Replaces the busy-wait engine loop with efficient event-driven execution.
"""

import time
import threading
from typing import Optional, Dict, Any, Callable, List
import logging

from .event_types import ScheduledEvent, EventResult, EventPriority, EventType
# Phase 5C: HybridEventQueue removed - using pure beat-based execution
from .audio_clock import AudioClock

logger = logging.getLogger(__name__)

class EventScheduler:
    """
    Coordinates event scheduling and execution.
    Provides the interface between the audio engine and event system.
    """
    
    def __init__(self, audio_clock: AudioClock):
        self.audio_clock = audio_clock
        logger.debug(f"EventScheduler - Using audio clock instance {id(self.audio_clock)}")
        
        # Phase 5A: Pure beat-based event storage
        # Structure: {deck_id: {beat_number: [events]}}
        self._beat_events: Dict[str, Dict[float, List[ScheduledEvent]]] = {}
        self._event_id_to_location: Dict[str, tuple] = {}  # event_id -> (deck_id, beat_number)
        
        # Phase 5D: Immediate execution queue for script_start actions
        self._immediate_events: List[ScheduledEvent] = []
        
        self._lock = threading.RLock()
        
        # Phase 3: Engine reference for deck-specific BeatManager access
        self.engine = None  # Will be set by Engine after creation
        
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
        
        logger.info("EventScheduler initialized (Phase 5A: Pure beat-based storage)")
    
    def set_engine_reference(self, engine) -> None:
        """
        Set reference to AudioEngine for deck-specific BeatManager access.
        Phase 3: Called by Engine after EventScheduler creation.
        """
        self.engine = engine
        logger.debug(f"EventScheduler - Engine reference set: {id(engine)}")
    
    def start(self) -> None:
        """Start the event scheduler - register beat callbacks instead of polling thread"""
        if self._running:
            logger.warning("EventScheduler already running")
            return
        
        self._running = True
        self._stop_event.clear()
        
        # Register beat callbacks with all existing deck BeatManagers
        self._register_beat_callbacks()
        
        # Start immediate event processing thread (still needed for script_start events)
        self._execution_thread = threading.Thread(
            target=self._immediate_event_loop,
            daemon=True,
            name="EventScheduler-Immediate"
        )
        self._execution_thread.start()
        
        logger.info("EventScheduler started (event-driven mode)")
    
    def stop(self) -> None:
        """Stop the event scheduler"""
        if not self._running:
            logger.warning("EventScheduler not running")
            return
        
        self._running = False
        self._stop_event.set()
        
        # Remove beat callbacks from all deck BeatManagers
        if self.engine and hasattr(self.engine, 'decks'):
            for deck_id, deck in self.engine.decks.items():
                if hasattr(deck, 'beat_manager'):
                    deck.beat_manager.remove_beat_callback(self._on_beat_boundary)
                    logger.debug(f"EventScheduler: Removed beat callback from deck {deck_id}")
        
        # Wait for execution thread to finish
        if self._execution_thread and self._execution_thread.is_alive():
            self._execution_thread.join(timeout=2.0)
            if self._execution_thread.is_alive():
                logger.warning("EventScheduler execution thread did not stop cleanly")
        
        logger.info("EventScheduler stopped")
    
    def schedule_beat_event(self, deck_id: str, beat_number: float, action: Dict[str, Any],
                           priority: int = EventPriority.NORMAL.value) -> str:
        """
        Phase 5A: Pure beat-based event scheduling.
        Schedule an event to execute when the specified deck reaches the specified beat.
        
        Args:
            deck_id: Deck ID for beat timing
            beat_number: Beat number where event should execute
            action: Action to execute
            priority: Event priority
            
        Returns:
            Event ID for cancellation
        """
        try:
            with self._lock:
                # Generate unique event ID
                event_id = action.get('action_id', f"beat_event_{self._stats['events_scheduled']}")
                
                # Create scheduled event
                event = ScheduledEvent(
                    action_id=event_id,
                    trigger_time=0.0,  # Not used in beat-based system
                    priority=priority,
                    action=action,
                    event_type=EventType.BEAT_ALIGNED,
                    metadata={"target_beat": beat_number, "deck_id": deck_id}
                )
                
                # Initialize deck storage if needed
                if deck_id not in self._beat_events:
                    self._beat_events[deck_id] = {}
                
                # Initialize beat storage if needed
                if beat_number not in self._beat_events[deck_id]:
                    self._beat_events[deck_id][beat_number] = []
                
                # Store event
                self._beat_events[deck_id][beat_number].append(event)
                self._event_id_to_location[event_id] = (deck_id, beat_number)
                
                self._stats["events_scheduled"] += 1
                
                logger.debug(f"EventScheduler: Scheduled beat event {event_id} for deck {deck_id} beat {beat_number}")
                
                # Debug: Log the current state of scheduled events
                total_events = sum(len(events) for events in self._beat_events.values())
                logger.info(f"EventScheduler: Total scheduled events: {total_events}")
                for d_id, beat_events in self._beat_events.items():
                    for beat, events in beat_events.items():
                        logger.info(f"EventScheduler: Deck {d_id} has {len(events)} events at beat {beat}")
                
                return event_id
                
        except Exception as e:
            logger.error(f"EventScheduler: Failed to schedule beat event: {e}")
            raise
    
    # Phase 5A: Time-based scheduling removed - everything is now beat-based
    # Use schedule_beat_event() instead
    
    def schedule_beat_action(self, action: Dict[str, Any], beat_number: float,
                           deck_id: str = None, priority: int = EventPriority.NORMAL.value) -> str:
        """
        Phase 5A: Updated to use pure beat-based scheduling.
        Schedule an action to execute at a specific beat for a specific deck.
        
        Args:
            action: Action to execute
            beat_number: Beat number where action should execute  
            deck_id: Deck ID for beat timing (if None, extracted from action)
            priority: Event priority
        """
        # Get deck_id from parameter or action
        if deck_id is None:
            deck_id = action.get('deck_id')
        
        if not deck_id:
            logger.error(f"EventScheduler - No deck_id specified for beat action: {action}")
            raise ValueError("deck_id is required for beat-based scheduling")
        
        # Phase 5A: Use pure beat-based scheduling - no time conversion
        return self.schedule_beat_event(deck_id, beat_number, action, priority)
        
    def schedule_immediate_action(self, action: Dict[str, Any],
                                priority: int = EventPriority.HIGH.value) -> str:
        """
        Phase 5D: Schedule an action for immediate execution (for script_start actions).
        These bypass beat timing and execute right away to bootstrap the system.
        """
        try:
            with self._lock:
                # Generate unique event ID
                event_id = action.get('action_id', f"immediate_event_{self._stats['events_scheduled']}")
                
                # Create immediate event
                event = ScheduledEvent(
                    action_id=event_id,
                    trigger_time=0.0,  # Not used for immediate events
                    priority=priority,
                    action=action,
                    event_type=EventType.IMMEDIATE,
                    metadata={"immediate": True}
                )
                
                # Store in immediate queue
                self._immediate_events.append(event)
                self._stats["events_scheduled"] += 1
                
                logger.debug(f"EventScheduler: Scheduled immediate action {event_id}")
                return event_id
                
        except Exception as e:
            logger.error(f"EventScheduler: Failed to schedule immediate action: {e}")
            raise
    
    def cancel_action(self, action_id: str) -> bool:
        """Phase 5A: Cancel a scheduled beat-based action"""
        try:
            with self._lock:
                if action_id not in self._event_id_to_location:
                    logger.warning(f"EventScheduler: Cannot cancel unknown action {action_id}")
                    return False
                
                deck_id, beat_number = self._event_id_to_location[action_id]
                
                # Find and remove the event
                if deck_id in self._beat_events and beat_number in self._beat_events[deck_id]:
                    events = self._beat_events[deck_id][beat_number]
                    for i, event in enumerate(events):
                        if event.action_id == action_id:
                            events.pop(i)
                            # Clean up empty beat entry
                            if not events:
                                del self._beat_events[deck_id][beat_number]
                            # Clean up empty deck entry  
                            if not self._beat_events[deck_id]:
                                del self._beat_events[deck_id]
                            # Clean up tracking
                            del self._event_id_to_location[action_id]
                            
                            logger.debug(f"EventScheduler: Cancelled beat action {action_id}")
                            return True
                
                logger.warning(f"EventScheduler: Could not find action {action_id} to cancel")
                return False
                
        except Exception as e:
            logger.error(f"EventScheduler: Error cancelling action {action_id}: {e}")
            return False
    
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
        """Phase 5A: Clear all scheduled beat-based events"""
        with self._lock:
            self._beat_events.clear()
            self._event_id_to_location.clear()
        logger.info("EventScheduler: Cleared all scheduled beat events")
    
    def get_stats(self) -> Dict[str, Any]:
        """Phase 5D: Get scheduler statistics including immediate events"""
        with self._lock:
            # Count pending events by deck
            pending_by_deck = {}
            total_pending = 0
            
            for deck_id, beat_events in self._beat_events.items():
                deck_pending = 0
                for beat_number, events in beat_events.items():
                    deck_pending += len(events)
                pending_by_deck[deck_id] = deck_pending
                total_pending += deck_pending
            
            # Add immediate events
            immediate_pending = len(self._immediate_events)
            total_pending += immediate_pending
            
            return {
                "scheduler": self._stats.copy(),
                "beat_events": {
                    "total_pending": total_pending,
                    "immediate_pending": immediate_pending,
                    "beat_pending": total_pending - immediate_pending,
                    "pending_by_deck": pending_by_deck,
                    "decks_with_events": list(self._beat_events.keys())
                }
            }
    
    def is_running(self) -> bool:
        """Check if scheduler is running"""
        return self._running
    
    def _on_beat_boundary(self, deck_id: str, current_beat: float, previous_beat: float = None) -> None:
        """
        Callback fired when beat position updates.
        Checks if any scheduled events should execute at the current beat position.
        
        Args:
            deck_id: The deck with updated beat position
            current_beat: The current beat position
        """
        try:
            with self._lock:
                logger.debug(f"EventScheduler: Beat callback for {deck_id} at {current_beat:.3f}")
                
                # Check if this deck has any events scheduled
                if deck_id not in self._beat_events:
                    logger.debug(f"EventScheduler: No events scheduled for deck {deck_id}")
                    return
                
                scheduled_beats = list(self._beat_events[deck_id].keys())
                logger.debug(f"EventScheduler: Deck {deck_id} has events at beats: {scheduled_beats}")
                
                # Check all scheduled beats for this deck
                for scheduled_beat in scheduled_beats:
                    # Execute events if scheduled beat is within the range we just crossed
                    should_execute = False
                    if previous_beat is not None:
                        # Check if scheduled beat is in the range [previous_beat, current_beat]
                        should_execute = previous_beat < scheduled_beat <= current_beat
                        logger.info(f"EventScheduler: Beat {scheduled_beat} in range {previous_beat:.3f} → {current_beat:.3f}: {should_execute}")
                    else:
                        # Fallback to simple comparison
                        should_execute = current_beat >= scheduled_beat
                        logger.info(f"EventScheduler: Beat {current_beat:.3f} >= {scheduled_beat}: {should_execute}")
                    
                    if should_execute:
                        events_to_execute = self._beat_events[deck_id][scheduled_beat].copy()
                        
                        print(f"🔥 EXECUTING EVENTS AT BEAT {scheduled_beat}: {[e.action_id for e in events_to_execute]}")
                        
                        # Remove executed events from schedule
                        for event in events_to_execute:
                            if event.action_id in self._event_id_to_location:
                                del self._event_id_to_location[event.action_id]
                        
                        del self._beat_events[deck_id][scheduled_beat]
                        if not self._beat_events[deck_id]:
                            del self._beat_events[deck_id]
                        
                        # Execute events outside lock to avoid deadlock
                        self._lock.release()
                        try:
                            for event in events_to_execute:
                                print(f"🎵 EXECUTING: {event.action_id} (command: {event.action.get('command', 'unknown')})")
                                logger.info(f"EventScheduler: Executing beat event {event.action_id} "
                                           f"for deck {deck_id} at beat {current_beat:.2f} "
                                           f"(scheduled for {scheduled_beat})")
                                self._execute_event(event)
                                print(f"✅ COMPLETED: {event.action_id}")
                        finally:
                            self._lock.acquire()
                        
        except Exception as e:
            logger.error(f"EventScheduler: Error in beat boundary callback: {e}")
            self._notify_error_callbacks(e)
    
    def _register_beat_callbacks(self) -> None:
        """Register beat callbacks with all existing deck BeatManagers"""
        if self.engine and hasattr(self.engine, 'decks'):
            registered_count = 0
            for deck_id, deck in self.engine.decks.items():
                if hasattr(deck, 'beat_manager'):
                    deck.beat_manager.add_beat_callback(self._on_beat_boundary)
                    registered_count += 1
                    logger.debug(f"EventScheduler: Registered beat callback with deck {deck_id}")
                else:
                    logger.warning(f"EventScheduler: Deck {deck_id} has no beat_manager")
            logger.info(f"EventScheduler: Registered beat callbacks with {registered_count} decks")
        else:
            logger.warning("EventScheduler: No engine reference or decks available for callback registration")
    
    def register_deck_callback(self, deck_id: str) -> None:
        """Register beat callback for a specific deck (called when new deck is created)"""
        logger.info(f"EventScheduler: register_deck_callback called for deck {deck_id}")
        logger.info(f"EventScheduler: _running={self._running}, has_engine={self.engine is not None}")
        
        if self._running and self.engine and hasattr(self.engine, 'decks'):
            logger.info(f"EventScheduler: Engine has {len(self.engine.decks)} decks")
            if deck_id in self.engine.decks:
                deck = self.engine.decks[deck_id]
                logger.info(f"EventScheduler: Found deck {deck_id}, has_beat_manager={hasattr(deck, 'beat_manager')}")
                if hasattr(deck, 'beat_manager'):
                    # Check current callback count before adding
                    before_count = len(deck.beat_manager._beat_callbacks)
                    logger.info(f"EventScheduler: Deck {deck_id} has {before_count} callbacks before registration")
                    
                    deck.beat_manager.add_beat_callback(self._on_beat_boundary)
                    print(f"DEBUG: Registered beat callback with new deck {deck_id}")
                    logger.info(f"EventScheduler: Registered beat callback with new deck {deck_id}")
                    
                    # Verify callback was added
                    after_count = len(deck.beat_manager._beat_callbacks)
                    logger.info(f"EventScheduler: Deck {deck_id} now has {after_count} callbacks after registration")
                    
                    if after_count == before_count:
                        logger.error(f"EventScheduler: FAILED to add callback to deck {deck_id}!")
                    else:
                        logger.info(f"EventScheduler: SUCCESSFULLY added callback to deck {deck_id}")
                else:
                    logger.warning(f"EventScheduler: New deck {deck_id} has no beat_manager")
            else:
                logger.warning(f"EventScheduler: Deck {deck_id} not found in engine.decks")
        else:
            logger.warning(f"EventScheduler: Cannot register callback - running={self._running}, engine={self.engine is not None}")
    
    def _immediate_event_loop(self) -> None:
        """
        Process immediate events (script_start, etc.) that don't depend on beat timing.
        This replaces the old polling loop for immediate events only.
        """
        logger.info("EventScheduler immediate event loop started")
        
        while not self._stop_event.is_set() and self._running:
            try:
                # Check for immediate events
                with self._lock:
                    if self._immediate_events:
                        immediate_event = self._immediate_events.pop(0)
                        logger.debug(f"EventScheduler: Executing immediate event {immediate_event.action_id}")
                        
                        # Execute outside lock to avoid deadlock
                        self._lock.release()
                        try:
                            self._execute_event(immediate_event)
                        finally:
                            self._lock.acquire()
                    else:
                        # No immediate events, wait for signal
                        self._lock.release()
                        try:
                            # Wait longer since immediate events are rare
                            if self._stop_event.wait(0.1):
                                break
                        finally:
                            self._lock.acquire()
                        
            except Exception as e:
                logger.error(f"EventScheduler immediate event loop error: {e}")
                self._notify_error_callbacks(e)
                time.sleep(0.01)  # Brief pause on error
        
        logger.info("EventScheduler immediate event loop stopped")
    
    
    def _get_current_beat_for_deck(self, deck_id: str) -> Optional[float]:
        """Get current beat position for specified deck"""
        if not self.engine:
            logger.warning(f"EventScheduler: No engine reference to get beat for deck {deck_id}")
            return None
        
        beat_manager = self.engine.get_beat_manager_for_deck(deck_id)
        if not beat_manager:
            logger.debug(f"EventScheduler: No BeatManager for deck {deck_id}")
            return None
        
        try:
            current_beat = beat_manager.get_current_beat()
            return current_beat
        except Exception as e:
            logger.warning(f"EventScheduler: Error getting current beat for deck {deck_id}: {e}")
            return None
    
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
        """Phase 5C: Wait for all beat-based events to complete or timeout"""
        start_time = time.time()
        
        while True:  # Check events regardless of running state
            # Check if any events are pending in beat-based storage
            with self._lock:
                total_pending = 0
                for deck_id, deck_events in self._beat_events.items():
                    for beat_num, events_list in deck_events.items():
                        total_pending += len(events_list)
                        logger.debug(f"EventScheduler: wait_for_completion - deck {deck_id} beat {beat_num}: {len(events_list)} events")
                
                logger.debug(f"EventScheduler: wait_for_completion - total pending: {total_pending}")
            
            if total_pending == 0:
                logger.debug("EventScheduler: wait_for_completion - no events pending, returning True")
                return True  # All events completed
            
            # If not running and events are pending, return False (won't be processed)
            if not self._running:
                logger.debug("EventScheduler: wait_for_completion - scheduler not running but events pending")
                return False
            
            # Check timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                logger.warning(f"EventScheduler: Wait timeout after {timeout}s with {total_pending} events pending")
                return False
            
            time.sleep(0.01)  # Brief pause
