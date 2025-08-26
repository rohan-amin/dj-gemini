#!/usr/bin/env python3
"""
Modular event queue system for scheduling and managing audio engine events.
Provides efficient event scheduling with multiple queue levels for different event types.
"""

import heapq
import time
import threading
from typing import Optional, List, Dict, Any, Callable
from collections import deque
import logging

from .event_types import ScheduledEvent, EventType, EventPriority, EventResult

logger = logging.getLogger(__name__)

class EventQueue:
    """
    Base class for event queues.
    Defines the interface that all queue implementations must follow.
    """
    
    def schedule(self, event: ScheduledEvent) -> str:
        """Schedule an event for execution"""
        raise NotImplementedError
    
    def get_next_event(self) -> Optional[ScheduledEvent]:
        """Get the next event that should execute now"""
        raise NotImplementedError
    
    def cancel_event(self, event_id: str) -> bool:
        """Cancel a scheduled event"""
        raise NotImplementedError
    
    def clear(self) -> None:
        """Clear all scheduled events"""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        raise NotImplementedError

class ImmediateEventQueue(EventQueue):
    """
    Queue for immediate events (execute now).
    O(1) insertion and removal.
    """
    
    def __init__(self):
        self._queue = deque()
        self._lock = threading.RLock()
        self._stats = {"scheduled": 0, "executed": 0, "cancelled": 0}
    
    def schedule(self, event: ScheduledEvent) -> str:
        """Schedule an immediate event"""
        with self._lock:
            self._queue.append(event)
            self._stats["scheduled"] += 1
            logger.debug(f"ImmediateEventQueue: Scheduled event {event.action_id}")
            return event.action_id
    
    def get_next_event(self) -> Optional[ScheduledEvent]:
        """Get next immediate event"""
        with self._lock:
            if self._queue:
                event = self._queue.popleft()
                self._stats["executed"] += 1
                return event
            return None
    
    def cancel_event(self, event_id: str) -> bool:
        """Cancel an immediate event"""
        with self._lock:
            for event in list(self._queue):
                if event.action_id == event_id:
                    self._queue.remove(event)
                    self._stats["cancelled"] += 1
                    return True
            return False
    
    def clear(self) -> None:
        """Clear all immediate events"""
        with self._lock:
            self._queue.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get immediate queue statistics"""
        with self._lock:
            return {
                "queue_type": "immediate",
                "pending_count": len(self._queue),
                **self._stats
            }

class BeatAlignedEventQueue(EventQueue):
    """
    Queue for beat-aligned events (execute at specific beat boundaries).
    Groups events by beat number for efficient access.
    """
    
    def __init__(self):
        self._beat_events: Dict[int, List[ScheduledEvent]] = {}
        self._lock = threading.RLock()
        self._stats = {"scheduled": 0, "executed": 0, "cancelled": 0}
    
    def schedule(self, event: ScheduledEvent) -> str:
        """Schedule a beat-aligned event"""
        with self._lock:
            # Extract beat number from trigger time (assuming trigger_time is beat number)
            beat_number = int(event.trigger_time)
            
            if beat_number not in self._beat_events:
                self._beat_events[beat_number] = []
            
            # Insert in priority order (higher priority first)
            events = self._beat_events[beat_number]
            insert_index = 0
            for i, existing_event in enumerate(events):
                if existing_event.priority < event.priority:
                    insert_index = i
                    break
                insert_index = i + 1
            
            events.insert(insert_index, event)
            self._stats["scheduled"] += 1
            
            logger.debug(f"BeatAlignedEventQueue: Scheduled event {event.action_id} for beat {beat_number}")
            return event.action_id
    
    def get_next_event(self) -> Optional[ScheduledEvent]:
        """Get next beat-aligned event (should be called with current beat)"""
        with self._lock:
            # This method should be called with the current beat number
            # For now, return None - the hybrid queue will handle this
            return None
    
    def get_events_for_beat(self, beat_number: int) -> List[ScheduledEvent]:
        """Get all events for a specific beat number"""
        with self._lock:
            if beat_number in self._beat_events:
                events = self._beat_events[beat_number].copy()
                # Remove events from the queue
                del self._beat_events[beat_number]
                self._stats["executed"] += len(events)
                return events
            return []
    
    def cancel_event(self, event_id: str) -> bool:
        """Cancel a beat-aligned event"""
        with self._lock:
            for beat_events in self._beat_events.values():
                for event in beat_events:
                    if event.action_id == event_id:
                        beat_events.remove(event)
                        self._stats["cancelled"] += 1
                        return True
            return False
    
    def clear(self) -> None:
        """Clear all beat-aligned events"""
        with self._lock:
            self._beat_events.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get beat-aligned queue statistics"""
        with self._lock:
            total_pending = sum(len(events) for events in self._beat_events.values())
            return {
                "queue_type": "beat_aligned",
                "pending_count": total_pending,
                "beat_count": len(self._beat_events),
                **self._stats
            }

class FutureEventQueue(EventQueue):
    """
    Queue for future events (execute at later times).
    Uses heap for efficient time-based ordering.
    """
    
    def __init__(self):
        self._queue = []  # [(trigger_time, -priority, unique_id, event), ...]
        self._lock = threading.RLock()
        self._stats = {"scheduled": 0, "executed": 0, "cancelled": 0}
        self._next_id = 0  # Unique ID counter for stable sorting
    
    def schedule(self, event: ScheduledEvent) -> str:
        """Schedule a future event"""
        with self._lock:
            # Use negative priority for max-heap (higher priority first)
            # Add unique ID to ensure stable sorting when times and priorities are equal
            unique_id = self._next_id
            self._next_id += 1
            heapq.heappush(self._queue, (event.trigger_time, -event.priority, unique_id, event))
            self._stats["scheduled"] += 1
            
            logger.debug(f"FutureEventQueue: Scheduled event {event.action_id} for time {event.trigger_time}")
            return event.action_id
    
    def get_next_event(self) -> Optional[ScheduledEvent]:
        """Get next future event that should execute now"""
        with self._lock:
            if not self._queue:
                return None
            
            # Check if next event should execute now
            next_time, priority, unique_id, event = self._queue[0]
            if next_time <= time.time():
                heapq.heappop(self._queue)
                self._stats["executed"] += 1
                return event
            
            return None
    
    def get_next_event_time(self) -> Optional[float]:
        """Get the time of the next scheduled event"""
        with self._lock:
            if self._queue:
                return self._queue[0][0]  # First element is still trigger_time
            return None
    
    def cancel_event(self, event_id: str) -> bool:
        """Cancel a future event"""
        with self._lock:
            # Find and remove the event
            for i, (_, _, _, event) in enumerate(self._queue):
                if event.action_id == event_id:
                    # Remove by index (not efficient but necessary for heap)
                    self._queue.pop(i)
                    heapq.heapify(self._queue)  # Re-heapify
                    self._stats["cancelled"] += 1
                    return True
            return False
    
    def clear(self) -> None:
        """Clear all future events"""
        with self._lock:
            self._queue.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get future queue statistics"""
        with self._lock:
            return {
                "queue_type": "future",
                "pending_count": len(self._queue),
                **self._stats
            }

class HybridEventQueue(EventQueue):
    """
    Hybrid event queue that combines multiple queue types for optimal performance.
    Routes events to appropriate queues based on their timing requirements.
    """
    
    def __init__(self, audio_clock):
        self.audio_clock = audio_clock
        
        # Initialize sub-queues
        self.immediate_queue = ImmediateEventQueue()
        self.beat_aligned_queue = BeatAlignedEventQueue()
        self.future_queue = FutureEventQueue()
        
        # Event routing configuration
        self.immediate_threshold = 0.001  # 1ms
        self.beat_aligned_threshold = 0.1  # 100ms
        
        # Statistics
        self._stats = {"total_scheduled": 0, "total_executed": 0, "total_cancelled": 0}
        self._lock = threading.RLock()
        
        logger.info("HybridEventQueue initialized")
    
    def schedule(self, event: ScheduledEvent) -> str:
        """Schedule an event, routing it to the appropriate queue"""
        with self._lock:
            # Determine which queue to use
            current_time = self.audio_clock.get_current_time()
            time_diff = event.trigger_time - current_time
            
            if time_diff <= self.immediate_threshold:
                # Execute immediately
                event.event_type = EventType.IMMEDIATE
                event_id = self.immediate_queue.schedule(event)
            elif time_diff <= self.beat_aligned_threshold:
                # Execute at beat boundary
                event.event_type = EventType.BEAT_ALIGNED
                event_id = self.beat_aligned_queue.schedule(event)
            else:
                # Execute in the future
                event.event_type = EventType.FUTURE
                event_id = self.future_queue.schedule(event)
            
            self._stats["total_scheduled"] += 1
            logger.debug(f"HybridEventQueue: Routed event {event_id} to {event.event_type.value} queue")
            
            return event_id
    
    def get_next_event(self) -> Optional[ScheduledEvent]:
        """Get the next event that should execute now"""
        with self._lock:
            # Check immediate queue first
            event = self.immediate_queue.get_next_event()
            if event:
                self._stats["total_executed"] += 1
                return event
            
            # Check beat-aligned queue
            current_beat = self.audio_clock.get_current_beat()
            if current_beat is not None:
                beat_events = self.beat_aligned_queue.get_events_for_beat(int(current_beat))
                if beat_events:
                    # Return highest priority event
                    event = max(beat_events, key=lambda e: e.priority)
                    self._stats["total_executed"] += 1
                    return event
            
            # Check future queue
            event = self.future_queue.get_next_event()
            if event:
                self._stats["total_executed"] += 1
                return event
            
            return None
    
    def get_next_event_time(self) -> Optional[float]:
        """Get the time of the next scheduled event"""
        with self._lock:
            # Check immediate and beat-aligned first
            if (self.immediate_queue.get_stats()["pending_count"] > 0 or 
                self.beat_aligned_queue.get_stats()["pending_count"] > 0):
                return self.audio_clock.get_current_time()
            
            # Check future queue
            return self.future_queue.get_next_event_time()
    
    def cancel_event(self, event_id: str) -> bool:
        """Cancel a scheduled event"""
        with self._lock:
            # Try to cancel from all queues
            if self.immediate_queue.cancel_event(event_id):
                self._stats["total_cancelled"] += 1
                return True
            if self.beat_aligned_queue.cancel_event(event_id):
                self._stats["total_cancelled"] += 1
                return True
            if self.future_queue.cancel_event(event_id):
                self._stats["total_cancelled"] += 1
                return True
            return False
    
    def clear(self) -> None:
        """Clear all scheduled events"""
        with self._lock:
            self.immediate_queue.clear()
            self.beat_aligned_queue.clear()
            self.future_queue.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics"""
        with self._lock:
            immediate_stats = self.immediate_queue.get_stats()
            beat_aligned_stats = self.beat_aligned_queue.get_stats()
            future_stats = self.future_queue.get_stats()
            
            return {
                "hybrid_queue": {
                    "immediate": immediate_stats,
                    "beat_aligned": beat_aligned_stats,
                    "future": future_stats,
                    "total": self._stats
                }
            }
    
    def get_queue_info(self) -> Dict[str, Any]:
        """Get detailed information about each queue"""
        with self._lock:
            return {
                "immediate": self.immediate_queue.get_stats(),
                "beat_aligned": self.beat_aligned_queue.get_stats(),
                "future": self.future_queue.get_stats()
            }
