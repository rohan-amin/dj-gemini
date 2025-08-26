#!/usr/bin/env python3
"""
Core event types and interfaces for the event-driven audio engine.
Provides the foundation for modular event scheduling and execution.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, Protocol
from enum import Enum
import time
import uuid

class EventType(Enum):
    """Types of events that can be scheduled"""
    NORMAL = "normal"                 # Standard event execution
    IMMEDIATE = "immediate"           # Execute now
    BEAT_ALIGNED = "beat_aligned"     # Execute at specific beat
    FUTURE = "future"                 # Execute at later time
    LOOP_COMPLETION = "loop_completion"  # Execute when loop completes

class EventPriority(Enum):
    """Priority levels for event execution"""
    CRITICAL = 100      # Must execute immediately (e.g., audio stop)
    HIGH = 75           # High priority (e.g., loop activation)
    NORMAL = 50         # Standard priority (e.g., beat triggers)
    LOW = 25            # Low priority (e.g., UI updates)
    BACKGROUND = 0      # Background tasks

@dataclass
class ScheduledEvent:
    """Represents a scheduled action with timing and metadata"""
    action_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trigger_time: float = 0.0         # When to execute (audio clock time)
    priority: int = EventPriority.NORMAL.value
    action: Dict[str, Any] = field(default_factory=dict)
    event_type: EventType = EventType.NORMAL
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate event data after initialization"""
        if self.trigger_time < 0:
            raise ValueError(f"Trigger time cannot be negative: {self.trigger_time}")
        if not isinstance(self.priority, int):
            raise ValueError(f"Priority must be integer: {self.priority}")
        if not self.action:
            raise ValueError("Action cannot be empty")

@dataclass
class EventResult:
    """Result of event execution"""
    success: bool
    action_id: str
    execution_time: float
    error_message: Optional[str] = None
    return_value: Any = None

class EventHandler(Protocol):
    """Protocol for event handlers"""
    def handle_event(self, event: ScheduledEvent) -> EventResult:
        """Handle a scheduled event"""
        ...

class EventFilter(Protocol):
    """Protocol for event filters"""
    def should_execute(self, event: ScheduledEvent, current_time: float) -> bool:
        """Determine if event should execute now"""
        ...

class EventValidator(Protocol):
    """Protocol for event validation"""
    def validate_event(self, event: ScheduledEvent) -> bool:
        """Validate event data"""
        ...

# Type aliases for cleaner code
EventID = str
EventCallback = Callable[[ScheduledEvent], EventResult]
EventFilterFunc = Callable[[ScheduledEvent, float], bool]
