"""
DJ Loop Management System

This module provides a centralized, thread-safe loop management system for DJ audio mixing.
It implements a single source of truth architecture to eliminate fragmented loop state
that was causing audio interruptions.

Key Components:
- LoopState: Immutable loop state objects
- LoopRegistry: Thread-safe registry of active loops  
- LoopController: Centralized loop management
- LoopEvents: Event system for loop lifecycle communication
- LoopCompletionSystem: Event-driven completion action handling
"""

from .loop_state import LoopState, LoopRegistry
from .loop_controller import LoopController
from .loop_events import LoopEvent, LoopEventType, LoopEventPublisher
from .completion_system import (
    LoopCompletionSystem,
    CompletionConfiguration,
    CompletionAction,
    ActionType,
    CompletionActionHandler,
    create_completion_configuration_from_json
)
# Action-based integration
from .action_adapter import (
    ActionLoopAdapter,
    PendingCompletionAction,
    create_action_adapter_for_deck
)
from .observability import (
    LoopSystemHealthMonitor,
    ObservabilityIntegrator,
    MetricsCollector,
    PerformanceTimer,
    HealthStatus,
    MetricType,
    timed_operation,
    counted_operation
)

__all__ = [
    # Core classes
    'LoopState',
    'LoopRegistry', 
    'LoopController',
    
    # Event system
    'LoopEvent',
    'LoopEventType',
    'LoopEventPublisher',
    
    # Completion system
    'LoopCompletionSystem',
    'CompletionConfiguration',
    'CompletionAction',
    'ActionType',
    'CompletionActionHandler',
    'create_completion_configuration_from_json',
    
    # Action-based integration
    'ActionLoopAdapter',
    'PendingCompletionAction',
    'create_action_adapter_for_deck',
    
    # Observability
    'LoopSystemHealthMonitor',
    'ObservabilityIntegrator',
    'MetricsCollector',
    'PerformanceTimer',
    'HealthStatus',
    'MetricType',
    'timed_operation',
    'counted_operation'
]