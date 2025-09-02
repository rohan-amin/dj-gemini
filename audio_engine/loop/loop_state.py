"""
Loop state management for DJ loop system.

This module defines immutable loop state objects and provides a thread-safe registry
for managing active loops. All loop state transitions use immutable objects to
prevent race conditions and ensure thread safety.
"""

from dataclasses import dataclass, replace
from typing import Dict, Optional, Set, Any
from threading import RLock
import logging
import time

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoopState:
    """
    Immutable loop state object.
    
    This dataclass represents the complete state of a loop at any point in time.
    Being frozen (immutable), it can be safely shared between threads without
    synchronization concerns.
    """
    
    action_id: str
    deck_id: str
    start_beat: float
    end_beat: float
    iterations_planned: int
    iterations_completed: int = 0
    is_active: bool = False
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    current_frame_position: Optional[int] = None
    
    @property
    def is_complete(self) -> bool:
        """Check if this loop has completed all planned iterations."""
        return self.iterations_completed >= self.iterations_planned
    
    @property
    def duration_beats(self) -> float:
        """Get the duration of this loop in beats."""
        return self.end_beat - self.start_beat
    
    @property
    def progress(self) -> float:
        """Get completion progress as a ratio (0.0 to 1.0)."""
        if self.iterations_planned <= 0:
            return 1.0
        return min(1.0, self.iterations_completed / self.iterations_planned)
    
    def with_completed_iteration(self) -> 'LoopState':
        """Return a new LoopState with incremented iteration count."""
        return replace(self, iterations_completed=self.iterations_completed + 1)
    
    def with_activation(self, start_frame: int, end_frame: int) -> 'LoopState':
        """Return a new LoopState marked as active with frame boundaries."""
        return replace(
            self,
            is_active=True,
            start_frame=start_frame,
            end_frame=end_frame,
            current_frame_position=start_frame
        )
    
    def with_deactivation(self) -> 'LoopState':
        """Return a new LoopState marked as inactive."""
        return replace(self, is_active=False)
    
    def with_frame_position(self, frame_position: int) -> 'LoopState':
        """Return a new LoopState with updated current frame position."""
        return replace(self, current_frame_position=frame_position)


class LoopRegistry:
    """
    Thread-safe registry for managing active loops with enhanced locking strategy.
    
    This class serves as the single source of truth for all loop state in the system.
    It uses a combination of reader-writer semantics and atomic operations to ensure
    thread safety while maintaining high performance for concurrent access patterns.
    """
    
    def __init__(self):
        self._loops: Dict[str, LoopState] = {}
        self._lock = RLock()  # Main registry lock
        self._state_lock = RLock()  # Separate lock for state transitions
        self._read_count = 0
        self._read_lock = RLock()  # Lock for read counter
        self._write_lock = RLock()  # Lock for exclusive writes
        
        # State validation cache
        self._validation_cache = {}
        self._cache_lock = RLock()
        
        logger.info("🔄 Loop registry initialized with enhanced thread safety")
    
    def register_loop(self, loop_state: LoopState) -> None:
        """Register a new loop in the registry."""
        with self._lock:
            if loop_state.action_id in self._loops:
                logger.warning(
                    f"🔄 Loop {loop_state.action_id} already registered, replacing"
                )
            self._loops[loop_state.action_id] = loop_state
            logger.debug(
                f"🔄 Loop registered: {loop_state.action_id} on {loop_state.deck_id}"
            )
    
    def update_loop(self, action_id: str, new_state: LoopState) -> None:
        """
        Update an existing loop with new state using atomic operations.
        
        This method ensures atomic state transitions by using dedicated locking
        for state changes and validation caching.
        """
        # Use atomic state transition locking
        with self._state_lock:
            with self._write_lock:
                if action_id not in self._loops:
                    raise KeyError(f"Loop {action_id} not found in registry")
                
                old_state = self._loops[action_id]
                
                # Validate state transition
                if not self._validate_state_transition(old_state, new_state):
                    logger.error(f"🔄 Invalid state transition for loop {action_id}")
                    raise ValueError(f"Invalid state transition for loop {action_id}")
                
                # Perform atomic update
                self._loops[action_id] = new_state
                
                # Update validation cache
                self._update_validation_cache(action_id, new_state)
                
                logger.debug(
                    f"🔄 Loop state updated: {action_id} - "
                    f"iterations: {old_state.iterations_completed} → {new_state.iterations_completed}, "
                    f"active: {old_state.is_active} → {new_state.is_active}"
                )
    
    def get_loop(self, action_id: str) -> Optional[LoopState]:
        """Get current state of a loop by action ID using optimized read access."""
        # Use read-optimized locking pattern
        with self._read_lock:
            self._read_count += 1
            if self._read_count == 1:
                # First reader acquires write lock to prevent writes during reads
                self._write_lock.acquire()
        
        try:
            return self._loops.get(action_id)
        finally:
            with self._read_lock:
                self._read_count -= 1
                if self._read_count == 0:
                    # Last reader releases write lock
                    self._write_lock.release()
    
    def get_active_loops(self) -> Dict[str, LoopState]:
        """Get all currently active loops."""
        with self._lock:
            return {
                action_id: loop_state 
                for action_id, loop_state in self._loops.items() 
                if loop_state.is_active
            }
    
    def get_loops_for_deck(self, deck_id: str) -> Dict[str, LoopState]:
        """Get all loops (active and inactive) for a specific deck."""
        with self._lock:
            return {
                action_id: loop_state 
                for action_id, loop_state in self._loops.items() 
                if loop_state.deck_id == deck_id
            }
    
    def get_active_loops_for_deck(self, deck_id: str) -> Dict[str, LoopState]:
        """Get only active loops for a specific deck."""
        with self._lock:
            return {
                action_id: loop_state 
                for action_id, loop_state in self._loops.items() 
                if loop_state.deck_id == deck_id and loop_state.is_active
            }
    
    def remove_loop(self, action_id: str) -> Optional[LoopState]:
        """Remove a loop from the registry and return its final state."""
        with self._lock:
            removed_loop = self._loops.pop(action_id, None)
            if removed_loop:
                logger.debug(f"🔄 Loop removed: {action_id}")
            return removed_loop
    
    def get_all_loops(self) -> Dict[str, LoopState]:
        """Get a snapshot of all loops in the registry."""
        with self._lock:
            return self._loops.copy()
    
    def clear(self) -> None:
        """Remove all loops from the registry."""
        with self._lock:
            count = len(self._loops)
            self._loops.clear()
            logger.info(f"🔄 Loop registry cleared - {count} loops removed")
    
    def get_completed_loops(self) -> Set[str]:
        """Get action IDs of all completed loops."""
        with self._lock:
            return {
                action_id 
                for action_id, loop_state in self._loops.items() 
                if loop_state.is_complete
            }
    
    def __len__(self) -> int:
        """Get the number of loops in the registry."""
        with self._lock:
            return len(self._loops)
    
    def __contains__(self, action_id: str) -> bool:
        """Check if a loop is in the registry."""
        with self._lock:
            return action_id in self._loops
    
    def _validate_state_transition(self, old_state: LoopState, new_state: LoopState) -> bool:
        """
        Validate that a state transition is legal and consistent.
        
        This method implements atomic state transition validation to prevent
        invalid state changes that could cause audio artifacts or system instability.
        
        Args:
            old_state: Current loop state
            new_state: Proposed new loop state
            
        Returns:
            True if state transition is valid
        """
        try:
            # Basic identity validation
            if old_state.action_id != new_state.action_id:
                logger.error(f"🔄 State transition error: action_id mismatch")
                return False
            
            if old_state.deck_id != new_state.deck_id:
                logger.error(f"🔄 State transition error: deck_id mismatch")
                return False
            
            # Loop boundary consistency
            if (old_state.start_beat != new_state.start_beat or 
                old_state.end_beat != new_state.end_beat or
                old_state.iterations_planned != new_state.iterations_planned):
                # Core loop parameters should not change during execution
                logger.warning(f"🔄 Core loop parameters changed for {old_state.action_id}")
            
            # Iteration count validation
            if new_state.iterations_completed < old_state.iterations_completed:
                logger.error(f"🔄 State transition error: iteration count decreased")
                return False
            
            if new_state.iterations_completed > new_state.iterations_planned:
                logger.warning(f"🔄 Loop {old_state.action_id} completed more iterations than planned")
            
            # State consistency validation
            if new_state.is_complete and new_state.is_active:
                logger.error(f"🔄 State transition error: loop cannot be both complete and active")
                return False
            
            # Frame boundary validation
            if new_state.is_active:
                if new_state.start_frame is None or new_state.end_frame is None:
                    logger.error(f"🔄 State transition error: active loop missing frame boundaries")
                    return False
                
                if new_state.start_frame >= new_state.end_frame:
                    logger.error(f"🔄 State transition error: invalid frame boundaries")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"🔄 Error validating state transition: {e}")
            return False
    
    def _update_validation_cache(self, action_id: str, loop_state: LoopState) -> None:
        """
        Update validation cache for performance optimization.
        
        Args:
            action_id: Loop action ID
            loop_state: Current loop state
        """
        try:
            with self._cache_lock:
                # Cache key loop properties for quick validation
                self._validation_cache[action_id] = {
                    'is_active': loop_state.is_active,
                    'is_complete': loop_state.is_complete,
                    'iterations_completed': loop_state.iterations_completed,
                    'last_updated': time.time()
                }
                
                # Limit cache size to prevent memory bloat
                if len(self._validation_cache) > 100:
                    # Remove oldest entries
                    oldest_items = sorted(
                        self._validation_cache.items(),
                        key=lambda x: x[1]['last_updated']
                    )[:50]
                    
                    for old_action_id, _ in oldest_items:
                        if old_action_id in self._validation_cache:
                            del self._validation_cache[old_action_id]
                            
        except Exception as e:
            logger.debug(f"🔄 Error updating validation cache: {e}")
    
    def get_state_consistency_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive state consistency report for debugging.
        
        Returns:
            Dictionary with consistency analysis
        """
        try:
            with self._lock:
                report = {
                    'total_loops': len(self._loops),
                    'active_loops': 0,
                    'completed_loops': 0,
                    'inconsistencies': [],
                    'validation_cache_size': len(self._validation_cache),
                    'timestamp': time.time()
                }
                
                for action_id, loop_state in self._loops.items():
                    if loop_state.is_active:
                        report['active_loops'] += 1
                    
                    if loop_state.is_complete:
                        report['completed_loops'] += 1
                    
                    # Check for inconsistencies
                    if loop_state.is_active and loop_state.is_complete:
                        report['inconsistencies'].append(f"{action_id}: active and complete")
                    
                    if loop_state.iterations_completed > loop_state.iterations_planned:
                        report['inconsistencies'].append(f"{action_id}: over-iterations")
                    
                    if loop_state.is_active and (loop_state.start_frame is None or loop_state.end_frame is None):
                        report['inconsistencies'].append(f"{action_id}: active without frame boundaries")
                
                return report
                
        except Exception as e:
            logger.error(f"🔄 Error generating consistency report: {e}")
            return {'error': str(e), 'timestamp': time.time()}