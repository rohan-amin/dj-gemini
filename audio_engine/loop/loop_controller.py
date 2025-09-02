"""
Centralized loop controller for DJ loop management system.

This module provides the core LoopController class that manages all loop operations
in a single thread context (producer thread). It serves as the single source of truth
for loop lifecycle management and coordinates with the BeatManager for frame-accurate
positioning.
"""

import logging
import time
from typing import Dict, List, Optional, Set, Tuple, Any
from .loop_state import LoopState, LoopRegistry
from .loop_events import LoopEvent, LoopEventType, LoopEventPublisher
from .completion_system import LoopCompletionSystem, CompletionConfiguration

logger = logging.getLogger(__name__)


class LoopController:
    """
    Centralized controller for all loop management operations.
    
    This class implements the single source of truth architecture for loop management.
    All loop lifecycle operations (activation, iteration, completion) are managed
    here in the producer thread context to eliminate race conditions and ensure
    frame-accurate timing.
    
    Key responsibilities:
    - Manage loop lifecycle (activate, iterate, complete)
    - Convert between beats and frames using BeatManager
    - Handle seamless audio position jumps for loops
    - Publish loop events for completion actions
    - Maintain thread-safe loop state registry
    """
    
    def __init__(self, deck, beat_manager_adapter):
        """
        Initialize the loop controller.
        
        Args:
            deck: The deck instance this controller manages loops for
            beat_manager_adapter: BeatManagerAdapter for beat/frame conversions
        """
        self._deck = deck
        self._deck_id = getattr(deck, 'deck_id', 'unknown')
        self._beat_converter = beat_manager_adapter
        
        # Core components
        self._registry = LoopRegistry()
        self._event_publisher = LoopEventPublisher()
        self._completion_system = None  # Will be initialized when deck_manager is available
        
        # State tracking
        self._is_initialized = True
        self._processing_enabled = True
        
        # Performance tracking
        self._stats = {
            'loops_activated': 0,
            'iterations_completed': 0,
            'loops_completed': 0,
            'seamless_jumps': 0,
            'processing_errors': 0,
            'timing_drift_events': 0,
            'last_update': time.time()
        }
        
        # Timing drift detection
        self._timing_history = {}  # action_id -> timing data
        self._drift_threshold_frames = 1000  # ~22ms at 44.1kHz
        self._last_beat_check = time.time()
        self._beat_sync_interval = 1.0  # Check beat sync every second
        
        logger.info(f"🔄 LoopController initialized for deck {self._deck_id}")
    
    def create_loop(self, action_id: str, start_beat: float, end_beat: float, 
                   iterations: int) -> bool:
        """
        Create a new loop definition.
        
        Args:
            action_id: Unique identifier for this loop
            start_beat: Loop start position in beats (1-based)
            end_beat: Loop end position in beats (1-based)
            iterations: Number of times to repeat the loop
            
        Returns:
            True if loop was created successfully
        """
        try:
            # Validate parameters
            if start_beat >= end_beat:
                logger.error(f"🔄 Invalid loop bounds: start_beat {start_beat} >= end_beat {end_beat}")
                return False
            
            if iterations <= 0:
                logger.error(f"🔄 Invalid iteration count: {iterations}")
                return False
            
            # Create loop state
            loop_state = LoopState(
                action_id=action_id,
                deck_id=self._deck_id,
                start_beat=start_beat,
                end_beat=end_beat,
                iterations_planned=iterations
            )
            
            # Register in the registry
            self._registry.register_loop(loop_state)
            
            logger.info(f"🔄 Loop created: {action_id} on {self._deck_id} - "
                       f"beats {start_beat:.2f}-{end_beat:.2f}, {iterations} iterations")
            
            return True
            
        except Exception as e:
            logger.error(f"🔄 Error creating loop {action_id}: {e}")
            self._stats['processing_errors'] += 1
            return False
    
    def activate_loop(self, action_id: str, start_at_beat: Optional[float] = None) -> bool:
        """
        Activate a loop, starting its playback cycle at beat boundaries.
        
        This converts the loop's beat boundaries to frames, validates timing,
        and performs seamless activation either immediately or at a specified beat.
        
        Args:
            action_id: ID of the loop to activate
            start_at_beat: Optional beat to start the loop at (None = immediate)
            
        Returns:
            True if loop was activated successfully
        """
        try:
            # Get current loop state
            loop_state = self._registry.get_loop(action_id)
            if not loop_state:
                logger.error(f"🔄 Loop not found for activation: {action_id}")
                return False
            
            if loop_state.is_active:
                logger.warning(f"🔄 Loop already active: {action_id}")
                return True
            
            # Validate loop boundaries with safety checks
            if not self._validate_loop_boundaries(loop_state):
                logger.error(f"🔄 Loop boundary validation failed: {action_id}")
                return False
            
            # Convert beats to frames using BeatManager with validation
            try:
                start_frame = self._beat_converter.get_frame_for_beat(loop_state.start_beat)
                end_frame = self._beat_converter.get_frame_for_beat(loop_state.end_beat)
                
                # Additional frame validation
                track_length = getattr(self._deck, 'total_frames', 0)
                if not self._validate_frame_boundaries(start_frame, end_frame, track_length):
                    return False
                    
            except Exception as e:
                logger.error(f"🔄 Failed to convert beats to frames for loop {action_id}: {e}")
                return False
            
            # Handle beat boundary activation
            current_frame = self._get_current_frame()
            if start_at_beat is not None:
                # Activate at specific beat - check if we need to seek
                target_frame = self._beat_converter.get_frame_for_beat(start_at_beat)
                if abs(current_frame - target_frame) > self._get_beat_tolerance_frames():
                    # Need to seek to start position
                    if not self._seek_to_beat_boundary(start_at_beat):
                        logger.error(f"🔄 Failed to seek to start beat {start_at_beat} for loop {action_id}")
                        return False
                    current_frame = target_frame
            
            # Check if we need to seek to loop start for seamless activation
            if current_frame < start_frame or current_frame >= end_frame:
                if not self._seek_to_frame_seamless(start_frame):
                    logger.error(f"🔄 Failed to seek to loop start frame {start_frame} for loop {action_id}")
                    return False
                current_frame = start_frame
            
            # Update loop state to active with current position
            active_state = loop_state.with_activation(start_frame, end_frame).with_frame_position(current_frame)
            self._registry.update_loop(action_id, active_state)
            
            # Publish activation event
            event = LoopEvent(
                event_type=LoopEventType.START,
                loop_id=action_id,
                deck_id=self._deck_id,
                current_iteration=0,
                total_iterations=loop_state.iterations_planned,
                timestamp=time.time()
            )
            self._event_publisher.publish(event)
            
            # Update statistics
            self._stats['loops_activated'] += 1
            self._stats['last_update'] = time.time()
            
            logger.info(f"🔄 LOOP START: {action_id} on {self._deck_id} - "
                       f"{loop_state.iterations_planned} iterations planned")
            
            return True
            
        except Exception as e:
            logger.error(f"🔄 Error activating loop {action_id}: {e}")
            self._stats['processing_errors'] += 1
            return False
    
    def deactivate_loop(self, action_id: str) -> bool:
        """
        Deactivate a currently active loop.
        
        This method gracefully deactivates a loop without audio artifacts,
        stopping its playback cycle while maintaining system stability.
        
        Args:
            action_id: ID of the loop to deactivate
            
        Returns:
            True if loop was deactivated successfully
        """
        try:
            # Get current loop state
            loop_state = self._registry.get_loop(action_id)
            if not loop_state:
                logger.error(f"🔄 Loop not found for deactivation: {action_id}")
                return False
            
            if not loop_state.is_active:
                logger.warning(f"🔄 Loop already inactive: {action_id}")
                return True
            
            # Perform graceful deactivation
            deactivated_state = loop_state.with_deactivation()
            self._registry.update_loop(action_id, deactivated_state)
            
            # Remove completion configuration to prevent spurious completions
            if self._completion_system:
                self._completion_system.unregister_completion_configuration(action_id)
            
            # Update statistics
            self._stats['last_update'] = time.time()
            
            logger.info(f"🔄 LOOP DEACTIVATED: {action_id} on {self._deck_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"🔄 Error deactivating loop {action_id}: {e}")
            self._stats['processing_errors'] += 1
            return False
    
    def process_active_loops(self, current_frame: int) -> List[Dict]:
        """
        Process all active loops for the current audio frame.
        
        This method should be called from the producer thread's audio processing
        loop to handle loop iterations and completions.
        
        Args:
            current_frame: Current audio frame position
            
        Returns:
            List of loop events (completions, iterations) that occurred
        """
        if not self._processing_enabled:
            return []
        
        events = []
        
        try:
            # Get all active loops
            active_loops = self._registry.get_active_loops()
            
            for action_id, loop_state in active_loops.items():
                try:
                    loop_events = self._process_single_loop(action_id, loop_state, current_frame)
                    events.extend(loop_events)
                except Exception as e:
                    logger.error(f"🔄 Error processing loop {action_id}: {e}")
                    self._stats['processing_errors'] += 1
            
            return events
            
        except Exception as e:
            logger.error(f"🔄 Error processing active loops: {e}")
            self._stats['processing_errors'] += 1
            return []
    
    def _process_single_loop(self, action_id: str, loop_state: LoopState, 
                           current_frame: int) -> List[Dict]:
        """
        Process a single loop for the current frame with accurate iteration tracking.
        
        This method implements the core loop logic with frame-accurate detection
        of loop boundaries and seamless jump-back for continuous iterations.
        
        Args:
            action_id: Loop action ID
            loop_state: Current loop state
            current_frame: Current audio frame position
            
        Returns:
            List of events that occurred for this loop
        """
        events = []
        
        try:
            # Validate timing accuracy - ensure we're still in sync
            if not self._validate_loop_timing(loop_state, current_frame):
                logger.warning(f"🔄 Timing validation failed for loop {action_id}")
                return []
            
            # Check for loop boundary crossing with frame accuracy
            if self._has_crossed_loop_boundary(loop_state, current_frame):
                # We've reached the end of the current loop iteration
                new_iterations = loop_state.iterations_completed + 1
                
                # FIXED: Create updated state with correct iteration count first
                # This prevents race conditions where the old state is used inconsistently
                updated_loop_state = loop_state.with_completed_iteration()
                
                # Verify the state update was successful
                if updated_loop_state.iterations_completed != new_iterations:
                    logger.error(f"🔄 State update mismatch: expected {new_iterations}, got {updated_loop_state.iterations_completed}")
                    return []
                
                # Check completion condition - iteration count vs target
                if self._is_loop_complete(new_iterations, loop_state.iterations_planned):
                    # Loop has completed all planned iterations - use updated state
                    events.append(self._complete_loop(action_id, updated_loop_state, new_iterations))
                else:
                    # Continue to next iteration - use updated state 
                    events.extend(self._iterate_loop(action_id, updated_loop_state, new_iterations, current_frame))
                
                # FIXED: Update position tracking with the updated state, not the old one
                self._update_loop_position_tracking(action_id, updated_loop_state, current_frame)
                
                # Check for timing drift and perform correction if needed - use updated state
                self._check_and_correct_timing_drift(action_id, updated_loop_state, current_frame)
            else:
                # No boundary crossing - just update position tracking with original state
                self._update_loop_position_tracking(action_id, loop_state, current_frame)
                self._check_and_correct_timing_drift(action_id, loop_state, current_frame)
        
        except Exception as e:
            logger.error(f"🔄 Error processing loop {action_id}: {e}")
            # Create error event
            events.append({
                'type': 'error',
                'action_id': action_id,
                'error': str(e),
                'current_frame': current_frame
            })
        
        return events
    
    def _validate_loop_timing(self, loop_state: LoopState, current_frame: int) -> bool:
        """
        Validate loop timing accuracy to detect drift or synchronization issues.
        
        Args:
            loop_state: Current loop state
            current_frame: Current frame position
            
        Returns:
            True if timing is accurate within tolerance
        """
        try:
            # Skip validation for inactive loops
            if not loop_state.is_active:
                return True
            
            # Check if current frame is within reasonable bounds for this loop
            if current_frame < 0:
                return False
            
            # For active loops, validate timing more intelligently
            tolerance = self._get_beat_tolerance_frames()
            loop_start = loop_state.start_frame
            loop_end = loop_state.end_frame
            
            # Allow normal linear playback past loop end - that's when we should jump back!
            # Only fail if we're unreasonably far beyond the loop (indicating real timing issues)
            max_reasonable_overshoot = tolerance * 8  # Allow up to 8x tolerance overshoot
            
            if current_frame < (loop_start - tolerance):
                logger.debug(f"🔄 Loop {loop_state.action_id} timing drift: "
                           f"frame={current_frame} too far before start {loop_start}")
                return False
            elif current_frame > (loop_end + max_reasonable_overshoot):
                logger.debug(f"🔄 Loop {loop_state.action_id} severe timing drift: "
                           f"frame={current_frame} extremely far past end {loop_end}")
                return False
            
            # Normal case: allow current_frame to advance past loop_end naturally
            return True
            
        except Exception as e:
            logger.error(f"🔄 Error validating loop timing: {e}")
            return False
    
    def _has_crossed_loop_boundary(self, loop_state: LoopState, current_frame: int) -> bool:
        """
        Check if playback has crossed the loop end boundary.
        
        Args:
            loop_state: Current loop state
            current_frame: Current frame position
            
        Returns:
            True if loop boundary has been crossed
        """
        try:
            # Only check boundary crossing for active loops
            if not loop_state.is_active or loop_state.end_frame is None:
                return False
            
            # Check if we've reached or passed the end frame
            # Use small tolerance to account for buffer boundaries
            tolerance = min(512, self._get_beat_tolerance_frames())  
            boundary_crossed = current_frame >= (loop_state.end_frame - tolerance)
            
            if boundary_crossed:
                logger.info(f"🔄 Loop boundary crossed: frame={current_frame}, loop_end={loop_state.end_frame}, tolerance={tolerance}")
            
            return boundary_crossed
            
        except Exception as e:
            logger.error(f"🔄 Error checking loop boundary crossing: {e}")
            return False
    
    def _is_loop_complete(self, current_iterations: int, planned_iterations: int) -> bool:
        """
        Check completion condition - iteration count vs target.
        
        We complete the loop when we've finished the planned number of iterations.
        The boundary crossing detection happens at the END of each iteration,
        so when current_iterations equals planned_iterations, we've completed
        all the required iterations and should stop.
        
        Args:
            current_iterations: Current number of completed iterations
            planned_iterations: Target number of iterations
            
        Returns:
            True if loop should be completed
        """
        # Complete when we've finished all planned iterations
        return current_iterations >= planned_iterations
    
    def _update_loop_position_tracking(self, action_id: str, loop_state: LoopState, current_frame: int) -> None:
        """
        Update loop position tracking for continuous monitoring.
        
        Args:
            action_id: Loop action ID
            loop_state: Current loop state  
            current_frame: Current frame position
        """
        # FIXED: Don't update registry here - it conflicts with iteration count updates
        # This was causing race conditions where position updates overwrote iteration counts
        # Position tracking is now handled by the caller who has the authoritative state
        try:
            # Just log the position for monitoring - don't update registry
            if loop_state.current_frame_position is None or \
               abs(current_frame - loop_state.current_frame_position) > 10000:  # Only log significant changes
                logger.debug(f"🔄 Loop {action_id} position: {current_frame} (was {loop_state.current_frame_position})")
                
        except Exception as e:
            logger.debug(f"🔄 Error in loop position tracking: {e}")
    
    def _iterate_loop(self, action_id: str, loop_state: LoopState, 
                     new_iteration_count: int, current_frame: int) -> List[Dict]:
        """
        Handle loop iteration with seamless jump-back logic.
        
        This method implements frame-accurate jump-back to the loop start
        position without causing audio artifacts, maintaining perfect
        musical timing throughout the loop cycle.
        
        Args:
            action_id: Loop action ID
            loop_state: Current loop state
            new_iteration_count: New iteration count after this iteration
            current_frame: Current frame position
            
        Returns:
            List containing iteration event
        """
        try:
            # Calculate precise jump target with musical timing alignment
            jump_target_frame = self._calculate_seamless_jump_target(loop_state)
            
            # Store current loop ID for event handling
            self._current_loop_id = action_id
            
            # Perform seamless jump back to loop start with timing validation
            jump_success = self._perform_seamless_jump_back(
                jump_target_frame, 
                loop_state, 
                current_frame
            )
            
            if not jump_success:
                logger.error(f"🔄 Failed seamless jump-back for loop {action_id}")
                # Try fallback jump method
                if not self._perform_fallback_jump(loop_state.start_frame):
                    logger.error(f"🔄 Fallback jump also failed for loop {action_id}")
                    return []
            
            # Update loop state with reset position (iteration count already updated)
            updated_state = loop_state.with_frame_position(jump_target_frame)
            self._registry.update_loop(action_id, updated_state)
            
            # Publish iteration event
            event = LoopEvent(
                event_type=LoopEventType.ITERATION,
                loop_id=action_id,
                deck_id=self._deck_id,
                current_iteration=new_iteration_count,
                total_iterations=loop_state.iterations_planned,
                timestamp=time.time()
            )
            self._event_publisher.publish(event)
            
            # Update statistics with timing information
            self._stats['iterations_completed'] += 1
            self._stats['seamless_jumps'] += 1
            self._stats['last_update'] = time.time()
            
            logger.info(f"🔄 LOOP ITERATION: {action_id} on {self._deck_id} - "
                       f"iteration {new_iteration_count}/{loop_state.iterations_planned} complete")
            
            # Log jump timing for debugging
            jump_distance_frames = abs(current_frame - jump_target_frame)
            jump_distance_ms = (jump_distance_frames / getattr(self._deck, 'sample_rate', 44100)) * 1000
            logger.debug(f"🔄 Seamless jump: {jump_distance_frames} frames ({jump_distance_ms:.2f}ms)")
            
            return [{
                'type': 'iteration',
                'action_id': action_id,
                'iteration': new_iteration_count,
                'total_iterations': loop_state.iterations_planned,
                'jump_distance_frames': jump_distance_frames,
                'jump_target_frame': jump_target_frame
            }]
            
        except Exception as e:
            logger.error(f"🔄 Error in loop iteration for {action_id}: {e}")
            self._stats['processing_errors'] += 1
            return []
    
    def _calculate_seamless_jump_target(self, loop_state: LoopState) -> int:
        """
        Calculate the precise frame target for seamless jump-back.
        
        This ensures the jump lands exactly on a beat boundary for
        musical accuracy and seamless audio transitions.
        
        Args:
            loop_state: Current loop state
            
        Returns:
            Target frame for jump-back
        """
        try:
            # Start with the loop's defined start frame
            target_frame = loop_state.start_frame
            
            # Align to nearest beat boundary if possible
            start_beat = loop_state.start_beat
            aligned_frame = self._beat_converter.get_frame_for_beat(start_beat)
            
            # Use aligned frame if it's close to the original
            tolerance = self._get_beat_tolerance_frames()
            if abs(aligned_frame - target_frame) <= tolerance:
                target_frame = aligned_frame
                logger.debug(f"🔄 Jump target aligned to beat {start_beat}: frame {target_frame}")
            
            return target_frame
            
        except Exception as e:
            logger.warning(f"🔄 Error calculating jump target, using loop start frame: {e}")
            return loop_state.start_frame or 0
    
    def _perform_seamless_jump_back(self, target_frame: int, loop_state: LoopState, 
                                   current_frame: int) -> bool:
        """
        Perform seamless jump-back with enhanced timing and validation.
        
        Args:
            target_frame: Target frame to jump to
            loop_state: Current loop state
            current_frame: Current frame before jump
            
        Returns:
            True if jump was successful
        """
        try:
            # Pre-jump validation
            if not self._validate_jump_target(target_frame, loop_state):
                logger.warning(f"🔄 Jump target validation failed: {target_frame}")
                return False
            
            # Perform the actual jump
            success = self._seek_to_frame_seamless(target_frame)
            
            if success:
                # FIXED: Trust deck.seek() - it's asynchronous but reliable
                # Removed premature accuracy validation that was causing audio breakup
                # The producer thread will process the seek command properly
                logger.debug(f"🔄 Loop jump initiated to frame {target_frame} via deck.seek()")
                return True
            
            return success
            
        except Exception as e:
            logger.error(f"🔄 Error in seamless jump-back: {e}")
            return False
    
    def _validate_jump_target(self, target_frame: int, loop_state: LoopState) -> bool:
        """
        Validate that the jump target is safe and accurate.
        
        Args:
            target_frame: Target frame for jump
            loop_state: Current loop state
            
        Returns:
            True if jump target is valid
        """
        try:
            # Basic frame validation
            if target_frame < 0:
                return False
            
            # Check against track length
            track_length = getattr(self._deck, 'total_frames', 0)
            if track_length > 0 and target_frame >= track_length:
                return False
            
            # Ensure target is within loop boundaries
            if loop_state.start_frame is not None and loop_state.end_frame is not None:
                if target_frame < loop_state.start_frame or target_frame >= loop_state.end_frame:
                    logger.warning(f"🔄 Jump target {target_frame} outside loop bounds [{loop_state.start_frame}, {loop_state.end_frame})")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"🔄 Error validating jump target: {e}")
            return False
    
    def _perform_fallback_jump(self, target_frame: int) -> bool:
        """
        Fallback jump method if primary seamless jump fails.
        
        Args:
            target_frame: Frame to jump to
            
        Returns:
            True if fallback jump succeeded
        """
        try:
            logger.info(f"🔄 Attempting fallback jump to frame {target_frame}")
            return self._seek_to_frame_seamless(target_frame)
        except Exception as e:
            logger.error(f"🔄 Fallback jump failed: {e}")
            return False
    
    def _complete_loop(self, action_id: str, loop_state: LoopState, 
                      final_iteration_count: int) -> Dict:
        """
        Complete a loop with artifact-free deactivation and completion event triggering.
        
        This method handles the final loop iteration, deactivates the loop without
        causing audio interruptions, and triggers any registered completion actions
        through the event system.
        
        Args:
            action_id: Loop action ID
            loop_state: Current loop state
            final_iteration_count: Final iteration count
            
        Returns:
            Dictionary describing the completion event
        """
        try:
            # Perform graceful loop deactivation without audio artifacts
            deactivation_success = self._deactivate_loop_gracefully(action_id, loop_state)
            
            if not deactivation_success:
                logger.warning(f"🔄 Graceful deactivation failed for loop {action_id}, using fallback")
            
            # Update loop state to completed and inactive (iteration count already updated)
            completed_state = loop_state.with_deactivation()
            self._registry.update_loop(action_id, completed_state)
            
            # Create and publish completion event to trigger completion actions
            completion_event = LoopEvent(
                event_type=LoopEventType.COMPLETE,
                loop_id=action_id,
                deck_id=self._deck_id,
                current_iteration=final_iteration_count,
                total_iterations=loop_state.iterations_planned,
                timestamp=time.time(),
                additional_data={
                    'deactivation_success': deactivation_success,
                    'loop_duration_beats': loop_state.duration_beats,
                    'completion_frame': self._get_current_frame()
                }
            )
            
            # Publish completion event for completion action triggering
            event_published = self._event_publisher.publish(completion_event)
            
            if not event_published:
                logger.warning(f"🔄 Failed to publish completion event for loop {action_id}")
            
            # Trigger completion system to handle completion actions
            if self._completion_system:
                try:
                    self._completion_system.handle_loop_completion(action_id, self._deck_id)
                except Exception as e:
                    logger.error(f"🔄 Error handling loop completion in completion system: {e}")
            else:
                logger.debug(f"🔄 No completion system available for loop {action_id}")
            
            # Update statistics with completion details
            self._stats['loops_completed'] += 1
            self._stats['last_update'] = time.time()
            
            # Calculate loop performance metrics
            loop_duration_ms = (loop_state.duration_beats * 60000) / self._beat_converter.get_bpm()
            total_loop_time_ms = loop_duration_ms * final_iteration_count
            
            logger.info(f"🔄 LOOP COMPLETE: {action_id} on {self._deck_id} - "
                       f"{final_iteration_count} iterations finished")
            logger.debug(f"🔄 Loop performance: {loop_duration_ms:.1f}ms per iteration, "
                        f"{total_loop_time_ms:.1f}ms total runtime")
            
            return {
                'type': 'completion',
                'action_id': action_id,
                'total_iterations': final_iteration_count,
                'deactivation_success': deactivation_success,
                'event_published': event_published,
                'loop_duration_beats': loop_state.duration_beats,
                'total_runtime_ms': total_loop_time_ms
            }
            
        except Exception as e:
            logger.error(f"🔄 Error completing loop {action_id}: {e}")
            self._stats['processing_errors'] += 1
            
            # Try to deactivate loop even if other operations failed
            try:
                self._deactivate_loop_emergency(action_id)
            except Exception as emergency_error:
                logger.error(f"🔄 Emergency deactivation also failed for {action_id}: {emergency_error}")
            
            return {
                'type': 'error', 
                'action_id': action_id,
                'error': str(e),
                'emergency_deactivation_attempted': True
            }
    
    def _deactivate_loop_gracefully(self, action_id: str, loop_state: LoopState) -> bool:
        """
        Deactivate loop without causing audio artifacts.
        
        This method ensures that loop deactivation happens smoothly without
        interrupting the audio stream or causing clicks/pops.
        
        Args:
            action_id: Loop action ID
            loop_state: Current loop state
            
        Returns:
            True if deactivation was successful
        """
        try:
            logger.debug(f"🔄 Starting graceful deactivation for loop {action_id}")
            
            # Ensure we're at a safe position for deactivation
            current_frame = self._get_current_frame()
            
            # Allow playback to continue naturally past loop end
            # No seeking required - just let it play through
            safe_deactivation_frame = loop_state.end_frame
            
            # If we're very close to the end frame, let it play naturally
            tolerance = self._get_beat_tolerance_frames()
            if abs(current_frame - safe_deactivation_frame) <= tolerance:
                logger.debug(f"🔄 Natural deactivation: current frame {current_frame} close to end frame {safe_deactivation_frame}")
                return True
            
            # If we're past the end frame, deactivation is already natural
            if current_frame > safe_deactivation_frame:
                logger.debug(f"🔄 Natural deactivation: already past end frame")
                return True
            
            # If we're before the end, we might need to wait or seek depending on timing
            frames_to_end = safe_deactivation_frame - current_frame
            sample_rate = getattr(self._deck, 'sample_rate', 44100)
            time_to_end_ms = (frames_to_end / sample_rate) * 1000
            
            # If it's a very short wait (< 50ms), let it play naturally
            if time_to_end_ms < 50:
                logger.debug(f"🔄 Natural deactivation: waiting {time_to_end_ms:.1f}ms for natural end")
                return True
            
            # For longer waits, perform gentle seek to end position
            logger.debug(f"🔄 Gentle seek to loop end for graceful deactivation")
            return self._seek_to_frame_seamless(safe_deactivation_frame)
            
        except Exception as e:
            logger.error(f"🔄 Error in graceful loop deactivation: {e}")
            return False
    
    def _deactivate_loop_emergency(self, action_id: str) -> bool:
        """
        Emergency loop deactivation when graceful methods fail.
        
        Args:
            action_id: Loop action ID
            
        Returns:
            True if emergency deactivation succeeded
        """
        try:
            logger.warning(f"🔄 Emergency deactivation for loop {action_id}")
            
            # Force deactivation in registry
            loop_state = self._registry.get_loop(action_id)
            if loop_state:
                emergency_state = loop_state.with_deactivation()
                self._registry.update_loop(action_id, emergency_state)
            
            # Stop any pending jumps or seeks
            # (Implementation depends on deck's command queue system)
            
            return True
            
        except Exception as e:
            logger.error(f"🔄 Emergency deactivation failed: {e}")
            return False
    
    def _perform_seamless_jump(self, target_frame: int) -> bool:
        """
        Perform a seamless audio jump to the target frame position.
        
        This coordinates with the deck's audio system to jump to the new position
        without causing audio artifacts.
        
        Args:
            target_frame: Frame position to jump to
            
        Returns:
            True if jump was successful
        """
        try:
            return self._seek_to_frame_seamless(target_frame)
        except Exception as e:
            logger.error(f"🔄 Error performing seamless jump to frame {target_frame}: {e}")
            return False
    
    def _validate_loop_boundaries(self, loop_state: LoopState) -> bool:
        """
        Validate loop boundaries for safety and musical accuracy.
        
        Args:
            loop_state: Loop state to validate
            
        Returns:
            True if boundaries are valid
        """
        try:
            # Check beat boundary logic
            if loop_state.start_beat >= loop_state.end_beat:
                logger.error(f"🔄 Invalid beat bounds: start={loop_state.start_beat} >= end={loop_state.end_beat}")
                return False
            
            # Check minimum loop length (should be at least one beat)
            loop_duration = loop_state.end_beat - loop_state.start_beat
            if loop_duration < 0.25:  # Quarter beat minimum
                logger.error(f"🔄 Loop too short: {loop_duration:.3f} beats")
                return False
            
            # Check maximum reasonable loop length (prevent memory issues)
            if loop_duration > 1000:  # 1000 beats maximum
                logger.error(f"🔄 Loop too long: {loop_duration:.3f} beats")
                return False
            
            # Check iteration count
            if loop_state.iterations_planned <= 0:
                logger.error(f"🔄 Invalid iteration count: {loop_state.iterations_planned}")
                return False
            
            # Check if loop is within track bounds
            current_beat = self._beat_converter.get_current_beat()
            max_beat = self._get_track_length_in_beats()
            
            if loop_state.end_beat > max_beat:
                logger.error(f"🔄 Loop extends beyond track: end_beat={loop_state.end_beat:.2f}, track_beats={max_beat:.2f}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"🔄 Error validating loop boundaries: {e}")
            return False
    
    def _validate_frame_boundaries(self, start_frame: int, end_frame: int, track_length: int) -> bool:
        """
        Validate frame boundaries against track length.
        
        Args:
            start_frame: Loop start frame
            end_frame: Loop end frame  
            track_length: Total track length in frames
            
        Returns:
            True if frame boundaries are valid
        """
        try:
            # Basic frame validation
            if start_frame < 0:
                logger.error(f"🔄 Negative start frame: {start_frame}")
                return False
                
            if end_frame <= start_frame:
                logger.error(f"🔄 Invalid frame bounds: start={start_frame} >= end={end_frame}")
                return False
            
            # Check against track length
            if track_length > 0:
                if end_frame > track_length:
                    logger.error(f"🔄 Loop extends beyond track: end_frame={end_frame}, track_length={track_length}")
                    return False
            
            # Check minimum loop length in frames (at least 1000 frames ~ 22ms at 44.1kHz)
            min_loop_frames = 1000
            if (end_frame - start_frame) < min_loop_frames:
                logger.error(f"🔄 Loop too short in frames: {end_frame - start_frame} < {min_loop_frames}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"🔄 Error validating frame boundaries: {e}")
            return False
    
    def _get_current_frame(self) -> int:
        """Get current playback frame position from deck."""
        try:
            if hasattr(self._deck, 'audio_thread_current_frame'):
                return int(self._deck.audio_thread_current_frame)
            elif hasattr(self._deck, '_current_playback_frame_for_display'):
                return int(self._deck._current_playback_frame_for_display)
            elif hasattr(self._beat_converter, 'get_current_frame'):
                return int(self._beat_converter.get_current_frame())
            else:
                logger.warning("🔄 No method available to get current frame, defaulting to 0")
                return 0
        except Exception as e:
            logger.error(f"🔄 Error getting current frame: {e}")
            return 0
    
    def _get_beat_tolerance_frames(self) -> int:
        """Get frame tolerance for beat boundary matching."""
        try:
            # Tolerance of 1/32 beat at current BPM
            sample_rate = getattr(self._deck, 'sample_rate', 44100)
            bpm = self._beat_converter.get_bpm()
            
            # Calculate frames per beat, then 1/32 of that
            frames_per_beat = (60.0 / bpm) * sample_rate
            tolerance = int(frames_per_beat / 32)
            
            # Minimum tolerance of 100 frames (~2ms at 44.1kHz)
            return max(100, tolerance)
        except Exception as e:
            logger.error(f"🔄 Error calculating beat tolerance: {e}")
            return 1000  # Safe default
    
    def _estimate_audio_latency(self) -> int:
        """
        Estimate the audio latency between internal frame position and actual audio output.
        
        This accounts for buffering delays in the audio pipeline that cause the internal
        frame position to be ahead of what the listener actually hears.
        
        Returns:
            Estimated latency in frames
        """
        try:
            sample_rate = getattr(self._deck, 'sample_rate', 44100)
            
            # Estimate various buffering sources:
            # 1. Ring buffer - typically holds several chunks of audio
            ring_buffer_frames = 8192  # Typical ring buffer size
            
            # 2. Audio device buffer - OS/hardware buffering  
            device_buffer_ms = 50  # Typical audio device latency
            device_buffer_frames = int((device_buffer_ms / 1000.0) * sample_rate)
            
            # 3. Processing delays (tempo, effects, etc.)
            processing_delay_frames = 1024  # Typical processing chunk size
            
            # Total estimated latency
            total_latency = ring_buffer_frames + device_buffer_frames + processing_delay_frames
            
            logger.debug(f"🔄 Estimated audio latency: {total_latency} frames "
                        f"({total_latency/sample_rate*1000:.1f}ms) - "
                        f"ring:{ring_buffer_frames}, device:{device_buffer_frames}, processing:{processing_delay_frames}")
            
            return total_latency
            
        except Exception as e:
            logger.error(f"🔄 Error estimating audio latency: {e}")
            # Conservative default - 2 seconds of latency
            return 88200  # ~2 seconds at 44.1kHz
    
    def _get_track_length_in_beats(self) -> float:
        """Get track length in beats."""
        try:
            if hasattr(self._deck, 'total_frames') and self._deck.total_frames > 0:
                end_frame = self._deck.total_frames - 1
                return self._beat_converter.get_beat_from_frame(end_frame)
            else:
                logger.warning("🔄 No track length available")
                return 1000.0  # Safe default
        except Exception as e:
            logger.error(f"🔄 Error getting track length in beats: {e}")
            return 1000.0
    
    def _seek_to_beat_boundary(self, target_beat: float) -> bool:
        """
        Seek to a specific beat boundary with frame accuracy.
        
        Args:
            target_beat: Beat position to seek to
            
        Returns:
            True if seek was successful
        """
        try:
            target_frame = self._beat_converter.get_frame_for_beat(target_beat)
            return self._seek_to_frame_seamless(target_frame)
        except Exception as e:
            logger.error(f"🔄 Error seeking to beat {target_beat}: {e}")
            return False
    
    def _seek_to_frame_seamless(self, target_frame: int) -> bool:
        """
        Perform frame-accurate seek without audio artifacts using event system.
        
        This integrates with the deck's event system to ensure seamless seeking 
        that works correctly with the producer thread.
        
        Args:
            target_frame: Frame position to seek to
            
        Returns:
            True if seek was successful
        """
        try:
            # Validate target frame
            track_length = getattr(self._deck, 'total_frames', 0)
            if track_length > 0:
                target_frame = max(0, min(target_frame, track_length - 1))
            
            # FIXED: Use seamless in-stream loop jump that coordinates audio pipeline
            # without stopping and restarting the audio stream
            if hasattr(self._deck, '_perform_seamless_loop_jump_in_stream'):
                # Use deck's seamless in-stream jump that coordinates RubberBand without restart
                old_frame = getattr(self._deck, 'audio_thread_current_frame', 0)
                success = self._deck._perform_seamless_loop_jump_in_stream(target_frame)
                
                if success:
                    logger.info(f"🔄 SEAMLESS LOOP JUMP: {old_frame} → {target_frame} (in-stream coordination)")
                    return True
                else:
                    logger.error(f"🔄 Seamless in-stream jump failed, falling back to command queue")
            
            # Fallback: command queue approach (causes restart but works)
            if hasattr(self._deck, 'seek_for_loop_jump'):
                old_frame = getattr(self._deck, 'audio_thread_current_frame', 0)
                success = self._deck.seek_for_loop_jump(target_frame)
                
                if success:
                    logger.warning(f"🔄 COMMAND QUEUE LOOP JUMP: {old_frame} → {target_frame} (causes restart)")
                    return True
            
            # Final fallback: direct frame manipulation 
            if hasattr(self._deck, 'audio_thread_current_frame'):
                old_frame = getattr(self._deck, 'audio_thread_current_frame', 0)
                self._deck.audio_thread_current_frame = target_frame
                self._deck._current_playback_frame_for_display = target_frame
                
                logger.warning(f"🔄 DIRECT FRAME JUMP: {old_frame} → {target_frame} (minimal coordination)")
                return True
            
            # Final fallback: use command queue for thread-safe seeking
            elif hasattr(self._deck, 'command_queue'):
                from ..deck import DECK_CMD_SEEK
                seek_data = {
                    'frame': target_frame,
                    'loop_context': True,  # Indicate this is a loop-related seek
                    'require_seamless': True  # Request seamless handling
                }
                self._deck.command_queue.put((DECK_CMD_SEEK, seek_data))
                logger.debug(f"🔄 Fallback queued seek to frame {target_frame}")
                return True
            
            logger.error(f"🔄 No available seek method for frame {target_frame}")
            return False
            
        except Exception as e:
            logger.error(f"🔄 Error in seamless seek to frame {target_frame}: {e}")
            return False
    
    def _prepare_ring_buffer_for_seek(self, target_frame: int) -> None:
        """
        Prepare ring buffer for seamless seek operation.
        
        This method coordinates with the ring buffer to minimize audio artifacts
        during seek operations by managing buffer state transitions.
        
        Args:
            target_frame: Target frame position for seek
        """
        try:
            # Check if deck has ring buffer system
            if hasattr(self._deck, 'out_ring') and self._deck.out_ring:
                # Get current buffer fill level
                available_data = self._deck.out_ring.available_read() if hasattr(self._deck.out_ring, 'available_read') else 0
                
                # If buffer is very full, we might want to reduce it slightly to allow for seek data
                if available_data > 8192:  # More than 8K frames buffered
                    logger.debug(f"🔄 Ring buffer has {available_data} frames before seek - may cause slight delay")
                
                # Mark that a seek is imminent (if deck supports this flag)
                if hasattr(self._deck, 'seek_in_progress_flag'):
                    self._deck.seek_in_progress_flag = True
            
            # Coordinate with producer thread if possible
            if hasattr(self._deck, '_producer_running') and self._deck._producer_running:
                logger.debug(f"🔄 Producer thread active during seek - coordinating transition")
                
        except Exception as e:
            logger.debug(f"🔄 Error preparing ring buffer for seek: {e}")
    
    def _finalize_ring_buffer_after_seek(self, target_frame: int) -> None:
        """
        Finalize ring buffer coordination after seek operation.
        
        Args:
            target_frame: Target frame position that was seeked to
        """
        try:
            # Update deck position tracking
            if hasattr(self._deck, 'audio_thread_current_frame'):
                # The deck's seek should have updated this, but ensure consistency
                expected_frame = target_frame
                actual_frame = getattr(self._deck, 'audio_thread_current_frame', target_frame)
                
                if abs(actual_frame - expected_frame) > 1000:  # More than ~22ms difference
                    logger.warning(f"🔄 Seek position mismatch: expected {expected_frame}, actual {actual_frame}")
            
            # Clear seek in progress flag
            if hasattr(self._deck, 'seek_in_progress_flag'):
                self._deck.seek_in_progress_flag = False
                
        except Exception as e:
            logger.debug(f"🔄 Error finalizing ring buffer after seek: {e}")
    
    def _coordinate_queued_seek(self, target_frame: int) -> None:
        """
        Coordinate ring buffer for queued seek operations.
        
        Args:
            target_frame: Target frame for the queued seek
        """
        try:
            # For queued seeks, we have less direct control, but can still prepare
            # Mark the seek as pending in any tracking systems
            logger.debug(f"🔄 Coordinating queued seek to frame {target_frame}")
            
            # If there's a way to signal the audio thread about the pending seek, do it here
            # This is deck-implementation specific
            
        except Exception as e:
            logger.debug(f"🔄 Error coordinating queued seek: {e}")
    
    def get_active_loops(self) -> Dict[str, LoopState]:
        """Get all currently active loops."""
        return self._registry.get_active_loops()
    
    def get_loop_state(self, action_id: str) -> Optional[LoopState]:
        """Get current state of a specific loop."""
        return self._registry.get_loop(action_id)
    
    def cancel_loop(self, action_id: str) -> bool:
        """
        Cancel a loop (remove it from registry).
        
        Args:
            action_id: ID of loop to cancel
            
        Returns:
            True if loop was found and cancelled
        """
        try:
            removed_state = self._registry.remove_loop(action_id)
            if removed_state:
                logger.info(f"🔄 Loop cancelled: {action_id}")
                return True
            else:
                logger.warning(f"🔄 Loop not found for cancellation: {action_id}")
                return False
        except Exception as e:
            logger.error(f"🔄 Error cancelling loop {action_id}: {e}")
            return False
    
    def clear_all_loops(self) -> int:
        """
        Clear all loops from the controller.
        
        Returns:
            Number of loops that were cleared
        """
        try:
            count = len(self._registry)
            self._registry.clear()
            logger.info(f"🔄 Cleared {count} loops from controller")
            return count
        except Exception as e:
            logger.error(f"🔄 Error clearing loops: {e}")
            return 0
    
    def subscribe_to_events(self, callback) -> None:
        """
        Subscribe to loop events.
        
        Args:
            callback: Function to call when loop events occur
                     Should accept (event_type, event_data) parameters
        """
        self._event_publisher.subscribe(callback)
    
    def enable_processing(self, enabled: bool = True) -> None:
        """
        Enable or disable loop processing.
        
        Args:
            enabled: Whether to enable loop processing
        """
        self._processing_enabled = enabled
        status = "enabled" if enabled else "disabled"
        logger.info(f"🔄 Loop processing {status} for deck {self._deck_id}")
    
    def get_stats(self) -> Dict:
        """Get loop controller statistics."""
        try:
            current_stats = self._stats.copy()
            current_stats.update({
                'total_loops': len(self._registry),
                'active_loops': len(self._registry.get_active_loops()),
                'completed_loops': len(self._registry.get_completed_loops()),
                'processing_enabled': self._processing_enabled,
                'deck_id': self._deck_id
            })
            return current_stats
        except Exception as e:
            logger.error(f"🔄 Error getting stats: {e}")
            return {'error': str(e), 'deck_id': self._deck_id}
    
    def _check_and_correct_timing_drift(self, action_id: str, loop_state: LoopState, current_frame: int) -> None:
        """
        Check for timing drift and perform correction if needed.
        
        This method monitors loop timing accuracy against the BeatManager and
        corrects drift to maintain musical synchronization.
        
        Args:
            action_id: Loop action ID
            loop_state: Current loop state
            current_frame: Current frame position
        """
        try:
            # Only check active loops
            if not loop_state.is_active:
                return
            
            # Periodic beat sync check (not every frame for performance)
            current_time = time.time()
            if current_time - self._last_beat_check < self._beat_sync_interval:
                return
            
            self._last_beat_check = current_time
            
            # Get expected frame position from beat manager
            try:
                current_beat = self._beat_converter.get_current_beat()
                expected_frame = self._beat_converter.get_frame_for_beat(current_beat)
            except Exception as e:
                logger.debug(f"🔄 Could not get beat timing for drift check: {e}")
                return
            
            # Calculate drift
            drift_frames = abs(current_frame - expected_frame)
            
            # Update timing history
            if action_id not in self._timing_history:
                self._timing_history[action_id] = {
                    'drift_samples': [],
                    'last_correction': 0,
                    'correction_count': 0
                }
            
            timing_data = self._timing_history[action_id]
            timing_data['drift_samples'].append({
                'timestamp': current_time,
                'drift_frames': drift_frames,
                'current_frame': current_frame,
                'expected_frame': expected_frame,
                'beat': current_beat
            })
            
            # Keep only recent samples (last 10 seconds)
            cutoff_time = current_time - 10.0
            timing_data['drift_samples'] = [
                sample for sample in timing_data['drift_samples'] 
                if sample['timestamp'] > cutoff_time
            ]
            
            # Check if drift exceeds threshold
            if drift_frames > self._drift_threshold_frames:
                # Only correct if we haven't corrected recently (avoid oscillation)
                time_since_correction = current_time - timing_data['last_correction']
                if time_since_correction > 2.0:  # Wait at least 2 seconds between corrections
                    
                    logger.warning(f"🔄 Timing drift detected for loop {action_id}: "
                                 f"{drift_frames} frames ({drift_frames / 44.1:.1f}ms)")
                    
                    # Perform gentle correction by updating our frame tracking
                    self._perform_timing_correction(action_id, loop_state, expected_frame)
                    
                    timing_data['last_correction'] = current_time
                    timing_data['correction_count'] += 1
                    self._stats['timing_drift_events'] += 1
                    
        except Exception as e:
            logger.debug(f"🔄 Error in timing drift check: {e}")
    
    def _perform_timing_correction(self, action_id: str, loop_state: LoopState, expected_frame: int) -> None:
        """
        Perform gentle timing correction to realign with musical timing.
        
        Args:
            action_id: Loop action ID
            loop_state: Current loop state
            expected_frame: Frame position we should be at according to beat timing
        """
        try:
            # Gentle correction: don't seek aggressively, just update our tracking
            # This allows the natural audio flow to gradually realign
            
            # Check if the expected frame is within our loop boundaries
            if (loop_state.start_frame <= expected_frame <= loop_state.end_frame):
                # Update loop position to match expected timing
                corrected_state = loop_state.with_frame_position(expected_frame)
                self._registry.update_loop(action_id, corrected_state)
                
                logger.debug(f"🔄 Applied gentle timing correction for loop {action_id}")
            else:
                # Expected frame is outside loop - more complex correction needed
                logger.debug(f"🔄 Timing drift outside loop boundaries - monitoring")
                
        except Exception as e:
            logger.debug(f"🔄 Error performing timing correction: {e}")
    
    def get_timing_drift_report(self) -> Dict[str, Any]:
        """
        Get comprehensive timing drift analysis report.
        
        Returns:
            Dictionary with timing analysis for all loops
        """
        try:
            current_time = time.time()
            report = {
                'timestamp': current_time,
                'deck_id': self._deck_id,
                'total_drift_events': self._stats['timing_drift_events'],
                'active_loops': len(self._registry.get_active_loops()),
                'loops': {}
            }
            
            for action_id, timing_data in self._timing_history.items():
                loop_report = {
                    'correction_count': timing_data['correction_count'],
                    'last_correction': timing_data.get('last_correction', 0),
                    'recent_samples': len(timing_data['drift_samples']),
                    'average_drift': 0,
                    'max_drift': 0
                }
                
                # Calculate drift statistics from recent samples
                if timing_data['drift_samples']:
                    recent_drifts = [s['drift_frames'] for s in timing_data['drift_samples']]
                    loop_report['average_drift'] = sum(recent_drifts) / len(recent_drifts)
                    loop_report['max_drift'] = max(recent_drifts)
                
                report['loops'][action_id] = loop_report
            
            return report
            
        except Exception as e:
            logger.error(f"🔄 Error generating timing drift report: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    def force_beat_sync_check(self) -> bool:
        """
        Force immediate beat synchronization check for all active loops.
        
        Returns:
            True if sync check completed successfully
        """
        try:
            logger.debug(f"🔄 Forcing beat sync check for deck {self._deck_id}")
            
            # Reset timing check interval to force immediate check
            self._last_beat_check = 0
            
            # Process all active loops for timing check
            current_frame = self._get_current_frame()
            active_loops = self._registry.get_active_loops()
            
            for action_id, loop_state in active_loops.items():
                self._check_and_correct_timing_drift(action_id, loop_state, current_frame)
            
            return True
            
        except Exception as e:
            logger.error(f"🔄 Error in forced beat sync check: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up controller resources."""
        try:
            logger.info(f"🔄 Cleaning up LoopController for deck {self._deck_id}")
            
            # Clear all loops
            self.clear_all_loops()
            
            # Clean up timing history
            self._timing_history.clear()
            
            # Clean up event publisher
            self._event_publisher.cleanup()
            
            # Stop completion system
            if self._completion_system:
                self._completion_system.stop()
            
            # Mark as not initialized
            self._is_initialized = False
            
            logger.info(f"🔄 LoopController cleanup complete for deck {self._deck_id}")
            
        except Exception as e:
            logger.error(f"🔄 Error during LoopController cleanup: {e}")
    
    def initialize_completion_system(self, deck_manager) -> None:
        """
        Initialize the completion system with deck manager reference.
        
        This method should be called after the deck manager is available
        to enable completion action execution.
        
        Args:
            deck_manager: Reference to the deck manager for completion actions
        """
        try:
            if self._completion_system is not None:
                logger.warning(f"🔄 Completion system already initialized for deck {self._deck_id}")
                return
            
            # Initialize completion system
            self._completion_system = LoopCompletionSystem(
                deck_manager=deck_manager,
                event_publisher=self._event_publisher
            )
            
            # Start the completion processing thread
            self._completion_system.start()
            
            logger.info(f"🔄 Completion system initialized for deck {self._deck_id}")
            
        except Exception as e:
            logger.error(f"🔄 Error initializing completion system: {e}")
    
    def register_completion_configuration(self, configuration: CompletionConfiguration) -> bool:
        """
        Register completion configuration for a loop.
        
        Args:
            configuration: CompletionConfiguration defining what should happen when loop completes
            
        Returns:
            True if configuration was registered successfully
        """
        try:
            if not self._completion_system:
                logger.error("🔄 Completion system not initialized - cannot register configuration")
                return False
            
            self._completion_system.register_completion_configuration(configuration)
            
            logger.debug(f"🔄 Completion configuration registered for loop {configuration.loop_id}")
            return True
            
        except Exception as e:
            logger.error(f"🔄 Error registering completion configuration: {e}")
            return False
    
    def unregister_completion_configuration(self, loop_id: str) -> bool:
        """
        Remove completion configuration for a loop.
        
        Args:
            loop_id: ID of the loop to remove configuration for
            
        Returns:
            True if configuration was removed successfully
        """
        try:
            if not self._completion_system:
                logger.warning("🔄 Completion system not initialized")
                return False
            
            self._completion_system.unregister_completion_configuration(loop_id)
            
            logger.debug(f"🔄 Completion configuration removed for loop {loop_id}")
            return True
            
        except Exception as e:
            logger.error(f"🔄 Error removing completion configuration: {e}")
            return False
    
    def get_completion_configuration(self, loop_id: str) -> Optional[CompletionConfiguration]:
        """
        Get completion configuration for a specific loop.
        
        Args:
            loop_id: ID of the loop
            
        Returns:
            CompletionConfiguration if found, None otherwise
        """
        try:
            if not self._completion_system:
                return None
            
            return self._completion_system.get_completion_configuration(loop_id)
            
        except Exception as e:
            logger.error(f"🔄 Error getting completion configuration: {e}")
            return None
    
    def register_custom_completion_handler(self, name: str, handler) -> bool:
        """
        Register a custom completion action handler.
        
        Args:
            name: Name identifier for the handler
            handler: Callable that implements the custom action
            
        Returns:
            True if handler was registered successfully
        """
        try:
            if not self._completion_system:
                logger.error("🔄 Completion system not initialized - cannot register custom handler")
                return False
            
            self._completion_system.register_custom_action_handler(name, handler)
            
            logger.debug(f"🔄 Custom completion handler '{name}' registered")
            return True
            
        except Exception as e:
            logger.error(f"🔄 Error registering custom completion handler: {e}")
            return False
    
    def get_completion_system_stats(self) -> Dict[str, Any]:
        """
        Get completion system statistics.
        
        Returns:
            Dictionary with completion system statistics
        """
        try:
            if not self._completion_system:
                return {'error': 'completion_system_not_initialized'}
            
            return self._completion_system.get_stats()
            
        except Exception as e:
            logger.error(f"🔄 Error getting completion system stats: {e}")
            return {'error': str(e)}
    
    def __repr__(self) -> str:
        """String representation of the loop controller."""
        try:
            active_count = len(self._registry.get_active_loops())
            total_count = len(self._registry)
            return f"LoopController(deck={self._deck_id}, active={active_count}, total={total_count})"
        except:
            return f"LoopController(deck={self._deck_id})"