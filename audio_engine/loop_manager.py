# dj-gemini/audio_engine/loop_manager.py
# Centralized Loop Management State Machine

import logging
import time
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

class LoopState(Enum):
    """Loop state enumeration for clean state management"""
    INACTIVE = "inactive"
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETING = "completing"

class Loop:
    """Represents a single loop with all its parameters"""
    def __init__(self, start_frame: int, end_frame: int, repetitions: int, action_id: str, start_beat: float = None, end_beat: float = None):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.start_beat = start_beat  # Store original beat numbers for tempo changes
        self.end_beat = end_beat
        self.repetitions_total = repetitions
        self.repetitions_done = 0
        self.action_id = action_id
        self.start_time = time.time()
        self.last_beat_time = None
        
    @property
    def length_frames(self) -> int:
        """Get loop length in frames"""
        return self.end_frame - self.start_frame
    
    @property
    def is_complete(self) -> bool:
        """Check if loop has completed all repetitions"""
        return self.repetitions_done >= self.repetitions_total
    
    def increment_repetition(self):
        """Increment the repetition counter"""
        self.repetitions_done += 1
        
    def reset_repetition_count(self):
        """Reset repetition counter (for loop restart)"""
        self.repetitions_done = 0

class LoopManager:
    """
    Centralized loop management state machine.
    
    This class provides a clean, maintainable state machine that handles:
    - Loop activation and deactivation
    - State transitions
    - Event-driven loop completion
    - Repetition counting
    - Completion signaling
    """
    
    def __init__(self, deck):
        self.deck = deck
        self.current_loop: Optional[Loop] = None
        self.state = LoopState.INACTIVE
        self.pending_loops = []
        
        # Callback for deck to clear ring buffer during position jumps
        self._clear_ring_buffer_callback = None
        
        # Reference to engine for event-driven loop completion notification
        self._engine = None
        
    def set_clear_ring_buffer_callback(self, callback):
        """Set callback for clearing ring buffer during position jumps"""
        self._clear_ring_buffer_callback = callback
        
    def set_engine_reference(self, engine):
        """Set reference to engine for event-driven loop completion notification"""
        self._engine = engine
        
    def activate_loop(self, start_beat: int, length_beats: int, repetitions: int, action_id: str) -> bool:
        """
        Activate a new loop with the given parameters.
        
        Args:
            start_beat: Beat number where loop should start
            length_beats: Length of loop in beats
            repetitions: Number of times to repeat the loop
            action_id: Unique identifier for this loop action
            
        Returns:
            True if loop was successfully activated, False otherwise
        """
        if not self.deck.total_frames or self.deck.beat_manager.get_bpm() <= 0:
            logger.error(f"Deck {self.deck.deck_id} - Cannot activate loop: invalid track or BPM")
            return False
            
        # Calculate frame positions using BeatManager for consistent timing
        start_frame = self.deck.beat_manager.get_frame_for_beat(start_beat)
        
        # CRITICAL FIX: Use BeatManager to calculate end frame for accurate timing
        # Instead of theoretical frames_per_beat calculation, use the actual beat position
        end_beat = start_beat + length_beats
        end_frame = self.deck.beat_manager.get_frame_for_beat(end_beat)
        
        # Validate frame positions
        if start_frame >= self.deck.total_frames:
            logger.error(f"Deck {self.deck.deck_id} - Loop start frame {start_frame} beyond track length {self.deck.total_frames}")
            return False
            
        if end_frame > self.deck.total_frames:
            logger.warning(f"Deck {self.deck.deck_id} - Loop end frame {end_frame} beyond track length, adjusting to {self.deck.total_frames}")
            end_frame = self.deck.total_frames
            
        # Create new loop with beat information for tempo change support
        new_loop = Loop(start_frame, end_frame, repetitions, action_id, start_beat, start_beat + length_beats)
        
        # Clear any existing loops
        self._clear_current_loop()
        
        # Set as current loop
        self.current_loop = new_loop
        
        # Schedule loop completion events using beat-aligned timing
        self._schedule_loop_events(start_frame, end_frame, repetitions)
        
        # Determine initial state based on current playback position
        current_frame = self.deck.audio_thread_current_frame
        
        if current_frame >= start_frame:
            # CRITICAL FIX: Always jump to loop start when activating, even if we're past it
            # This prevents the "fast-forward" effect and ensures loops start cleanly
            logger.info(f"Deck {self.deck.deck_id} - Loop activation: current_frame={current_frame}, start_frame={start_frame}, jumping to start")
            self.state = LoopState.ACTIVE
            
            # Force immediate jump to loop start to prevent mid-cycle activation
            self._loop_position_jump_pending = True
            self._pending_jump_frame = start_frame
            logger.info(f"Deck {self.deck.deck_id} - Loop marked as active: {action_id} (frames {start_frame}-{end_frame}) - jumping to start immediately")
        else:
            # We're before the start frame, wait for activation
            self.state = LoopState.PENDING
            logger.info(f"Deck {self.deck.deck_id} - Loop queued for activation: {action_id} (will activate at frame {start_frame})")
            
        # Clear ring buffer to prevent artifacts
        if self._clear_ring_buffer_callback:
            self._clear_ring_buffer_callback("loop activation")
        
        # Initialize position jump tracking
        self._loop_position_jump_pending = False
            
        return True

    def activate_loop_direct(self, start_frame: int, end_frame: int, repetitions: int, action_id: str) -> bool:
        """
        Directly activate a loop using exact frame positions - for frame-accurate execution.
        
        This method bypasses beat calculations and uses pre-calculated frame positions
        for sample-accurate loop timing. Called from the audio thread context.
        
        Args:
            start_frame: Exact frame where loop should start
            end_frame: Exact frame where loop should end
            repetitions: Number of times to repeat the loop
            action_id: Unique identifier for this loop action
            
        Returns:
            True if loop was successfully activated, False otherwise
        """
        # Validate frame positions
        if not self.deck.total_frames:
            logger.error(f"Deck {self.deck.deck_id} - Cannot activate loop: no track loaded")
            return False
            
        if start_frame >= self.deck.total_frames or end_frame > self.deck.total_frames:
            logger.error(f"Deck {self.deck.deck_id} - Loop frames {start_frame}-{end_frame} beyond track length {self.deck.total_frames}")
            return False
            
        if start_frame >= end_frame:
            logger.error(f"Deck {self.deck.deck_id} - Invalid loop: start_frame {start_frame} >= end_frame {end_frame}")
            return False
        
        # Create new loop without beat calculations (frames are already exact)
        new_loop = Loop(start_frame, end_frame, repetitions, action_id)
        
        # Clear any existing loops
        self._clear_current_loop()
        
        # Set as current loop
        self.current_loop = new_loop
        
        # Schedule loop completion events using beat-aligned timing
        self._schedule_loop_events(start_frame, end_frame, repetitions)
        
        # Determine initial state - since this is called frame-accurately, always activate immediately
        current_frame = self.deck.audio_thread_current_frame
        
        # Set as active immediately since we're executing at the exact frame
        self.state = LoopState.ACTIVE
        
        # CRITICAL FIX: Always jump to loop start for clean loop activation
        # This ensures loops start at the correct musical position, not partway through
        frame_difference = current_frame - start_frame
        if abs(frame_difference) > 100:  # Allow small timing tolerance (2.3ms at 44.1kHz)
            logger.info(f"Deck {self.deck.deck_id} - Frame-accurate loop activation: jumping from {current_frame} to {start_frame} (diff: {frame_difference} frames)")
        else:
            logger.info(f"Deck {self.deck.deck_id} - Frame-accurate loop activation: jumping to exact start {start_frame} (current: {current_frame})")
        
        # Always jump to loop start to ensure clean musical positioning
        self._loop_position_jump_pending = True
        self._pending_jump_frame = start_frame
        
        # Clear ring buffer to prevent artifacts
        if self._clear_ring_buffer_callback:
            self._clear_ring_buffer_callback("frame-accurate loop activation")
        
        logger.info(f"Deck {self.deck.deck_id} - Frame-accurate loop activated: {action_id} (frames {start_frame}-{end_frame})")
        return True
        
    def deactivate_loop(self):
        """Deactivate the current loop"""
        if self.current_loop:
            logger.info(f"Deck {self.deck.deck_id} - Deactivating loop: {self.current_loop.action_id}")
            self._clear_current_loop()
            
    def check_boundaries(self, current_frame: int) -> Dict[str, Any]:
        """
        Check loop boundaries and handle state transitions.
        
        This is the main method called from the audio callback to handle
        all loop-related logic in one place.
        
        Args:
            current_frame: Current playback frame position
            
        Returns:
            Dict with loop state information and any actions needed
        """
        if not self.current_loop:
            return {"action": "none", "state": self.state.value}
            
        result = {
            "action": "none",
            "state": self.state.value,
            "loop_id": self.current_loop.action_id,
            "repetitions": f"{self.current_loop.repetitions_done}/{self.current_loop.repetitions_total}"
        }
        
        # Handle state transitions based on current position
        if self.state == LoopState.PENDING:
            # Check if we've reached the loop start frame
            if current_frame >= self.current_loop.start_frame:
                self.state = LoopState.ACTIVE
                result["action"] = "activated"
                result["state"] = self.state.value
                logger.info(f"Deck {self.deck.deck_id} - Loop now ACTIVE: {self.current_loop.action_id}")
                
        elif self.state == LoopState.ACTIVE:
            # Check if we need to jump back to loop start (when we reach the END)
            if current_frame >= self.current_loop.end_frame:
                # We've reached the loop end, jump back to loop start frame at next beat boundary
                # This prevents audio skipping by jumping at a natural break point
                if hasattr(self, '_loop_position_jump_pending') and self._loop_position_jump_pending:
                    # Position jump already handled
                    pass
                else:
                    # Mark position jump as pending for next beat boundary
                    self._loop_position_jump_pending = True
                    logger.info(f"Deck {self.deck.deck_id} - Loop end reached, position jump pending: will jump to {self.current_loop.start_frame} at next beat")
            
            # Execute any pending position jump at beat boundaries
            self.execute_pending_position_jump()
            
            # Only log current position for debugging
            frames_played = current_frame - self.current_loop.start_frame
            logger.debug(f"Deck {self.deck.deck_id} - Loop active: current_frame={current_frame}, frames_played={frames_played}")
            
            # All loop boundaries are handled by beat-aligned events
            # No manual boundary detection - events fire at exact beat positions
                    
        elif self.state == LoopState.COMPLETING:
            # In completing state, we can either restart or finish
            if not self.current_loop.is_complete:
                # Restart loop
                self.current_loop.reset_repetition_count()
                self.state = LoopState.ACTIVE
                result["action"] = "restarted"
                result["state"] = self.state.value
                logger.debug(f"Deck {self.deck.deck_id} - Loop restarted: {self.current_loop.action_id}")
            else:
                # Loop is truly complete, clean up and notify engine
                logger.info(f"Deck {self.deck.deck_id} - Loop truly finished: {self.current_loop.action_id}")
                self._notify_engine_loop_completed()
                self._clear_current_loop()
                result["action"] = "finished"
                result["state"] = self.state.value
                
        return result
        
    def has_active_loop(self) -> bool:
        """Check if there's an active loop"""
        return self.current_loop is not None and self.state in [LoopState.ACTIVE, LoopState.COMPLETING]
        
    def has_pending_loop(self) -> bool:
        """Check if there's a pending loop waiting to activate"""
        return self.current_loop is not None and self.state == LoopState.PENDING
        
    def get_current_loop_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current loop"""
        if not self.current_loop:
            return None
            
        # Use stored beat numbers if available (these don't change with tempo)
        if self.current_loop.start_beat is not None and self.current_loop.end_beat is not None:
            start_beat = self.current_loop.start_beat
            end_beat = self.current_loop.end_beat
        else:
            # Get beat information using BeatManager for consistency
            start_beat = self.deck.beat_manager.get_beat_from_frame(self.current_loop.start_frame)
            end_beat = self.deck.beat_manager.get_beat_from_frame(self.current_loop.end_frame)
            
        length_beats = end_beat - start_beat
            
        return {
            "action_id": self.current_loop.action_id,
            "start_frame": self.current_loop.start_frame,
            "end_frame": self.current_loop.end_frame,
            "start_beat": start_beat,
            "end_beat": end_beat,
            "length_beats": length_beats,
            "repetitions_total": self.current_loop.repetitions_total,
            "repetitions_done": self.current_loop.repetitions_done,
            "state": self.state.value,
            "length_frames": self.current_loop.length_frames
        }
        
    def _clear_current_loop(self):
        """Clear the current loop and reset state"""
        if self.current_loop:
            logger.debug(f"Deck {self.deck.deck_id} - Clearing loop: {self.current_loop.action_id}")
            
        self.current_loop = None
        self.state = LoopState.INACTIVE
        self.pending_loops.clear()
        
        # Clear position jump tracking
        if hasattr(self, '_loop_position_jump_pending'):
            self._loop_position_jump_pending = False
        
    def _schedule_frame_accurate_loop_events(self, start_frame: int, end_frame: int, repetitions: int):
        """
        Schedule frame-accurate loop completion events directly on the deck.
        This bypasses the event scheduler for sample-accurate timing.
        
        Args:
            start_frame: Start frame of the loop
            end_frame: End frame of the loop
            repetitions: Number of repetitions for this loop
        """
        if not self.current_loop:
            logger.warning(f"Deck {self.deck.deck_id} - Cannot schedule loop events: no current loop")
            return
            
        loop_length_frames = end_frame - start_frame
        
        logger.debug(f"Deck {self.deck.deck_id} - Scheduling frame-accurate loop events: {repetitions} reps, {loop_length_frames} frames each")
        
        # Schedule frame-accurate completion events for each repetition
        for rep in range(repetitions):
            # Calculate exact frame where this repetition should end
            rep_end_frame = start_frame + (loop_length_frames * (rep + 1))
            
            # Schedule frame-accurate loop completion
            completion_data = {
                'action_id': self.current_loop.action_id,
                'repetition': rep + 1,
                'total_repetitions': repetitions,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'rep_end_frame': rep_end_frame
            }
            
            self.deck.schedule_frame_action(rep_end_frame, 'loop_repetition_complete', completion_data)
            
            logger.debug(f"Deck {self.deck.deck_id} - Scheduled loop completion {rep + 1}/{repetitions} at frame {rep_end_frame}")
    
    def _schedule_loop_events(self, start_frame: int, end_frame: int, repetitions: int):
        """
        Schedule loop completion events using the event scheduler.
        Uses beat-aligned timing for precise loop management.
        
        Args:
            start_frame: Start frame of the loop
            end_frame: End frame of the loop
            repetitions: Number of repetitions for this loop
        """
        if not self._engine or not hasattr(self._engine, 'event_scheduler'):
            logger.warning(f"Deck {self.deck.deck_id} - Cannot schedule loop events: no engine reference")
            return
            
        try:
            # Get beat information for accurate timing
            start_beat = self.deck.beat_manager.get_beat_from_frame(start_frame)
            end_beat = self.deck.beat_manager.get_beat_from_frame(end_frame)
            loop_length_beats = end_beat - start_beat
            
            logger.debug(f"Deck {self.deck.deck_id} - Loop timing: start_beat={start_beat:.2f}, end_beat={end_beat:.2f}, length={loop_length_beats:.2f} beats")
            
            # Schedule events for each repetition based on BEAT position
            for rep in range(repetitions):
                # Calculate the exact beat where this repetition should end
                rep_end_beat = start_beat + (loop_length_beats * (rep + 1))
                
                # Schedule loop completion event using TRUE beat-aligned scheduling
                # Phase 3: Pass deck_id for deck-specific beat alignment
                self._engine.event_scheduler.schedule_beat_action({
                    "command": "loop_repetition_complete",
                    "deck_id": self.deck.deck_id,
                    "parameters": {
                        "action_id": self.current_loop.action_id,
                        "repetition": rep + 1,
                        "total_repetitions": repetitions,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "target_beat": rep_end_beat  # Key: exact beat to trigger on
                    }
                }, rep_end_beat, deck_id=self.deck.deck_id, priority=80)
                
                logger.debug(f"Deck {self.deck.deck_id} - Scheduled loop repetition {rep + 1}/{repetitions} completion at beat {rep_end_beat:.2f}")
                
        except Exception as e:
            logger.error(f"Deck {self.deck.deck_id} - Failed to schedule loop events: {e}")
            
    def execute_pending_position_jump(self):
        """Execute any pending position jump at beat boundaries"""
        if (hasattr(self, '_loop_position_jump_pending') and 
            self._loop_position_jump_pending and 
            self.current_loop and 
            self.state == LoopState.ACTIVE):
            
            # Execute the position jump
            logger.info(f"Deck {self.deck.deck_id} - Executing pending position jump to loop start: {self.current_loop.start_frame}")
            self.deck.audio_thread_current_frame = self.current_loop.start_frame
            self.deck._current_playback_frame_for_display = self.current_loop.start_frame
            
            # Clear the pending flag
            self._loop_position_jump_pending = False
            
            # Clear ring buffer to prevent artifacts
            if self._clear_ring_buffer_callback:
                self._clear_ring_buffer_callback("loop position jump")
            
            logger.info(f"Deck {self.deck.deck_id} - Position jump completed: now at frame {self.current_loop.start_frame}")
            
    def handle_loop_repetition_complete(self, repetition: int, total_repetitions: int):
        """
        Handle loop repetition completion events from the event scheduler.
        This is called when a loop_repetition_complete event fires.
        
        Args:
            repetition: The repetition number that just completed
            total_repetitions: Total number of repetitions for this loop
        """
        if not self.current_loop or self.state != LoopState.ACTIVE:
            logger.warning(f"Deck {self.deck.deck_id} - Received repetition completion event but no active loop")
            return
            
        logger.info(f"Deck {self.deck.deck_id} - Loop repetition {repetition}/{total_repetitions} completed: {self.current_loop.action_id}")
        
        # Update the repetition count
        self.current_loop.repetitions_done = repetition
        
        if repetition >= total_repetitions:
            # Final repetition completed - finish the loop
            logger.info(f"Deck {self.deck.deck_id} - Final loop repetition completed: {self.current_loop.action_id}")
            self.state = LoopState.COMPLETING
            self._notify_engine_loop_completed()
            self._clear_current_loop()
        else:
            # More repetitions to go - jump back to start
            logger.debug(f"Deck {self.deck.deck_id} - Continuing loop: repetition {repetition}/{total_repetitions}")
            # The deck will handle the jump_to_start action when processing the result
            
    def _notify_engine_loop_completed(self):
        """Notify engine that a loop has completed via event queue"""
        if self._engine and hasattr(self._engine, 'event_scheduler'):
            try:
                # Post loop completion event to engine's event queue
                self._engine.event_scheduler.schedule_immediate_action({
                    "command": "loop_completed",
                    "deck_id": self.deck.deck_id,
                    "parameters": {
                        "action_id": self.current_loop.action_id,
                        "completed_at": time.time(),
                        "total_repetitions": self.current_loop.repetitions_total
                    }
                }, priority=75)
                logger.debug(f"Deck {self.deck.deck_id} - Loop completion event posted to engine queue")
            except Exception as e:
                logger.warning(f"Deck {self.deck.deck_id} - Failed to post loop completion event: {e}")
        else:
            logger.warning(f"Deck {self.deck.deck_id} - Cannot notify engine: no engine reference")
            
    def get_current_loop_position_beats(self) -> Optional[float]:
        """
        Get the current position within the active loop in beats.
        Useful for synchronization and display purposes.
        
        Returns:
            Current position in beats within the loop, or None if no active loop
        """
        if not self.current_loop or self.state != LoopState.ACTIVE:
            return None
            
        current_frame = self.deck.audio_thread_current_frame
        start_frame = self.current_loop.start_frame
        
        if current_frame < start_frame:
            return 0.0
            
        # Calculate how many beats we've played in the loop
        frames_played_in_loop = current_frame - start_frame
        current_beat = self.deck.beat_manager.get_beat_from_frame(current_frame)
        start_beat = self.deck.beat_manager.get_beat_from_frame(start_frame)
        
        return current_beat - start_beat
        
    def is_loop_synchronized_with_beat_grid(self) -> bool:
        """
        Check if the current loop is properly synchronized with the beat grid.
        This ensures loops start and end on proper beat boundaries.
        
        Returns:
            True if loop is synchronized, False otherwise
        """
        if not self.current_loop:
            return False
            
        # Get beat positions using BeatManager
        start_beat = self.deck.beat_manager.get_beat_from_frame(self.current_loop.start_frame)
        end_beat = self.deck.beat_manager.get_beat_from_frame(self.current_loop.end_frame)
        
        # Check if both start and end are close to integer beat boundaries
        start_beat_rounded = round(start_beat)
        end_beat_rounded = round(end_beat)
        
        start_sync = abs(start_beat - start_beat_rounded) < 0.1  # Allow 0.1 beat tolerance
        end_sync = abs(end_beat - end_beat_rounded) < 0.1
        
        return start_sync and end_sync
        
    def refresh_loop_frame_positions(self):
        """
        Refresh loop frame positions after tempo changes.
        This ensures the loop boundaries are updated when the tempo changes.
        """
        if not self.current_loop:
            return
            
        # Use the stored beat numbers (these don't change with tempo)
        if self.current_loop.start_beat is not None and self.current_loop.end_beat is not None:
            start_beat = self.current_loop.start_beat
            end_beat = self.current_loop.end_beat
        else:
            # Derive from current frame positions
            start_beat = self.deck.beat_manager.get_beat_from_frame(self.current_loop.start_frame)
            end_beat = self.deck.beat_manager.get_beat_from_frame(self.current_loop.end_frame)
        
        # Recalculate frame positions using current BeatManager state
        new_start_frame = self.deck.beat_manager.get_frame_for_beat(start_beat)
        new_end_frame = self.deck.beat_manager.get_frame_for_beat(end_beat)
        
        # Update the loop with new frame positions
        self.current_loop.start_frame = new_start_frame
        self.current_loop.end_frame = new_end_frame
        
        logger.debug(f"Deck {self.deck.deck_id} - Loop frame positions refreshed: {start_beat}→{end_beat} beats = {new_start_frame}→{new_end_frame} frames")
