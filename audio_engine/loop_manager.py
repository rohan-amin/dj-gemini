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
    def __init__(self, start_frame: int, end_frame: int, repetitions: int, action_id: str):
        self.start_frame = start_frame
        self.end_frame = end_frame
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
    
    This class replaces all the scattered loop detection logic in the deck
    with a clean, maintainable state machine that handles:
    - Loop activation and deactivation
    - State transitions
    - Boundary detection
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
        
        # Flag to track when a loop has completed for engine notification
        self._loop_completed_for_engine = False
        
    def set_clear_ring_buffer_callback(self, callback):
        """Set callback for clearing ring buffer during position jumps"""
        self._clear_ring_buffer_callback = callback
        
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
        if not self.deck.total_frames or self.deck.bpm <= 0:
            logger.error(f"Deck {self.deck.deck_id} - Cannot activate loop: invalid track or BPM")
            return False
            
        # Calculate frame positions
        start_frame = self.deck.get_frame_from_beat(start_beat)
        frames_per_beat = (self.deck.sample_rate * 60) / self.deck.bpm
        loop_length_frames = int(length_beats * frames_per_beat)
        end_frame = start_frame + loop_length_frames
        
        # Validate frame positions
        if start_frame >= self.deck.total_frames:
            logger.error(f"Deck {self.deck.deck_id} - Loop start frame {start_frame} beyond track length {self.deck.total_frames}")
            return False
            
        if end_frame > self.deck.total_frames:
            logger.warning(f"Deck {self.deck.deck_id} - Loop end frame {end_frame} beyond track length, adjusting to {self.deck.total_frames}")
            end_frame = self.deck.total_frames
            
        # Create new loop
        new_loop = Loop(start_frame, end_frame, repetitions, action_id)
        
        # Clear any existing loops
        self._clear_current_loop()
        
        # Set as current loop
        self.current_loop = new_loop
        
        # Determine initial state based on current playback position
        current_frame = self.deck.audio_thread_current_frame
        
        if current_frame >= start_frame:
            # We're already at or past the start frame, activate immediately
            self.state = LoopState.ACTIVE
            logger.info(f"Deck {self.deck.deck_id} - Loop activated immediately: {action_id} (frames {start_frame}-{end_frame})")
        else:
            # We're before the start frame, wait for activation
            self.state = LoopState.PENDING
            logger.info(f"Deck {self.deck.deck_id} - Loop queued for activation: {action_id} (will activate at frame {start_frame})")
            
        # Clear ring buffer to prevent artifacts
        if self._clear_ring_buffer_callback:
            self._clear_ring_buffer_callback("loop activation")
            
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
            # Check if we've reached the loop end frame
            if current_frame >= self.current_loop.end_frame:
                # Always increment repetition count first
                self.current_loop.increment_repetition()
                
                if not self.current_loop.is_complete:
                    # Continue looping - jump back to start
                    result["action"] = "jump_to_start"
                    result["jump_frame"] = self.current_loop.start_frame
                    result["repetitions"] = f"{self.current_loop.repetitions_done}/{self.current_loop.repetitions_total}"
                    
                    logger.debug(f"Deck {self.deck.deck_id} - Loop repetition {self.current_loop.repetitions_done}/{self.current_loop.repetitions_total}")
                    
                    # Don't clear ring buffer here - let the deck handle it only when needed
                    # The deck will clear it when processing the jump_to_start action
                        
                else:
                    # Loop complete - transition to completing state
                    self.state = LoopState.COMPLETING
                    result["action"] = "completed"
                    result["state"] = self.state.value
                    logger.info(f"Deck {self.deck.deck_id} - Loop completed: {self.current_loop.action_id}")
                    
                    # Mark this loop as completed for engine notification
                    self._loop_completed_for_engine = True
                    
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
                # Loop is truly complete, clean up
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
            
        return {
            "action_id": self.current_loop.action_id,
            "start_frame": self.current_loop.start_frame,
            "end_frame": self.current_loop.end_frame,
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
        
    def get_loop_completion_status(self) -> Optional[Dict[str, Any]]:
        """
        Get loop completion status for engine notification.
        This replaces the scattered _loop_just_completed flags.
        """
        if self._loop_completed_for_engine and self.current_loop:
            # Reset the flag and return completion info
            self._loop_completed_for_engine = False
            return {
                "action_id": self.current_loop.action_id,
                "completed_at": time.time(),
                "total_repetitions": self.current_loop.repetitions_total
            }
        return None
