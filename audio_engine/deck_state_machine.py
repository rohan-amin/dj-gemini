# Centralized deck state machine for dj-gemini
# Manages all deck state transitions and coordination

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Callable, Any
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class DeckState(Enum):
    """All possible deck states"""
    UNLOADED = auto()       # No track loaded
    LOADING = auto()        # Loading track/processing
    LOADED = auto()         # Track loaded, ready to play
    PLAYING = auto()        # Currently playing
    PAUSED = auto()         # Paused (can resume)
    STOPPED = auto()        # Stopped (position reset)
    SEEKING = auto()        # Seeking to new position
    LOOPING = auto()        # Active loop mode
    PROCESSING = auto()     # Heavy processing (tempo/pitch change)
    ERROR = auto()          # Error state

class DeckEvent(Enum):
    """Events that can trigger state transitions"""
    LOAD_TRACK = auto()
    LOAD_COMPLETE = auto()
    LOAD_FAILED = auto()
    PLAY = auto()
    PAUSE = auto()
    STOP = auto()
    SEEK = auto()
    SEEK_COMPLETE = auto()
    LOOP_START = auto()
    LOOP_END = auto()
    PROCESS_START = auto()
    PROCESS_COMPLETE = auto()
    PROCESS_FAILED = auto()
    ERROR = auto()
    RESET = auto()

@dataclass
class DeckStateData:
    """Complete deck state information"""
    # Basic state
    current_state: DeckState = DeckState.UNLOADED
    previous_state: DeckState = DeckState.UNLOADED
    
    # Track information
    filepath: Optional[str] = None
    total_frames: int = 0
    sample_rate: int = 44100
    original_bpm: float = 120.0
    
    # Playback state
    current_frame: int = 0
    current_beat: float = 0.0
    is_playing: bool = False
    volume: float = 1.0
    
    # Tempo/BPM state
    current_bpm: float = 120.0
    tempo_ratio: float = 1.0
    
    # EQ state
    eq_low: float = 1.0
    eq_mid: float = 1.0
    eq_high: float = 1.0
    
    # Loop state
    loop_active: bool = False
    loop_start_frame: int = 0
    loop_end_frame: int = 0
    loop_repetitions: int = 0
    loop_repetitions_remaining: int = 0
    
    # Effect state
    effects_active: List[str] = field(default_factory=list)
    fade_active: bool = False
    
    # Processing state
    processing_operation: Optional[str] = None
    processing_progress: float = 0.0
    
    # Timestamps
    state_entered_time: float = field(default_factory=time.time)
    last_updated_time: float = field(default_factory=time.time)
    
    # Error information
    last_error: Optional[str] = None
    error_count: int = 0

class DeckStateMachine:
    """Centralized state machine for deck management"""
    
    # Valid state transitions
    VALID_TRANSITIONS = {
        DeckState.UNLOADED: [DeckState.LOADING, DeckState.ERROR],
        DeckState.LOADING: [DeckState.LOADED, DeckState.ERROR, DeckState.UNLOADED],
        DeckState.LOADED: [DeckState.PLAYING, DeckState.PROCESSING, DeckState.UNLOADED, DeckState.ERROR],
        DeckState.PLAYING: [DeckState.PAUSED, DeckState.STOPPED, DeckState.SEEKING, 
                           DeckState.LOOPING, DeckState.PROCESSING, DeckState.ERROR],
        DeckState.PAUSED: [DeckState.PLAYING, DeckState.STOPPED, DeckState.SEEKING, DeckState.ERROR],
        DeckState.STOPPED: [DeckState.PLAYING, DeckState.LOADED, DeckState.SEEKING, DeckState.ERROR],
        DeckState.SEEKING: [DeckState.PLAYING, DeckState.PAUSED, DeckState.STOPPED, DeckState.ERROR],
        DeckState.LOOPING: [DeckState.PLAYING, DeckState.PAUSED, DeckState.STOPPED, DeckState.ERROR],
        DeckState.PROCESSING: [DeckState.LOADED, DeckState.PLAYING, DeckState.ERROR],
        DeckState.ERROR: [DeckState.UNLOADED, DeckState.LOADED]  # Recovery states
    }
    
    def __init__(self, deck_id: str):
        self.deck_id = deck_id
        self.state_data = DeckStateData()
        self._state_lock = threading.RLock()
        self._observers: List[Callable[[DeckState, DeckState, DeckStateData], None]] = []
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"Deck{deck_id}State")
        
        logger.info(f"Deck {deck_id} - State machine initialized in {self.state_data.current_state}")
    
    def add_observer(self, callback: Callable[[DeckState, DeckState, DeckStateData], None]):
        """Add observer for state changes"""
        self._observers.append(callback)
    
    def get_state(self) -> DeckStateData:
        """Get current state (thread-safe)"""
        with self._state_lock:
            # Return a copy to prevent external modification
            return DeckStateData(
                current_state=self.state_data.current_state,
                previous_state=self.state_data.previous_state,
                filepath=self.state_data.filepath,
                total_frames=self.state_data.total_frames,
                sample_rate=self.state_data.sample_rate,
                original_bpm=self.state_data.original_bpm,
                current_frame=self.state_data.current_frame,
                current_beat=self.state_data.current_beat,
                is_playing=self.state_data.is_playing,
                volume=self.state_data.volume,
                current_bpm=self.state_data.current_bpm,
                tempo_ratio=self.state_data.tempo_ratio,
                eq_low=self.state_data.eq_low,
                eq_mid=self.state_data.eq_mid,
                eq_high=self.state_data.eq_high,
                loop_active=self.state_data.loop_active,
                loop_start_frame=self.state_data.loop_start_frame,
                loop_end_frame=self.state_data.loop_end_frame,
                loop_repetitions=self.state_data.loop_repetitions,
                loop_repetitions_remaining=self.state_data.loop_repetitions_remaining,
                effects_active=self.state_data.effects_active.copy(),
                fade_active=self.state_data.fade_active,
                processing_operation=self.state_data.processing_operation,
                processing_progress=self.state_data.processing_progress,
                state_entered_time=self.state_data.state_entered_time,
                last_updated_time=self.state_data.last_updated_time,
                last_error=self.state_data.last_error,
                error_count=self.state_data.error_count
            )
    
    def handle_event(self, event: DeckEvent, **kwargs) -> bool:
        """Handle state machine event"""
        with self._state_lock:
            current_state = self.state_data.current_state
            new_state = self._get_next_state(current_state, event)
            
            if new_state is None:
                logger.warning(f"Deck {self.deck_id} - Invalid transition: {event} from {current_state}")
                return False
            
            # Execute transition
            self._transition_to_state(new_state, event, **kwargs)
            return True
    
    def _get_next_state(self, current_state: DeckState, event: DeckEvent) -> Optional[DeckState]:
        """Determine next state based on current state and event"""
        
        # State transition logic
        if event == DeckEvent.LOAD_TRACK:
            if current_state in [DeckState.UNLOADED, DeckState.LOADED, DeckState.STOPPED]:
                return DeckState.LOADING
                
        elif event == DeckEvent.LOAD_COMPLETE:
            if current_state == DeckState.LOADING:
                return DeckState.LOADED
                
        elif event == DeckEvent.LOAD_FAILED:
            if current_state == DeckState.LOADING:
                return DeckState.ERROR
                
        elif event == DeckEvent.PLAY:
            if current_state in [DeckState.LOADED, DeckState.PAUSED, DeckState.STOPPED]:
                return DeckState.PLAYING
                
        elif event == DeckEvent.PAUSE:
            if current_state == DeckState.PLAYING:
                return DeckState.PAUSED
                
        elif event == DeckEvent.STOP:
            if current_state in [DeckState.PLAYING, DeckState.PAUSED, DeckState.LOOPING]:
                return DeckState.STOPPED
                
        elif event == DeckEvent.SEEK:
            if current_state in [DeckState.PLAYING, DeckState.PAUSED, DeckState.STOPPED]:
                return DeckState.SEEKING
                
        elif event == DeckEvent.SEEK_COMPLETE:
            if current_state == DeckState.SEEKING:
                # Return to previous state
                if self.state_data.previous_state in [DeckState.PLAYING, DeckState.PAUSED]:
                    return self.state_data.previous_state
                return DeckState.STOPPED
                
        elif event == DeckEvent.LOOP_START:
            if current_state == DeckState.PLAYING:
                return DeckState.LOOPING
                
        elif event == DeckEvent.LOOP_END:
            if current_state == DeckState.LOOPING:
                return DeckState.PLAYING
                
        elif event == DeckEvent.PROCESS_START:
            if current_state in [DeckState.LOADED, DeckState.PLAYING, DeckState.PAUSED]:
                return DeckState.PROCESSING
                
        elif event == DeckEvent.PROCESS_COMPLETE:
            if current_state == DeckState.PROCESSING:
                # Return to appropriate state based on previous
                if self.state_data.previous_state == DeckState.PLAYING:
                    return DeckState.PLAYING
                return DeckState.LOADED
                
        elif event == DeckEvent.PROCESS_FAILED:
            if current_state == DeckState.PROCESSING:
                return DeckState.ERROR
                
        elif event == DeckEvent.ERROR:
            return DeckState.ERROR
            
        elif event == DeckEvent.RESET:
            if current_state == DeckState.ERROR:
                return DeckState.UNLOADED
        
        # Check if transition is valid
        if new_state and new_state in self.VALID_TRANSITIONS.get(current_state, []):
            return new_state
            
        return None
    
    def _transition_to_state(self, new_state: DeckState, event: DeckEvent, **kwargs):
        """Execute state transition"""
        old_state = self.state_data.current_state
        
        # Update state
        self.state_data.previous_state = old_state
        self.state_data.current_state = new_state
        self.state_data.state_entered_time = time.time()
        self.state_data.last_updated_time = time.time()
        
        # Handle state-specific actions
        self._handle_state_entry(new_state, event, **kwargs)
        
        # Notify observers
        self._notify_observers(old_state, new_state)
        
        logger.info(f"Deck {self.deck_id} - State transition: {old_state} -> {new_state} (event: {event})")
    
    def _handle_state_entry(self, state: DeckState, event: DeckEvent, **kwargs):
        """Handle actions when entering a new state"""
        
        if state == DeckState.LOADING:
            filepath = kwargs.get('filepath')
            self.state_data.filepath = filepath
            self.state_data.processing_operation = "loading"
            self.state_data.processing_progress = 0.0
            
        elif state == DeckState.LOADED:
            self.state_data.processing_operation = None
            self.state_data.processing_progress = 1.0
            # Update track info from kwargs
            self.state_data.total_frames = kwargs.get('total_frames', 0)
            self.state_data.original_bpm = kwargs.get('bpm', 120.0)
            self.state_data.current_bpm = self.state_data.original_bpm
            
        elif state == DeckState.PLAYING:
            self.state_data.is_playing = True
            start_frame = kwargs.get('start_frame', self.state_data.current_frame)
            self.state_data.current_frame = start_frame
            
        elif state == DeckState.PAUSED:
            self.state_data.is_playing = False
            
        elif state == DeckState.STOPPED:
            self.state_data.is_playing = False
            self.state_data.current_frame = 0
            self.state_data.current_beat = 0.0
            self.state_data.loop_active = False
            
        elif state == DeckState.SEEKING:
            target_frame = kwargs.get('frame', 0)
            self.state_data.current_frame = target_frame
            
        elif state == DeckState.LOOPING:
            self.state_data.loop_active = True
            self.state_data.loop_start_frame = kwargs.get('start_frame', 0)
            self.state_data.loop_end_frame = kwargs.get('end_frame', 0)
            self.state_data.loop_repetitions = kwargs.get('repetitions', 1)
            self.state_data.loop_repetitions_remaining = self.state_data.loop_repetitions
            
        elif state == DeckState.PROCESSING:
            operation = kwargs.get('operation', 'unknown')
            self.state_data.processing_operation = operation
            self.state_data.processing_progress = 0.0
            
        elif state == DeckState.ERROR:
            error_message = kwargs.get('error', 'Unknown error')
            self.state_data.last_error = error_message
            self.state_data.error_count += 1
            self.state_data.processing_operation = None
    
    def _notify_observers(self, old_state: DeckState, new_state: DeckState):
        """Notify all observers of state change"""
        for observer in self._observers:
            try:
                observer(old_state, new_state, self.state_data)
            except Exception as e:
                logger.error(f"Deck {self.deck_id} - Observer error: {e}")
    
    def update_playback_state(self, frame: int, beat: float, is_playing: bool):
        """Update playback state from audio callback"""
        with self._state_lock:
            self.state_data.current_frame = frame
            self.state_data.current_beat = beat
            self.state_data.is_playing = is_playing
            self.state_data.last_updated_time = time.time()
    
    def update_audio_state(self, volume: float = None, eq_low: float = None, 
                          eq_mid: float = None, eq_high: float = None):
        """Update audio state"""
        with self._state_lock:
            if volume is not None: self.state_data.volume = volume
            if eq_low is not None: self.state_data.eq_low = eq_low
            if eq_mid is not None: self.state_data.eq_mid = eq_mid
            if eq_high is not None: self.state_data.eq_high = eq_high
            self.state_data.last_updated_time = time.time()
    
    def can_transition_to(self, target_state: DeckState) -> bool:
        """Check if transition to target state is valid"""
        with self._state_lock:
            current_state = self.state_data.current_state
            return target_state in self.VALID_TRANSITIONS.get(current_state, [])
    
    def is_in_state(self, *states: DeckState) -> bool:
        """Check if currently in one of the specified states"""
        with self._state_lock:
            return self.state_data.current_state in states
    
    def get_time_in_current_state(self) -> float:
        """Get time spent in current state (seconds)"""
        with self._state_lock:
            return time.time() - self.state_data.state_entered_time
    
    def shutdown(self):
        """Clean shutdown of state machine"""
        logger.info(f"Deck {self.deck_id} - Shutting down state machine")
        self._executor.shutdown(wait=True)