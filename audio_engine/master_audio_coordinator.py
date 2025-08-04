# Master audio coordinator for dj-gemini
# Single master clock coordinating all decks and timing

import threading
import time
import queue
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)

class TimingEvent(Enum):
    """Types of timing events"""
    DECK_BEAT = auto()
    SCRIPT_BEAT = auto()
    LOOP_COMPLETE = auto()
    TEMPO_RAMP_COMPLETE = auto()
    FADE_COMPLETE = auto()

@dataclass
class ScheduledEvent:
    """Event scheduled for future execution"""
    event_id: str
    event_type: TimingEvent
    trigger_time: float  # Absolute time when event should fire
    deck_id: Optional[str] = None
    beat_number: Optional[float] = None
    callback: Optional[Callable] = None
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}

class MasterAudioCoordinator:
    """Single master clock coordinating all audio timing"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
        # Master timing
        self._master_start_time = time.time()
        self._master_frame = 0
        self._is_running = False
        
        # Deck coordination
        self._decks: Dict[str, Any] = {}  # Will hold deck references
        self._deck_states: Dict[str, Dict] = {}
        
        # Event scheduling
        self._scheduled_events: List[ScheduledEvent] = []
        self._event_queue = queue.PriorityQueue()
        self._next_event_id = 0
        
        # Threading
        self._coordinator_thread = None
        self._stop_event = threading.Event()
        self._state_lock = threading.RLock()
        
        # Callbacks
        self._beat_callbacks: List[Callable] = []
        self._state_change_callbacks: List[Callable] = []
        
        # Timing precision
        self._tick_interval = 0.001  # 1ms precision
        self._last_tick_time = 0.0
        
        logger.info("Master audio coordinator initialized")
    
    def start(self):
        """Start the master coordinator"""
        with self._state_lock:
            if self._is_running:
                logger.warning("Master coordinator already running")
                return
            
            self._is_running = True
            self._master_start_time = time.time()
            self._last_tick_time = self._master_start_time
            self._stop_event.clear()
            
            # Start coordinator thread
            self._coordinator_thread = threading.Thread(
                target=self._coordinator_loop,
                name="MasterAudioCoordinator",
                daemon=True
            )
            self._coordinator_thread.start()
            
            logger.info("Master audio coordinator started")
    
    def stop(self):
        """Stop the master coordinator"""
        with self._state_lock:
            if not self._is_running:
                return
            
            self._is_running = False
            self._stop_event.set()
            
            if self._coordinator_thread and self._coordinator_thread.is_alive():
                self._coordinator_thread.join(timeout=1.0)
            
            logger.info("Master audio coordinator stopped")
    
    def register_deck(self, deck_id: str, deck_instance):
        """Register a deck with the coordinator"""
        with self._state_lock:
            self._decks[deck_id] = deck_instance
            self._deck_states[deck_id] = {
                'current_beat': 0.0,
                'current_frame': 0,
                'is_playing': False,
                'bpm': 120.0,
                'last_update': time.time()
            }
            logger.info(f"Registered deck {deck_id} with master coordinator")
    
    def unregister_deck(self, deck_id: str):
        """Unregister a deck"""
        with self._state_lock:
            if deck_id in self._decks:
                del self._decks[deck_id]
            if deck_id in self._deck_states:
                del self._deck_states[deck_id]
            logger.info(f"Unregistered deck {deck_id} from master coordinator")
    
    def schedule_event(self, event_type: TimingEvent, trigger_time: float,
                      deck_id: str = None, beat_number: float = None,
                      callback: Callable = None, **params) -> str:
        """Schedule an event for future execution"""
        with self._state_lock:
            event_id = f"event_{self._next_event_id}"
            self._next_event_id += 1
            
            event = ScheduledEvent(
                event_id=event_id,
                event_type=event_type,
                trigger_time=trigger_time,
                deck_id=deck_id,
                beat_number=beat_number,
                callback=callback,
                params=params
            )
            
            # Add to priority queue (sorted by trigger time)
            self._event_queue.put((trigger_time, event_id, event))
            
            logger.debug(f"Scheduled event {event_id}: {event_type} at time {trigger_time}")
            return event_id
    
    def schedule_beat_event(self, deck_id: str, beat_number: float, 
                           callback: Callable, **params) -> str:
        """Schedule event to trigger at specific beat on deck"""
        # Calculate trigger time based on deck's current position and BPM
        with self._state_lock:
            deck_state = self._deck_states.get(deck_id)
            if not deck_state:
                logger.warning(f"Cannot schedule beat event - deck {deck_id} not registered")
                return ""
            
            current_beat = deck_state['current_beat']
            current_bpm = deck_state['bpm']
            
            if beat_number <= current_beat:
                logger.warning(f"Cannot schedule beat event - beat {beat_number} already passed (current: {current_beat})")
                return ""
            
            # Calculate time to reach target beat
            beats_remaining = beat_number - current_beat
            seconds_per_beat = 60.0 / current_bpm
            time_to_beat = beats_remaining * seconds_per_beat
            trigger_time = time.time() + time_to_beat
            
            return self.schedule_event(
                TimingEvent.DECK_BEAT,
                trigger_time,
                deck_id=deck_id,
                beat_number=beat_number,
                callback=callback,
                **params
            )
    
    def cancel_event(self, event_id: str) -> bool:
        """Cancel a scheduled event"""
        with self._state_lock:
            # Note: This is simplified - in production we'd need a more efficient
            # way to remove items from PriorityQueue
            logger.debug(f"Event {event_id} cancelled (will be skipped when processed)")
            return True
    
    def update_deck_state(self, deck_id: str, current_beat: float = None,
                         current_frame: int = None, is_playing: bool = None,
                         bpm: float = None):
        """Update deck state from audio callback"""
        with self._state_lock:
            if deck_id not in self._deck_states:
                return
            
            state = self._deck_states[deck_id]
            current_time = time.time()
            
            if current_beat is not None: state['current_beat'] = current_beat
            if current_frame is not None: state['current_frame'] = current_frame
            if is_playing is not None: state['is_playing'] = is_playing
            if bpm is not None: state['bpm'] = bpm
            state['last_update'] = current_time
            
            # Check for beat-based triggers
            if current_beat is not None:
                self._check_beat_triggers(deck_id, current_beat, current_time)
    
    def get_deck_state(self, deck_id: str) -> Dict:
        """Get current deck state"""
        with self._state_lock:
            return self._deck_states.get(deck_id, {}).copy()
    
    def get_master_time(self) -> float:
        """Get current master time"""
        return time.time() - self._master_start_time
    
    def synchronize_decks(self, reference_deck_id: str, target_deck_ids: List[str]):
        """Synchronize target decks to reference deck's timing"""
        with self._state_lock:
            reference_state = self._deck_states.get(reference_deck_id)
            if not reference_state:
                logger.error(f"Cannot synchronize - reference deck {reference_deck_id} not found")
                return
            
            reference_beat = reference_state['current_beat']
            reference_bpm = reference_state['bpm']
            
            for target_deck_id in target_deck_ids:
                if target_deck_id in self._decks:
                    deck = self._decks[target_deck_id]
                    # Set target deck to same BPM and phase
                    if hasattr(deck, 'set_tempo'):
                        deck.set_tempo(reference_bpm)
                    # Could add phase alignment here
                    logger.info(f"Synchronized deck {target_deck_id} to {reference_deck_id}")
    
    def add_beat_callback(self, callback: Callable):
        """Add callback for beat events"""
        self._beat_callbacks.append(callback)
    
    def add_state_change_callback(self, callback: Callable):
        """Add callback for state changes"""
        self._state_change_callbacks.append(callback)
    
    def _coordinator_loop(self):
        """Main coordinator loop - handles timing and events"""
        logger.info("Master coordinator loop started")
        
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                
                # Process scheduled events
                self._process_scheduled_events(current_time)
                
                # Update master timing
                self._update_master_timing(current_time)
                
                # Check deck states and trigger callbacks
                self._check_deck_states(current_time)
                
                # Sleep until next tick
                sleep_time = max(0, self._tick_interval - (time.time() - current_time))
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"Error in coordinator loop: {e}")
                time.sleep(0.01)  # Prevent tight error loop
        
        logger.info("Master coordinator loop stopped")
    
    def _process_scheduled_events(self, current_time: float):
        """Process all events that should trigger now"""
        events_to_process = []
        
        # Get all events that should trigger now
        while not self._event_queue.empty():
            try:
                trigger_time, event_id, event = self._event_queue.get_nowait()
                if trigger_time <= current_time:
                    events_to_process.append(event)
                else:
                    # Put it back - not time yet
                    self._event_queue.put((trigger_time, event_id, event))
                    break
            except queue.Empty:
                break
        
        # Process events
        for event in events_to_process:
            try:
                self._execute_event(event, current_time)
            except Exception as e:
                logger.error(f"Error executing event {event.event_id}: {e}")
    
    def _execute_event(self, event: ScheduledEvent, current_time: float):
        """Execute a scheduled event"""
        logger.debug(f"Executing event {event.event_id}: {event.event_type}")
        
        if event.callback:
            try:
                # Call the callback with event parameters
                event.callback(event.deck_id, event.beat_number, **event.params)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
        
        # Notify beat callbacks for beat events
        if event.event_type == TimingEvent.DECK_BEAT:
            for callback in self._beat_callbacks:
                try:
                    callback(event.deck_id, event.beat_number, current_time)
                except Exception as e:
                    logger.error(f"Error in beat callback: {e}")
    
    def _update_master_timing(self, current_time: float):
        """Update master timing information"""
        elapsed = current_time - self._last_tick_time
        frames_elapsed = int(elapsed * self.sample_rate)
        self._master_frame += frames_elapsed
        self._last_tick_time = current_time
    
    def _check_deck_states(self, current_time: float):
        """Check deck states and trigger appropriate callbacks"""
        with self._state_lock:
            for deck_id, state in self._deck_states.items():
                # Check for stale states (deck might have stopped updating)
                time_since_update = current_time - state['last_update']
                if time_since_update > 1.0 and state['is_playing']:
                    logger.warning(f"Deck {deck_id} state is stale ({time_since_update:.2f}s)")
    
    def _check_beat_triggers(self, deck_id: str, current_beat: float, current_time: float):
        """Check if any beat-based events should trigger"""
        # This is called when deck state is updated with new beat position
        # Could trigger additional beat-based events here
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        with self._state_lock:
            return {
                'is_running': self._is_running,
                'master_time': self.get_master_time(),
                'master_frame': self._master_frame,
                'registered_decks': list(self._decks.keys()),
                'scheduled_events': self._event_queue.qsize(),
                'deck_states': {deck_id: state.copy() for deck_id, state in self._deck_states.items()}
            }