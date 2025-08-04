# Business logic layer for dj-gemini
# Separates audio engine from high-level script execution and coordination

from typing import Dict, List, Optional, Any, Callable
import threading
import logging
from dataclasses import dataclass

from .lockfree_audio import LockFreeAudioEngine, AudioCommand
from .deck_state_machine import DeckStateMachine, DeckState, DeckEvent
from .master_audio_coordinator import MasterAudioCoordinator, TimingEvent
from .effect_coordinator import EffectCoordinator, EffectType
from .script_state_machine import ScriptStateMachine, ScriptState, ScriptEvent

logger = logging.getLogger(__name__)

@dataclass
class BusinessLayerConfig:
    """Configuration for business layer"""
    sample_rate: int = 44100
    max_decks: int = 4
    enable_effects: bool = True
    enable_state_persistence: bool = True
    enable_auto_sync: bool = True

class AudioBusinessLayer:
    """
    High-level business logic layer that coordinates:
    - Script execution
    - Deck management  
    - Audio processing
    - Effect coordination
    - State management
    """
    
    def __init__(self, config: BusinessLayerConfig = None):
        self.config = config or BusinessLayerConfig()
        
        # Core components
        self._master_coordinator = MasterAudioCoordinator(self.config.sample_rate)
        self._effect_coordinator = EffectCoordinator() if self.config.enable_effects else None
        self._script_state_machine = ScriptStateMachine()
        
        # Deck management
        self._decks: Dict[str, LockFreeAudioEngine] = {}
        self._deck_state_machines: Dict[str, DeckStateMachine] = {}
        self._deck_lock = threading.RLock()
        
        # Business logic state
        self._is_initialized = False
        self._is_running = False
        
        # Callbacks and event handlers
        self._setup_event_handlers()
        
        logger.info("Audio business layer initialized")
    
    def initialize(self) -> bool:
        """Initialize all components"""
        try:
            # Start core components
            self._master_coordinator.start()
            
            if self._effect_coordinator:
                self._effect_coordinator.start()
            
            # Setup script callbacks
            self._script_state_machine.add_state_change_callback(self._on_script_state_change)
            self._script_state_machine.add_action_callback(self._on_script_action)
            self._script_state_machine.add_error_callback(self._on_script_error)
            
            self._is_initialized = True
            self._is_running = True
            
            logger.info("Audio business layer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize audio business layer: {e}")
            return False
    
    def shutdown(self):
        """Shutdown all components"""
        logger.info("Shutting down audio business layer")
        
        self._is_running = False
        
        # Stop script execution
        if self._script_state_machine.state.current_state == ScriptState.RUNNING:
            self._script_state_machine.stop_script()
        
        # Stop all decks
        with self._deck_lock:
            for deck in self._decks.values():
                deck.stop()
        
        # Shutdown components
        if self._effect_coordinator:
            self._effect_coordinator.stop()
        
        self._master_coordinator.stop()
        
        # Shutdown deck state machines
        for state_machine in self._deck_state_machines.values():
            state_machine.shutdown()
        
        logger.info("Audio business layer shutdown complete")
    
    def create_deck(self, deck_id: str) -> bool:
        """Create a new deck"""
        with self._deck_lock:
            if deck_id in self._decks:
                logger.warning(f"Deck {deck_id} already exists")
                return False
            
            if len(self._decks) >= self.config.max_decks:
                logger.error(f"Maximum number of decks ({self.config.max_decks}) reached")
                return False
            
            try:
                # Create audio engine for deck
                audio_engine = LockFreeAudioEngine(deck_id, self.config.sample_rate)
                
                # Create state machine for deck
                state_machine = DeckStateMachine(deck_id)
                
                # Register with master coordinator
                self._master_coordinator.register_deck(deck_id, audio_engine)
                
                # Setup state machine callbacks
                state_machine.add_observer(self._on_deck_state_change)
                
                # Store references
                self._decks[deck_id] = audio_engine
                self._deck_state_machines[deck_id] = state_machine
                
                logger.info(f"Created deck: {deck_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to create deck {deck_id}: {e}")
                return False
    
    def remove_deck(self, deck_id: str) -> bool:
        """Remove a deck"""
        with self._deck_lock:
            if deck_id not in self._decks:
                logger.warning(f"Deck {deck_id} does not exist")
                return False
            
            try:
                # Stop deck
                self._decks[deck_id].stop()
                
                # Unregister from coordinator
                self._master_coordinator.unregister_deck(deck_id)
                
                # Shutdown state machine
                self._deck_state_machines[deck_id].shutdown()
                
                # Remove references
                del self._decks[deck_id]
                del self._deck_state_machines[deck_id]
                
                logger.info(f"Removed deck: {deck_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to remove deck {deck_id}: {e}")
                return False
    
    def load_script(self, script_path: str, script_data: Dict[str, Any]) -> bool:
        """Load a mix script"""
        try:
            # Load script into state machine
            if not self._script_state_machine.load_script(script_path, script_data):
                return False
            
            # Pre-create decks mentioned in script
            self._create_script_decks(script_data)
            
            logger.info(f"Loaded script: {script_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load script {script_path}: {e}")
            return False
    
    def start_script(self) -> bool:
        """Start script execution"""
        if not self._script_state_machine.start_script():
            return False
        
        # Process initial script_start actions
        self._process_ready_actions()
        
        return True
    
    def pause_script(self) -> bool:
        """Pause script execution"""
        return self._script_state_machine.pause_script()
    
    def resume_script(self) -> bool:
        """Resume script execution"""
        if not self._script_state_machine.resume_script():
            return False
        
        # Resume processing actions
        self._process_ready_actions()
        
        return True
    
    def stop_script(self) -> bool:
        """Stop script execution"""
        # Stop all decks
        with self._deck_lock:
            for deck in self._decks.values():
                deck.stop()
        
        return self._script_state_machine.stop_script()
    
    def load_track(self, deck_id: str, filepath: str, audio_data: Any, 
                   beat_timestamps: Any, bpm: float) -> bool:
        """Load track onto deck (high-level interface)"""
        with self._deck_lock:
            if deck_id not in self._decks:
                logger.error(f"Deck {deck_id} does not exist")
                return False
            
            try:
                # Update deck state machine
                state_machine = self._deck_state_machines[deck_id]
                state_machine.handle_event(DeckEvent.LOAD_TRACK, filepath=filepath)
                
                # Load audio data
                audio_engine = self._decks[deck_id]
                audio_engine.load_audio_data(audio_data, beat_timestamps, bpm)
                
                # Update state machine with successful load
                state_machine.handle_event(DeckEvent.LOAD_COMPLETE, 
                                         total_frames=len(audio_data), bpm=bpm)
                
                logger.info(f"Loaded track on deck {deck_id}: {filepath}")
                return True
                
            except Exception as e:
                # Update state machine with failure
                self._deck_state_machines[deck_id].handle_event(DeckEvent.LOAD_FAILED, error=str(e))
                logger.error(f"Failed to load track on deck {deck_id}: {e}")
                return False
    
    def play_deck(self, deck_id: str, start_frame: int = 0) -> bool:
        """Play deck (high-level interface)"""
        with self._deck_lock:
            if deck_id not in self._decks:
                logger.error(f"Deck {deck_id} does not exist")
                return False
            
            try:
                # Update state machine
                state_machine = self._deck_state_machines[deck_id]
                state_machine.handle_event(DeckEvent.PLAY, start_frame=start_frame)
                
                # Send command to audio engine
                audio_engine = self._decks[deck_id]
                audio_engine.play(start_frame)
                
                logger.debug(f"Started playback on deck {deck_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to play deck {deck_id}: {e}")
                return False
    
    def set_deck_volume(self, deck_id: str, volume: float, fade_duration: float = 0.0) -> bool:
        """Set deck volume with optional fade"""
        with self._deck_lock:
            if deck_id not in self._decks:
                logger.error(f"Deck {deck_id} does not exist")
                return False
            
            try:
                if fade_duration > 0.0 and self._effect_coordinator:
                    # Use effect coordinator for fade
                    current_volume = self._decks[deck_id].get_current_state().volume
                    self._effect_coordinator.add_effect(
                        EffectType.VOLUME_FADE,
                        deck_id,
                        fade_duration,
                        start_values={'volume': current_volume},
                        target_values={'volume': volume},
                        on_update=lambda eid, did, values: self._decks[deck_id].set_volume(values['volume'])
                    )
                else:
                    # Immediate volume change
                    self._decks[deck_id].set_volume(volume)
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to set volume on deck {deck_id}: {e}")
                return False
    
    def set_deck_eq(self, deck_id: str, low: float = None, mid: float = None, 
                    high: float = None, fade_duration: float = 0.0) -> bool:
        """Set deck EQ with optional fade"""
        with self._deck_lock:
            if deck_id not in self._decks:
                logger.error(f"Deck {deck_id} does not exist")
                return False
            
            try:
                if fade_duration > 0.0 and self._effect_coordinator:
                    # Use effect coordinator for fade
                    current_state = self._decks[deck_id].get_current_state()
                    start_values = {
                        'eq_low': current_state.eq_low,
                        'eq_mid': current_state.eq_mid,
                        'eq_high': current_state.eq_high
                    }
                    target_values = {}
                    if low is not None: target_values['eq_low'] = low
                    if mid is not None: target_values['eq_mid'] = mid
                    if high is not None: target_values['eq_high'] = high
                    
                    self._effect_coordinator.add_effect(
                        EffectType.EQ_FADE,
                        deck_id,
                        fade_duration,
                        start_values=start_values,
                        target_values=target_values,
                        on_update=lambda eid, did, values: self._decks[deck_id].set_eq(
                            values.get('eq_low'), values.get('eq_mid'), values.get('eq_high')
                        )
                    )
                else:
                    # Immediate EQ change
                    self._decks[deck_id].set_eq(low, mid, high)
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to set EQ on deck {deck_id}: {e}")
                return False
    
    def get_deck_state(self, deck_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive deck state"""
        with self._deck_lock:
            if deck_id not in self._decks:
                return None
            
            try:
                # Combine audio state and state machine state
                audio_state = self._decks[deck_id].get_current_state()
                state_machine_state = self._deck_state_machines[deck_id].get_state()
                
                return {
                    'deck_id': deck_id,
                    'audio_state': {
                        'current_frame': audio_state.current_frame,
                        'current_beat': audio_state.current_beat,
                        'is_playing': audio_state.is_playing,
                        'volume': audio_state.volume,
                        'eq_low': audio_state.eq_low,
                        'eq_mid': audio_state.eq_mid,
                        'eq_high': audio_state.eq_high,
                        'current_bpm': audio_state.current_bpm,
                        'tempo_ratio': audio_state.tempo_ratio
                    },
                    'deck_state': state_machine_state.current_state.name,
                    'track_info': {
                        'filepath': state_machine_state.filepath,
                        'total_frames': state_machine_state.total_frames,
                        'original_bpm': state_machine_state.original_bpm
                    }
                }
                
            except Exception as e:
                logger.error(f"Failed to get state for deck {deck_id}: {e}")
                return None
    
    def get_script_progress(self) -> Dict[str, Any]:
        """Get script execution progress"""
        return self._script_state_machine.get_script_progress()
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        with self._deck_lock:
            stats = {
                'business_layer': {
                    'is_running': self._is_running,
                    'deck_count': len(self._decks),
                    'max_decks': self.config.max_decks
                },
                'master_coordinator': self._master_coordinator.get_statistics(),
                'script_state': self._script_state_machine.get_script_progress(),
                'decks': {}
            }
            
            # Add deck states
            for deck_id in self._decks:
                stats['decks'][deck_id] = self.get_deck_state(deck_id)
            
            # Add effect coordinator stats
            if self._effect_coordinator:
                stats['effects'] = self._effect_coordinator.get_statistics()
            
            return stats
    
    def _setup_event_handlers(self):
        """Setup event handlers between components"""
        # Master coordinator beat callback
        self._master_coordinator.add_beat_callback(self._on_beat_event)
        
        # Master coordinator state change callback
        self._master_coordinator.add_state_change_callback(self._on_coordinator_state_change)
    
    def _create_script_decks(self, script_data: Dict[str, Any]):
        """Create decks mentioned in script"""
        actions = script_data.get("actions", [])
        deck_ids = set()
        
        # Extract deck IDs from actions
        for action in actions:
            deck_id = action.get("deck_id")
            if deck_id:
                deck_ids.add(deck_id)
        
        # Create decks
        for deck_id in deck_ids:
            if deck_id not in self._decks:
                self.create_deck(deck_id)
    
    def _process_ready_actions(self):
        """Process actions that are ready for execution"""
        ready_actions = self._script_state_machine.get_ready_actions()
        
        for action in ready_actions:
            self._execute_script_action(action)
    
    def _execute_script_action(self, action) -> bool:
        """Execute a script action"""
        try:
            command = action.command
            deck_id = action.deck_id
            parameters = action.parameters
            
            # Route command to appropriate handler
            if command == "load_track":
                # This would integrate with your existing loading logic
                pass
            elif command == "play":
                start_frame = parameters.get("start_at_beat", 0)  # Convert beat to frame
                return self.play_deck(deck_id, start_frame)
            elif command == "set_volume":
                volume = parameters.get("volume", 1.0)
                return self.set_deck_volume(deck_id, volume)
            elif command == "set_eq":
                return self.set_deck_eq(
                    deck_id,
                    parameters.get("low"),
                    parameters.get("mid"), 
                    parameters.get("high")
                )
            elif command == "fade_eq":
                duration = parameters.get("duration_seconds", 1.0)
                return self.set_deck_eq(
                    deck_id,
                    parameters.get("target_low"),
                    parameters.get("target_mid"),
                    parameters.get("target_high"),
                    fade_duration=duration
                )
            # Add more command handlers as needed
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute action {action.action_id}: {e}")
            return False
    
    def _on_script_state_change(self, old_state, new_state, script_state):
        """Handle script state changes"""
        logger.info(f"Script state change: {old_state.name} -> {new_state.name}")
        
        if new_state == ScriptState.RUNNING:
            self._process_ready_actions()
    
    def _on_script_action(self, action_id, action_state):
        """Handle script action state changes"""
        logger.debug(f"Action {action_id} state: {action_state}")
    
    def _on_script_error(self, action_id, error_message):
        """Handle script errors"""
        logger.error(f"Script error in action {action_id}: {error_message}")
    
    def _on_deck_state_change(self, old_state, new_state, deck_state):
        """Handle deck state changes"""
        logger.debug(f"Deck {deck_state.current_state} state change: {old_state.name} -> {new_state.name}")
    
    def _on_beat_event(self, deck_id, beat_number, current_time):
        """Handle beat events from master coordinator"""
        logger.debug(f"Beat event: deck {deck_id}, beat {beat_number}")
        
        # This could trigger script actions that are waiting for specific beats
        # Update script state machine with beat trigger
    
    def _on_coordinator_state_change(self, state_change_info):
        """Handle master coordinator state changes"""
        logger.debug(f"Coordinator state change: {state_change_info}")

# Convenience function for creating and configuring the business layer
def create_audio_system(config: BusinessLayerConfig = None) -> AudioBusinessLayer:
    """Create and initialize complete audio system"""
    business_layer = AudioBusinessLayer(config)
    
    if business_layer.initialize():
        return business_layer
    else:
        raise RuntimeError("Failed to initialize audio business layer")