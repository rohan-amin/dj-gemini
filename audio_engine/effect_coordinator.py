# Effect coordination system for dj-gemini
# Manages all effects and their interactions

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import threading
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

class EffectType(Enum):
    """Types of effects"""
    VOLUME_FADE = auto()
    EQ_FADE = auto()
    TEMPO_RAMP = auto()
    LOOP = auto()
    CROSSFADE = auto()
    FILTER_SWEEP = auto()
    SCRATCH = auto()

class EffectState(Enum):
    """Effect state"""
    PENDING = auto()    # Queued but not started
    ACTIVE = auto()     # Currently running
    COMPLETED = auto()  # Finished
    CANCELLED = auto()  # Cancelled before completion

@dataclass
class EffectInstance:
    """Individual effect instance"""
    effect_id: str
    effect_type: EffectType
    deck_id: str
    state: EffectState = EffectState.PENDING
    
    # Timing
    start_time: Optional[float] = None
    duration: float = 1.0
    progress: float = 0.0
    
    # Parameters
    start_values: Dict[str, float] = field(default_factory=dict)
    target_values: Dict[str, float] = field(default_factory=dict)
    current_values: Dict[str, float] = field(default_factory=dict)
    
    # Control
    curve_type: str = "linear"  # linear, exponential, smooth
    priority: int = 0  # Higher priority effects take precedence
    
    # Callbacks
    on_complete: Optional[Callable] = None
    on_update: Optional[Callable] = None
    
    # Effect-specific data
    custom_data: Dict[str, Any] = field(default_factory=dict)

class EffectCoordinator:
    """Coordinates all effects across all decks"""
    
    def __init__(self):
        self._effects: Dict[str, EffectInstance] = {}
        self._deck_effects: Dict[str, List[str]] = {}  # deck_id -> effect_ids
        self._effect_priorities: Dict[EffectType, int] = {
            EffectType.LOOP: 100,           # Highest priority
            EffectType.TEMPO_RAMP: 90,
            EffectType.CROSSFADE: 80,
            EffectType.SCRATCH: 70,
            EffectType.EQ_FADE: 60,
            EffectType.VOLUME_FADE: 50,
            EffectType.FILTER_SWEEP: 40
        }
        
        self._next_effect_id = 0
        self._coordinator_lock = threading.RLock()
        
        # Update thread
        self._is_running = False
        self._update_thread = None
        self._stop_event = threading.Event()
        
        logger.info("Effect coordinator initialized")
    
    def start(self):
        """Start the effect coordinator"""
        with self._coordinator_lock:
            if self._is_running:
                return
            
            self._is_running = True
            self._stop_event.clear()
            
            self._update_thread = threading.Thread(
                target=self._update_loop,
                name="EffectCoordinator",
                daemon=True
            )
            self._update_thread.start()
            
            logger.info("Effect coordinator started")
    
    def stop(self):
        """Stop the effect coordinator"""
        with self._coordinator_lock:
            if not self._is_running:
                return
            
            self._is_running = False
            self._stop_event.set()
            
            if self._update_thread and self._update_thread.is_alive():
                self._update_thread.join(timeout=1.0)
            
            logger.info("Effect coordinator stopped")
    
    def add_effect(self, effect_type: EffectType, deck_id: str, duration: float,
                   start_values: Dict[str, float] = None,
                   target_values: Dict[str, float] = None,
                   curve_type: str = "linear",
                   priority: int = None,
                   on_complete: Callable = None,
                   on_update: Callable = None,
                   **custom_data) -> str:
        """Add new effect"""
        
        with self._coordinator_lock:
            effect_id = f"effect_{self._next_effect_id}"
            self._next_effect_id += 1
            
            # Set default priority based on effect type
            if priority is None:
                priority = self._effect_priorities.get(effect_type, 50)
            
            effect = EffectInstance(
                effect_id=effect_id,
                effect_type=effect_type,
                deck_id=deck_id,
                duration=duration,
                start_values=start_values or {},
                target_values=target_values or {},
                current_values=(start_values or {}).copy(),
                curve_type=curve_type,
                priority=priority,
                on_complete=on_complete,
                on_update=on_update,
                custom_data=custom_data
            )
            
            self._effects[effect_id] = effect
            
            # Add to deck's effect list
            if deck_id not in self._deck_effects:
                self._deck_effects[deck_id] = []
            self._deck_effects[deck_id].append(effect_id)
            
            # Check for conflicts and resolve
            self._resolve_effect_conflicts(deck_id, effect)
            
            logger.info(f"Added effect {effect_id}: {effect_type} on deck {deck_id} (duration: {duration}s)")
            return effect_id
    
    def cancel_effect(self, effect_id: str) -> bool:
        """Cancel an effect"""
        with self._coordinator_lock:
            if effect_id not in self._effects:
                return False
            
            effect = self._effects[effect_id]
            effect.state = EffectState.CANCELLED
            
            logger.info(f"Cancelled effect {effect_id}")
            return True
    
    def cancel_deck_effects(self, deck_id: str, effect_types: List[EffectType] = None) -> int:
        """Cancel effects on a deck"""
        with self._coordinator_lock:
            if deck_id not in self._deck_effects:
                return 0
            
            cancelled_count = 0
            for effect_id in self._deck_effects[deck_id][:]:  # Copy list to avoid modification during iteration
                effect = self._effects.get(effect_id)
                if effect and effect.state == EffectState.ACTIVE:
                    if effect_types is None or effect.effect_type in effect_types:
                        effect.state = EffectState.CANCELLED
                        cancelled_count += 1
            
            logger.info(f"Cancelled {cancelled_count} effects on deck {deck_id}")
            return cancelled_count
    
    def get_deck_effect_values(self, deck_id: str) -> Dict[str, float]:
        """Get current combined effect values for deck"""
        with self._coordinator_lock:
            if deck_id not in self._deck_effects:
                return {}
            
            # Combine values from all active effects
            combined_values = {}
            active_effects = []
            
            for effect_id in self._deck_effects[deck_id]:
                effect = self._effects.get(effect_id)
                if effect and effect.state == EffectState.ACTIVE:
                    active_effects.append(effect)
            
            # Sort by priority (highest first)
            active_effects.sort(key=lambda e: e.priority, reverse=True)
            
            # Apply effects in priority order
            for effect in active_effects:
                for param, value in effect.current_values.items():
                    if param not in combined_values:
                        combined_values[param] = value
                    else:
                        # Combine effects based on parameter type
                        combined_values[param] = self._combine_effect_values(
                            param, combined_values[param], value, effect.effect_type
                        )
            
            return combined_values
    
    def start_effect(self, effect_id: str) -> bool:
        """Start an effect immediately"""
        with self._coordinator_lock:
            if effect_id not in self._effects:
                return False
            
            effect = self._effects[effect_id]
            if effect.state != EffectState.PENDING:
                return False
            
            effect.state = EffectState.ACTIVE
            effect.start_time = time.time()
            effect.progress = 0.0
            
            logger.debug(f"Started effect {effect_id}")
            return True
    
    def get_effect_state(self, effect_id: str) -> Optional[EffectInstance]:
        """Get effect state"""
        with self._coordinator_lock:
            return self._effects.get(effect_id)
    
    def get_deck_effects(self, deck_id: str) -> List[EffectInstance]:
        """Get all effects for a deck"""
        with self._coordinator_lock:
            if deck_id not in self._deck_effects:
                return []
            
            return [self._effects[effect_id] for effect_id in self._deck_effects[deck_id]
                    if effect_id in self._effects]
    
    def _update_loop(self):
        """Main update loop for effects"""
        logger.info("Effect coordinator update loop started")
        
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                self._update_all_effects(current_time)
                time.sleep(0.01)  # 100Hz update rate
                
            except Exception as e:
                logger.error(f"Error in effect update loop: {e}")
                time.sleep(0.1)
        
        logger.info("Effect coordinator update loop stopped")
    
    def _update_all_effects(self, current_time: float):
        """Update all active effects"""
        with self._coordinator_lock:
            completed_effects = []
            
            for effect_id, effect in self._effects.items():
                if effect.state == EffectState.ACTIVE:
                    if self._update_effect(effect, current_time):
                        completed_effects.append(effect_id)
            
            # Clean up completed effects
            for effect_id in completed_effects:
                self._cleanup_effect(effect_id)
    
    def _update_effect(self, effect: EffectInstance, current_time: float) -> bool:
        """Update single effect, return True if completed"""
        if effect.start_time is None:
            return False
        
        elapsed = current_time - effect.start_time
        
        if elapsed >= effect.duration:
            # Effect completed
            effect.progress = 1.0
            effect.state = EffectState.COMPLETED
            
            # Set final values
            for param, target in effect.target_values.items():
                effect.current_values[param] = target
            
            # Call completion callback
            if effect.on_complete:
                try:
                    effect.on_complete(effect.effect_id, effect.deck_id)
                except Exception as e:
                    logger.error(f"Error in effect completion callback: {e}")
            
            return True
        
        # Update progress
        effect.progress = elapsed / effect.duration
        
        # Apply curve
        curved_progress = self._apply_curve(effect.progress, effect.curve_type)
        
        # Interpolate values
        for param in effect.target_values:
            start_val = effect.start_values.get(param, 0.0)
            target_val = effect.target_values[param]
            current_val = start_val + (target_val - start_val) * curved_progress
            effect.current_values[param] = current_val
        
        # Call update callback
        if effect.on_update:
            try:
                effect.on_update(effect.effect_id, effect.deck_id, effect.current_values)
            except Exception as e:
                logger.error(f"Error in effect update callback: {e}")
        
        return False
    
    def _apply_curve(self, progress: float, curve_type: str) -> float:
        """Apply curve to progress value"""
        if curve_type == "linear":
            return progress
        elif curve_type == "exponential":
            return progress * progress
        elif curve_type == "smooth":
            # Smoothstep
            return 3 * progress * progress - 2 * progress * progress * progress
        elif curve_type == "ease_in":
            return 1 - np.cos(progress * np.pi / 2)
        elif curve_type == "ease_out":
            return np.sin(progress * np.pi / 2)
        else:
            return progress
    
    def _combine_effect_values(self, param: str, value1: float, value2: float, 
                              effect_type: EffectType) -> float:
        """Combine two effect values for the same parameter"""
        
        # Parameter-specific combination rules
        if param in ['volume', 'eq_low', 'eq_mid', 'eq_high']:
            # Multiplicative for gain-like parameters
            return value1 * value2
        elif param in ['tempo_ratio']:
            # Multiplicative for ratios
            return value1 * value2
        elif param in ['crossfade_position']:
            # Take the more recent/higher priority value
            return value2
        else:
            # Additive for other parameters
            return value1 + value2 - 1.0  # Assuming 1.0 is neutral
    
    def _resolve_effect_conflicts(self, deck_id: str, new_effect: EffectInstance):
        """Resolve conflicts between effects"""
        if deck_id not in self._deck_effects:
            return
        
        # Check for conflicting effects
        conflicts = []
        for effect_id in self._deck_effects[deck_id]:
            existing_effect = self._effects.get(effect_id)
            if (existing_effect and 
                existing_effect.state in [EffectState.PENDING, EffectState.ACTIVE] and
                existing_effect.effect_id != new_effect.effect_id):
                
                # Check for parameter conflicts
                common_params = set(existing_effect.target_values.keys()) & set(new_effect.target_values.keys())
                if common_params and existing_effect.priority < new_effect.priority:
                    conflicts.append(effect_id)
        
        # Cancel conflicting effects
        for effect_id in conflicts:
            self.cancel_effect(effect_id)
            logger.info(f"Cancelled conflicting effect {effect_id} for new effect {new_effect.effect_id}")
    
    def _cleanup_effect(self, effect_id: str):
        """Clean up completed effect"""
        with self._coordinator_lock:
            effect = self._effects.get(effect_id)
            if not effect:
                return
            
            # Remove from deck effects list
            if effect.deck_id in self._deck_effects:
                if effect_id in self._deck_effects[effect.deck_id]:
                    self._deck_effects[effect.deck_id].remove(effect_id)
            
            # Remove from effects dict
            del self._effects[effect_id]
            
            logger.debug(f"Cleaned up effect {effect_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        with self._coordinator_lock:
            effect_counts = {}
            for effect in self._effects.values():
                effect_type = effect.effect_type.name
                if effect_type not in effect_counts:
                    effect_counts[effect_type] = 0
                effect_counts[effect_type] += 1
            
            return {
                'total_effects': len(self._effects),
                'effect_counts': effect_counts,
                'deck_effect_counts': {deck_id: len(effects) for deck_id, effects in self._deck_effects.items()},
                'is_running': self._is_running
            }