# Multi-deck stem mixing system for dj-gemini
# Enables advanced stem mixing techniques between multiple decks

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import threading
import time

from .stem_processor import StemProcessor, create_stem_processor

logger = logging.getLogger(__name__)

@dataclass
class CrossfadeState:
    """State of crossfading between decks"""
    active: bool = False
    from_deck: str = ""
    to_deck: str = ""
    position: float = 0.5        # 0.0 = fully from_deck, 1.0 = fully to_deck
    curve_type: str = "linear"   # linear, smooth, sharp
    
    # Per-stem crossfade (can crossfade different stems at different rates)
    stem_positions: Dict[str, float] = field(default_factory=dict)
    
    # Automatic crossfade
    auto_crossfade: bool = False
    auto_duration: float = 4.0
    auto_start_time: float = 0.0

@dataclass
class StemRoutingRule:
    """Rule for routing stems between decks"""
    source_deck: str
    source_stem: str
    target_deck: str
    target_stem: str
    mix_ratio: float = 1.0       # How much to mix (0.0 to 1.0)
    active: bool = True

class MultiDeckStemMixer:
    """Advanced multi-deck stem mixing system"""
    
    STEM_NAMES = ['vocals', 'drums', 'bass', 'other']
    
    def __init__(self, max_decks: int = 4, sample_rate: int = 44100):
        self.max_decks = max_decks
        self.sample_rate = sample_rate
        
        # Deck stem processors
        self._deck_processors: Dict[str, StemProcessor] = {}
        self._deck_lock = threading.RLock()
        
        # Crossfade state
        self._crossfade_state = CrossfadeState()
        
        # Stem routing system
        self._routing_rules: List[StemRoutingRule] = []
        
        # Master output mixing
        self._master_volume = 1.0
        self._master_eq = {'low': 1.0, 'mid': 1.0, 'high': 1.0}
        
        # Advanced mixing features
        self._stem_sync_enabled = False  # Sync stem playback across decks
        self._harmonic_mixing = False    # Harmonic mixing between compatible stems
        
        logger.info(f"Multi-deck stem mixer initialized (max decks: {max_decks})")
    
    def register_deck(self, deck_id: str) -> bool:
        """Register a deck with the stem mixer"""
        try:
            with self._deck_lock:
                if deck_id in self._deck_processors:
                    logger.warning(f"Deck {deck_id} already registered")
                    return True
                
                if len(self._deck_processors) >= self.max_decks:
                    logger.error(f"Maximum number of decks ({self.max_decks}) reached")
                    return False
                
                # Create stem processor for deck
                processor = create_stem_processor(deck_id, self.sample_rate)
                self._deck_processors[deck_id] = processor
                
                # Initialize crossfade stem positions
                for stem_name in self.STEM_NAMES:
                    if stem_name not in self._crossfade_state.stem_positions:
                        self._crossfade_state.stem_positions[stem_name] = 0.5
                
                logger.info(f"Registered deck {deck_id} with stem mixer")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register deck {deck_id}: {e}")
            return False
    
    def unregister_deck(self, deck_id: str):
        """Unregister deck from stem mixer"""
        with self._deck_lock:
            if deck_id in self._deck_processors:
                del self._deck_processors[deck_id]
                
                # Remove routing rules involving this deck
                self._routing_rules = [
                    rule for rule in self._routing_rules 
                    if rule.source_deck != deck_id and rule.target_deck != deck_id
                ]
                
                logger.info(f"Unregistered deck {deck_id}")
    
    def get_deck_processor(self, deck_id: str) -> Optional[StemProcessor]:
        """Get stem processor for specific deck"""
        with self._deck_lock:
            return self._deck_processors.get(deck_id)
    
    def mix_all_decks(self, frames: int, deck_positions: Dict[str, int]) -> np.ndarray:
        """Mix audio from all registered decks with stem routing"""
        try:
            with self._deck_lock:
                if not self._deck_processors:
                    return np.zeros(frames, dtype=np.float32)
                
                # Process each deck's stems
                deck_outputs = {}
                for deck_id, processor in self._deck_processors.items():
                    global_position = deck_positions.get(deck_id, 0)
                    deck_outputs[deck_id] = processor.process_stems(frames, global_position)
                
                # Apply stem routing
                routed_outputs = self._apply_stem_routing(deck_outputs, frames)
                
                # Apply crossfading
                crossfaded_output = self._apply_crossfading(routed_outputs, frames)
                
                # Apply master processing
                final_output = self._apply_master_processing(crossfaded_output)
                
                return final_output
                
        except Exception as e:
            logger.error(f"Error mixing decks: {e}")
            return np.zeros(frames, dtype=np.float32)
    
    def _apply_stem_routing(self, deck_outputs: Dict[str, np.ndarray], frames: int) -> Dict[str, np.ndarray]:
        """Apply stem routing rules between decks"""
        try:
            if not self._routing_rules:
                return deck_outputs
            
            # Create copy to modify
            routed_outputs = {deck_id: audio.copy() for deck_id, audio in deck_outputs.items()}
            
            # Apply each routing rule
            for rule in self._routing_rules:
                if not rule.active:
                    continue
                
                source_deck_id = rule.source_deck
                target_deck_id = rule.target_deck
                
                if source_deck_id not in deck_outputs or target_deck_id not in routed_outputs:
                    continue
                
                # Get source stem audio
                source_processor = self._deck_processors.get(source_deck_id)
                if not source_processor:
                    continue
                
                # For now, mix entire deck output
                # In full implementation, would extract individual stems
                source_audio = deck_outputs[source_deck_id]
                target_audio = routed_outputs[target_deck_id]
                
                # Mix with specified ratio
                mixed_audio = target_audio + (source_audio * rule.mix_ratio)
                routed_outputs[target_deck_id] = mixed_audio
                
                logger.debug(f"Applied routing: {source_deck_id}->{target_deck_id} (ratio: {rule.mix_ratio})")
            
            return routed_outputs
            
        except Exception as e:
            logger.error(f"Error applying stem routing: {e}")
            return deck_outputs
    
    def _apply_crossfading(self, deck_outputs: Dict[str, np.ndarray], frames: int) -> np.ndarray:
        """Apply crossfading between decks"""
        try:
            if not self._crossfade_state.active or len(deck_outputs) < 2:
                # No crossfading, sum all decks
                mixed = np.zeros(frames, dtype=np.float32)
                for audio in deck_outputs.values():
                    mixed += audio
                return mixed
            
            # Update automatic crossfade if active
            if self._crossfade_state.auto_crossfade:
                self._update_auto_crossfade()
            
            from_deck = self._crossfade_state.from_deck
            to_deck = self._crossfade_state.to_deck
            position = self._crossfade_state.position
            
            if from_deck not in deck_outputs or to_deck not in deck_outputs:
                # Fallback to summing all decks
                mixed = np.zeros(frames, dtype=np.float32)
                for audio in deck_outputs.values():
                    mixed += audio
                return mixed
            
            # Apply crossfade curve
            if self._crossfade_state.curve_type == "linear":
                from_gain = 1.0 - position
                to_gain = position
            elif self._crossfade_state.curve_type == "smooth":
                # S-curve crossfade
                smooth_pos = 3 * position * position - 2 * position * position * position
                from_gain = 1.0 - smooth_pos
                to_gain = smooth_pos
            elif self._crossfade_state.curve_type == "sharp":
                # Sharp cut at center
                if position < 0.5:
                    from_gain = 1.0
                    to_gain = 0.0
                else:
                    from_gain = 0.0
                    to_gain = 1.0
            else:
                from_gain = 1.0 - position
                to_gain = position
            
            # Mix the two decks
            from_audio = deck_outputs[from_deck] * from_gain
            to_audio = deck_outputs[to_deck] * to_gain
            crossfaded = from_audio + to_audio
            
            # Add any other decks at reduced volume
            for deck_id, audio in deck_outputs.items():
                if deck_id not in [from_deck, to_deck]:
                    crossfaded += audio * 0.3  # Reduced volume for non-crossfaded decks
            
            return crossfaded
            
        except Exception as e:
            logger.error(f"Error applying crossfade: {e}")
            # Fallback to summing all decks
            mixed = np.zeros(frames, dtype=np.float32)
            for audio in deck_outputs.values():
                mixed += audio
            return mixed
    
    def _update_auto_crossfade(self):
        """Update automatic crossfade position"""
        try:
            if not self._crossfade_state.auto_crossfade:
                return
            
            elapsed = time.time() - self._crossfade_state.auto_start_time
            progress = elapsed / self._crossfade_state.auto_duration
            
            if progress >= 1.0:
                # Auto crossfade complete
                self._crossfade_state.position = 1.0
                self._crossfade_state.auto_crossfade = False
                logger.info("Auto crossfade completed")
            else:
                self._crossfade_state.position = progress
                
        except Exception as e:
            logger.error(f"Error updating auto crossfade: {e}")
    
    def _apply_master_processing(self, audio: np.ndarray) -> np.ndarray:
        """Apply master processing to final mix"""
        try:
            # Apply master volume
            processed = audio * self._master_volume
            
            # Apply master EQ (simplified)
            eq_factor = (self._master_eq['low'] + self._master_eq['mid'] + self._master_eq['high']) / 3.0
            processed *= eq_factor
            
            # Soft limiting to prevent clipping
            max_val = np.max(np.abs(processed))
            if max_val > 1.0:
                processed = processed / max_val * 0.95
            
            return processed
            
        except Exception as e:
            logger.error(f"Error in master processing: {e}")
            return audio
    
    def start_crossfade(self, from_deck: str, to_deck: str, duration: float = 0.0, 
                       curve_type: str = "linear") -> bool:
        """Start crossfade between two decks"""
        try:
            with self._deck_lock:
                if from_deck not in self._deck_processors or to_deck not in self._deck_processors:
                    logger.error(f"Cannot crossfade - invalid decks: {from_deck}, {to_deck}")
                    return False
                
                self._crossfade_state.active = True
                self._crossfade_state.from_deck = from_deck
                self._crossfade_state.to_deck = to_deck
                self._crossfade_state.position = 0.0
                self._crossfade_state.curve_type = curve_type
                
                if duration > 0:
                    # Automatic crossfade
                    self._crossfade_state.auto_crossfade = True
                    self._crossfade_state.auto_duration = duration
                    self._crossfade_state.auto_start_time = time.time()
                else:
                    # Manual crossfade
                    self._crossfade_state.auto_crossfade = False
                
                logger.info(f"Started crossfade: {from_deck} -> {to_deck} "
                          f"({'auto' if duration > 0 else 'manual'}, {curve_type})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start crossfade: {e}")
            return False
    
    def set_crossfade_position(self, position: float):
        """Set crossfade position manually (0.0 to 1.0)"""
        self._crossfade_state.position = max(0.0, min(1.0, position))
        self._crossfade_state.auto_crossfade = False  # Disable auto when manually controlled
    
    def stop_crossfade(self):
        """Stop current crossfade"""
        self._crossfade_state.active = False
        self._crossfade_state.auto_crossfade = False
        logger.info("Stopped crossfade")
    
    def add_stem_routing(self, source_deck: str, source_stem: str, 
                        target_deck: str, target_stem: str, mix_ratio: float = 1.0) -> bool:
        """Add stem routing rule"""
        try:
            if source_deck not in self._deck_processors or target_deck not in self._deck_processors:
                logger.error(f"Cannot add routing - invalid decks: {source_deck}, {target_deck}")
                return False
            
            if source_stem not in self.STEM_NAMES or target_stem not in self.STEM_NAMES:
                logger.error(f"Cannot add routing - invalid stems: {source_stem}, {target_stem}")
                return False
            
            rule = StemRoutingRule(
                source_deck=source_deck,
                source_stem=source_stem,
                target_deck=target_deck,
                target_stem=target_stem,
                mix_ratio=max(0.0, min(1.0, mix_ratio))
            )
            
            self._routing_rules.append(rule)
            logger.info(f"Added stem routing: {source_deck}:{source_stem} -> {target_deck}:{target_stem} ({mix_ratio})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add stem routing: {e}")
            return False
    
    def remove_stem_routing(self, source_deck: str, source_stem: str, 
                           target_deck: str, target_stem: str):
        """Remove stem routing rule"""
        self._routing_rules = [
            rule for rule in self._routing_rules 
            if not (rule.source_deck == source_deck and rule.source_stem == source_stem and
                   rule.target_deck == target_deck and rule.target_stem == target_stem)
        ]
        logger.info(f"Removed stem routing: {source_deck}:{source_stem} -> {target_deck}:{target_stem}")
    
    def clear_all_routing(self):
        """Clear all stem routing rules"""
        self._routing_rules.clear()
        logger.info("Cleared all stem routing rules")
    
    def stem_crossfade(self, deck_a: str, deck_b: str, stem_name: str, 
                      position: float, duration: float = 0.0) -> bool:
        """Crossfade specific stem between two decks"""
        try:
            if deck_a not in self._deck_processors or deck_b not in self._deck_processors:
                logger.error(f"Cannot stem crossfade - invalid decks: {deck_a}, {deck_b}")
                return False
            
            if stem_name not in self.STEM_NAMES:
                logger.error(f"Cannot stem crossfade - invalid stem: {stem_name}")
                return False
            
            # Set per-stem crossfade position
            self._crossfade_state.stem_positions[stem_name] = max(0.0, min(1.0, position))
            
            logger.info(f"Stem crossfade: {deck_a}:{stem_name} <-> {deck_b}:{stem_name} (pos: {position})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stem crossfade: {e}")
            return False
    
    def isolate_deck_stem(self, deck_id: str, stem_names: List[str]):
        """Isolate specific stems on a deck (mute others)"""
        processor = self.get_deck_processor(deck_id)
        if processor:
            processor.isolate_stems(stem_names)
            logger.info(f"Isolated stems {stem_names} on deck {deck_id}")
    
    def sync_stems_across_decks(self, stem_name: str, reference_deck: str):
        """Sync specific stem timing across all decks"""
        # This would implement beat-sync functionality
        # For now, just log the intent
        logger.info(f"Syncing {stem_name} across decks to reference: {reference_deck}")
        self._stem_sync_enabled = True
    
    def set_master_volume(self, volume: float):
        """Set master output volume"""
        self._master_volume = max(0.0, min(2.0, volume))
    
    def set_master_eq(self, low: float = None, mid: float = None, high: float = None):
        """Set master EQ"""
        if low is not None: self._master_eq['low'] = max(0.0, min(3.0, low))
        if mid is not None: self._master_eq['mid'] = max(0.0, min(3.0, mid))
        if high is not None: self._master_eq['high'] = max(0.0, min(3.0, high))
    
    def get_crossfade_state(self) -> Dict[str, Any]:
        """Get current crossfade state"""
        return {
            'active': self._crossfade_state.active,
            'from_deck': self._crossfade_state.from_deck,
            'to_deck': self._crossfade_state.to_deck,
            'position': self._crossfade_state.position,
            'curve_type': self._crossfade_state.curve_type,
            'auto_crossfade': self._crossfade_state.auto_crossfade,
            'stem_positions': self._crossfade_state.stem_positions.copy()
        }
    
    def get_routing_rules(self) -> List[Dict[str, Any]]:
        """Get all stem routing rules"""
        return [
            {
                'source_deck': rule.source_deck,
                'source_stem': rule.source_stem,
                'target_deck': rule.target_deck,
                'target_stem': rule.target_stem,
                'mix_ratio': rule.mix_ratio,
                'active': rule.active
            }
            for rule in self._routing_rules
        ]
    
    def get_deck_states(self) -> Dict[str, Any]:
        """Get states of all registered decks"""
        with self._deck_lock:
            states = {}
            for deck_id, processor in self._deck_processors.items():
                states[deck_id] = processor.get_mix_state()
            return states
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mixer statistics"""
        with self._deck_lock:
            return {
                'registered_decks': list(self._deck_processors.keys()),
                'max_decks': self.max_decks,
                'active_routing_rules': len([r for r in self._routing_rules if r.active]),
                'total_routing_rules': len(self._routing_rules),
                'crossfade_active': self._crossfade_state.active,
                'stem_sync_enabled': self._stem_sync_enabled,
                'harmonic_mixing': self._harmonic_mixing,
                'master_volume': self._master_volume,
                'master_eq': self._master_eq.copy()
            }

# Factory function
def create_multi_deck_stem_mixer(max_decks: int = 4, sample_rate: int = 44100) -> MultiDeckStemMixer:
    """Create multi-deck stem mixer"""
    return MultiDeckStemMixer(max_decks, sample_rate)