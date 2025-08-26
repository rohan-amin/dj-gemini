# Stem processing and mixing system for dj-gemini
# Handles per-stem effects, mixing, and real-time control

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import threading
import time

from .pedalboard_effects import PedalboardAudioProcessor, create_audio_processor
from .stem_separation import StemData, StemSeparationResult

logger = logging.getLogger(__name__)

@dataclass
class StemState:
    """State of an individual stem"""
    name: str
    volume: float = 1.0           # 0.0 = muted, 1.0 = original level
    solo: bool = False            # Solo this stem (mute others)
    muted: bool = False           # Mute this stem
    
    # EQ settings per stem
    eq_low: float = 1.0
    eq_mid: float = 1.0
    eq_high: float = 1.0
    
    # Pan and width
    pan: float = 0.0              # -1.0 = left, 0.0 = center, 1.0 = right
    width: float = 1.0            # Stereo width (0.0 = mono, 1.0 = full stereo)
    
    # Effects
    reverb_send: float = 0.0      # Send to reverb (0.0 to 1.0)
    delay_send: float = 0.0       # Send to delay (0.0 to 1.0)
    
    # Processing state
    audio_processor: Optional[PedalboardAudioProcessor] = None
    last_rms: float = 0.0         # Last measured RMS level

@dataclass
class StemMixState:
    """Overall stem mixing state"""
    stems: Dict[str, StemState] = field(default_factory=dict)
    master_volume: float = 1.0
    crossfade_position: float = 0.5  # For deck crossfading
    
    # Master effects
    master_eq_low: float = 1.0
    master_eq_mid: float = 1.0
    master_eq_high: float = 1.0
    
    # Mix modes
    stem_isolation_mode: bool = False  # Only play selected stems
    isolated_stems: List[str] = field(default_factory=list)

class StemProcessor:
    """Processes and mixes individual stems with effects"""
    
    STEM_NAMES = ['vocals', 'drums', 'bass', 'other']
    
    def __init__(self, deck_id: str, sample_rate: int = 44100):
        self.deck_id = deck_id
        self.sample_rate = sample_rate
        
        # Stem data
        self._stem_data: Dict[str, StemData] = {}
        self._stem_positions: Dict[str, int] = {}  # Current playback position per stem
        
        # Mixing state
        self._mix_state = StemMixState()
        self._mix_lock = threading.RLock()
        
        # Audio processors per stem
        self._stem_processors: Dict[str, PedalboardAudioProcessor] = {}
        
        # Master processor for final mix
        self._master_processor = create_audio_processor(sample_rate)
        
        # Initialize stem states and processors
        for stem_name in self.STEM_NAMES:
            self._mix_state.stems[stem_name] = StemState(name=stem_name)
            self._stem_processors[stem_name] = create_audio_processor(sample_rate)
        
        logger.info(f"Stem processor initialized for deck {deck_id}")
    
    def load_stems(self, stem_result: StemSeparationResult) -> bool:
        """Load separated stems"""
        try:
            with self._mix_lock:
                self._stem_data.clear()
                self._stem_positions.clear()
                
                # Load each stem
                for stem_name, stem_data in stem_result.stems.items():
                    if stem_name in self.STEM_NAMES:
                        self._stem_data[stem_name] = stem_data
                        self._stem_positions[stem_name] = 0
                        
                        logger.debug(f"Loaded stem {stem_name}: {len(stem_data.audio_data)} samples, "
                                   f"RMS: {stem_data.rms_level:.4f}")
                
                logger.info(f"Loaded {len(self._stem_data)} stems for deck {self.deck_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load stems: {e}")
            return False
    
    def process_stems(self, frames: int, global_position: int) -> np.ndarray:
        """Process and mix all stems for current position"""
        try:
            with self._mix_lock:
                if not self._stem_data:
                    return np.zeros(frames, dtype=np.float32)
                
                # Process each stem individually
                stem_outputs = {}
                for stem_name in self.STEM_NAMES:
                    if stem_name in self._stem_data:
                        stem_audio = self._process_individual_stem(stem_name, frames, global_position)
                        stem_outputs[stem_name] = stem_audio
                    else:
                        stem_outputs[stem_name] = np.zeros(frames, dtype=np.float32)
                
                # Mix stems together
                mixed_audio = self._mix_stems(stem_outputs, frames)
                
                # Apply master processing
                final_audio = self._apply_master_processing(mixed_audio)
                
                return final_audio
                
        except Exception as e:
            logger.error(f"Error processing stems: {e}")
            return np.zeros(frames, dtype=np.float32)
    
    def _process_individual_stem(self, stem_name: str, frames: int, global_position: int) -> np.ndarray:
        """Process individual stem with effects"""
        try:
            stem_data = self._stem_data[stem_name]
            stem_state = self._mix_state.stems[stem_name]
            processor = self._stem_processors[stem_name]
            
            # Get audio chunk from stem
            start_frame = global_position
            end_frame = start_frame + frames
            
            if start_frame >= len(stem_data.audio_data):
                return np.zeros(frames, dtype=np.float32)
            
            # Handle end of stem
            if end_frame > len(stem_data.audio_data):
                available_frames = len(stem_data.audio_data) - start_frame
                audio_chunk = np.zeros(frames, dtype=np.float32)
                audio_chunk[:available_frames] = stem_data.audio_data[start_frame:end_frame]
            else:
                audio_chunk = stem_data.audio_data[start_frame:end_frame].copy()
            
            # Apply stem-specific processing
            
            # 1. Volume control
            if stem_state.muted:
                audio_chunk *= 0.0
            else:
                audio_chunk *= stem_state.volume
            
            # 2. Solo logic
            any_soloed = any(state.solo for state in self._mix_state.stems.values())
            if any_soloed and not stem_state.solo:
                audio_chunk *= 0.0
            
            # 3. Stem isolation mode
            if self._mix_state.stem_isolation_mode:
                if stem_name not in self._mix_state.isolated_stems:
                    audio_chunk *= 0.0
            
            # 4. Apply EQ via processor
            processor.set_eq(
                low=stem_state.eq_low,
                mid=stem_state.eq_mid,
                high=stem_state.eq_high
            )
            audio_chunk = processor.process_audio(audio_chunk)
            
            # 5. Apply pan (convert to stereo if needed)
            if stem_state.pan != 0.0:
                audio_chunk = self._apply_pan(audio_chunk, stem_state.pan)
            
            # 6. Update RMS level for metering
            stem_state.last_rms = np.sqrt(np.mean(audio_chunk ** 2))
            
            return audio_chunk
            
        except Exception as e:
            logger.error(f"Error processing stem {stem_name}: {e}")
            return np.zeros(frames, dtype=np.float32)
    
    def _mix_stems(self, stem_outputs: Dict[str, np.ndarray], frames: int) -> np.ndarray:
        """Mix processed stems together"""
        try:
            mixed = np.zeros(frames, dtype=np.float32)
            
            for stem_name, stem_audio in stem_outputs.items():
                if stem_audio is not None and len(stem_audio) > 0:
                    mixed += stem_audio
            
            # Apply master volume
            mixed *= self._mix_state.master_volume
            
            return mixed
            
        except Exception as e:
            logger.error(f"Error mixing stems: {e}")
            return np.zeros(frames, dtype=np.float32)
    
    def _apply_master_processing(self, audio: np.ndarray) -> np.ndarray:
        """Apply master effects to final mix"""
        try:
            # Apply master EQ
            self._master_processor.set_eq(
                low=self._mix_state.master_eq_low,
                mid=self._mix_state.master_eq_mid,
                high=self._mix_state.master_eq_high
            )
            
            return self._master_processor.process_audio(audio)
            
        except Exception as e:
            logger.error(f"Error in master processing: {e}")
            return audio
    
    def _apply_pan(self, audio: np.ndarray, pan: float) -> np.ndarray:
        """Apply panning to audio (convert to stereo)"""
        try:
            if audio.ndim == 1:
                # Convert mono to stereo with panning
                left_gain = np.sqrt((1.0 - pan) / 2.0)
                right_gain = np.sqrt((1.0 + pan) / 2.0)
                
                stereo_audio = np.column_stack([
                    audio * left_gain,
                    audio * right_gain
                ])
                return stereo_audio
            else:
                # Already stereo, apply pan
                left_gain = np.sqrt((1.0 - pan) / 2.0)
                right_gain = np.sqrt((1.0 + pan) / 2.0)
                
                audio[:, 0] *= left_gain
                audio[:, 1] *= right_gain
                
                return audio
                
        except Exception as e:
            logger.error(f"Error applying pan: {e}")
            return audio
    
    def set_stem_volume(self, stem_name: str, volume: float):
        """Set volume for specific stem"""
        with self._mix_lock:
            if stem_name in self._mix_state.stems:
                self._mix_state.stems[stem_name].volume = max(0.0, min(2.0, volume))
                logger.debug(f"Set {stem_name} volume: {volume}")
    
    def set_stem_eq(self, stem_name: str, low: float = None, mid: float = None, high: float = None):
        """Set EQ for specific stem"""
        with self._mix_lock:
            if stem_name in self._mix_state.stems:
                stem_state = self._mix_state.stems[stem_name]
                if low is not None: stem_state.eq_low = max(0.0, min(3.0, low))
                if mid is not None: stem_state.eq_mid = max(0.0, min(3.0, mid))
                if high is not None: stem_state.eq_high = max(0.0, min(3.0, high))
                logger.debug(f"Set {stem_name} EQ: L={stem_state.eq_low}, M={stem_state.eq_mid}, H={stem_state.eq_high}")
    
    def solo_stem(self, stem_name: str, solo: bool = True):
        """Solo/unsolo specific stem"""
        with self._mix_lock:
            if stem_name in self._mix_state.stems:
                self._mix_state.stems[stem_name].solo = solo
                logger.debug(f"{'Solo' if solo else 'Unsolo'} stem: {stem_name}")
    
    def mute_stem(self, stem_name: str, muted: bool = True):
        """Mute/unmute specific stem"""
        with self._mix_lock:
            if stem_name in self._mix_state.stems:
                self._mix_state.stems[stem_name].muted = muted
                logger.debug(f"{'Mute' if muted else 'Unmute'} stem: {stem_name}")
    
    def isolate_stems(self, stem_names: List[str]):
        """Isolate specific stems (only these will play)"""
        with self._mix_lock:
            self._mix_state.stem_isolation_mode = True
            self._mix_state.isolated_stems = [name for name in stem_names if name in self.STEM_NAMES]
            logger.info(f"Isolated stems: {self._mix_state.isolated_stems}")
    
    def clear_isolation(self):
        """Clear stem isolation (all stems play)"""
        with self._mix_lock:
            self._mix_state.stem_isolation_mode = False
            self._mix_state.isolated_stems.clear()
            logger.info("Cleared stem isolation")
    
    def set_master_volume(self, volume: float):
        """Set master volume for all stems"""
        with self._mix_lock:
            self._mix_state.master_volume = max(0.0, min(2.0, volume))
    
    def set_master_eq(self, low: float = None, mid: float = None, high: float = None):
        """Set master EQ for final mix"""
        with self._mix_lock:
            if low is not None: self._mix_state.master_eq_low = max(0.0, min(3.0, low))
            if mid is not None: self._mix_state.master_eq_mid = max(0.0, min(3.0, mid))
            if high is not None: self._mix_state.master_eq_high = max(0.0, min(3.0, high))
    
    def get_stem_levels(self) -> Dict[str, float]:
        """Get current RMS levels for all stems"""
        with self._mix_lock:
            return {
                stem_name: state.last_rms 
                for stem_name, state in self._mix_state.stems.items()
            }
    
    def get_stem_state(self, stem_name: str) -> Optional[Dict[str, Any]]:
        """Get complete state for specific stem"""
        with self._mix_lock:
            if stem_name not in self._mix_state.stems:
                return None
            
            state = self._mix_state.stems[stem_name]
            return {
                'name': state.name,
                'volume': state.volume,
                'solo': state.solo,
                'muted': state.muted,
                'eq_low': state.eq_low,
                'eq_mid': state.eq_mid,
                'eq_high': state.eq_high,
                'pan': state.pan,
                'width': state.width,
                'last_rms': state.last_rms,
                'available': stem_name in self._stem_data
            }
    
    def get_mix_state(self) -> Dict[str, Any]:
        """Get complete mixing state"""
        with self._mix_lock:
            return {
                'master_volume': self._mix_state.master_volume,
                'master_eq': {
                    'low': self._mix_state.master_eq_low,
                    'mid': self._mix_state.master_eq_mid,
                    'high': self._mix_state.master_eq_high
                },
                'isolation_mode': self._mix_state.stem_isolation_mode,
                'isolated_stems': self._mix_state.isolated_stems.copy(),
                'stems': {
                    name: self.get_stem_state(name) 
                    for name in self.STEM_NAMES
                },
                'available_stems': list(self._stem_data.keys()),
                'crossfade_position': self._mix_state.crossfade_position
            }
    
    def reset_all_stems(self):
        """Reset all stem states to defaults"""
        with self._mix_lock:
            for stem_name in self.STEM_NAMES:
                state = self._mix_state.stems[stem_name]
                state.volume = 1.0
                state.solo = False
                state.muted = False
                state.eq_low = 1.0
                state.eq_mid = 1.0
                state.eq_high = 1.0
                state.pan = 0.0
                state.width = 1.0
                state.reverb_send = 0.0
                state.delay_send = 0.0
            
            # Reset master settings
            self._mix_state.master_volume = 1.0
            self._mix_state.master_eq_low = 1.0
            self._mix_state.master_eq_mid = 1.0
            self._mix_state.master_eq_high = 1.0
            
            # Clear isolation
            self.clear_isolation()
            
            logger.info(f"Reset all stem states for deck {self.deck_id}")
    
    def has_stems(self) -> bool:
        """Check if stems are loaded"""
        with self._mix_lock:
            return len(self._stem_data) > 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics"""
        with self._mix_lock:
            return {
                'deck_id': self.deck_id,
                'sample_rate': self.sample_rate,
                'loaded_stems': list(self._stem_data.keys()),
                'total_stems': len(self._stem_data),
                'stem_processors': len(self._stem_processors),
                'isolation_active': self._mix_state.stem_isolation_mode,
                'solo_active': any(state.solo for state in self._mix_state.stems.values()),
                'stem_levels': self.get_stem_levels()
            }

# Factory function
def create_stem_processor(deck_id: str, sample_rate: int = 44100) -> StemProcessor:
    """Create stem processor for deck"""
    return StemProcessor(deck_id, sample_rate)