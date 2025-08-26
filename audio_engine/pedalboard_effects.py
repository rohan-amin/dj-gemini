# Professional audio effects using Spotify Pedalboard
# Replaces SciPy-based EQ with studio-quality processing

import numpy as np
import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field

try:
    from pedalboard import Pedalboard, LadderFilter, Gain, HighpassFilter, LowpassFilter
    from pedalboard import Reverb, Delay, Chorus, Phaser, Distortion
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False
    logging.warning("Pedalboard not available - falling back to basic processing")

logger = logging.getLogger(__name__)

@dataclass
class EQSettings:
    """Professional 3-band DJ EQ settings"""
    low_gain: float = 1.0      # 0.0 = kill, 1.0 = neutral, 2.0 = boost
    mid_gain: float = 1.0      # Linear gain values
    high_gain: float = 1.0
    
    # EQ frequency points (DJ mixer style)
    low_freq: float = 320.0    # Low/mid crossover
    high_freq: float = 2500.0  # Mid/high crossover
    
    # Filter characteristics
    low_q: float = 0.7         # Quality factor for filters
    mid_q: float = 0.7
    high_q: float = 0.7

@dataclass
class ScratchSettings:
    """Scratch effect settings"""
    active: bool = False
    pattern: str = "chirp"     # Pattern type
    speed: float = 1.0         # Playback speed multiplier
    direction: int = 1         # 1 = forward, -1 = reverse
    crossfader_position: float = 1.0  # 0.0 = cut, 1.0 = open

class PedalboardAudioProcessor:
    """Professional audio processor using Pedalboard"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
        if not PEDALBOARD_AVAILABLE:
            logger.error("Pedalboard not available - audio processing will be limited")
            self._use_fallback = True
        else:
            self._use_fallback = False
        
        # EQ state
        self._eq_settings = EQSettings()
        self._eq_board = None
        self._eq_dirty = True  # Rebuild EQ chain when True
        
        # Effect state
        self._effects_board = None
        self._effects_dirty = True
        
        # Scratch state
        self._scratch_settings = ScratchSettings()
        
        # Audio buffers for processing
        self._temp_buffer = None
        
        logger.info(f"Pedalboard processor initialized (fallback: {self._use_fallback})")
    
    def set_eq(self, low: float = None, mid: float = None, high: float = None):
        """Set EQ levels (0.0 = kill, 1.0 = neutral, 2.0 = boost)"""
        changed = False
        
        if low is not None and abs(low - self._eq_settings.low_gain) > 1e-6:
            self._eq_settings.low_gain = max(0.0, min(3.0, low))
            changed = True
        
        if mid is not None and abs(mid - self._eq_settings.mid_gain) > 1e-6:
            self._eq_settings.mid_gain = max(0.0, min(3.0, mid))
            changed = True
        
        if high is not None and abs(high - self._eq_settings.high_gain) > 1e-6:
            self._eq_settings.high_gain = max(0.0, min(3.0, high))
            changed = True
        
        if changed:
            self._eq_dirty = True
            logger.debug(f"EQ updated: L={self._eq_settings.low_gain:.2f}, "
                        f"M={self._eq_settings.mid_gain:.2f}, H={self._eq_settings.high_gain:.2f}")
    
    def get_eq(self) -> Dict[str, float]:
        """Get current EQ settings"""
        return {
            'low': self._eq_settings.low_gain,
            'mid': self._eq_settings.mid_gain,
            'high': self._eq_settings.high_gain
        }
    
    def set_scratch_effect(self, active: bool, pattern: str = "chirp", 
                          speed: float = 1.0, crossfader: float = 1.0):
        """Configure scratch effect"""
        self._scratch_settings.active = active
        self._scratch_settings.pattern = pattern
        self._scratch_settings.speed = speed
        self._scratch_settings.crossfader_position = crossfader
        
        logger.debug(f"Scratch effect: active={active}, pattern={pattern}, speed={speed}")
    
    def process_audio(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Process audio chunk with EQ and effects"""
        try:
            if audio_chunk is None or len(audio_chunk) == 0:
                return audio_chunk
            
            # Ensure proper audio format
            if audio_chunk.ndim == 1:
                # Mono to stereo
                stereo_audio = np.column_stack([audio_chunk, audio_chunk])
            else:
                stereo_audio = audio_chunk.copy()
            
            # Apply EQ processing
            processed_audio = self._apply_eq(stereo_audio)
            
            # Apply scratch effects if active
            if self._scratch_settings.active:
                processed_audio = self._apply_scratch_effect(processed_audio)
            
            # Return in original format
            if audio_chunk.ndim == 1:
                return processed_audio[:, 0]  # Convert back to mono
            else:
                return processed_audio
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return audio_chunk  # Return original on error
    
    def _apply_eq(self, audio: np.ndarray) -> np.ndarray:
        """Apply EQ processing using Pedalboard"""
        if self._use_fallback:
            return self._apply_eq_fallback(audio)
        
        try:
            # Rebuild EQ chain if needed
            if self._eq_dirty or self._eq_board is None:
                self._rebuild_eq_chain()
                self._eq_dirty = False
            
            if self._eq_board is None:
                return audio
            
            # Process with Pedalboard
            # Convert to float32 if needed
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Apply EQ
            processed = self._eq_board(audio, sample_rate=self.sample_rate)
            
            return processed
            
        except Exception as e:
            logger.error(f"Pedalboard EQ processing failed: {e}")
            return self._apply_eq_fallback(audio)
    
    def _rebuild_eq_chain(self):
        """Rebuild Pedalboard EQ chain"""
        if self._use_fallback:
            return
        
        try:
            eq_chain = []
            
            # Low band processing
            if self._eq_settings.low_gain != 1.0:
                if self._eq_settings.low_gain < 0.01:  # Kill low
                    eq_chain.append(HighpassFilter(cutoff_frequency_hz=self._eq_settings.low_freq))
                else:
                    # Low shelf filter simulation using ladder filter
                    eq_chain.append(LadderFilter(
                        mode=LadderFilter.Mode.LPF24,
                        cutoff_hz=self._eq_settings.low_freq,
                        resonance=self._eq_settings.low_q
                    ))
                    if self._eq_settings.low_gain != 1.0:
                        gain_db = 20 * np.log10(self._eq_settings.low_gain)
                        eq_chain.append(Gain(gain_db=gain_db))
            
            # Mid band processing (most complex - need bandpass simulation)
            if self._eq_settings.mid_gain != 1.0:
                if self._eq_settings.mid_gain < 0.01:  # Kill mids
                    # Create notch by combining high-pass and low-pass
                    eq_chain.append(LadderFilter(
                        mode=LadderFilter.Mode.BPF12,
                        cutoff_hz=(self._eq_settings.low_freq + self._eq_settings.high_freq) / 2,
                        resonance=0.1  # Narrow notch
                    ))
                    eq_chain.append(Gain(gain_db=-40))  # Heavy cut
                else:
                    # Mid boost/cut using bandpass filter
                    eq_chain.append(LadderFilter(
                        mode=LadderFilter.Mode.BPF12,
                        cutoff_hz=(self._eq_settings.low_freq + self._eq_settings.high_freq) / 2,
                        resonance=self._eq_settings.mid_q
                    ))
                    if self._eq_settings.mid_gain != 1.0:
                        gain_db = 20 * np.log10(self._eq_settings.mid_gain)
                        eq_chain.append(Gain(gain_db=gain_db))
            
            # High band processing
            if self._eq_settings.high_gain != 1.0:
                if self._eq_settings.high_gain < 0.01:  # Kill highs
                    eq_chain.append(LowpassFilter(cutoff_frequency_hz=self._eq_settings.high_freq))
                else:
                    # High shelf filter simulation
                    eq_chain.append(LadderFilter(
                        mode=LadderFilter.Mode.HPF12,
                        cutoff_hz=self._eq_settings.high_freq,
                        resonance=self._eq_settings.high_q
                    ))
                    if self._eq_settings.high_gain != 1.0:
                        gain_db = 20 * np.log10(self._eq_settings.high_gain)
                        eq_chain.append(Gain(gain_db=gain_db))
            
            # Create pedalboard
            if eq_chain:
                self._eq_board = Pedalboard(eq_chain)
            else:
                self._eq_board = None
            
            logger.debug(f"Rebuilt EQ chain with {len(eq_chain)} processors")
            
        except Exception as e:
            logger.error(f"Failed to rebuild EQ chain: {e}")
            self._eq_board = None
    
    def _apply_eq_fallback(self, audio: np.ndarray) -> np.ndarray:
        """Fallback EQ processing when Pedalboard unavailable"""
        # Simple gain-based EQ (not ideal but functional)
        try:
            # Rough frequency separation using simple filters
            # This is a basic fallback - not as good as Pedalboard
            low_factor = self._eq_settings.low_gain
            mid_factor = self._eq_settings.mid_gain  
            high_factor = self._eq_settings.high_gain
            
            # Apply average EQ (very basic)
            eq_factor = (low_factor + mid_factor + high_factor) / 3.0
            return audio * eq_factor
            
        except Exception as e:
            logger.error(f"Fallback EQ processing failed: {e}")
            return audio
    
    def _apply_scratch_effect(self, audio: np.ndarray) -> np.ndarray:
        """Apply scratch effect to audio"""
        try:
            if not self._scratch_settings.active:
                return audio
            
            # Apply crossfader
            crossfader = self._scratch_settings.crossfader_position
            
            # Simple crossfader curve (can be made more sophisticated)
            if crossfader < 0.01:
                return np.zeros_like(audio)  # Cut audio
            elif crossfader < 1.0:
                return audio * crossfader
            else:
                return audio
                
        except Exception as e:
            logger.error(f"Scratch effect processing failed: {e}")
            return audio
    
    def add_reverb(self, room_size: float = 0.5, damping: float = 0.5, 
                   wet_level: float = 0.3, dry_level: float = 0.7):
        """Add reverb effect to the chain"""
        if self._use_fallback:
            logger.warning("Reverb not available in fallback mode")
            return
        
        try:
            # This would be added to a separate effects chain
            reverb = Reverb(
                room_size=room_size,
                damping=damping,
                wet_level=wet_level,
                dry_level=dry_level
            )
            
            if self._effects_board is None:
                self._effects_board = Pedalboard([reverb])
            else:
                # Add to existing chain
                effects = list(self._effects_board)
                effects.append(reverb)
                self._effects_board = Pedalboard(effects)
            
            logger.info("Added reverb effect")
            
        except Exception as e:
            logger.error(f"Failed to add reverb: {e}")
    
    def clear_effects(self):
        """Clear all effects except EQ"""
        self._effects_board = None
        logger.info("Cleared all effects")
    
    def get_available_effects(self) -> List[str]:
        """Get list of available effects"""
        if self._use_fallback:
            return ["basic_eq"]
        
        return [
            "ladder_filter", "gain", "highpass_filter", "lowpass_filter",
            "reverb", "delay", "chorus", "phaser", "distortion"
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            'pedalboard_available': PEDALBOARD_AVAILABLE,
            'using_fallback': self._use_fallback,
            'sample_rate': self.sample_rate,
            'eq_settings': {
                'low': self._eq_settings.low_gain,
                'mid': self._eq_settings.mid_gain,
                'high': self._eq_settings.high_gain
            },
            'scratch_active': self._scratch_settings.active,
            'effects_active': self._effects_board is not None,
            'available_effects': self.get_available_effects()
        }

# Factory function for creating processors
def create_audio_processor(sample_rate: int = 44100) -> PedalboardAudioProcessor:
    """Create and initialize audio processor"""
    return PedalboardAudioProcessor(sample_rate)