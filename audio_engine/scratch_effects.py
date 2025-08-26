# Advanced scratching effects system for dj-gemini
# Implements professional DJ scratch techniques with precise timing

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import math
import time

logger = logging.getLogger(__name__)

class ScratchType(Enum):
    """Types of scratch patterns"""
    BABY = auto()           # Basic forward scratch
    CHIRP = auto()          # Quick forward scratch with release
    SCRIBBLE = auto()       # Rapid back-and-forth
    TRANSFORMER = auto()    # On/off pattern while moving
    CRAB = auto()           # Multiple rapid cuts
    FLARE = auto()          # Multiple crossfader cuts during movement
    ORBIT = auto()          # Circular motion with speed variation
    TEAR = auto()           # Aggressive forward cut
    HYDROPLANE = auto()     # Long smooth movement
    DRILL = auto()          # Staccato pattern

class CrossfaderCurve(Enum):
    """Crossfader curve types"""
    LINEAR = auto()         # Linear fade
    SHARP = auto()          # Sharp cut (DJ style)
    SMOOTH = auto()         # Smooth S-curve
    EXPONENTIAL = auto()    # Exponential curve

@dataclass
class ScratchSegment:
    """Individual segment of a scratch pattern"""
    duration: float         # Duration in seconds
    start_speed: float      # Starting playback speed (-3.0 to +3.0)
    end_speed: float        # Ending playback speed
    crossfader_start: float # Crossfader position at start (0.0 = cut, 1.0 = open)
    crossfader_end: float   # Crossfader position at end
    curve_type: str = "linear"  # Speed curve: linear, exponential, smooth

@dataclass
class ScratchPattern:
    """Complete scratch pattern definition"""
    name: str
    segments: List[ScratchSegment]
    loop: bool = False      # Whether pattern should loop
    bpm_sync: bool = False  # Whether to sync to deck BPM
    total_duration: float = 0.0  # Total pattern duration
    
    def __post_init__(self):
        self.total_duration = sum(seg.duration for seg in self.segments)

@dataclass
class ScratchState:
    """Current state of scratch effect"""
    active: bool = False
    pattern: Optional[ScratchPattern] = None
    start_time: float = 0.0
    current_segment: int = 0
    segment_start_time: float = 0.0
    
    # Playback state
    current_position: float = 0.0  # Current audio position
    base_position: float = 0.0     # Starting audio position
    current_speed: float = 1.0     # Current playback speed
    current_crossfader: float = 1.0  # Current crossfader position
    
    # Loop state
    loop_count: int = 0
    max_loops: int = 1

class ScratchEngine:
    """Professional scratching effects engine"""
    
    # Pre-defined scratch patterns
    PATTERNS = {}
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._state = ScratchState()
        
        # Crossfader settings
        self._crossfader_curve = CrossfaderCurve.SHARP
        self._crossfader_cut_threshold = 0.01  # Below this is considered "cut"
        
        # Initialize built-in patterns
        self._init_patterns()
        
        logger.info("Scratch engine initialized")
    
    def _init_patterns(self):
        """Initialize built-in scratch patterns"""
        
        # Baby scratch - simple forward movement
        self.PATTERNS['baby'] = ScratchPattern(
            name="Baby Scratch",
            segments=[
                ScratchSegment(0.15, 1.5, 1.5, 1.0, 1.0),    # Forward
                ScratchSegment(0.15, -1.5, -1.5, 1.0, 1.0),  # Backward
            ],
            loop=True,
            bpm_sync=True
        )
        
        # Chirp scratch - quick forward with release
        self.PATTERNS['chirp'] = ScratchPattern(
            name="Chirp Scratch",
            segments=[
                ScratchSegment(0.08, 2.0, 2.0, 1.0, 1.0),    # Quick forward
                ScratchSegment(0.05, 0.0, 0.0, 1.0, 1.0),    # Stop
                ScratchSegment(0.12, -1.0, -1.0, 1.0, 1.0),  # Backward
            ],
            loop=True,
            bpm_sync=True
        )
        
        # Transformer scratch - on/off cuts while moving
        self.PATTERNS['transformer_2beat'] = ScratchPattern(
            name="2-Beat Transformer",
            segments=[
                ScratchSegment(0.1, 1.5, 1.5, 1.0, 1.0),     # Forward, open
                ScratchSegment(0.05, 1.5, 1.5, 0.0, 0.0),    # Forward, cut
                ScratchSegment(0.1, 1.5, 1.5, 1.0, 1.0),     # Forward, open
                ScratchSegment(0.05, 1.5, 1.5, 0.0, 0.0),    # Forward, cut
                ScratchSegment(0.2, -1.2, -1.2, 1.0, 1.0),   # Backward, open
            ],
            loop=True,
            bpm_sync=True
        )
        
        # Crab scratch - multiple rapid cuts
        self.PATTERNS['crab'] = ScratchPattern(
            name="Crab Scratch",
            segments=[
                ScratchSegment(0.03, 1.8, 1.8, 1.0, 1.0),    # Forward, open
                ScratchSegment(0.02, 1.8, 1.8, 0.0, 0.0),    # Forward, cut
                ScratchSegment(0.03, 1.8, 1.8, 1.0, 1.0),    # Forward, open
                ScratchSegment(0.02, 1.8, 1.8, 0.0, 0.0),    # Forward, cut
                ScratchSegment(0.03, 1.8, 1.8, 1.0, 1.0),    # Forward, open
                ScratchSegment(0.02, 1.8, 1.8, 0.0, 0.0),    # Forward, cut
                ScratchSegment(0.15, -1.5, -1.5, 1.0, 1.0),  # Backward, open
            ],
            loop=True,
            bpm_sync=True
        )
        
        # Flare scratch - multiple crossfader cuts during one movement
        self.PATTERNS['flare'] = ScratchPattern(
            name="Flare Scratch",
            segments=[
                ScratchSegment(0.05, 1.5, 1.5, 1.0, 1.0),    # Forward start
                ScratchSegment(0.03, 1.5, 1.5, 0.0, 0.0),    # Cut
                ScratchSegment(0.05, 1.5, 1.5, 1.0, 1.0),    # Open
                ScratchSegment(0.03, 1.5, 1.5, 0.0, 0.0),    # Cut
                ScratchSegment(0.04, 1.5, 1.5, 1.0, 1.0),    # Open
                ScratchSegment(0.2, -1.2, -1.2, 1.0, 1.0),   # Backward
            ],
            loop=True,
            bpm_sync=True
        )
        
        # Orbit scratch - circular motion with speed variation
        self.PATTERNS['orbit'] = ScratchPattern(
            name="Orbit Scratch",
            segments=[
                ScratchSegment(0.1, 0.5, 2.0, 1.0, 1.0, "exponential"),  # Accelerate forward
                ScratchSegment(0.1, 2.0, 0.5, 1.0, 1.0, "exponential"),  # Decelerate forward
                ScratchSegment(0.1, -0.5, -2.0, 1.0, 1.0, "exponential"), # Accelerate backward
                ScratchSegment(0.1, -2.0, -0.5, 1.0, 1.0, "exponential"), # Decelerate backward
            ],
            loop=True,
            bpm_sync=True
        )
        
        # Tear scratch - aggressive forward cut
        self.PATTERNS['tear'] = ScratchPattern(
            name="Tear Scratch",
            segments=[
                ScratchSegment(0.05, 0.0, 2.5, 1.0, 1.0, "exponential"),  # Aggressive start
                ScratchSegment(0.1, 2.5, 2.5, 1.0, 1.0),                  # Sustain
                ScratchSegment(0.2, -1.5, -1.5, 1.0, 1.0),                # Return
                ScratchSegment(0.15, 0.0, 0.0, 1.0, 1.0),                 # Pause
            ],
            loop=True,
            bpm_sync=True
        )
        
        logger.info(f"Initialized {len(self.PATTERNS)} scratch patterns")
    
    def start_scratch(self, pattern_name: str, start_position: float, 
                     max_loops: int = 1, bpm: float = 120.0) -> bool:
        """Start scratch effect with specified pattern"""
        try:
            if pattern_name not in self.PATTERNS:
                logger.error(f"Unknown scratch pattern: {pattern_name}")
                return False
            
            pattern = self.PATTERNS[pattern_name]
            
            # Adjust timing for BPM if pattern is BPM-synced
            if pattern.bpm_sync and bpm > 0:
                # Scale pattern duration to match BPM
                # Assume patterns are designed for 120 BPM
                bpm_scale = 120.0 / bpm
                scaled_pattern = self._scale_pattern_for_bpm(pattern, bpm_scale)
            else:
                scaled_pattern = pattern
            
            # Initialize scratch state
            self._state.active = True
            self._state.pattern = scaled_pattern
            self._state.start_time = time.time()
            self._state.current_segment = 0
            self._state.segment_start_time = self._state.start_time
            self._state.base_position = start_position
            self._state.current_position = start_position
            self._state.current_speed = 1.0
            self._state.current_crossfader = 1.0
            self._state.loop_count = 0
            self._state.max_loops = max_loops
            
            logger.info(f"Started '{pattern_name}' scratch at position {start_position}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start scratch: {e}")
            return False
    
    def stop_scratch(self):
        """Stop current scratch effect"""
        self._state.active = False
        self._state.pattern = None
        logger.info("Stopped scratch effect")
    
    def process_scratch(self, audio_chunk: np.ndarray, frames: int, 
                       global_position: int, full_audio: np.ndarray) -> np.ndarray:
        """Process audio chunk with scratch effect"""
        try:
            if not self._state.active or not self._state.pattern:
                return audio_chunk
            
            current_time = time.time()
            
            # Update scratch state
            self._update_scratch_state(current_time)
            
            if not self._state.active:  # Scratch may have ended
                return audio_chunk
            
            # Generate scratched audio
            scratched_audio = self._generate_scratched_audio(
                frames, full_audio, global_position
            )
            
            # Apply crossfader
            final_audio = self._apply_crossfader(scratched_audio, self._state.current_crossfader)
            
            return final_audio
            
        except Exception as e:
            logger.error(f"Error processing scratch: {e}")
            return audio_chunk
    
    def _scale_pattern_for_bpm(self, pattern: ScratchPattern, scale: float) -> ScratchPattern:
        """Scale pattern timing for different BPM"""
        scaled_segments = []
        for segment in pattern.segments:
            scaled_segments.append(ScratchSegment(
                duration=segment.duration * scale,
                start_speed=segment.start_speed,
                end_speed=segment.end_speed,
                crossfader_start=segment.crossfader_start,
                crossfader_end=segment.crossfader_end,
                curve_type=segment.curve_type
            ))
        
        return ScratchPattern(
            name=pattern.name + f" (BPM scaled {scale:.2f})",
            segments=scaled_segments,
            loop=pattern.loop,
            bpm_sync=pattern.bpm_sync
        )
    
    def _update_scratch_state(self, current_time: float):
        """Update scratch state based on current time"""
        if not self._state.pattern:
            return
        
        elapsed = current_time - self._state.start_time
        segment_elapsed = current_time - self._state.segment_start_time
        
        # Check if current segment is complete
        current_segment = self._state.pattern.segments[self._state.current_segment]
        
        if segment_elapsed >= current_segment.duration:
            # Move to next segment
            self._state.current_segment += 1
            
            # Check if pattern is complete
            if self._state.current_segment >= len(self._state.pattern.segments):
                if self._state.pattern.loop and self._state.loop_count < self._state.max_loops - 1:
                    # Loop pattern
                    self._state.current_segment = 0
                    self._state.loop_count += 1
                    logger.debug(f"Looping scratch pattern (loop {self._state.loop_count + 1}/{self._state.max_loops})")
                else:
                    # End scratch
                    self.stop_scratch()
                    return
            
            self._state.segment_start_time = current_time
            segment_elapsed = 0.0
        
        # Update current speed and crossfader based on segment progress
        current_segment = self._state.pattern.segments[self._state.current_segment]
        progress = segment_elapsed / current_segment.duration if current_segment.duration > 0 else 1.0
        progress = min(1.0, progress)
        
        # Apply curve to progress
        curved_progress = self._apply_curve(progress, current_segment.curve_type)
        
        # Interpolate speed
        self._state.current_speed = self._interpolate(
            current_segment.start_speed, 
            current_segment.end_speed, 
            curved_progress
        )
        
        # Interpolate crossfader
        self._state.current_crossfader = self._interpolate(
            current_segment.crossfader_start,
            current_segment.crossfader_end,
            curved_progress
        )
    
    def _apply_curve(self, progress: float, curve_type: str) -> float:
        """Apply curve to progress value"""
        if curve_type == "linear":
            return progress
        elif curve_type == "exponential":
            return progress * progress
        elif curve_type == "smooth":
            # Smoothstep
            return 3 * progress * progress - 2 * progress * progress * progress
        elif curve_type == "sharp":
            return 1.0 if progress > 0.5 else 0.0
        else:
            return progress
    
    def _interpolate(self, start: float, end: float, progress: float) -> float:
        """Linear interpolation"""
        return start + (end - start) * progress
    
    def _generate_scratched_audio(self, frames: int, full_audio: np.ndarray, 
                                global_position: int) -> np.ndarray:
        """Generate scratched audio based on current scratch state"""
        try:
            scratched = np.zeros(frames, dtype=np.float32)
            
            speed = self._state.current_speed
            
            # Calculate audio position movement
            for i in range(frames):
                # Update position based on speed
                position_increment = speed * (1.0 / self.sample_rate)
                self._state.current_position += position_increment
                
                # Clamp position to audio bounds
                audio_length = len(full_audio)
                if self._state.current_position < 0:
                    self._state.current_position = 0
                elif self._state.current_position >= audio_length - 1:
                    self._state.current_position = audio_length - 1
                
                # Get sample with interpolation
                pos_int = int(self._state.current_position)
                pos_frac = self._state.current_position - pos_int
                
                if pos_int < audio_length - 1:
                    # Linear interpolation
                    sample = (1.0 - pos_frac) * full_audio[pos_int] + pos_frac * full_audio[pos_int + 1]
                else:
                    sample = full_audio[pos_int] if pos_int < audio_length else 0.0
                
                scratched[i] = sample
            
            return scratched
            
        except Exception as e:
            logger.error(f"Error generating scratched audio: {e}")
            return np.zeros(frames, dtype=np.float32)
    
    def _apply_crossfader(self, audio: np.ndarray, crossfader_position: float) -> np.ndarray:
        """Apply crossfader effect to audio"""
        try:
            if crossfader_position >= 1.0:
                return audio  # Fully open
            elif crossfader_position <= self._crossfader_cut_threshold:
                return np.zeros_like(audio)  # Cut
            
            # Apply crossfader curve
            if self._crossfader_curve == CrossfaderCurve.LINEAR:
                gain = crossfader_position
            elif self._crossfader_curve == CrossfaderCurve.SHARP:
                # Sharp curve - stays high until close to cut point
                gain = (crossfader_position - self._crossfader_cut_threshold) / (1.0 - self._crossfader_cut_threshold)
                gain = max(0.0, min(1.0, gain))
            elif self._crossfader_curve == CrossfaderCurve.SMOOTH:
                # S-curve
                normalized = (crossfader_position - self._crossfader_cut_threshold) / (1.0 - self._crossfader_cut_threshold)
                normalized = max(0.0, min(1.0, normalized))
                gain = 3 * normalized * normalized - 2 * normalized * normalized * normalized
            else:
                gain = crossfader_position
            
            return audio * gain
            
        except Exception as e:
            logger.error(f"Error applying crossfader: {e}")
            return audio
    
    def is_active(self) -> bool:
        """Check if scratch effect is currently active"""
        return self._state.active
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current scratch state"""
        if not self._state.active or not self._state.pattern:
            return {'active': False}
        
        return {
            'active': True,
            'pattern_name': self._state.pattern.name,
            'current_segment': self._state.current_segment,
            'total_segments': len(self._state.pattern.segments),
            'current_speed': self._state.current_speed,
            'current_crossfader': self._state.current_crossfader,
            'current_position': self._state.current_position,
            'loop_count': self._state.loop_count,
            'max_loops': self._state.max_loops,
            'elapsed_time': time.time() - self._state.start_time if self._state.start_time else 0.0
        }
    
    def get_available_patterns(self) -> List[str]:
        """Get list of available scratch patterns"""
        return list(self.PATTERNS.keys())
    
    def add_custom_pattern(self, name: str, pattern: ScratchPattern):
        """Add custom scratch pattern"""
        self.PATTERNS[name] = pattern
        logger.info(f"Added custom scratch pattern: {name}")
    
    def set_crossfader_curve(self, curve: CrossfaderCurve):
        """Set crossfader curve type"""
        self._crossfader_curve = curve
        logger.debug(f"Set crossfader curve: {curve}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scratch engine statistics"""
        return {
            'sample_rate': self.sample_rate,
            'available_patterns': len(self.PATTERNS),
            'pattern_names': list(self.PATTERNS.keys()),
            'current_state': self.get_current_state(),
            'crossfader_curve': self._crossfader_curve.name,
            'cut_threshold': self._crossfader_cut_threshold
        }

# Factory function
def create_scratch_engine(sample_rate: int = 44100) -> ScratchEngine:
    """Create scratch engine"""
    return ScratchEngine(sample_rate)