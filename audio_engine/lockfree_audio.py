# Lock-free audio communication system for dj-gemini
# Replaces blocking locks with atomic operations and lock-free queues

import threading
import queue
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Any, Dict
from enum import Enum
import logging

from .pedalboard_effects import PedalboardAudioProcessor, create_audio_processor

logger = logging.getLogger(__name__)

class AudioCommand(Enum):
    """Commands that can be sent to audio thread"""
    PLAY = "play"
    PAUSE = "pause" 
    STOP = "stop"
    SEEK = "seek"
    SET_VOLUME = "set_volume"
    SET_EQ = "set_eq"
    FADE_EQ = "fade_eq"
    SET_TEMPO = "set_tempo"
    LOAD_AUDIO = "load_audio"

@dataclass
class AudioState:
    """Atomic audio state - updated by audio callback, read by main thread"""
    current_frame: int = 0
    current_beat: float = 0.0
    is_playing: bool = False
    volume: float = 1.0
    eq_low: float = 1.0
    eq_mid: float = 1.0
    eq_high: float = 1.0
    
    # Tempo/BPM state
    current_bpm: float = 120.0
    tempo_ratio: float = 1.0
    
    # Loop state
    loop_active: bool = False
    loop_start_frame: int = 0
    loop_end_frame: int = 0
    loop_repetitions_remaining: int = 0

class LockFreeAudioEngine:
    """Lock-free audio engine using atomic operations and message passing"""
    
    def __init__(self, deck_id: str, sample_rate: int = 44100):
        self.deck_id = deck_id
        self.sample_rate = sample_rate
        
        # Audio data (written by main thread, read by audio callback)
        self._audio_data: Optional[np.ndarray] = None
        self._audio_data_ready = threading.Event()
        
        # Command queue for main thread -> audio thread communication
        # Lock-free queue, audio thread consumes commands
        self._command_queue = queue.Queue(maxsize=100)
        
        # State shared between threads (atomic reads/writes)
        self._shared_state = AudioState()
        self._state_lock = threading.RLock()  # Only for state updates, never held in audio callback
        
        # Audio callback state (only accessed by audio callback)
        self._callback_frame_position = 0
        self._callback_audio_data: Optional[np.ndarray] = None
        self._callback_volume = 1.0
        self._callback_playing = False
        
        # Beat detection data
        self._beat_timestamps: Optional[np.ndarray] = None
        self._original_bpm = 120.0
        
        # Pedalboard audio processor
        self._audio_processor = create_audio_processor(sample_rate)
        
        # Effect state
        self._fade_active = False
        self._fade_start_time = None
        self._fade_duration = 0.0
        self._fade_start_values = {}
        self._fade_target_values = {}
        
    def send_command(self, command: AudioCommand, params: Dict[str, Any] = None) -> bool:
        """Send command to audio thread (non-blocking)"""
        try:
            self._command_queue.put_nowait((command, params or {}))
            return True
        except queue.Full:
            logger.warning(f"Deck {self.deck_id} - Command queue full, dropping command: {command}")
            return False
    
    def get_current_state(self) -> AudioState:
        """Get current audio state (thread-safe read)"""
        # Fast read without locking - audio callback updates atomically
        return AudioState(
            current_frame=self._shared_state.current_frame,
            current_beat=self._shared_state.current_beat,
            is_playing=self._shared_state.is_playing,
            volume=self._shared_state.volume,
            eq_low=self._shared_state.eq_low,
            eq_mid=self._shared_state.eq_mid,
            eq_high=self._shared_state.eq_high,
            current_bpm=self._shared_state.current_bpm,
            tempo_ratio=self._shared_state.tempo_ratio,
            loop_active=self._shared_state.loop_active,
            loop_start_frame=self._shared_state.loop_start_frame,
            loop_end_frame=self._shared_state.loop_end_frame,
            loop_repetitions_remaining=self._shared_state.loop_repetitions_remaining
        )
    
    def load_audio_data(self, audio_data: np.ndarray, beat_timestamps: np.ndarray, bpm: float):
        """Load audio data (called from main thread)"""
        logger.info(f"Deck {self.deck_id} - Loading audio data: {len(audio_data)} samples, BPM: {bpm}")
        
        # Store data that audio callback will use
        self._audio_data = audio_data.copy()
        self._beat_timestamps = beat_timestamps.copy()
        self._original_bpm = bpm
        
        # Signal audio callback that data is ready
        self._audio_data_ready.set()
        
        # Send load command to audio thread
        self.send_command(AudioCommand.LOAD_AUDIO, {
            'total_frames': len(audio_data),
            'bpm': bpm
        })
    
    def play(self, start_frame: int = 0):
        """Start playback"""
        self.send_command(AudioCommand.PLAY, {'start_frame': start_frame})
    
    def pause(self):
        """Pause playback"""
        self.send_command(AudioCommand.PAUSE)
    
    def stop(self):
        """Stop playback"""
        self.send_command(AudioCommand.STOP)
    
    def seek(self, frame: int):
        """Seek to specific frame"""
        self.send_command(AudioCommand.SEEK, {'frame': frame})
    
    def set_volume(self, volume: float):
        """Set volume (0.0 to 1.0)"""
        self.send_command(AudioCommand.SET_VOLUME, {'volume': volume})
    
    def set_eq(self, low: float = None, mid: float = None, high: float = None):
        """Set EQ instantly"""
        params = {}
        if low is not None: params['low'] = low
        if mid is not None: params['mid'] = mid  
        if high is not None: params['high'] = high
        self.send_command(AudioCommand.SET_EQ, params)
    
    def fade_eq(self, target_low: float = None, target_mid: float = None, 
                target_high: float = None, duration_seconds: float = 1.0):
        """Fade EQ over time"""
        params = {'duration': duration_seconds}
        if target_low is not None: params['target_low'] = target_low
        if target_mid is not None: params['target_mid'] = target_mid
        if target_high is not None: params['target_high'] = target_high
        self.send_command(AudioCommand.FADE_EQ, params)
    
    def audio_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """Audio callback - NEVER BLOCKS, lock-free operation"""
        
        # Process all pending commands (non-blocking)
        self._process_commands()
        
        # Update effects (fades, etc.)
        self._update_effects()
        
        # Generate audio
        if self._callback_playing and self._callback_audio_data is not None:
            audio_chunk = self._get_audio_chunk(frames)
            
            # Apply effects
            audio_chunk = self._apply_eq(audio_chunk)
            
            # Apply volume
            audio_chunk *= self._callback_volume
            
            # Output
            if audio_chunk.ndim == 1:
                outdata[:, 0] = audio_chunk
            else:
                outdata[:] = audio_chunk
                
            # Update position
            self._callback_frame_position += frames
            
        else:
            # Silence
            outdata.fill(0)
        
        # Update shared state (atomic write)
        self._update_shared_state()
    
    def get_current_eq(self) -> Dict[str, float]:
        """Get current EQ settings"""
        return self._audio_processor.get_eq()
    
    def _process_commands(self):
        """Process all pending commands (called from audio callback)"""
        processed = 0
        while processed < 10:  # Limit processing to prevent audio dropouts
            try:
                command, params = self._command_queue.get_nowait()
                self._handle_command(command, params)
                processed += 1
            except queue.Empty:
                break
    
    def _handle_command(self, command: AudioCommand, params: Dict[str, Any]):
        """Handle single command (called from audio callback)"""
        if command == AudioCommand.PLAY:
            start_frame = params.get('start_frame', 0)
            self._callback_frame_position = start_frame
            self._callback_playing = True
            
        elif command == AudioCommand.PAUSE:
            self._callback_playing = False
            
        elif command == AudioCommand.STOP:
            self._callback_playing = False
            self._callback_frame_position = 0
            
        elif command == AudioCommand.SEEK:
            frame = params.get('frame', 0)
            self._callback_frame_position = max(0, frame)
            
        elif command == AudioCommand.SET_VOLUME:
            self._callback_volume = params.get('volume', 1.0)
            
        elif command == AudioCommand.SET_EQ:
            # Update Pedalboard processor EQ
            self._audio_processor.set_eq(
                low=params.get('low'),
                mid=params.get('mid'),
                high=params.get('high')
            )
            
        elif command == AudioCommand.FADE_EQ:
            self._start_eq_fade(params)
            
        elif command == AudioCommand.LOAD_AUDIO:
            # Audio data is already loaded, just copy reference
            self._callback_audio_data = self._audio_data
    
    def _get_audio_chunk(self, frames: int) -> np.ndarray:
        """Get audio chunk for current position (called from audio callback)"""
        if self._callback_audio_data is None:
            return np.zeros(frames, dtype=np.float32)
        
        start_frame = self._callback_frame_position
        end_frame = start_frame + frames
        
        # Handle end of track
        if start_frame >= len(self._callback_audio_data):
            self._callback_playing = False
            return np.zeros(frames, dtype=np.float32)
        
        # Handle partial read at end
        if end_frame > len(self._callback_audio_data):
            available_frames = len(self._callback_audio_data) - start_frame
            chunk = np.zeros(frames, dtype=np.float32)
            chunk[:available_frames] = self._callback_audio_data[start_frame:start_frame + available_frames]
            return chunk
        
        return self._callback_audio_data[start_frame:end_frame].copy()
    
    def _apply_eq(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Apply EQ to audio chunk using Pedalboard (called from audio callback)"""
        try:
            return self._audio_processor.process_audio(audio_chunk)
        except Exception as e:
            logger.error(f"EQ processing failed: {e}")
            return audio_chunk
    
    def _update_effects(self):
        """Update time-based effects like fades (called from audio callback)"""
        if self._fade_active and self._fade_start_time is not None:
            elapsed = time.time() - self._fade_start_time
            if elapsed >= self._fade_duration:
                # Fade complete
                self._fade_active = False
                # Apply final target values to processor
                self._audio_processor.set_eq(
                    low=self._fade_target_values.get('low'),
                    mid=self._fade_target_values.get('mid'),
                    high=self._fade_target_values.get('high')
                )
            else:
                # Interpolate fade
                progress = elapsed / self._fade_duration
                current_values = {}
                for param, target in self._fade_target_values.items():
                    if param in self._fade_start_values:
                        start_val = self._fade_start_values[param]
                        current_val = start_val + (target - start_val) * progress
                        current_values[param] = current_val
                
                # Apply interpolated values to processor
                self._audio_processor.set_eq(
                    low=current_values.get('low'),
                    mid=current_values.get('mid'),
                    high=current_values.get('high')
                )
    
    def _start_eq_fade(self, params: Dict[str, Any]):
        """Start EQ fade (called from audio callback)"""
        self._fade_active = True
        self._fade_start_time = time.time()
        self._fade_duration = params.get('duration', 1.0)
        
        # Store current values as start
        self._fade_start_values = self._audio_processor.get_eq().copy()
        
        # Store targets
        self._fade_target_values = {}
        if 'target_low' in params: self._fade_target_values['low'] = params['target_low']
        if 'target_mid' in params: self._fade_target_values['mid'] = params['target_mid']  
        if 'target_high' in params: self._fade_target_values['high'] = params['target_high']
    
    def _update_shared_state(self):
        """Update shared state from audio callback (atomic write)"""
        # Calculate current beat
        current_beat = 0.0
        if self._beat_timestamps is not None and len(self._beat_timestamps) > 0:
            current_time = self._callback_frame_position / self.sample_rate
            current_beat = float(np.searchsorted(self._beat_timestamps, current_time, side='left'))
        
        # Get current EQ from processor
        current_eq = self._audio_processor.get_eq()
        
        # Atomic state update (no locks needed for simple assignments)
        self._shared_state.current_frame = self._callback_frame_position
        self._shared_state.current_beat = current_beat
        self._shared_state.is_playing = self._callback_playing
        self._shared_state.volume = self._callback_volume
        self._shared_state.eq_low = current_eq.get('low', 1.0)
        self._shared_state.eq_mid = current_eq.get('mid', 1.0)
        self._shared_state.eq_high = current_eq.get('high', 1.0)
        self._shared_state.current_bpm = self._original_bpm