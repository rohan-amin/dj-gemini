# Real-time tempo processing for dj-gemini
# Similar to professional DJ software like Mixxx

import numpy as np
import logging
import threading
from typing import Optional, Callable
from collections import deque
import time

# PyRubberBand removed - using native rubberband library for consistency
PYRUBBERBAND_AVAILABLE = False

try:
    import rubberband
    RUBBERBAND_AVAILABLE = True
except ImportError:
    RUBBERBAND_AVAILABLE = False

# SoundTouch is not easily available on PyPI, so we'll focus on PyRubberBand + fallbacks
SOUNDTOUCH_AVAILABLE = False

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class RealtimeTempoProcessor:
    """
    Real-time tempo processing engine similar to professional DJ software.
    Processes audio in small chunks with minimal latency.
    """
    
    def __init__(self, sample_rate: int = 44100, channels: int = 2, 
                 buffer_size: int = 1024, max_buffered_chunks: int = 8):
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = buffer_size
        self.max_buffered_chunks = max_buffered_chunks
        
        # Tempo control
        self._tempo_ratio = 1.0
        self._pitch_shift = 0.0  # semitones
        self._target_tempo_ratio = 1.0
        self._tempo_smoothing = 0.95  # Smooth tempo changes to avoid artifacts
        
        # Audio processing engine
        self._processor = None
        self._initialize_processor()
        
        # Buffer management
        self._input_buffer = deque(maxlen=max_buffered_chunks * 2)
        self._output_buffer = deque(maxlen=max_buffered_chunks)
        self._processing_thread = None
        self._should_process = False
        self._lock = threading.RLock()
        
        # Performance monitoring
        self._processing_times = deque(maxlen=100)
        self._underruns = 0
        
        logger.info(f"Real-time tempo processor initialized: {sample_rate}Hz, {channels}ch, {buffer_size} samples")
    
    def _initialize_processor(self):
        """Initialize the best available audio processing engine"""
        if RUBBERBAND_AVAILABLE:
            try:
                # Native RubberBand - high quality (same as beat_viewer)
                self._processor_type = "rubberband_native"
                self._processor = None  # RubberBand is used as function calls
                logger.info("Using native RubberBand for real-time processing (high quality)")
            except Exception as e:
                logger.warning(f"RubberBand initialization failed: {e}")
                self._processor_type = None
        
        if not self._processor_type:
            # Fallback to simple linear interpolation
            self._processor_type = "linear_interpolation"
            self._processor = None
            logger.info("Using linear interpolation for real-time processing (fast fallback)")
            logger.warning("For better quality, install RubberBand: pip install rubberband")
    
    def start_processing(self):
        """Start the real-time processing thread"""
        if self._processing_thread and self._processing_thread.is_alive():
            return
        
        self._should_process = True
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()
        logger.info("Real-time tempo processing started")
    
    def stop_processing(self):
        """Stop the real-time processing thread"""
        self._should_process = False
        if self._processing_thread:
            self._processing_thread.join(timeout=1.0)
        logger.info("Real-time tempo processing stopped")
    
    def set_tempo_ratio(self, ratio: float):
        """Set tempo ratio (1.0 = original speed, 1.2 = 20% faster)"""
        with self._lock:
            # Clamp to reasonable range (0.5x to 2.0x speed)
            self._target_tempo_ratio = max(0.5, min(2.0, ratio))
    
    def set_pitch_shift(self, semitones: float):
        """Set pitch shift in semitones (-12 to +12)"""
        with self._lock:
            self._pitch_shift = max(-12.0, min(12.0, semitones))
    
    def get_tempo_ratio(self) -> float:
        """Get current tempo ratio"""
        return self._tempo_ratio
    
    def get_pitch_shift(self) -> float:
        """Get current pitch shift"""
        return self._pitch_shift
    
    def process_audio_chunk(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Process a chunk of audio with current tempo/pitch settings.
        This is the main entry point for real-time audio processing.
        """
        if audio_data is None or len(audio_data) == 0:
            return np.zeros((self.buffer_size, self.channels), dtype=np.float32)
        
        start_time = time.time()
        
        try:
            # Add to input buffer
            with self._lock:
                self._input_buffer.append(audio_data.copy())
            
            # Try to get processed output
            processed_audio = self._get_processed_chunk()
            
            # Performance monitoring
            processing_time = time.time() - start_time
            self._processing_times.append(processing_time)
            
            return processed_audio
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            # Return original audio as fallback
            return audio_data
    
    def _get_processed_chunk(self) -> np.ndarray:
        """Get a processed audio chunk from the output buffer"""
        with self._lock:
            if self._output_buffer:
                return self._output_buffer.popleft()
            else:
                # Buffer underrun - return silence and count it
                self._underruns += 1
                if self._underruns % 10 == 1:  # Log occasionally
                    logger.warning(f"Audio buffer underrun #{self._underruns}")
                return np.zeros((self.buffer_size, self.channels), dtype=np.float32)
    
    def _processing_loop(self):
        """Background processing loop that handles audio transformation"""
        logger.info("Real-time processing loop started")
        
        while self._should_process:
            try:
                # Check if we have input to process
                with self._lock:
                    if not self._input_buffer or len(self._output_buffer) >= self.max_buffered_chunks:
                        time.sleep(0.001)  # 1ms sleep to avoid busy waiting
                        continue
                    
                    # Get input chunk
                    input_chunk = self._input_buffer.popleft()
                
                # Process the chunk
                processed_chunk = self._process_chunk(input_chunk)
                
                # Add to output buffer
                with self._lock:
                    if len(self._output_buffer) < self.max_buffered_chunks:
                        self._output_buffer.append(processed_chunk)
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                time.sleep(0.01)  # Brief pause before retrying
        
        logger.info("Real-time processing loop stopped")
    
    def _process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Process a single audio chunk with tempo/pitch transformation"""
        # Smooth tempo changes to avoid artifacts
        with self._lock:
            tempo_diff = self._target_tempo_ratio - self._tempo_ratio
            self._tempo_ratio += tempo_diff * (1.0 - self._tempo_smoothing)
        
        # Skip processing if no change needed
        if abs(self._tempo_ratio - 1.0) < 0.001 and abs(self._pitch_shift) < 0.001:
            return audio_chunk
        
        try:
            if self._processor_type == "rubberband_native":
                return self._process_with_rubberband(audio_chunk)
            elif self._processor_type == "linear_interpolation":
                return self._process_with_linear_interpolation(audio_chunk)
            else:
                return audio_chunk
                
        except Exception as e:
            logger.error(f"Chunk processing failed: {e}")
            return audio_chunk
    
    def _process_with_linear_interpolation(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Process audio chunk using simple linear interpolation (fast fallback)"""
        try:
            # Only process tempo changes (pitch requires more complex processing)
            if abs(self._tempo_ratio - 1.0) < 0.001:
                return audio_chunk
            
            # Linear interpolation for tempo change
            original_length = len(audio_chunk)
            new_length = int(original_length / self._tempo_ratio)
            
            if new_length <= 0:
                return np.zeros_like(audio_chunk)
            
            # Create new indices for interpolation
            old_indices = np.linspace(0, original_length - 1, new_length)
            old_indices = np.clip(old_indices, 0, original_length - 1)
            
            # Interpolate
            if len(audio_chunk.shape) == 1:
                # Mono audio
                processed = np.interp(old_indices, np.arange(original_length), audio_chunk)
            else:
                # Multi-channel audio
                processed = np.zeros((new_length, audio_chunk.shape[1]), dtype=np.float32)
                for channel in range(audio_chunk.shape[1]):
                    processed[:, channel] = np.interp(old_indices, np.arange(original_length), audio_chunk[:, channel])
            
            # Ensure output is same size as input (pad or truncate)
            if len(processed) != original_length:
                if len(processed) > original_length:
                    processed = processed[:original_length]
                else:
                    # Pad with zeros
                    if len(audio_chunk.shape) == 1:
                        padding = np.zeros(original_length - len(processed), dtype=np.float32)
                        processed = np.concatenate([processed, padding])
                    else:
                        padding = np.zeros((original_length - len(processed), audio_chunk.shape[1]), dtype=np.float32)
                        processed = np.concatenate([processed, padding])
            
            return processed.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Linear interpolation processing error: {e}")
            return audio_chunk
    
    def _process_with_rubberband(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Process audio chunk using native RubberBand (same as beat_viewer)"""
        try:
            # Native RubberBand processing - same approach as beat_viewer
            processed = audio_chunk
            
            if abs(self._tempo_ratio - 1.0) > 0.001:
                processed = rubberband.stretch(
                    audio_chunk,
                    rate=self.sample_rate,
                    ratio=self._tempo_ratio,
                    crispness=2,
                    formants=False
                )
            
            # Note: Pitch shifting would require separate processing
            # For now, focus on tempo only (consistent with main audio engine)
            if abs(self._pitch_shift) > 0.001:
                logger.warning("Pitch shifting not yet implemented with native RubberBand")
            
            # Ensure output size matches input for real-time consistency
            if len(processed) != len(audio_chunk):
                if len(processed) > len(audio_chunk):
                    processed = processed[:len(audio_chunk)]
                else:
                    # Pad with zeros
                    padding_shape = list(audio_chunk.shape)
                    padding_shape[0] = len(audio_chunk) - len(processed)
                    padding = np.zeros(padding_shape, dtype=np.float32)
                    processed = np.concatenate([processed, padding])
            
            return processed.astype(np.float32)
            
        except Exception as e:
            logger.error(f"RubberBand processing error: {e}")
            return audio_chunk
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        with self._lock:
            avg_processing_time = np.mean(self._processing_times) if self._processing_times else 0
            max_processing_time = np.max(self._processing_times) if self._processing_times else 0
            
            return {
                'processor_type': self._processor_type,
                'tempo_ratio': self._tempo_ratio,
                'target_tempo_ratio': self._target_tempo_ratio,
                'pitch_shift': self._pitch_shift,
                'input_buffer_size': len(self._input_buffer),
                'output_buffer_size': len(self._output_buffer),
                'avg_processing_time_ms': avg_processing_time * 1000,
                'max_processing_time_ms': max_processing_time * 1000,
                'buffer_underruns': self._underruns,
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'buffer_size': self.buffer_size,
                'rubberband_available': RUBBERBAND_AVAILABLE,
                'scipy_available': SCIPY_AVAILABLE
            }
    
    def reset_stats(self):
        """Reset performance statistics"""
        with self._lock:
            self._processing_times.clear()
            self._underruns = 0
    
    def flush_buffers(self):
        """Clear all audio buffers"""
        with self._lock:
            self._input_buffer.clear()
            self._output_buffer.clear()


# Factory function
def create_realtime_tempo_processor(sample_rate: int = 44100, channels: int = 2, 
                                  buffer_size: int = 1024) -> RealtimeTempoProcessor:
    """Create a real-time tempo processor with optimal settings"""
    return RealtimeTempoProcessor(
        sample_rate=sample_rate,
        channels=channels,
        buffer_size=buffer_size,
        max_buffered_chunks=8  # Keep small for low latency
    )