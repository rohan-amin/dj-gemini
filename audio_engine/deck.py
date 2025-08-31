# dj-gemini/audio_engine/deck.py
# Hybrid Ring Buffer + Professional EQ + Seamless Loop Architecture

import os
import threading
import time 
import queue 
import json 
import logging
from concurrent.futures import ThreadPoolExecutor
logger = logging.getLogger(__name__)
import essentia.standard as es
import numpy as np
import sounddevice as sd
import librosa
import threading
import time
from scipy import signal
import scipy.signal as sps
from scipy.signal import lfilter_zi
from config import EQ_SMOOTHING_MS
from .realtime_tempo import create_realtime_tempo_processor
from .ring_buffer import RingBuffer as ExternalRingBuffer
from .professional_eq import ToneEQ3, IsolatorEQ
# REMOVED: Legacy LoopManager import - using frame-accurate system only
from .beat_manager import BeatManager
from .scheduling.musical_timing_system import MusicalTimingSystem

# Check for RubberBand availability
try:
    import rubberband_ctypes
    RUBBERBAND_STREAMING_AVAILABLE = True
    logger.info("RubberBand streaming ctypes wrapper loaded successfully!")
except ImportError as e:
    RUBBERBAND_STREAMING_AVAILABLE = False
    logger.warning(f"RubberBand streaming not available: {e}")

# PyRubberBand removed - using native rubberband library for consistency with beat_viewer
PYRUBBERBAND_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import rubberband
    RUBBERBAND_AVAILABLE = True
except ImportError:
    RUBBERBAND_AVAILABLE = False

PITCH_PRESERVATION_AVAILABLE = RUBBERBAND_STREAMING_AVAILABLE or LIBROSA_AVAILABLE or RUBBERBAND_AVAILABLE

# Command constants for the Deck's internal audio thread
DECK_CMD_LOAD_AUDIO = "LOAD_AUDIO"
DECK_CMD_PLAY = "PLAY" 
DECK_CMD_PAUSE = "PAUSE"
DECK_CMD_STOP = "STOP"   
DECK_CMD_SEEK = "SEEK" 
DECK_CMD_ACTIVATE_LOOP = "ACTIVATE_LOOP"
DECK_CMD_DEACTIVATE_LOOP = "DEACTIVATE_LOOP"
DECK_CMD_SHUTDOWN = "SHUTDOWN"
DECK_CMD_STOP_AT_BEAT = "STOP_AT_BEAT"
DECK_CMD_SET_TEMPO = "SET_TEMPO"
DECK_CMD_SET_PITCH = "SET_PITCH"
DECK_CMD_SET_VOLUME = "SET_VOLUME"
DECK_CMD_FADE_VOLUME = "FADE_VOLUME"

# --- Scratch cut effect config ---
SCRATCH_CUT_FRAMES = 0  # Disabled for debugging

class ToneEQ3:
    """
    Serial 3-band tone EQ: low shelf, mid peaking, high shelf.
    - Stateful across blocks (per-stage zi)
    - Smooth parameter updates via short crossfade (default 10 ms)
    - Designed for per-stem shaping (not a kill EQ)
    """
    def __init__(self, sample_rate: int, xfade_ms: float = 10.0):
        self.sr = int(sample_rate)
        self._xfade = int(max(1, xfade_ms * 1e-3 * self.sr))
        self._left = 0
        self._cur = self._design(0.0, 0.0, 0.0, 200, 1000, 0.7, 4000)
        self._pend = None

    def set_params_db(self, low_db: float, mid_db: float, high_db: float,
                      f_low: float = 200, f_mid: float = 1000, q_mid: float = 0.7, f_high: float = 4000):
        self._pend = self._design(low_db, mid_db, high_db, f_low, f_mid, q_mid, f_high)
        self._left = self._xfade

    def process_block(self, x: np.ndarray) -> np.ndarray:
        # Check if EQ is flat (all gains near 0dB) - bypass processing if so
        if self._is_flat():
            return x.astype(np.float32).reshape(-1, 1)
        
        xin = x.astype(np.float64, copy=False).reshape(-1)
        if self._pend is not None and self._left > 0:
            n = len(xin); nxf = min(self._left, n)
            y0 = self._run(xin, self._cur, copy_state=True)
            y1 = self._run(xin, self._pend, copy_state=True)
            if nxf > 0:
                w = np.linspace(0.0, 1.0, nxf, dtype=np.float64)
                y = y0.copy()
                y[:nxf] = (1.0 - w) * y0[:nxf] + w * y1[:nxf]
                if nxf < n: y[nxf:] = y1[nxf:]
                self._left -= nxf
            else:
                y = y1
            if self._left <= 0:
                self._cur = self._pend; self._pend = None
        else:
            y = self._run(xin, self._cur, copy_state=False)
        return y.astype(np.float32).reshape(-1, 1)
    
    def _is_flat(self) -> bool:
        """Check if current EQ settings are effectively flat (bypass worthy)"""
        # Check current configuration for flat response
        if 'gains' not in self._cur:
            return True
        gains = self._cur['gains']  # [low_db, mid_db, high_db]
        return all(abs(gain) < 0.01 for gain in gains)  # All gains within 0.01dB of flat

    # ---- internals ----
    def _design(self, low_db, mid_db, high_db, f_low, f_mid, q_mid, f_high):
        def shelf_low(db, fc):
            A = 10**(db/40.0)
            w0 = 2*np.pi*fc/self.sr; cosw = np.cos(w0); sinw = np.sin(w0)
            S = np.sqrt(2)
            alpha = sinw/2*np.sqrt((A+1/A)*(1/S -1)+2)
            b0 =    A*((A+1) - (A-1)*cosw + 2*np.sqrt(A)*alpha)
            b1 =  2*A*((A-1) - (A+1)*cosw)
            b2 =    A*((A+1) - (A-1)*cosw - 2*np.sqrt(A)*alpha)
            a0 =        (A+1) + (A-1)*cosw + 2*np.sqrt(A)*alpha
            a1 =   -2*((A-1) + (A+1)*cosw)
            a2 =        (A+1) + (A-1)*cosw - 2*np.sqrt(A)*alpha
            return sps.tf2sos([b0/a0,b1/a0,b2/a0],[1.0,a1/a0,a2/a0])
        def peak(db, fc, Q):
            A = 10**(db/40.0); w0 = 2*np.pi*fc/self.sr
            alpha = np.sin(w0)/(2*Q); cosw = np.cos(w0)
            b0 = 1 + alpha*A; b1 = -2*cosw; b2 = 1 - alpha*A
            a0 = 1 + alpha/A; a1 = -2*cosw; a2 = 1 - alpha/A
            return sps.tf2sos([b0/a0,b1/a0,b2/a0],[1.0,a1/a0,a2/a0])
        def shelf_high(db, fc):
            A = 10**(db/40.0)
            w0 = 2*np.pi*fc/self.sr; cosw = np.cos(w0); sinw = np.sin(w0)
            S = np.sqrt(2)
            alpha = sinw/2*np.sqrt((A+1/A)*(1/S -1)+2)
            b0 =    A*((A+1) + (A-1)*cosw + 2*np.sqrt(A)*alpha)
            b1 = -2*A*((A-1) + (A+1)*cosw)
            b2 =    A*((A+1) + (A-1)*cosw - 2*np.sqrt(A)*alpha)
            a0 =        (A+1) - (A-1)*cosw + 2*np.sqrt(A)*alpha
            a1 =    2*((A-1) - (A+1)*cosw)
            a2 =        (A+1) - (A-1)*cosw - 2*np.sqrt(A)*alpha
            return sps.tf2sos([b0/a0,b1/a0,b2/a0],[1.0,a1/a0,a2/a0])

        sosL = shelf_low (low_db,  f_low)
        sosM = peak      (mid_db,  f_mid, q_mid)
        sosH = shelf_high(high_db, f_high)

        return {
            "sos": [sosL, sosM, sosH],
            "zi":  [sps.sosfilt_zi(sosL)*0.0, sps.sosfilt_zi(sosM)*0.0, sps.sosfilt_zi(sosH)*0.0],
            "gains": [low_db, mid_db, high_db]  # Store gains for bypass check
        }

    def _run(self, xin: np.ndarray, cfg: dict, copy_state: bool) -> np.ndarray:
        y = xin
        for i, sos in enumerate(cfg["sos"]):
            zi = cfg["zi"][i].copy() if copy_state else cfg["zi"][i]
            y, zi = sps.sosfilt(sos, y, zi=zi)
            if not copy_state:
                cfg["zi"][i] = zi
        return y

class RingBuffer:
    """Thread-safe ring buffer for audio data"""
    def __init__(self, capacity_frames, channels=1):
        self.channels = channels
        self.cap = int(capacity_frames)
        self.buf = np.zeros((self.cap, channels), dtype=np.float32)
        self.w = 0  # write index
        self.r = 0  # read index
        self.size = 0
        self.lock = threading.Lock()

    def write(self, x):
        """Write audio data to ring buffer. Returns number of frames written."""
        n = x.shape[0]
        with self.lock:
            to_write = min(n, self.cap - self.size)
            if to_write <= 0:
                return 0
            first = min(to_write, self.cap - self.w)
            self.buf[self.w:self.w+first] = x[:first]
            second = to_write - first
            if second:
                self.buf[0:second] = x[first:first+second]
            self.w = (self.w + to_write) % self.cap
            self.size += to_write
            return to_write

    def read(self, n):
        """Read n frames from ring buffer. Returns (data, frames_read)."""
        out = np.zeros((n, self.channels), dtype=np.float32)
        with self.lock:
            to_read = min(n, self.size)
            if to_read <= 0:
                return out, 0
            first = min(to_read, self.cap - self.r)
            out[:first] = self.buf[self.r:self.r+first]
            second = to_read - first
            if second:
                out[first:first+second] = self.buf[:second]
            self.r = (self.r + to_read) % self.cap
            self.size -= to_read
            return out, to_read
            
    def available_data(self):
        """Get number of frames available for reading"""
        with self.lock:
            return self.size
            
    def available_space(self):
        """Get number of frames available for writing"""
        with self.lock:
            return self.cap - self.size


class Deck:
    def __init__(self, deck_id, analyzer_instance, engine_instance=None):
        self.deck_id = deck_id
        self.analyzer = analyzer_instance
        self.engine = engine_instance  # Reference to the engine for global samples
        logger.debug(f"Deck {self.deck_id} - Initializing...")

        self.filepath = None
        self.sample_rate = 44100  # Default sample rate
        self.beat_timestamps = np.array([])
        self.bpm = 0.0
        self.total_frames = 0 
        self.cue_points = {} 
        
        # === RING BUFFER ARCHITECTURE FOR STEREO AUDIO ===
        # Ring buffer for real-time audio (5 seconds at 44.1kHz stereo)
        self.ring_buffer = RingBuffer(capacity_frames=5 * 44100, channels=2)
        
        # Producer thread for heavy processing
        self._producer_thread = None
        self._producer_running = False
        self._producer_stop_event = threading.Event()
        
        # === PROFESSIONAL STEREO EQ SYSTEM ===
        # Per-stem tone EQs for musical shaping (stereo)
        self.stem_tone_eqs = {
            'vocals': ToneEQ3(self.sample_rate),
            'drums': ToneEQ3(self.sample_rate),
            'bass': ToneEQ3(self.sample_rate),
            'other': ToneEQ3(self.sample_rate)
        }
        
        # Master isolator EQ for kill switches (stereo)
        self.master_isolator = IsolatorEQ(self.sample_rate, f_lo=200.0, f_hi=2000.0)
        
        # Stem EQ control
        self.stem_eq_enabled = {'vocals': False, 'drums': False, 'bass': False, 'other': False}
        self.stem_volumes = {'vocals': 1.0, 'drums': 1.0, 'bass': 1.0, 'other': 1.0}
        self.master_eq_enabled = True
        
        # === STEREO AUDIO DATA ===
        self.audio_data = None  # Will be stereo (frames, 2)
        self.stems_available = False
        self.stem_data = {}  # Each stem will be stereo
        
        # Data for Audio Thread's direct use by its callback
        self.audio_thread_data = None
        self.audio_thread_sample_rate = 0
        self.audio_thread_total_samples = 0
        self.audio_thread_current_frame = 0 # Frame counter for audio thread's playback logic

        self.command_queue = queue.Queue()
        self.audio_thread_stop_event = threading.Event()
        self.audio_thread = threading.Thread(target=self._audio_management_loop, daemon=True)
        
        # Thread pool for heavy processing tasks
        self._processing_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"Deck{deck_id}")

        self._stream_lock = threading.Lock() 
        self._current_playback_frame_for_display = 0 
        self._user_wants_to_play = False 
        self._is_actually_playing_stream_state = False 
        self.seek_in_progress_flag = False 

                # === REMOVED: Legacy loop manager initialization ===
        # Now using frame-accurate loop system only
        
        # === CENTRALIZED BEAT MANAGEMENT ===
        self.beat_manager = BeatManager(self)
        
        # === MUSICAL TIMING SYSTEM (New Frame-Accurate Scheduling) ===
        try:
            self.musical_timing_system = MusicalTimingSystem(self)
            logger.info(f"Deck {self.deck_id} - Musical timing system initialized")
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - Failed to initialize musical timing system: {e}")
            self.musical_timing_system = None
        
        # Note: Engine reference will be set when deck is added to engine
        
        # --- Synchronized beat state (DEPRECATED - use beat_manager instead) ---
        self._last_synchronized_beat = 0

        # Loop queue handled by LoopManager

        # Add tempo state
        self._playback_tempo = 1.0  # 1.0 = original speed
        self._original_bpm = 0.0    # Store original BPM for calculations

        # Tempo ramp state now handled by BeatManager
        # Old fields removed to eliminate confusion


        # Add volume control
        self._volume = 1.0  # 0.0 to 1.0
        self._fade_target_volume = None
        self._fade_start_volume = None
        self._fade_start_time = None
        self._fade_duration = None

        # Add EQ control
        self._eq_low = 1.0    # 0.0 to 2.0 (cut/boost)
        self._eq_mid = 1.0
        self._eq_high = 1.0
        self._eq_filters_initialized = False
        self._eq_low_filter = None
        self._eq_mid_filter = None
        self._eq_high_filter = None
        
        # Add EQ smoothing to reduce artifacts
        self._eq_smoothing_frames = max(1, int(self.sample_rate * (EQ_SMOOTHING_MS / 1000.0)))  # 0.5ms smoothing window
        self._eq_smoothing_counter = 0
        self._eq_smoothing_active = False
        self._eq_smoothing_start = {'low': 1.0, 'mid': 1.0, 'high': 1.0}
        self._eq_smoothing_target = {'low': 1.0, 'mid': 1.0, 'high': 1.0}
        
        # Add filter state for continuous filtering
        self._eq_low_state = None
        self._eq_mid_state = None
        self._eq_high_state = None
        
        # Add EQ transition smoothing to eliminate artifacts
        # self._eq_transition_frames = 64  # Short transition to eliminate click
        # self._eq_transition_counter = 0
        # self._eq_previous_low = 1.0
        # self._eq_previous_mid = 1.0
        # self._eq_previous_high = 1.0
        
        # EQ fade state (similar to volume fade)
        self._fade_target_eq_low = None
        self._fade_target_eq_mid = None
        self._fade_target_eq_high = None
        self._fade_start_eq_low = None
        self._fade_start_eq_mid = None
        self._fade_start_eq_high = None
        self._fade_start_time_eq = None
        self._fade_duration_eq = None
        self._fade_pending_eq = False

        # Buffer-synchronized EQ changes to eliminate clicking
        self._eq_pending_low = None
        self._eq_pending_mid = None
        self._eq_pending_high = None
        self._eq_pending_change = False
        self.playback_stream_obj = None

        # Add to Deck class initialization
        self._tempo_cache = {}  # Cache processed audio for different tempos

        # Real-time tempo processing
        self._realtime_tempo_processor = None
        self._use_realtime_tempo = True  # Flag to enable/disable real-time tempo

        # Phase offset storage for bpm_match
        self._pending_phase_offset_beats = 0.0
        self._phase_offset_applied = False

        self.enable_hard_seek_on_loop = True  # PATCH: Toggle for hard seek buffer zeroing on loop activation
        self._just_activated_loop_flush_count = 0  # PATCH: For double-flush after loop activation

        # --- Scratch state (isolated) ---
        self._scratch_active = False
        self._scratch_pattern = []
        self._scratch_duration_frames = 0
        self._scratch_start_frame = 0
        self._scratch_start_time = 0.0
        self._scratch_pattern_index = 0
        self._scratch_pattern_frames = int(0.5 * self.sample_rate)  # 0.5s per segment for dramatically slower hand movement
        self._scratch_elapsed_frames = 0
        self._scratch_window_frames = int(2.0 * self.sample_rate)  # 2.0s window for much larger movement
        self._scratch_pointer = 0.0
        self._scratch_cut_remaining = 0

        # --- Stem EQ System ---
        self.stem_data = {}  # {'vocals': np.array, 'drums': np.array, ...}
        self.stem_volumes = {'vocals': 1.0, 'drums': 1.0, 'bass': 1.0, 'other': 1.0}
        self.stem_eq_enabled = {'vocals': False, 'drums': False, 'bass': False, 'other': False}
        self.stem_tone_eqs = {}  # stem_name -> ToneEQ3 instance
        self.stems_available = False
        self.enable_stem_eqs = True  # Global enable flag for stem EQ processing
        
        # Initialize per-stem EQ processors (will be populated when sample rate is known)
        self.STEM_NAMES = ['vocals', 'drums', 'bass', 'other']

        # --- Ring Buffer Architecture ---
        self.RING_BUFFER_SIZE = 8192  # 8k frames buffer for tighter control and fewer artifacts
        self.out_ring = None  # Will be initialized when sample rate is known
        self.master_isolator = None  # Will be initialized when sample rate is known
        
        # Producer thread for filling ring buffer
        self._producer_error = None
        self.device_output_channels = 1  # Default to mono, updated when stream starts
        
        # RubberBand streaming processor
        self.rubberband_stretcher = None
        self._pending_reinit = False
        self._pending_disable = False
        self._pending_time_ratio = None
        self._rb_lock = threading.Lock()
        
        # Playback position for producer thread
        self.playback_position = 0.0
        
        # Current tempo ratio for processing
        self.current_tempo_ratio = 1.0
        
        # Pitch preservation setting - Always enabled for consistent RubberBand processing
        self.preserve_pitch_enabled = True  # Always use RubberBand for consistency

        self.audio_thread.start()
        logger.debug(f"Deck {self.deck_id} - Initialized and audio thread started.")

    def _wait_for_ring_buffer_ready(self):
        """Wait for ring buffer to have sufficient audio data before starting stream"""
        if not self.out_ring:
            return
            
        # Wait for at least 2 chunks of audio data to prevent startup artifacts
        target_frames = 8192  # 2 chunks of 4096 frames
        max_wait_time = 1.0  # Maximum 1 second wait
        
        start_time = time.time()
        while (self.out_ring.available_read() < target_frames and 
               time.time() - start_time < max_wait_time):
            time.sleep(0.001)  # 1ms sleep for tight polling
            
        available = self.out_ring.available_read()
        logger.info(f"Deck {self.deck_id} - Ring buffer ready: {available} frames available")
        
        if available < target_frames:
            logger.warning(f"Deck {self.deck_id} - Ring buffer not fully ready ({available} frames), but proceeding")

    def _clear_ring_buffer_for_position_jump(self, reason="position jump"):
        """
        Clear ring buffer to prevent audio artifacts during position jumps.
        
        This is critical for preventing "double beat" and other artifacts when:
        - Loop activation causes immediate position jumps
        - Seeking to new positions
        - Starting playback at specific beats
        """
        if self.out_ring:
            frames_cleared = self.out_ring.available_read()
            self.out_ring.clear()
            
            # Also clear any pending audio data to prevent artifacts
            if hasattr(self, '_pending_out') and self._pending_out is not None:
                self._pending_out = None
                logger.debug(f"Deck {self.deck_id} - Pending audio data cleared for {reason}")
            
            logger.info(f"Deck {self.deck_id} - Ring buffer cleared for {reason} ({frames_cleared} frames)")
        else:
            logger.debug(f"Deck {self.deck_id} - Ring buffer clear requested for {reason} but no ring buffer active")

    # === PROFESSIONAL STEREO EQ API METHODS ===
    
    def set_stem_eq(self, stem_name, low_db=0.0, mid_db=0.0, high_db=0.0, enabled=None):
        """Set per-stem 3-band EQ parameters with smooth transitions"""
        if stem_name not in self.stem_tone_eqs:
            logger.warning(f"Deck {self.deck_id} - Unknown stem: {stem_name}")
            return False
            
        # Update EQ parameters with smooth crossfade
        self.stem_tone_eqs[stem_name].set_params_db(low_db, mid_db, high_db)
        
        # Update enabled state if specified
        if enabled is not None:
            self.stem_eq_enabled[stem_name] = bool(enabled)
            
        logger.info(f"Deck {self.deck_id} - Stem EQ updated: {stem_name} L:{low_db:+.1f}dB M:{mid_db:+.1f}dB H:{high_db:+.1f}dB enabled:{self.stem_eq_enabled[stem_name]}")
        return True
    
    def set_stem_volume(self, stem_name, volume):
        """Set per-stem volume (0.0 to 1.0)"""
        if stem_name not in self.stem_volumes:
            logger.warning(f"Deck {self.deck_id} - Unknown stem: {stem_name}")
            return False
            
        self.stem_volumes[stem_name] = max(0.0, min(1.0, float(volume)))
        logger.info(f"Deck {self.deck_id} - Stem volume updated: {stem_name} = {self.stem_volumes[stem_name]:.2f}")
        return True
    
    def set_master_eq(self, low_gain=1.0, mid_gain=1.0, high_gain=1.0, enabled=None):
        """Set master isolator EQ gains (0.0 = kill, 1.0 = full)"""
        self.master_isolator.set_gains(low_gain, mid_gain, high_gain)
        
        if enabled is not None:
            self.master_eq_enabled = bool(enabled)
            self.master_isolator.set_enabled(self.master_eq_enabled)
            
        logger.info(f"Deck {self.deck_id} - Master EQ updated: L:{low_gain:.2f} M:{mid_gain:.2f} H:{high_gain:.2f} enabled:{self.master_eq_enabled}")
        return True
    
    def enable_stem_eq(self, stem_name, enabled):
        """Enable/disable per-stem EQ with smooth bypass"""
        if stem_name not in self.stem_eq_enabled:
            logger.warning(f"Deck {self.deck_id} - Unknown stem: {stem_name}")
            return False
            
        self.stem_eq_enabled[stem_name] = bool(enabled)
        
        # If disabling, set EQ to flat for smooth bypass
        if not enabled:
            self.stem_tone_eqs[stem_name].set_params_db(0.0, 0.0, 0.0)
            
        logger.info(f"Deck {self.deck_id} - Stem EQ {stem_name}: {'enabled' if enabled else 'disabled'}")
        return True
    
    def _process_stereo_stem_mix(self, target_frames):
        """
        Process stems with professional stereo EQ system
        
        Returns:
            np.ndarray: Stereo audio chunk, shape (frames, 2)
        """
        if not self.stems_available or not self.stem_data:
            return None
            
        # Get current position
        start_frame = int(self.audio_thread_current_frame)
        
        # Initialize stereo mix
        mix_chunk = np.zeros((target_frames, 2), dtype=np.float32)
        
        # Process each stem with individual EQ and volume
        for stem_name, stem_audio in self.stem_data.items():
            if stem_name not in self.stem_volumes:
                continue
                
            # Extract stem chunk
            end_frame = start_frame + target_frames
            if start_frame >= len(stem_audio):
                continue
                
            # Get stereo stem data
            if end_frame > len(stem_audio):
                # Handle end of track
                available_frames = len(stem_audio) - start_frame
                if available_frames <= 0:
                    continue
                stem_chunk = np.zeros((target_frames, 2), dtype=np.float32)
                if stem_audio.ndim == 1:
                    # Convert mono to stereo
                    mono_data = stem_audio[start_frame:start_frame + available_frames]
                    stem_chunk[:available_frames, 0] = mono_data
                    stem_chunk[:available_frames, 1] = mono_data
                else:
                    # Already stereo
                    stem_chunk[:available_frames] = stem_audio[start_frame:start_frame + available_frames]
            else:
                # Normal case
                if stem_audio.ndim == 1:
                    # Convert mono to stereo
                    mono_data = stem_audio[start_frame:end_frame]
                    stem_chunk = np.column_stack([mono_data, mono_data])
                else:
                    # Already stereo
                    stem_chunk = stem_audio[start_frame:end_frame]
            
            # Apply per-stem ToneEQ3 (musical shaping)
            if self.stem_eq_enabled.get(stem_name, False) and stem_name in self.stem_tone_eqs:
                try:
                    stem_chunk = self.stem_tone_eqs[stem_name].process_block(stem_chunk)
                except Exception as e:
                    logger.warning(f"Deck {self.deck_id} - Stem EQ error for {stem_name}: {e}")
            
            # Apply stem volume
            stem_volume = self.stem_volumes.get(stem_name, 1.0)
            stem_chunk *= stem_volume
            
            # Add to stereo mix
            mix_chunk += stem_chunk
        
        # Apply master isolator EQ (kill switches)
        if self.master_eq_enabled:
            try:
                mix_chunk = self.master_isolator.process_block(mix_chunk)
            except Exception as e:
                logger.warning(f"Deck {self.deck_id} - Master EQ error: {e}")
        
        return mix_chunk

    def _initialize_eq_filters(self):
        """Initialize EQ filters for real-time processing"""
        if self._eq_filters_initialized or self.sample_rate == 0:
            return
        
        try:
            from scipy import signal
            
            # True DJ-style shelving filters (like Pioneer DJM mixers)
            low_shelf_freq = 120.0   # Hz - low shelf frequency
            high_shelf_freq = 6000.0 # Hz - high shelf frequency
            mid_low_freq = 400.0     # Hz - mid band low frequency
            mid_high_freq = 2500.0   # Hz - mid band high frequency
            
            nyquist = self.sample_rate / 2.0
            
            # True DJ-style EQ filters (like Pioneer DJM, Serato, Traktor)
            # Low: Low shelf filter (affects frequencies below cutoff)
            # Mid: Peak/Bell filter (affects frequencies around center)
            # High: High shelf filter (affects frequencies above cutoff)
            
            # Complementary filter design - filters sum to unity to avoid spectral holes
            # This eliminates clicking by ensuring no frequency content is lost
            
            # Low band: frequencies below 120Hz
            self._eq_low_filter = signal.butter(2, low_shelf_freq/nyquist, btype='low')
            
            # Mid band: frequencies between 120-6000Hz (complementary to low/high)
            self._eq_mid_filter = signal.butter(2, [low_shelf_freq/nyquist, high_shelf_freq/nyquist], btype='band')
            
            # High band: frequencies above 6000Hz
            self._eq_high_filter = signal.butter(2, high_shelf_freq/nyquist, btype='high')

            # Initialize persistent filter state for each band
            if self._eq_low_filter is not None:
                self._eq_low_state = signal.lfilter_zi(self._eq_low_filter[0], self._eq_low_filter[1])
            if self._eq_mid_filter is not None:
                self._eq_mid_state = signal.lfilter_zi(self._eq_mid_filter[0], self._eq_mid_filter[1])
            if self._eq_high_filter is not None:
                self._eq_high_state = signal.lfilter_zi(self._eq_high_filter[0], self._eq_high_filter[1])
            
            self._eq_filters_initialized = True
            
            # Initialize ring buffer and master isolator EQ
            if self.out_ring is None:
                self.out_ring = ExternalRingBuffer(self.RING_BUFFER_SIZE, channels=2)
                logger.debug(f"Deck {self.deck_id} - Ring buffer initialized: {self.RING_BUFFER_SIZE} frames")
            
            if self.master_isolator is None:
                self.master_isolator = IsolatorEQ(self.sample_rate)
                logger.debug(f"Deck {self.deck_id} - Master isolator EQ initialized")
            
            logger.debug(f"Deck {self.deck_id} - EQ filters initialized for sample rate {self.sample_rate}")
            
        except ImportError:
            logger.warning(f"Deck {self.deck_id} - SciPy not available, EQ disabled")
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - Failed to initialize EQ filters: {e}")
            # Fallback to simple EQ without filters
            self._eq_filters_initialized = True
            self._eq_low_filter = None
            self._eq_mid_filter = None
            self._eq_high_filter = None
            self._eq_low_state = None
            self._eq_mid_state = None
            self._eq_high_state = None

    def _initialize_stem_eq_processors(self):
        """Initialize per-stem EQ processors"""
        if self.sample_rate == 0:
            logger.warning(f"Deck {self.deck_id} - Cannot initialize stem EQ: sample rate is 0")
            return
            
        try:
            for stem_name in self.STEM_NAMES:
                if stem_name not in self.stem_tone_eqs:
                    self.stem_tone_eqs[stem_name] = ToneEQ3(self.sample_rate)
                    logger.debug(f"Deck {self.deck_id} - Initialized {stem_name} EQ processor")
            logger.info(f"Deck {self.deck_id} - All stem EQ processors initialized")
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - Failed to initialize stem EQ processors: {e}")

    def _load_or_generate_stems(self, audio_filepath, audio_data):
        """Load existing stems or generate new ones"""
        try:
            # Check if stems already exist (stems must be pre-cached via preprocess_stems.py)
            if self._check_stems_exist(audio_filepath):
                logger.info(f"Deck {self.deck_id} - Loading cached stems for {os.path.basename(audio_filepath)}")
                self._load_cached_stems(audio_filepath)
            else:
                logger.info(f"Deck {self.deck_id} - No cached stems found. Run preprocess_stems.py first to generate stems.")
                self.stems_available = False
                
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - Failed to load stems: {e}")
            # Continue without stems - fallback to master EQ only
            self.stems_available = False

    def _get_stems_cache_dir(self, audio_filepath):
        """Get the directory for cached stems"""
        # Create a hash-based cache directory like beat viewer does
        import hashlib
        file_hash = hashlib.md5(audio_filepath.encode()).hexdigest()[:8]
        filename = os.path.splitext(os.path.basename(audio_filepath))[0]
        cache_dir = os.path.join(os.path.dirname(audio_filepath), f"stems_cache_{filename}_{file_hash}")
        return cache_dir

    def _check_stems_exist(self, audio_filepath):
        """Check if stems exist for this audio file"""
        stems_dir = self._get_stems_cache_dir(audio_filepath)
        if not os.path.exists(stems_dir):
            return False
        
        for stem_name in self.STEM_NAMES:
            stem_file = os.path.join(stems_dir, f"{stem_name}.npy")
            if not os.path.exists(stem_file):
                return False
        return True

    def _load_cached_stems(self, audio_filepath):
        """Load cached stems"""
        stems_dir = self._get_stems_cache_dir(audio_filepath)
        loaded_stems = {}
        
        try:
            for stem_name in self.STEM_NAMES:
                stem_file = os.path.join(stems_dir, f"{stem_name}.npy")
                if os.path.exists(stem_file):
                    stem_audio = np.load(stem_file)
                    
                    # Handle stem dimensions - transpose if needed to get (samples, channels)
                    if stem_audio.ndim == 2 and stem_audio.shape[0] == 2:
                        stem_audio = stem_audio.T  # Transpose from (2, samples) to (samples, 2)
                        logger.debug(f"Deck {self.deck_id} - Transposed stereo stem {stem_name} to (samples, channels)")
                    elif stem_audio.ndim == 1:
                        # Convert mono to stereo
                        stem_audio = np.column_stack([stem_audio, stem_audio])
                        logger.debug(f"Deck {self.deck_id} - Converted mono stem {stem_name} to stereo")
                    
                    loaded_stems[stem_name] = stem_audio
                    logger.debug(f"Deck {self.deck_id} - Loaded cached {stem_name}: {len(stem_audio)} samples")
            
            if loaded_stems:
                # Validate stems - check if they have reasonable length
                expected_min_samples = len(self.audio_thread_data) // 4  # At least 1/4 of original length
                valid_stems = {}
                for stem_name, stem_audio in loaded_stems.items():
                    # Get correct sample count for stereo data
                    sample_count = stem_audio.shape[0] if stem_audio.ndim == 2 else len(stem_audio)
                    if sample_count >= expected_min_samples:
                        valid_stems[stem_name] = stem_audio
                    else:
                        logger.warning(f"Deck {self.deck_id} - Stem {stem_name} is corrupted: {sample_count} samples (expected >= {expected_min_samples})")
                
                if len(valid_stems) >= 2:  # Need at least 2 valid stems
                    self.stem_data = valid_stems
                    self.stems_available = True
                    logger.info(f"Deck {self.deck_id} - Successfully loaded {len(valid_stems)} valid cached stems")
                else:
                    logger.warning(f"Deck {self.deck_id} - Too many corrupted stems, disabling stem processing")
                    self.stems_available = False
            else:
                logger.warning(f"Deck {self.deck_id} - No cached stems found")
                self.stems_available = False
                
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - Failed to load cached stems: {e}")
            self.stems_available = False


    def _build_stem_mix_chunk(self, start_frame, num_frames):
        """Build audio chunk from stems with per-stem EQ applied"""
        try:
            if not self.stems_available or not self.stem_data:
                # Fallback to original audio
                return self.audio_thread_data[start_frame:start_frame + num_frames]
            
            # Initialize output buffer
            mixed_chunk = np.zeros(num_frames, dtype=np.float32)
            
            # Process each stem with its EQ and volume
            for stem_name, stem_audio in self.stem_data.items():
                if stem_name not in self.STEM_NAMES:
                    continue
                
                # Check bounds
                end_frame = start_frame + num_frames
                if start_frame >= len(stem_audio):
                    continue
                
                # Get stem chunk (handle end of audio gracefully)
                if end_frame > len(stem_audio):
                    available_frames = len(stem_audio) - start_frame
                    stem_chunk = np.zeros(num_frames, dtype=np.float32)
                    stem_chunk[:available_frames] = stem_audio[start_frame:start_frame + available_frames]
                else:
                    stem_chunk = stem_audio[start_frame:end_frame].copy()
                
                # Apply per-stem EQ if enabled
                if (self.enable_stem_eqs and stem_name in self.stem_tone_eqs and 
                    self.stem_eq_enabled.get(stem_name, False)):
                    try:
                        # Process through ToneEQ3
                        stem_chunk_eq = self.stem_tone_eqs[stem_name].process_block(stem_chunk)
                        stem_chunk = stem_chunk_eq.flatten()  # Convert (N,1) back to (N,)
                    except Exception as e:
                        logger.warning(f"Deck {self.deck_id} - Stem EQ error for {stem_name}: {e}")
                        # Continue without EQ processing
                
                # Apply stem volume
                stem_volume = self.stem_volumes.get(stem_name, 1.0)
                stem_chunk *= stem_volume
                
                # Mix into output
                mixed_chunk += stem_chunk
            
            return mixed_chunk
            
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - Error building stem mix chunk: {e}")
            # Fallback to original audio on error
            return self.audio_thread_data[start_frame:start_frame + num_frames]

    def apply_eq(self, audio_chunk):
        """Apply EQ filtering to audio chunk with gain smoothing"""
        logger.debug(f"Deck {self.deck_id} - apply_eq called, chunk shape: {audio_chunk.shape}")
        
        # TEMPORARILY DISABLED: EQ processing to fix audio output
        logger.debug(f"Deck {self.deck_id} - EQ temporarily disabled, returning unprocessed audio")
        return audio_chunk
        
        try:
            # Ensure audio_chunk is 1D for processing
            if audio_chunk.ndim == 2:
                audio_1d = audio_chunk[:, 0]  # Take first channel
            else:
                audio_1d = audio_chunk
            
            # Add safety checks for audio data
            if len(audio_1d) == 0:
                logger.warning(f"Deck {self.deck_id} - Empty audio chunk, skipping EQ")
                return audio_chunk
            
            # Check for NaN or inf values
            if np.any(np.isnan(audio_1d)) or np.any(np.isinf(audio_1d)):
                logger.warning(f"Deck {self.deck_id} - Audio contains NaN/inf values, skipping EQ")
                return audio_chunk
            
            if (self._eq_low_filter is not None and 
                self._eq_mid_filter is not None and 
                self._eq_high_filter is not None):
                logger.debug(f"Deck {self.deck_id} - Using SciPy filtering")
                try:
                    # Apply each EQ band with error handling and persistent state
                    try:
                        low_band, self._eq_low_state = signal.lfilter(
                            self._eq_low_filter[0], self._eq_low_filter[1], audio_1d, zi=self._eq_low_state)
                        logger.debug(f"Deck {self.deck_id} - Low band filter applied successfully")
                    except Exception as e:
                        logger.error(f"Deck {self.deck_id} - Low band filter failed: {e}")
                        low_band = audio_1d.copy() * 0.5  # Fallback to simple scaling
                        self._eq_low_state = None
                    try:
                        mid_band, self._eq_mid_state = signal.lfilter(
                            self._eq_mid_filter[0], self._eq_mid_filter[1], audio_1d, zi=self._eq_mid_state)
                        logger.debug(f"Deck {self.deck_id} - Mid band filter applied successfully")
                    except Exception as e:
                        logger.error(f"Deck {self.deck_id} - Mid band filter failed: {e}")
                        mid_band = audio_1d.copy() * 0.5  # Fallback to simple scaling
                        self._eq_mid_state = None
                    try:
                        high_band, self._eq_high_state = signal.lfilter(
                            self._eq_high_filter[0], self._eq_high_filter[1], audio_1d, zi=self._eq_high_state)
                        logger.debug(f"Deck {self.deck_id} - High band filter applied successfully")
                    except Exception as e:
                        logger.error(f"Deck {self.deck_id} - High band filter failed: {e}")
                        high_band = audio_1d.copy() * 0.5  # Fallback to simple scaling
                        self._eq_high_state = None

                    # Use direct EQ values - no smoothing
                    low_val = self._eq_low
                    mid_val = self._eq_mid
                    high_val = self._eq_high

                    # DJ-style EQ mixing - direct band mixing
                    # Each band is mixed with its gain value
                    # When gain is 0, that band contributes nothing
                    
                    eq_audio_1d = (low_band * low_val) + (mid_band * mid_val) + (high_band * high_val)
                    
                    # Calculate RMS levels for debugging
                    original_rms = np.sqrt(np.mean(audio_1d**2))
                    eq_rms = np.sqrt(np.mean(eq_audio_1d**2))
                    rms_ratio = eq_rms / original_rms if original_rms > 0 else 1.0
                    
                    # Calculate individual band contributions
                    low_rms = np.sqrt(np.mean((low_band * low_val)**2)) if low_val > 0 else 0
                    mid_rms = np.sqrt(np.mean((mid_band * mid_val)**2)) if mid_val > 0 else 0
                    high_rms = np.sqrt(np.mean((high_band * high_val)**2)) if high_val > 0 else 0
                    
                    # Check for potential clicking indicators
                    max_amplitude_change = np.max(np.abs(eq_audio_1d - audio_1d))
                    amplitude_ratio = np.max(np.abs(eq_audio_1d)) / np.max(np.abs(audio_1d)) if np.max(np.abs(audio_1d)) > 0 else 1.0
                    
                    logger.info(f"Deck {self.deck_id} - EQ levels: low={low_val:.2f}, mid={mid_val:.2f}, high={high_val:.2f}")
                    logger.info(f"Deck {self.deck_id} - Applied values: low_val={low_val:.3f}, mid_val={mid_val:.3f}, high_val={high_val:.3f}")
                    logger.info(f"Deck {self.deck_id} - RMS: original={original_rms:.4f}, eq={eq_rms:.4f}, ratio={rms_ratio:.2f}")
                    logger.info(f"Deck {self.deck_id} - Band RMS: low={low_rms:.4f}, mid={mid_rms:.4f}, high={high_rms:.4f}")
                    logger.info(f"Deck {self.deck_id} - Click indicators: max_change={max_amplitude_change:.4f}, amp_ratio={amplitude_ratio:.2f}")
                except Exception as e:
                    logger.error(f"Deck {self.deck_id} - SciPy filtering failed, using simple EQ: {e}")
                    eq_audio_1d = audio_1d * ((self._eq_low + self._eq_mid + self._eq_high) / 3.0)
            else:
                eq_audio_1d = audio_1d * ((self._eq_low + self._eq_mid + self._eq_high) / 3.0)
            if np.any(np.isnan(eq_audio_1d)) or np.any(np.isinf(eq_audio_1d)):
                logger.warning(f"Deck {self.deck_id} - EQ output contains NaN/inf values, using original audio")
                return audio_chunk
            if audio_chunk.ndim == 2:
                eq_audio = eq_audio_1d.reshape(-1, 1)
            else:
                eq_audio = eq_audio_1d
            return eq_audio
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - EQ processing failed: {e}")
            import traceback
            traceback.print_exc()
            return audio_chunk  # Return original audio on error

    def load_track(self, audio_filepath):
        """Load audio track and analyze for BPM and beat detection"""
        logger.debug(f"Deck {self.deck_id} - load_track requested for: {audio_filepath}")
        self.stop() # Send CMD_STOP to ensure any previous playback is fully handled

        analysis_data = self.analyzer.analyze_track(audio_filepath)
        if not analysis_data:
            logger.error(f"Deck {self.deck_id} - Analysis failed for {audio_filepath}")
            self.filepath = None 
            return False

        # Debug: log what the analyzer returned (without verbose arrays)
        logger.debug(f"Deck {self.deck_id} - Analyzer returned: BPM={analysis_data.get('bpm')}, Beats={len(analysis_data.get('beat_timestamps', []))}, Key={analysis_data.get('key')}")

        self.filepath = audio_filepath 
        self.sample_rate = int(analysis_data.get('sample_rate', 0))
        self.beat_timestamps = np.array(analysis_data.get('beat_timestamps', []))
        self.bpm = float(analysis_data.get('bpm', 0.0))
        
        # Debug: log what we assigned (without verbose arrays)
        logger.debug(f"Deck {self.deck_id} - Assigned beat_timestamps: {type(self.beat_timestamps)}, count: {len(self.beat_timestamps)}")
        logger.debug(f"Deck {self.deck_id} - Assigned bpm: {type(self.bpm)}, value: {self.bpm}")
        
        self.cue_points = analysis_data.get('cue_points', {}) 
        # Store key information
        self.key = analysis_data.get('key', 'unknown')
        self.key_confidence = float(analysis_data.get('key_confidence', 0.0))
        logger.debug(f"Deck {self.deck_id} - Loaded cue points: {list(self.cue_points.keys())}")
        logger.debug(f"Deck {self.deck_id} - Track key: {self.key} (confidence: {self.key_confidence:.2f})")

        if self.sample_rate == 0:
            logger.error(f"Deck {self.deck_id} - Invalid sample rate from analysis for {audio_filepath}")
            return False
        # Safety check: ensure bpm is a scalar float
        if hasattr(self.bpm, '__len__') and len(self.bpm) > 1:
            logger.error(f"Deck {self.deck_id} - BPM is an array instead of scalar: {type(self.bpm)}, shape: {getattr(self.bpm, 'shape', 'no shape')}")
            return False
        
        if self.bpm <= 0: 
            logger.warning(f"Deck {self.deck_id} - BPM is {self.bpm}. Beat-length loops require a positive BPM.")

        # Store the ORIGINAL beat positions (before any tempo changes)
        self.original_beat_positions = {}
        for beat in range(1, len(self.beat_timestamps) + 1):
            original_frame = int(self.beat_timestamps[beat - 1] * self.sample_rate)
            self.original_beat_positions[beat] = original_frame
        
        # Store original BPM and beat timestamps for tempo scaling
        self.original_bpm = self.bpm
        self.original_beat_timestamps = self.beat_timestamps.copy()  # Store original timestamps
        
        # Phase 2: Use BeatManager as primary beat authority, AudioClock as fallback
        beat_data_set = False
        
        # Primary: Set beat data in BeatManager (Phase 2)
        if hasattr(self, 'beat_manager') and self.beat_manager:
            try:
                self.beat_manager.set_beat_data(self.beat_timestamps, self.bpm)
                beat_data_set = True
                logger.debug(f"Deck {self.deck_id} - Set beat data in BeatManager: {len(self.beat_timestamps)} timestamps, BPM {self.bpm}")
            except Exception as e:
                logger.error(f"Deck {self.deck_id} - Failed to set beat data in BeatManager: {e}")
        
        # Phase 4: AudioClock no longer stores beat data - BeatManager is single source of truth
        if not beat_data_set:
            logger.error(f"Deck {self.deck_id} - CRITICAL: Failed to set beat data in BeatManager!")
            return False
        else:
            logger.debug(f"Deck {self.deck_id} - Beat data successfully set in BeatManager (Phase 4: AudioClock no longer used)")
        # Legacy sync (will be updated in next step)
        if hasattr(self, 'beat_manager') and self.beat_manager:
            try:
                self.beat_manager._sync_with_deck()
                logger.debug(f"Deck {self.deck_id} - BeatManager synced with track data: BPM={self.bpm}, beats={len(self.beat_timestamps)}")
            except Exception as e:
                logger.warning(f"Deck {self.deck_id} - Failed to sync BeatManager: {e}")

        try:
            logger.debug(f"Deck {self.deck_id} - Loading audio samples with MonoLoader...")
            loader = es.MonoLoader(filename=audio_filepath, sampleRate=self.sample_rate)
            loaded_audio_samples = loader()
            current_total_frames = len(loaded_audio_samples)
            if current_total_frames == 0:
                logger.error(f"Deck {self.deck_id} - Loaded audio data is empty for {audio_filepath}")
                return False
            self.total_frames = current_total_frames 
            self.command_queue.put((DECK_CMD_LOAD_AUDIO, {
                'audio_data': loaded_audio_samples, 
                'sample_rate': self.sample_rate,    
                'total_frames': current_total_frames
            }))
            # Initialize EQ filters for this track
            self._initialize_eq_filters()
            # Initialize EQ smoothing frames now that sample_rate is known
            self._eq_smoothing_frames = max(1, int(self.sample_rate * (EQ_SMOOTHING_MS / 1000.0)))  # 0.5ms smoothing window
            
            # Initialize stem EQ processors now that sample rate is known
            self._initialize_stem_eq_processors()
            
            # Load or generate stems for per-stem processing
            self._load_or_generate_stems(audio_filepath, loaded_audio_samples)
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - Failed to load audio samples for {audio_filepath}: {e}")
            return False

        with self._stream_lock:
            self._current_playback_frame_for_display = 0 
            self._user_wants_to_play = False 
            # Loop state handled by LoopManager 
        
        # Initialize real-time tempo processor
        if self._use_realtime_tempo:
            try:
                # Check if audio_thread_data is available before accessing its shape
                if self.audio_thread_data is not None:
                    channels = 1 if len(self.audio_thread_data.shape) == 1 else self.audio_thread_data.shape[1]
                else:
                    channels = 1  # Default to mono if no audio data yet
                
                self._realtime_tempo_processor = create_realtime_tempo_processor(
                    sample_rate=self.sample_rate,
                    channels=channels,
                    buffer_size=1024
                )
                self._realtime_tempo_processor.start_processing()
                logger.info(f"Deck {self.deck_id} - Real-time tempo processor initialized")
            except Exception as e:
                logger.warning(f"Deck {self.deck_id} - Failed to initialize real-time tempo processor: {e}")
                self._use_realtime_tempo = False

        logger.info(f"Deck {self.deck_id} - Track '{os.path.basename(audio_filepath)}' data sent to audio thread. BPM: {self.bpm:.2f}")
        return True

    def get_frame_from_beat(self, beat_number):
        """Get frame number for a specific beat - DEPRECATED: Use beat_manager.get_frame_for_beat() instead"""
        logger.warning(f"Deck {self.deck_id} - get_frame_from_beat() is deprecated. Use beat_manager.get_frame_for_beat() instead.")
        return self.beat_manager.get_frame_for_beat(beat_number)

    def schedule_frame_action(self, target_frame, action_type, action_data):
        """Schedule an action to be executed at an exact frame position"""
        with self._frame_actions_lock:
            action = {
                'type': action_type,
                'data': action_data,
                'frame': target_frame
            }
            # Use negative frame as priority so earlier frames execute first
            self._pending_frame_actions.put((-target_frame, target_frame, action))
            logger.debug(f"Deck {self.deck_id} - Scheduled {action_type} action at frame {target_frame}")

    def schedule_beat_action_frame_accurate(self, target_beat, action_type, action_data):
        """Schedule an action at a specific beat with frame accuracy"""
        target_frame = self.beat_manager.get_frame_for_beat(target_beat)
        self.schedule_frame_action(target_frame, action_type, action_data)
        logger.debug(f"Deck {self.deck_id} - Scheduled {action_type} action at beat {target_beat} (frame {target_frame})")

    def _execute_pending_frame_actions(self, current_frame):
        """Execute any pending frame actions that should occur at or before current_frame"""
        with self._frame_actions_lock:
            executed_actions = []
            
            while not self._pending_frame_actions.empty():
                try:
                    # Peek at next action without removing it
                    neg_frame, target_frame, action = self._pending_frame_actions.queue[0]
                    
                    # Check if it's time to execute this action
                    if target_frame <= current_frame:
                        # Remove and execute the action
                        self._pending_frame_actions.get_nowait()
                        
                        try:
                            if action['type'] == 'activate_loop':
                                # Execute loop activation directly in audio thread context
                                loop_data = action['data']
                                # Legacy loop_manager call removed - using frame-accurate system only
                                logger.debug(f"Deck {self.deck_id} - Loop repetition handled by frame-accurate system")
                                success = True
                                if success:
                                    logger.info(f"Deck {self.deck_id} - Frame-accurate loop activated at frame {current_frame} (target: {target_frame})")
                                else:
                                    logger.warning(f"Deck {self.deck_id} - Frame-accurate loop activation failed at frame {current_frame}")
                            
                            elif action['type'] == 'loop_repetition_complete':
                                # Execute loop repetition completion directly in audio thread context
                                completion_data = action['data']
                                # Legacy loop_manager call removed - completion handled by frame-accurate system
                                logger.debug(f"Deck {self.deck_id} - Loop completion handled by frame-accurate system")
                                logger.info(f"Deck {self.deck_id} - Frame-accurate loop repetition {completion_data['repetition']}/{completion_data['total_repetitions']} completed at frame {current_frame}")
                            
                            executed_actions.append((target_frame, action['type']))
                            
                        except Exception as e:
                            logger.error(f"Deck {self.deck_id} - Error executing frame action {action['type']}: {e}")
                    else:
                        # No more actions to execute at this frame
                        break
                        
                except (IndexError, queue.Empty):
                    # Queue is empty
                    break
            
            if executed_actions:
                logger.debug(f"Deck {self.deck_id} - Executed {len(executed_actions)} frame actions at frame {current_frame}")
        
        return len(executed_actions)

    def get_frame_from_cue(self, cue_name):
        if not self.cue_points or cue_name not in self.cue_points:
            logger.warning(f"Deck {self.deck_id} - Cue point '{cue_name}' not found.")
            return None
        cue_info = self.cue_points[cue_name]
        start_beat = cue_info.get("start_beat")
        if start_beat is None:
            logger.warning(f"Deck {self.deck_id} - Cue point '{cue_name}' has no 'start_beat'.")
            return None
        return self.get_frame_from_beat(start_beat)

    def get_current_beat_count(self):
        """Get current beat count - DEPRECATED: Use beat_manager.get_current_beat() instead"""
        logger.warning(f"Deck {self.deck_id} - get_current_beat_count() is deprecated. Use beat_manager.get_current_beat() instead.")
        return int(self.beat_manager.get_current_beat())
    
    def get_synchronized_beat_count(self):
        """Get current beat count - DEPRECATED: Use beat_manager.get_current_beat() instead"""
        logger.warning(f"Deck {self.deck_id} - get_synchronized_beat_count() is deprecated. Use beat_manager.get_current_beat() instead.")
        return int(self.beat_manager.get_current_beat())

    def set_phase_offset(self, offset_beats):
        """Set a phase offset to be applied when the deck starts playing"""
        logger.debug(f"Deck {self.deck_id} - Setting phase offset: {offset_beats} beats")
        self._pending_phase_offset_beats = offset_beats
        self._phase_offset_applied = False
    
    def apply_phase_offset(self):
        """Apply the pending phase offset by seeking the appropriate number of frames"""
        if self._phase_offset_applied or self._pending_phase_offset_beats == 0.0:
            return
        
        # Safety check: ensure bpm is a scalar float
        if hasattr(self.bpm, '__len__') and len(self.bpm) > 1:
            logger.error(f"Deck {self.deck_id} - BPM is an array instead of scalar: {type(self.bpm)}, shape: {getattr(self.bpm, 'shape', 'no shape')}")
            return
        
        if self.bpm <= 0:
            logger.warning(f"Deck {self.deck_id} - Cannot apply phase offset: BPM is {self.bpm}")
            return
        
        # Calculate frames to offset
        frames_per_beat = (60.0 / self.bpm) * self.sample_rate
        offset_frames = round(self._pending_phase_offset_beats * frames_per_beat)
        
        # Apply the offset
        current_frame = self._current_playback_frame_for_display
        new_frame = current_frame + offset_frames
        
        if 0 <= new_frame < self.total_frames:
            self._current_playback_frame_for_display = new_frame
            logger.debug(f"Deck {self.deck_id} - Applied phase offset: {self._pending_phase_offset_beats} beats ({offset_frames} frames)")
            self._phase_offset_applied = True
        else:
            logger.warning(f"Deck {self.deck_id} - Phase offset would seek beyond track bounds: {new_frame}")

    def play(self, start_at_beat=None, start_at_cue_name=None):
        logger.info(f"Deck {self.deck_id} - Engine requests PLAY. Cue: '{start_at_cue_name}', Beat: {start_at_beat}, BPM: {self.bpm:.2f}")
        target_start_frame = None
        operation_description = "resuming/starting from current position"

        if start_at_cue_name:
            frame_from_cue = self.get_frame_from_cue(start_at_cue_name)
            if frame_from_cue is not None:
                target_start_frame = frame_from_cue
                operation_description = f"starting from cue '{start_at_cue_name}' (frame {target_start_frame})"
            else:
                logger.warning(f"Deck {self.deck_id} - Cue '{start_at_cue_name}' not found/invalid. Checking other options.")
        
        if target_start_frame is None and start_at_beat is not None: 
            target_start_frame = self.beat_manager.get_frame_for_beat(start_at_beat)
            operation_description = f"starting from beat {start_at_beat} (frame {target_start_frame})"

        final_start_frame_for_command = 0
        with self._stream_lock:
            self._user_wants_to_play = True 
            if target_start_frame is not None: 
                self._current_playback_frame_for_display = target_start_frame
                final_start_frame_for_command = target_start_frame
            else: 
                # Use current position (which may have been set by bpm_match)
                current_display_frame = self._current_playback_frame_for_display
                is_at_end = self.total_frames > 0 and current_display_frame >= self.total_frames
                if is_at_end:
                    final_start_frame_for_command = 0 
                    self._current_playback_frame_for_display = 0
                else:
                    final_start_frame_for_command = current_display_frame
            
            # Apply any pending phase offset
            self.apply_phase_offset()
            
            logger.debug(f"Deck {self.deck_id} - Finalizing PLAY from frame {final_start_frame_for_command} ({operation_description})")
        self.command_queue.put((DECK_CMD_PLAY, {'start_frame': final_start_frame_for_command}))

    def pause(self):
        logger.debug(f"Deck {self.deck_id} - Engine requests PAUSE.")
        with self._stream_lock:
            self._user_wants_to_play = False
        self.command_queue.put((DECK_CMD_PAUSE, None))

    def stop(self): 
        logger.debug(f"Deck {self.deck_id} - Engine requests STOP.")
        with self._stream_lock:
            self._user_wants_to_play = False
            self._current_playback_frame_for_display = 0 
            # Loop state handled by LoopManager 
        self.command_queue.put((DECK_CMD_STOP, None))

    def seek(self, target_frame): 
        logger.debug(f"Deck {self.deck_id} - Engine requests SEEK to frame: {target_frame}")
        was_playing_intent = False
        valid_target_frame = max(0, min(target_frame, self.total_frames -1 if self.total_frames > 0 else 0))
        with self._stream_lock:
            was_playing_intent = self._user_wants_to_play 
            self._current_playback_frame_for_display = valid_target_frame
            if was_playing_intent: 
                self.seek_in_progress_flag = True
            # Loop state handled by LoopManager 
        self.command_queue.put((DECK_CMD_SEEK, {'frame': valid_target_frame, 
                                               'was_playing_intent': was_playing_intent}))

    def activate_loop(self, start_beat, length_beats, repetitions=None, action_id=None):
        """Activate a loop at the specified beat position using frame-accurate scheduling"""
        logger.info(f"[LOOP ACTIVATION] Deck {self.deck_id} - Activating loop: {length_beats} beats, {repetitions} reps, Action ID: {action_id}")
        
        if self.total_frames == 0:
            logger.warning(f"Deck {self.deck_id} - Cannot activate loop: no track loaded.")
            return False
        
        # Safety check: ensure bpm is a scalar float
        if hasattr(self.bpm, '__len__') and len(self.bpm) > 1:
            logger.error(f"Deck {self.deck_id} - BPM is an array instead of scalar: {type(self.bpm)}, shape: {getattr(self.bpm, 'shape', 'no shape')}")
            return False
        
        if self.bpm <= 0:
            logger.warning(f"Deck {self.deck_id} - Cannot activate loop: BPM is {self.bpm}.")
            return False
        
        # === NEW: Use Musical Timing System for Frame-Accurate Loop Activation ===
        # Check if we're already using frame-accurate looping to avoid conflicts
        if hasattr(self, '_frame_accurate_loop') and self._frame_accurate_loop and self._frame_accurate_loop.get('active'):
            logger.info(f"Deck {self.deck_id} - Frame-accurate loop already active, ignoring old loop manager call")
            return True
        
        if hasattr(self, 'musical_timing_system') and self.musical_timing_system:
            try:
                # Get current beat position to determine if we need immediate vs scheduled activation
                current_beat = 0.0
                if hasattr(self, 'beat_manager') and self.beat_manager:
                    try:
                        current_beat = self.beat_manager.get_current_beat()
                    except:
                        current_beat = 0.0
                
                # If the target beat has already passed or is very close, activate immediately
                beat_threshold = 0.1  # If within 0.1 beats, activate immediately
                
                if current_beat >= (start_beat - beat_threshold):
                    # Target beat has passed or is imminent - activate immediately via old system
                    logger.warning(f"Deck {self.deck_id} - Beat {start_beat} has passed (current: {current_beat:.3f}) - using immediate activation")
                    # Fall through to legacy system for immediate activation
                else:
                    # Schedule loop activation using the new frame-accurate system
                    scheduled_action_id = self.musical_timing_system.schedule_beat_action(
                        beat_number=start_beat,
                        action_type='activate_loop',
                        parameters={
                            'start_at_beat': start_beat,
                            'length_beats': length_beats,
                            'repetitions': repetitions
                        },
                        action_id=action_id,
                        priority=0  # High priority for loop activation
                    )
                    
                    if scheduled_action_id:
                        logger.info(f"Deck {self.deck_id} - Loop scheduled with frame-accurate timing: {scheduled_action_id} at beat {start_beat} (current: {current_beat:.3f})")
                        return True
                    else:
                        logger.error(f"Deck {self.deck_id} - Failed to schedule frame-accurate loop activation")
                        # Fall back to old system
                    
            except Exception as e:
                logger.error(f"Deck {self.deck_id} - Error in frame-accurate loop scheduling: {e}")
                # Fall back to old system
        
        # Legacy LoopManager system removed - only frame-accurate system available
        logger.error(f"Deck {self.deck_id} - Frame-accurate timing system unavailable, cannot activate loop: {action_id}")
        return False

    def deactivate_loop(self):
        """Deactivate the current loop using frame-accurate system"""
        logger.debug(f"Deck {self.deck_id} - Engine requests DEACTIVATE_LOOP.")
        if hasattr(self, '_frame_accurate_loop') and self._frame_accurate_loop:
            self._frame_accurate_loop['active'] = False
            logger.info(f"Deck {self.deck_id} - Frame-accurate loop deactivated")
        else:
            logger.warning(f"Deck {self.deck_id} - No frame-accurate loop to deactivate")

    def stop_at_beat(self, beat_number):
        """Stop playback when reaching a specific beat"""
        logger.debug(f"Deck {self.deck_id} - Engine requests STOP_AT_BEAT. Beat: {beat_number}")
        if self.total_frames == 0:
            logger.error(f"Deck {self.deck_id} - Cannot stop at beat: track not loaded.")
            return
        # Safety check: ensure bpm is a scalar float
        if hasattr(self.bpm, '__len__') and len(self.bpm) > 1:
            logger.error(f"Deck {self.deck_id} - BPM is an array instead of scalar: {type(self.bpm)}, shape: {getattr(self.bpm, 'shape', 'no shape')}")
            return
        
        if self.bpm <= 0:
            logger.error(f"Deck {self.deck_id} - Invalid BPM ({self.bpm}) for beat calculation.")
            return
        
        target_frame = self.get_frame_from_beat(beat_number)
        if target_frame >= self.total_frames:
            logger.warning(f"Deck {self.deck_id} - Beat {beat_number} is beyond track length. Stopping immediately.")
            self.stop()
            return
        
        logger.debug(f"Deck {self.deck_id} - Scheduling stop at beat {beat_number} (frame {target_frame})")
        self.command_queue.put((DECK_CMD_STOP_AT_BEAT, {
            'target_frame': target_frame,
            'beat_number': beat_number
        }))

    # === MUSICAL TIMING SYSTEM API ===
    # Public methods for scheduling sample-accurate musical actions
    
    def schedule_musical_action(self, beat_number: float, action_type: str, 
                               parameters: dict = None, action_id: str = None, priority: int = 0) -> str:
        """
        Schedule a sample-accurate musical action at a specific beat.
        
        This is the main public API for the new musical timing system that provides
        frame-accurate scheduling that maintains musical reference across tempo changes.
        
        Args:
            beat_number: Musical beat number where action should execute (1-based)
            action_type: Type of action ('play', 'pause', 'stop', 'activate_loop', etc.)
            parameters: Action-specific parameters (optional)
            action_id: Optional unique identifier (auto-generated if None)
            priority: Action priority (lower number = higher priority)
            
        Returns:
            Action ID for cancellation/tracking, or empty string if failed
            
        Example:
            # Schedule loop activation at beat 32
            action_id = deck.schedule_musical_action(32.0, 'activate_loop', {
                'start_at_beat': 32.0,
                'length_beats': 8.0,
                'repetitions': 4
            })
        """
        if not hasattr(self, 'musical_timing_system') or not self.musical_timing_system:
            logger.error(f"Deck {self.deck_id} - Musical timing system not available")
            return ""
        
        if parameters is None:
            parameters = {}
        
        try:
            return self.musical_timing_system.schedule_beat_action(
                beat_number=beat_number,
                action_type=action_type,
                parameters=parameters,
                action_id=action_id,
                priority=priority
            )
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - Failed to schedule musical action: {e}")
            return ""
    
    def cancel_musical_action(self, action_id: str) -> bool:
        """
        Cancel a scheduled musical action.
        
        Args:
            action_id: ID of action to cancel (returned from schedule_musical_action)
            
        Returns:
            True if action was found and cancelled
        """
        if not hasattr(self, 'musical_timing_system') or not self.musical_timing_system:
            return False
        
        return self.musical_timing_system.cancel_action(action_id)
    
    def get_scheduled_musical_actions(self) -> dict:
        """
        Get all currently scheduled musical actions.
        
        Returns:
            Dictionary mapping action_id to ScheduledAction objects
        """
        if not hasattr(self, 'musical_timing_system') or not self.musical_timing_system:
            return {}
        
        return self.musical_timing_system.get_scheduled_actions()
    
    def get_musical_timing_stats(self) -> dict:
        """
        Get comprehensive musical timing system statistics.
        
        Returns:
            Dictionary with timing system statistics and performance metrics
        """
        if not hasattr(self, 'musical_timing_system') or not self.musical_timing_system:
            return {'error': 'Musical timing system not available'}
        
        return self.musical_timing_system.get_system_stats()

    def _get_tempo_cache_filepath(self, target_bpm):
        """Generate cache filepath for tempo-processed audio using new structure"""
        if not self.filepath:
            return None
        try:
            import config as app_config
            return app_config.get_tempo_cache_filepath(self.filepath, target_bpm)
        except ImportError:
            # Fallback to old method
            import hashlib
            original_filename = os.path.splitext(os.path.basename(self.filepath))[0]
            cache_key = f"{original_filename}_{target_bpm:.1f}"
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            return os.path.join(self.analyzer.cache_dir, f"{original_filename}.tempo_{cache_hash}.npy")

    def _get_pitch_cache_filepath(self, semitones):
        """Generate cache filepath for pitch-processed audio using new structure"""
        if not self.filepath:
            return None
        try:
            import config as app_config
            return app_config.get_pitch_cache_filepath(self.filepath, semitones)
        except ImportError:
            # Fallback to old method
            import hashlib
            original_filename = os.path.splitext(os.path.basename(self.filepath))[0]
            cache_key = f"{original_filename}_{semitones}"
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            return os.path.join(self.analyzer.cache_dir, f"{original_filename}.pitch_{cache_hash}.npy")




    def set_tempo(self, target_bpm):
        """Set playback tempo using real-time processing (like professional DJ software)"""
        logger.info(f"Deck {self.deck_id} - Setting tempo to {target_bpm} BPM")
        
        if self.total_frames == 0:
            logger.error(f"Deck {self.deck_id} - Cannot set tempo: track not loaded.")
            return False
        
        if not hasattr(self, 'original_bpm') or self.original_bpm == 0:
            logger.error(f"Deck {self.deck_id} - Original BPM not available for tempo calculation.")
            return False
        
        # Real-time tempo processing (like Mixxx)
        if self._use_realtime_tempo and self._realtime_tempo_processor:
            try:
                # Calculate tempo ratio
                tempo_ratio = target_bpm / self.original_bpm
                logger.debug(f"Deck {self.deck_id} - Setting real-time tempo ratio: {tempo_ratio:.3f}")
                
                # Set the tempo ratio in real-time processor (instant change)
                self._realtime_tempo_processor.set_tempo_ratio(tempo_ratio)
                
                # Update deck state
                with self._stream_lock:
                    # Safety check: ensure target_bpm is a scalar
                    if hasattr(target_bpm, '__len__') and len(target_bpm) > 1:
                        logger.error(f"Deck {self.deck_id} - target_bpm is an array instead of scalar: {type(target_bpm)}")
                        return False
                    
                    self.bpm = float(target_bpm)  # Ensure it's a float
                    self._playback_tempo = tempo_ratio
                    # Scale beat timestamps for UI display
                    self._scale_beat_positions(tempo_ratio)
                
                # NEW: Sync BeatManager with the new BPM
                if hasattr(self, 'beat_manager') and self.beat_manager:
                    try:
                        self.beat_manager._sync_with_deck()
                        logger.debug(f"Deck {self.deck_id} - BeatManager synced after tempo change: BPM={self.bpm}")
                    except Exception as e:
                        logger.warning(f"Deck {self.deck_id} - Failed to sync BeatManager after tempo change: {e}")
                
                logger.info(f"Deck {self.deck_id} - Real-time tempo set to {target_bpm} BPM (ratio: {tempo_ratio:.3f})")
                return True
                
            except Exception as e:
                logger.error(f"Deck {self.deck_id} - Real-time tempo setting failed: {e}")
                return False
        
        # No cached tempo fallback - real-time processing only
        logger.error(f"Deck {self.deck_id} - Real-time tempo processor not available")
        return False

    def set_pitch(self, semitones):
        """Set playback pitch using pre-processed cached audio (cache-only)"""
        logger.info(f"Deck {self.deck_id} - Setting pitch to {semitones} semitones")
        
        if self.total_frames == 0:
            logger.error(f"Deck {self.deck_id} - Cannot set pitch: track not loaded.")
            return False
        
        # Load from cache (cache-only approach)
        cache_filepath = self._get_pitch_cache_filepath(semitones)
        if not cache_filepath or not os.path.exists(cache_filepath):
            logger.error(f"Deck {self.deck_id} - Pitch cache not found for {semitones} semitones")
            logger.error(f"Expected cache file: {cache_filepath}")
            logger.error(f"Run preprocessing first: python preprocess.py <script.json>")
            return False
        
        try:
            logger.debug(f"Deck {self.deck_id} - Loading pitch cache...")
            processed_audio = np.load(cache_filepath)
            self._apply_pitch_shift_result(processed_audio, semitones)
            logger.info(f"Deck {self.deck_id} - Successfully loaded pitch {semitones} semitones from cache")
            return True
            
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - Failed to load pitch cache: {e}")
            return False



    def _apply_pitch_shift_result(self, processed_audio, semitones):
        """Apply pitch shift result in main thread"""
        logger.info(f"Deck {self.deck_id} - MAIN THREAD: Applying pitch shift result...")
        with self._stream_lock:
            # Store current position as a ratio to maintain it across audio changes
            current_position_ratio = self._current_playback_frame_for_display / len(self.audio_thread_data) if (self.audio_thread_data is not None and len(self.audio_thread_data) > 0) else 0
            
            old_audio_length = len(self.audio_thread_data) if self.audio_thread_data is not None else 0
            self.audio_thread_data = processed_audio
            self.audio_thread_total_samples = len(processed_audio)
            
            # Restore position in the new audio data
            new_position = int(current_position_ratio * len(processed_audio))
            self._current_playback_frame_for_display = new_position
            logger.info(f"Deck {self.deck_id} - MAIN THREAD: Audio data updated: {old_audio_length} -> {len(processed_audio)} frames, position: {new_position} (ratio: {current_position_ratio:.3f})")
            logger.info(f"Deck {self.deck_id} - MAIN THREAD: Pitch shift applied successfully! Audio now playing at new pitch.")

    def _scale_beat_positions(self, tempo_ratio):
        """Scale all beat positions, cue points, and loop positions to match the new tempo"""
        logger.debug(f"Deck {self.deck_id} - Scaling positions by tempo ratio {tempo_ratio:.3f}")
        
        # Scale beat timestamps - FIXED: use original timestamps to avoid cumulative scaling
        if hasattr(self, 'original_beat_timestamps') and len(self.original_beat_timestamps) > 0:
            self.beat_timestamps = self.original_beat_timestamps / tempo_ratio
            logger.debug(f"Deck {self.deck_id} - Scaled {len(self.beat_timestamps)} beat timestamps from original")
        
        # Scale cue points - FIXED: multiply by tempo ratio
        if hasattr(self, 'cue_points') and self.cue_points:
            scaled_cue_points = {}
            for cue_name, cue_data in self.cue_points.items():
                if 'start_beat' in cue_data:
                    # Scale the beat position - FIXED: multiply by tempo ratio
                    scaled_cue_points[cue_name] = {
                        'start_beat': cue_data['start_beat'] * tempo_ratio
                    }
            self.cue_points = scaled_cue_points
            logger.debug(f"Deck {self.deck_id} - Scaled {len(self.cue_points)} cue points")
        
        # Don't modify original_beat_positions - they should stay original!
        # The get_frame_from_beat method will handle tempo scaling when needed
        logger.debug(f"Deck {self.deck_id} - Keeping original beat positions unchanged for tempo scaling")

    def set_volume(self, volume):
        """Set deck volume (0.0 to 1.0)"""
        volume = max(0.0, min(1.0, float(volume)))
        logger.debug(f"Deck {self.deck_id} - Setting volume to {volume}")
        with self._stream_lock:
            self._volume = volume  # This sets _volume, not _current_volume
            # Clear any active fade
            self._fade_target_volume = None
            self._fade_start_volume = None
            self._fade_start_time = None
            self._fade_duration = None

    def fade_volume(self, target_volume, duration_seconds):
        """Fade volume to target over specified duration"""
        target_volume = max(0.0, min(1.0, float(target_volume)))
        duration_seconds = max(0.0, float(duration_seconds))
        
        logger.debug(f"Deck {self.deck_id} - Fading volume from {self._volume} to {target_volume} over {duration_seconds}s")
        
        with self._stream_lock:
            self._fade_target_volume = target_volume
            self._fade_start_volume = self._volume
            self._fade_start_time = time.time()
            self._fade_duration = duration_seconds

    def get_current_volume(self):
        """Get current volume level"""
        with self._stream_lock:
            return self._volume

    def _update_volume_fade(self):
        """Update volume based on active fade"""
        if self._fade_target_volume is None or self._fade_start_volume is None or self._fade_start_time is None or self._fade_duration is None:
            return
            
        current_time = time.time()
        elapsed = current_time - self._fade_start_time
        
        if elapsed >= self._fade_duration:
            # Fade complete
            self._volume = self._fade_target_volume
            self._fade_target_volume = None
            self._fade_start_volume = None
            self._fade_start_time = None
            self._fade_duration = None
            logger.debug(f"Deck {self.deck_id} - Fade complete, volume: {self._volume}")
        else:
            # Calculate current volume during fade
            progress = elapsed / self._fade_duration
            self._volume = self._fade_start_volume + (self._fade_target_volume - self._fade_start_volume) * progress

    def set_eq(self, low=None, mid=None, high=None):
        """Set EQ levels (0.0 to 2.0 for each band) with fast smoothing to prevent clicks"""
        logger.info(f"Deck {self.deck_id} - set_eq called with: low={low}, mid={mid}, high={high}")
        with self._stream_lock:
            eq_changed = False
            if low is not None and abs(low - self._eq_low) > 1e-6:
                eq_changed = True
            if mid is not None and abs(mid - self._eq_mid) > 1e-6:
                eq_changed = True
            if high is not None and abs(high - self._eq_high) > 1e-6:
                eq_changed = True
            if eq_changed:
                # Disable any active fade when starting smoothing
                self._fade_target_eq_low = None
                self._fade_target_eq_mid = None
                self._fade_target_eq_high = None
                self._fade_start_eq_low = None
                self._fade_start_eq_mid = None
                self._fade_start_eq_high = None
                self._fade_start_time_eq = None
                self._fade_duration_eq = None
                
                # Start smoothing from current values to new targets
                self._eq_smoothing_start = {
                    'low': self._eq_low,
                    'mid': self._eq_mid,
                    'high': self._eq_high
                }
                self._eq_smoothing_target = {
                    'low': low if low is not None else self._eq_low,
                    'mid': mid if mid is not None else self._eq_mid,
                    'high': high if high is not None else self._eq_high
                }
                self._eq_smoothing_counter = 0
                self._eq_smoothing_active = True
            logger.info(f"Deck {self.deck_id} - EQ set: Low={self._eq_low:.2f}, Mid={self._eq_mid:.2f}, High={self._eq_high:.2f}")
            logger.info(f"Deck {self.deck_id} - EQ filters initialized: {self._eq_filters_initialized}")
            logger.info(f"Deck {self.deck_id} - EQ filters exist: low={self._eq_low_filter is not None}, mid={self._eq_mid_filter is not None}, high={self._eq_high_filter is not None}")

    def fade_eq(self, target_low=None, target_mid=None, target_high=None, duration_seconds=1.0):
        """Fade EQ levels over time"""
        with self._stream_lock:
            # Disable any active smoothing when starting a fade
            self._eq_smoothing_active = False
            
            # Store target values but don't start fade yet
            self._fade_target_eq_low = target_low
            self._fade_target_eq_mid = target_mid
            self._fade_target_eq_high = target_high
            self._fade_duration_eq = duration_seconds
            
            # Mark fade as pending - will start at next buffer boundary
            self._fade_pending_eq = True
            
            logger.debug(f"Deck {self.deck_id} - EQ fade queued: {duration_seconds}s")

    def get_current_eq(self):
        """Get current EQ levels"""
        with self._stream_lock:
            return {
                'low': self._eq_low,
                'mid': self._eq_mid,
                'high': self._eq_high
            }

    # --- Per-Stem EQ Control Methods ---
    

    def get_stem_eq(self, stem_name):
        """Get current EQ levels for specific stem"""
        if stem_name not in self.STEM_NAMES:
            return {'low': 1.0, 'mid': 1.0, 'high': 1.0}
            
        with self._stream_lock:
            if stem_name in self.stem_tone_eqs:
                # Extract current gains from ToneEQ3 processor
                eq_processor = self.stem_tone_eqs[stem_name]
                if hasattr(eq_processor, '_cur') and 'gains' in eq_processor._cur:
                    gains_db = eq_processor._cur['gains']  # [low_db, mid_db, high_db]
                    return {
                        'low': self._db_to_linear(gains_db[0]),
                        'mid': self._db_to_linear(gains_db[1]),
                        'high': self._db_to_linear(gains_db[2])
                    }
            
            # Default flat response
            return {'low': 1.0, 'mid': 1.0, 'high': 1.0}

    def enable_stem_eq(self, stem_name, enabled=True):
        """Enable/disable EQ processing for specific stem"""
        if stem_name not in self.STEM_NAMES:
            logger.error(f"Deck {self.deck_id} - Invalid stem name: {stem_name}")
            return False
            
        with self._stream_lock:
            self.stem_eq_enabled[stem_name] = enabled
            logger.info(f"Deck {self.deck_id} - {stem_name} EQ {'enabled' if enabled else 'disabled'}")
            return True

    def set_stem_volume(self, stem_name, volume):
        """Set volume for specific stem (0.0 to 2.0)"""
        if stem_name not in self.STEM_NAMES:
            logger.error(f"Deck {self.deck_id} - Invalid stem name: {stem_name}")
            return False
            
        with self._stream_lock:
            self.stem_volumes[stem_name] = max(0.0, min(2.0, volume))
            logger.debug(f"Deck {self.deck_id} - {stem_name} volume set to {volume:.2f}")
            return True

    def get_stem_volume(self, stem_name):
        """Get current volume for specific stem"""
        with self._stream_lock:
            return self.stem_volumes.get(stem_name, 1.0)

    def set_all_stem_eq(self, stem_eq_settings):
        """Set EQ for multiple stems at once
        
        Args:
            stem_eq_settings: Dict like {'vocals': {'low': 1.0, 'mid': 2.0, 'high': 0.5, 'enabled': True}}
        """
        results = {}
        for stem_name, settings in stem_eq_settings.items():
            if stem_name not in self.STEM_NAMES:
                results[stem_name] = False
                continue
                
            success = True
            
            # Set EQ if provided
            if 'low' in settings or 'mid' in settings or 'high' in settings:
                success &= self.set_stem_eq(
                    stem_name, 
                    low=settings.get('low'),
                    mid=settings.get('mid'), 
                    high=settings.get('high')
                )
            
            # Set enabled state if provided
            if 'enabled' in settings:
                success &= self.enable_stem_eq(stem_name, settings['enabled'])
                
            results[stem_name] = success
            
        return results

    def get_all_stem_states(self):
        """Get complete state for all stems"""
        with self._stream_lock:
            states = {}
            for stem_name in self.STEM_NAMES:
                states[stem_name] = {
                    'eq': self.get_stem_eq(stem_name),
                    'eq_enabled': self.stem_eq_enabled.get(stem_name, False),
                    'volume': self.get_stem_volume(stem_name),
                    'available': stem_name in self.stem_data
                }
            return states

    def _linear_to_db(self, linear_gain):
        """Convert linear gain (0.0-3.0) to dB (-inf to +9.5dB)"""
        if linear_gain <= 0.0:
            return -60.0  # Effective silence
        elif linear_gain == 1.0:
            return 0.0    # Unity gain
        else:
            return 20 * np.log10(linear_gain)

    def _db_to_linear(self, db_gain):
        """Convert dB gain to linear (0.0-3.0)"""
        if db_gain <= -60.0:
            return 0.0
        else:
            return 10 ** (db_gain / 20.0)

    def _update_eq_fade(self):
        """Update EQ based on active fade"""
        # Check if fade is pending and start it at buffer boundary
        if self._fade_pending_eq:
            self._fade_start_eq_low = self._eq_low
            self._fade_start_eq_mid = self._eq_mid
            self._fade_start_eq_high = self._eq_high
            self._fade_start_time_eq = time.time()
            self._fade_pending_eq = False
            logger.debug(f"Deck {self.deck_id} - EQ fade started at buffer boundary")
            
        if (self._fade_target_eq_low is None and 
            self._fade_target_eq_mid is None and 
            self._fade_target_eq_high is None):
            return
            
        if (self._fade_start_time_eq is None or 
            self._fade_duration_eq is None):
            return
            
        current_time = time.time()
        elapsed = current_time - self._fade_start_time_eq
        
        # REMOVED: Previous value storage - no longer needed
        
        if elapsed >= self._fade_duration_eq:
            # Fade complete
            if self._fade_target_eq_low is not None:
                self._eq_low = self._fade_target_eq_low
            if self._fade_target_eq_mid is not None:
                self._eq_mid = self._fade_target_eq_mid
            if self._fade_target_eq_high is not None:
                self._eq_high = self._fade_target_eq_high
            
            # Reset fade state
            self._fade_target_eq_low = None
            self._fade_target_eq_mid = None
            self._fade_target_eq_high = None
            self._fade_start_eq_low = None
            self._fade_start_eq_mid = None
            self._fade_start_eq_high = None
            self._fade_start_time_eq = None
            self._fade_duration_eq = None
            
            logger.debug(f"Deck {self.deck_id} - EQ fade complete: Low={self._eq_low:.2f}, Mid={self._eq_mid:.2f}, High={self._eq_high:.2f}")
        else:
            # Calculate current EQ during fade with micro-interpolation
            progress = elapsed / self._fade_duration_eq
            
            # Apply smooth curve to reduce artifacts
            # Use S-curve (smoothstep) for more natural transitions
            smooth_progress = 3 * progress ** 2 - 2 * progress ** 3
            
            if (self._fade_target_eq_low is not None and 
                self._fade_start_eq_low is not None):
                self._eq_low = self._fade_start_eq_low + (self._fade_target_eq_low - self._fade_start_eq_low) * smooth_progress
            if (self._fade_target_eq_mid is not None and 
                self._fade_start_eq_mid is not None):
                self._eq_mid = self._fade_start_eq_mid + (self._fade_target_eq_mid - self._fade_start_eq_mid) * smooth_progress
            if (self._fade_target_eq_high is not None and 
                self._fade_start_eq_high is not None):
                self._eq_high = self._fade_start_eq_high + (self._fade_target_eq_high - self._fade_start_eq_high) * smooth_progress
        
        # REMOVED: Filter state reset - it was causing artifacts
        # The IIR filters maintain their own state continuity

    # TEMPORARILY REMOVED: _apply_eq_with_gradual_activation method to fix hang

    # REMOVED: _recalculate_beat_positions_for_ramp method - now handled by BeatManager
    # This eliminates duplicate beat position calculation logic

    def ramp_tempo(self, start_beat, end_beat, start_bpm, end_bpm, curve="linear"):
        """Start a tempo ramp - DEPRECATED: Use beat_manager.handle_tempo_change() instead"""
        logger.warning(f"Deck {self.deck_id} - ramp_tempo() is deprecated. Use beat_manager.handle_tempo_change() instead.")
        
        # Calculate ramp duration in beats
        ramp_duration_beats = end_beat - start_beat
        
        # Use BeatManager for tempo ramp
        self.beat_manager.handle_tempo_change(end_bpm, ramp_duration_beats)
        
        logger.debug(f"Deck {self.deck_id} - Tempo ramp started via BeatManager: {start_bpm}→{end_bpm} BPM over {ramp_duration_beats} beats")

    # --- Audio Management Thread ---
    def _audio_management_loop(self):
        logger.debug(f"Deck {self.deck_id} AudioThread - Started")
        _current_stream_in_thread = None 
        # Instance variables self.audio_thread_... are used by _sd_callback

        while not self.audio_thread_stop_event.is_set():
            try:
                command, data = self.command_queue.get(timeout=0.1)
                logger.debug(f"Deck {self.deck_id} AudioThread - Received command: {command}")

                if command in [DECK_CMD_LOAD_AUDIO, DECK_CMD_PLAY, DECK_CMD_SEEK, DECK_CMD_STOP, DECK_CMD_SHUTDOWN]:
                    if _current_stream_in_thread:
                        logger.debug(f"Deck {self.deck_id} AudioThread - Command {command} clearing existing stream.")
                        _current_stream_in_thread.abort(ignore_errors=True)
                        _current_stream_in_thread.close(ignore_errors=True)
                        _current_stream_in_thread = None
                        with self._stream_lock: self._is_actually_playing_stream_state = False
                
                if command == DECK_CMD_LOAD_AUDIO:
                    logger.debug(f"Deck {self.deck_id} AudioThread - Processing LOAD_AUDIO")
                    with self._stream_lock: 
                        self.audio_thread_data = data['audio_data']
                        self.audio_thread_sample_rate = data['sample_rate']
                        self.audio_thread_total_samples = data['total_frames'] 
                        self.audio_thread_current_frame = 0 
                        self._current_playback_frame_for_display = 0
                        # Loop state handled by LoopManager
                    
                    # Initialize RubberBand stretcher for consistent audio processing
                    if RUBBERBAND_STREAMING_AVAILABLE:
                        self._init_rubberband_stretcher()
                        logger.debug(f"Deck {self.deck_id} - RubberBand initialized for consistent processing")
                    
                    logger.debug(f"Deck {self.deck_id} AudioThread - Audio data set internally for playback.")

                elif command == DECK_CMD_PLAY:
                    logger.debug(f"Deck {self.deck_id} AudioThread - Processing PLAY")
                    with self._stream_lock: 
                        if self.audio_thread_data is None: 
                            logger.debug(f"Deck {self.deck_id} AudioThread - No audio data for PLAY.")
                            self._user_wants_to_play = False 
                            continue 

                        self._user_wants_to_play = True 
                        self.audio_thread_current_frame = data.get('start_frame', self._current_playback_frame_for_display)
                        
                        if self.audio_thread_current_frame >= self.audio_thread_total_samples: 
                            self.audio_thread_current_frame = 0
                        
                        self._current_playback_frame_for_display = self.audio_thread_current_frame

                        # CRITICAL: Clear ring buffer for new stream to prevent artifacts
                        self._clear_ring_buffer_for_position_jump("new stream creation")
                        
                        # Reset startup fade for new stream
                        if hasattr(self, '_startup_fade_active'):
                            self._startup_fade_active = False
                            logger.debug(f"Deck {self.deck_id} AudioThread - Startup fade reset for new stream.")

                        logger.debug(f"Deck {self.deck_id} AudioThread - Creating new stream. SR: {self.audio_thread_sample_rate}, Frame: {self.audio_thread_current_frame}")
                        _current_stream_in_thread = sd.OutputStream(
                            samplerate=self.audio_thread_sample_rate, channels=2,
                            callback=self._sd_callback, 
                            finished_callback=self._on_stream_finished_from_audio_thread, 
                            blocksize=2048,  # Same as beat_viewer_fixed - prevents startup artifacts
                            device=None  # Use default device
                        )
                        self.playback_stream_obj = _current_stream_in_thread 
                        
                        # Set device output channels for ring buffer processing
                        self.device_output_channels = 2  # Stereo output
                        
                        # CRITICAL FIX: Start producer thread FIRST to pre-fill ring buffer
                        self._start_producer_thread()
                        
                        # Give producer thread a moment to start producing audio
                        time.sleep(0.01)  # 10ms startup delay
                        
                        # Wait for ring buffer to have some data before starting audio stream
                        self._wait_for_ring_buffer_ready()
                        
                        # CRITICAL FIX: Enable startup fade-in to prevent pop/click
                        self._startup_fade_active = True
                        self._startup_fade_frames = 0
                        self._startup_fade_total_frames = int(0.01 * self.audio_thread_sample_rate)  # 10ms fade-in
                        
                        # Now start audio stream with pre-filled buffer
                        _current_stream_in_thread.start()
                        self._is_actually_playing_stream_state = True
                        
                        logger.debug(f"Deck {self.deck_id} AudioThread - New stream and producer thread started for PLAY.")
                    
                elif command == DECK_CMD_PAUSE:
                    logger.debug(f"Deck {self.deck_id} AudioThread - Processing PAUSE")
                    # Stop producer thread
                    self._producer_stop_event.set()
                    self._producer_running = False
                    if _current_stream_in_thread and _current_stream_in_thread.active:
                        logger.debug(f"Deck {self.deck_id} AudioThread - Calling stream.stop() for PAUSE.")
                        _current_stream_in_thread.stop(ignore_errors=True) 
                        logger.debug(f"Deck {self.deck_id} AudioThread - stream.stop() called for PAUSE.")
                    else: 
                         with self._stream_lock: self._is_actually_playing_stream_state = False

                elif command == DECK_CMD_STOP:
                    logger.debug(f"Deck {self.deck_id} AudioThread - Processing STOP")
                    # Stop producer thread
                    self._producer_stop_event.set()
                    self._producer_running = False
                    # Clean up RubberBand resources
                    self._cleanup_rubberband_on_stop()
                    with self._stream_lock: 
                        # CRITICAL: Clear ring buffer to prevent artifacts
                        self._clear_ring_buffer_for_position_jump("stop command")
                        self.audio_thread_current_frame = 0 
                        self._current_playback_frame_for_display = 0 
                        self._is_actually_playing_stream_state = False
                        # Loop state handled by LoopManager
                        
                        # Reset startup fade
                        if hasattr(self, '_startup_fade_active'):
                            self._startup_fade_active = False
                    logger.debug(f"Deck {self.deck_id} AudioThread - State reset for STOP.")

                elif command == DECK_CMD_SEEK:
                    new_frame = data['frame']
                    was_playing_intent = data['was_playing_intent']
                    logger.debug(f"Deck {self.deck_id} AudioThread - Processing SEEK to frame {new_frame}, was_playing_intent: {was_playing_intent}")
                    with self._stream_lock:
                        # CRITICAL: Clear ring buffer on seek to prevent old audio artifacts
                        self._clear_ring_buffer_for_position_jump("seek operation")
                        self.audio_thread_current_frame = new_frame
                        self._current_playback_frame_for_display = new_frame 
                        self._user_wants_to_play = was_playing_intent 
                        # Loop state handled by LoopManager
                        
                        # Reset startup fade
                        if hasattr(self, '_startup_fade_active'):
                            self._startup_fade_active = False
                    if was_playing_intent: 
                        logger.debug(f"Deck {self.deck_id} AudioThread - SEEK: Re-queueing PLAY command.")
                        self.command_queue.put((DECK_CMD_PLAY, {'start_frame': new_frame}))
                    else: 
                        with self._stream_lock: self._is_actually_playing_stream_state = False
                        logger.debug(f"Deck {self.deck_id} AudioThread - SEEK: Was paused, position updated.")
                
                elif command == DECK_CMD_ACTIVATE_LOOP:
                    logger.debug(f"Deck {self.deck_id} AudioThread - Processing ACTIVATE_LOOP")
                    
                    # Loop activation is now handled by LoopManager in the main activate_loop method
                    # This command is no longer needed since LoopManager handles everything centrally
                    logger.debug(f"Deck {self.deck_id} AudioThread - Loop activation handled by LoopManager")

                elif command == DECK_CMD_DEACTIVATE_LOOP:
                    logger.debug(f"Deck {self.deck_id} AudioThread - Processing DEACTIVATE_LOOP")
                    # Use LoopManager for centralized loop deactivation
                    # Legacy loop_manager call removed - using frame-accurate system only
                    if hasattr(self, '_frame_accurate_loop') and self._frame_accurate_loop:
                        self._frame_accurate_loop['active'] = False
                        logger.info(f"Deck {self.deck_id} - Frame-accurate loop deactivated on stop")
                    logger.debug(f"Deck {self.deck_id} AudioThread - Loop deactivated by LoopManager.")

                elif command == DECK_CMD_STOP_AT_BEAT:
                    logger.debug(f"Deck {self.deck_id} AudioThread - Processing STOP_AT_BEAT")
                    target_frame = data['target_frame']
                    beat_number = data['beat_number']
                    
                    with self._stream_lock:
                        # Store the stop target for the callback to check
                        self._stop_at_beat_frame = target_frame
                        self._stop_at_beat_number = beat_number
                        logger.debug(f"Deck {self.deck_id} AudioThread - Will stop at frame {target_frame} (beat {beat_number})")

                elif command == DECK_CMD_SET_TEMPO:
                    logger.debug(f"Deck {self.deck_id} AudioThread - Processing SET_TEMPO")
                    target_bpm = data.get('target_bpm')
                    if target_bpm is not None:
                        self.set_tempo(target_bpm)
                elif command == DECK_CMD_SET_PITCH:
                    logger.debug(f"Deck {self.deck_id} AudioThread - Processing SET_PITCH")
                    semitones = data.get('semitones')
                    if semitones is not None:
                        logger.info(f"Deck {self.deck_id} AudioThread - Starting pitch shift to {semitones} semitones")
                        self.set_pitch(semitones)
                        logger.info(f"Deck {self.deck_id} AudioThread - Pitch shift completed")
                elif command == DECK_CMD_SET_VOLUME:
                    logger.debug(f"Deck {self.deck_id} AudioThread - Processing SET_VOLUME")
                    volume = data.get('volume')
                    if volume is not None:
                        self.set_volume(volume)
                elif command == DECK_CMD_FADE_VOLUME:
                    logger.debug(f"Deck {self.deck_id} AudioThread - Processing FADE_VOLUME")
                    target_volume = data.get('target_volume')
                    duration_seconds = data.get('duration_seconds')
                    if target_volume is not None and duration_seconds is not None:
                        self.fade_volume(target_volume, duration_seconds)

                elif command == DECK_CMD_SHUTDOWN:
                    logger.debug(f"Deck {self.deck_id} AudioThread - Processing SHUTDOWN")
                    break 

                self.command_queue.task_done() 
            except queue.Empty:
                with self._stream_lock:
                    stream_obj_ref = self.playback_stream_obj 
                    user_wants_play_check = self._user_wants_to_play
                    is_active_check = self._is_actually_playing_stream_state if stream_obj_ref and hasattr(stream_obj_ref, 'active') else False # Guard access

                if stream_obj_ref and not is_active_check and not user_wants_play_check:
                    if not stream_obj_ref.closed:
                        logger.debug(f"Deck {self.deck_id} AudioThread - Inactive stream cleanup (timeout).")
                        stream_obj_ref.close(ignore_errors=True)
                        with self._stream_lock: 
                            if self.playback_stream_obj == stream_obj_ref: 
                                self.playback_stream_obj = None
                with self._stream_lock: _current_stream_in_thread = self.playback_stream_obj
                continue
            except Exception as e_loop:
                logger.error(f"ERROR in Deck {self.deck_id} _audio_management_loop: {e_loop}")
                import traceback
                traceback.print_exc()
                if _current_stream_in_thread: 
                    try: _current_stream_in_thread.abort(ignore_errors=True); _current_stream_in_thread.close(ignore_errors=True)
                    except: pass
                _current_stream_in_thread = None
                with self._stream_lock: 
                    self.playback_stream_obj = None 
                    self._is_actually_playing_stream_state = False
                    self._user_wants_to_play = False
        
        with self._stream_lock: 
            self.playback_stream_obj = _current_stream_in_thread 
        logger.debug(f"DEBUG: Deck {self.deck_id} AudioThread - Loop finished, thread ending.")

    def _start_producer_thread(self):
        """Start the producer thread to fill ring buffer"""
        logger.info(f"🎵 Deck {self.deck_id} - _start_producer_thread called")
        
        # Clean up any existing producer thread
        if hasattr(self, '_producer_thread') and self._producer_thread and self._producer_thread.is_alive():
            logger.debug(f"🎵 Deck {self.deck_id} - Cleaning up existing producer thread")
            self._producer_stop_event.set()
            self._producer_thread.join(timeout=1.0)
        
        # Reset state and start new producer thread
        self._producer_stop_event.clear()
        self._producer_running = True
        self._producer_startup_mode = True  # Flag for startup mode
        
        logger.debug(f"🎵 Deck {self.deck_id} - Creating new producer thread")
        self._producer_thread = threading.Thread(target=self._producer_loop, daemon=True)
        
        logger.debug(f"🎵 Deck {self.deck_id} - Starting producer thread")
        self._producer_thread.start()
        
        logger.info(f"🎵 Deck {self.deck_id} - Producer thread started in startup mode")
        logger.debug(f"🎵 Deck {self.deck_id} - Producer thread alive: {self._producer_thread.is_alive()}")
    
    def _producer_loop(self):
        """Continuously fill out_ring with processed audio."""
        logger.info(f"🎵 Deck {self.deck_id} - PRODUCER LOOP STARTED")
        TARGET_BLOCK = 4096  # frames per chunk we produce - smaller chunks for tighter control
        WATERMARK = self.sample_rate // 16 if self.sample_rate > 0 else 2756  # Lower watermark for smaller buffer
        
        # Ring buffer variables for partial writes
        self._pending_out = None

        logger.info(f"🎵 Deck {self.deck_id} - Producer loop condition check: stop_event={self._producer_stop_event.is_set()}, wants_to_play={self._user_wants_to_play}")
        
        while not self._producer_stop_event.is_set() and self._user_wants_to_play:
            try:
                # CRITICAL FIX: Don't back off on startup - we need to fill the buffer quickly
                available_data = self.out_ring.available_read() if self.out_ring else 0
                
                # Only back off if we have sufficient data and buffer is getting full
                if available_data > WATERMARK and available_data > 4096:  # Must have at least 1 chunk
                    # Exit startup mode once we have sufficient data
                    if hasattr(self, '_producer_startup_mode') and self._producer_startup_mode:
                        self._producer_startup_mode = False
                        logger.info(f"🎵 Deck {self.deck_id} - Producer startup mode complete, switching to normal pacing")
                    
                    backoff = 0.002
                    logger.debug(f"🎵 Deck {self.deck_id} - RING BUFFER FULL: available={available_data}, watermark={WATERMARK} - backing off")
                    time.sleep(backoff)
                    continue
                
                # In startup mode, log every chunk to track progress
                if hasattr(self, '_producer_startup_mode') and self._producer_startup_mode:
                    logger.info(f"🎵 Deck {self.deck_id} - STARTUP MODE: Producing chunk, available_data={available_data}")
                else:
                    logger.debug(f"🎵 Deck {self.deck_id} - PROCEEDING WITH CHUNK PRODUCTION: available_data={available_data}, watermark={WATERMARK}")
                
                # DEBUG: Log audio data status
                if hasattr(self, 'audio_thread_data') and self.audio_thread_data is not None:
                    logger.debug(f"🎵 Deck {self.deck_id} - Audio data: {len(self.audio_thread_data)} samples, current frame: {self.audio_thread_current_frame}")
                else:
                    logger.error(f"🎵 Deck {self.deck_id} - NO AUDIO DATA AVAILABLE!")
                    time.sleep(0.01)
                    continue

                # Compute a chunk - choose processing path
                # Check current RubberBand state and pending operations
                has_active_rb = False
                has_pending_rb_ops = False
                if hasattr(self, '_rb_lock') and self._rb_lock.acquire(blocking=False):
                    try:
                        has_active_rb = (hasattr(self, 'rubberband_stretcher') and self.rubberband_stretcher)
                        has_pending_rb_ops = (self._pending_reinit or self._pending_disable)
                    finally:
                        self._rb_lock.release()
                
                # Always use RubberBand for consistent audio processing architecture
                logger.debug(f"🎵 Deck {self.deck_id} - PRODUCER PATH: using RubberBand (consistent architecture)")
                
                if RUBBERBAND_STREAMING_AVAILABLE:
                    chunk = self._produce_chunk_rubberband(TARGET_BLOCK)
                    if chunk is not None:
                        logger.debug(f"🎵 Deck {self.deck_id} - Produced chunk: {chunk.shape}, sum: {np.sum(chunk):.3f}")
                    else:
                        logger.warning(f"🎵 Deck {self.deck_id} - RubberBand produced no chunk")
                else:
                    # If RubberBand is not available, we should fail gracefully rather than use inconsistent processing
                    logger.error(f"🎵 Deck {self.deck_id} - RubberBand not available! Cannot process audio with consistent architecture.")
                    chunk = np.zeros((TARGET_BLOCK, 2), dtype=np.float32)  # Return silence

                if chunk is None or (isinstance(chunk, np.ndarray) and chunk.size == 0):
                    # End: pad zeros to drain
                    chunk = np.zeros((TARGET_BLOCK, 2), dtype=np.float32)

                # Safety - remove any NaN values
                np.nan_to_num(chunk, copy=False)

                # Write to ring with partial write handling
                to_write = chunk if self._pending_out is None else np.vstack([self._pending_out, chunk])
                
                written = self.out_ring.write(to_write)
                if written < len(to_write):
                    # Save remainder as pending
                    self._pending_out = to_write[written:]
                else:
                    self._pending_out = None
                
                # Update BeatManager with current frame position (single source of truth)
                try:
                    self.beat_manager.update_from_frame(self.audio_thread_current_frame, self.audio_thread_sample_rate)
                    
                    # Update tempo ramp if active
                    if self.beat_manager.is_tempo_ramp_active():
                        current_beat = self.beat_manager.get_current_beat()
                        self.beat_manager.update_tempo_ramp(current_beat)
                    
                    # Keep deprecated field for backward compatibility
                    self._last_synchronized_beat = int(self.beat_manager.get_current_beat())
                    
                    # NEW: Update audio clock if we have access to it
                    if hasattr(self, 'engine') and hasattr(self.engine, 'audio_clock'):
                        self.engine.audio_clock.update_frame_count(TARGET_BLOCK)
                    
                    # === MUSICAL TIMING SYSTEM PROCESSING ===
                    # Process scheduled musical actions for this audio buffer
                    if hasattr(self, 'musical_timing_system') and self.musical_timing_system:
                        try:
                            # Calculate the frame range for this audio chunk
                            start_frame = int(self.audio_thread_current_frame)
                            end_frame = start_frame + TARGET_BLOCK
                            
                            # Process musical actions for this buffer
                            timing_result = self.musical_timing_system.process_audio_buffer(start_frame, end_frame)
                            
                            # Log execution statistics periodically (every 1000 chunks)
                            if not hasattr(self, '_timing_stats_counter'):
                                self._timing_stats_counter = 0
                            self._timing_stats_counter += 1
                            
                            if self._timing_stats_counter % 1000 == 0:
                                if timing_result.get('actions_executed', 0) > 0:
                                    logger.info(f"Deck {self.deck_id} - Musical timing stats: {timing_result}")
                        
                        except Exception as e:
                            logger.error(f"Deck {self.deck_id} - Error in musical timing system: {e}")
                        
                except Exception as e:
                    logger.debug(f"Deck {self.deck_id} - Error updating BeatManager: {e}")
                
            except Exception as e:
                # Set error flag but keep producer running
                self._producer_error = str(e)
                logger.error(f"🎵 Deck {self.deck_id} - Producer error: {e}")
                # Write silence to prevent underrun
                if self.out_ring:
                    silence_chunk = np.zeros((TARGET_BLOCK, 2), dtype=np.float32)
                    self.out_ring.write(silence_chunk)
                self._pending_out = None  # Clear pending on error
                time.sleep(0.01)  # Brief pause on error

        # Clean up producer state on exit
        self._producer_running = False
        if hasattr(self, '_producer_startup_mode'):
            self._producer_startup_mode = False
        logger.info(f"🎵 Deck {self.deck_id} - PRODUCER LOOP ENDED")
    def _produce_chunk_rubberband(self, out_frames):
        """Produce audio chunk using streaming RubberBand - ONLY processing method for consistency"""
        if not RUBBERBAND_STREAMING_AVAILABLE:
            # RubberBand is required for consistent architecture
            logger.error(f"Deck {self.deck_id} - RubberBand streaming not available! Cannot process audio.")
            return np.zeros((out_frames, 2), dtype=np.float32)
            
        try:
            # Initialize RubberBand stretcher if needed
            if not hasattr(self, 'rubberband_stretcher') or not self.rubberband_stretcher:
                self._init_rubberband_stretcher()
            
            # Get current playback position
            start_frame = int(self.audio_thread_current_frame)
            
            # === REMOVED: Legacy CENTRALIZED LOOP MANAGEMENT ===
            # Now using frame-accurate system only - no dual system conflicts
            
            # Check bounds
            if start_frame >= len(self.audio_thread_data):
                return None
                
            # Calculate input chunk size based on tempo ratio from BeatManager
            # RubberBand streaming needs input chunks to generate output
            input_chunk_size = int(out_frames * self.beat_manager.get_tempo_ratio()) + 1024  # Extra buffer for RB
            input_chunk_size = min(input_chunk_size, len(self.audio_thread_data) - start_frame)
            
            if input_chunk_size <= 0:
                return None
            
            # Build stem mix with per-stem EQ processing (mono for RubberBand compatibility)
            if self.stems_available and self.stem_data:
                input_chunk = np.zeros(input_chunk_size, dtype=np.float32)  # Mono
                for stem_name, stem_audio in self.stem_data.items():
                    if start_frame + input_chunk_size <= len(stem_audio):
                        # Get raw stem chunk
                        stem_chunk = stem_audio[start_frame:start_frame + input_chunk_size]
                        
                        # Convert to mono if stereo (average both channels like beat_viewer)
                        if stem_chunk.ndim == 2:
                            if stem_chunk.shape[1] == 2:
                                stem_chunk = np.mean(stem_chunk, axis=1)  # Average left + right
                            elif stem_chunk.shape[0] == 2:
                                stem_chunk = np.mean(stem_chunk, axis=0)  # Average channels
                        
                        # Apply per-stem tone EQ if enabled (EQ expects mono)
                        if (stem_name in self.stem_tone_eqs and 
                            self.stem_eq_enabled.get(stem_name, False)):
                            try:
                                stem_chunk_eq = self.stem_tone_eqs[stem_name].process_block(stem_chunk)
                                stem_chunk = stem_chunk_eq.flatten() if stem_chunk_eq.ndim > 1 else stem_chunk_eq
                            except Exception as e:
                                logger.warning(f"Deck {self.deck_id} - Stem EQ error for {stem_name}: {e}")
                        
                        # Apply stem volume
                        stem_volume = self.stem_volumes.get(stem_name, 1.0)
                        input_chunk += stem_chunk * stem_volume
            else:
                # Use main audio data (convert to mono by averaging like beat_viewer)
                main_audio = self.audio_thread_data[start_frame:start_frame + input_chunk_size]
                if main_audio.ndim == 2:
                    if main_audio.shape[1] == 2:
                        input_chunk = np.mean(main_audio, axis=1)  # Average left + right
                    elif main_audio.shape[0] == 2:
                        input_chunk = np.mean(main_audio, axis=0)  # Average channels
                else:
                    input_chunk = main_audio
            
            # Process with streaming RubberBand (same as beat_viewer primary method)
            with self._rb_lock:
                if self.rubberband_stretcher:
                    # Convert mono to 2D format for streaming RubberBand (like beat_viewer)
                    # Beat_viewer uses: input_block = np.column_stack([input_chunk, input_chunk])
                    input_block = np.column_stack([input_chunk, input_chunk]).astype(np.float32)
                    
                    # Validate buffer (same as beat_viewer)
                    input_block = self._validate_rubberband_buffer(input_block)
                    
                    # Process with streaming RubberBand (same as beat_viewer)
                    self.rubberband_stretcher.process(input_block, final=False)
                    
                    # Retrieve available output
                    available = self.rubberband_stretcher.available()
                    if available > 0:
                        output_block = self.rubberband_stretcher.retrieve(min(available, out_frames))
                        if output_block is not None and len(output_block) > 0:
                            # Convert back to mono like beat_viewer: average stereo channels
                            if output_block.shape[1] == 2:
                                processed_mono = np.mean(output_block, axis=1)  # Average left + right
                            else:
                                processed_mono = output_block[:, 0]
                            
                            # Convert back to stereo for output
                            processed_chunk = np.column_stack([processed_mono, processed_mono])
                        else:
                            logger.warning(f"Deck {self.deck_id} - RubberBand retrieve returned None/empty - returning silence")
                            processed_chunk = np.zeros((out_frames, 2), dtype=np.float32)
                    else:
                        # No output available yet, return silence or fall back
                        processed_chunk = np.zeros((out_frames, 2), dtype=np.float32)
                else:
                    logger.error(f"Deck {self.deck_id} - RubberBand stretcher not available - returning silence")
                    return np.zeros((out_frames, 2), dtype=np.float32)
                
                # Update playback position based on actual input consumed
                self.audio_thread_current_frame += input_chunk_size
                
                # === NEW: Frame-accurate seamless looping ===
                if hasattr(self, '_frame_accurate_loop') and self._frame_accurate_loop and self._frame_accurate_loop.get('active'):
                    loop_info = self._frame_accurate_loop
                    
                    # Debug logging for all loops every 0.5 seconds
                    if self.audio_thread_current_frame % 22050 == 0:  # Log every 0.5 seconds
                        logger.info(f"🔍 DEBUG Loop {loop_info['action_id']}: current_frame={self.audio_thread_current_frame}, end_frame={loop_info['end_frame']}, diff={loop_info['end_frame'] - self.audio_thread_current_frame}")
                    
                    if self.audio_thread_current_frame >= loop_info['end_frame']:
                        logger.info(f"🔄 Deck {self.deck_id}: Seamless loop jump - {loop_info['action_id']} "
                                   f"(frame {self.audio_thread_current_frame} → {loop_info['start_frame']})")
                        
                        # Seamlessly jump back to loop start
                        self.audio_thread_current_frame = loop_info['start_frame']
                        
                        # Track repetitions (jumps)
                        loop_info['current_repetition'] += 1
                        # CRITICAL FIX: repetitions = total plays = jumps + 1, so stop when jumps >= repetitions
                        if loop_info['repetitions'] > 0 and loop_info['current_repetition'] >= loop_info['repetitions']:
                            logger.info(f"🔄 Deck {self.deck_id}: Loop completed after {loop_info['repetitions']} total plays")
                            
                            # Store the completed loop's action_id before triggering completion handlers
                            completed_action_id = loop_info['action_id']
                            
                            # CRITICAL FIX: Offload loop completion actions from audio thread
                            # Heavy actions like crossfades should not block audio processing
                            if hasattr(self, 'musical_timing_system') and self.musical_timing_system:
                                def handle_completion_async():
                                    try:
                                        triggered_count = self.musical_timing_system.handle_loop_completion(completed_action_id)
                                        logger.info(f"🔄 Deck {self.deck_id}: Loop completion triggered {triggered_count} dependent actions")
                                    except Exception as e:
                                        logger.error(f"Deck {self.deck_id}: Error handling loop completion for {completed_action_id}: {e}")
                                
                                # Execute loop completion actions in background thread to avoid blocking audio
                                import threading
                                completion_thread = threading.Thread(target=handle_completion_async, daemon=True)
                                completion_thread.start()
                                logger.debug(f"🔄 Deck {self.deck_id}: Loop completion offloaded to background thread")
                            
                            # CRITICAL FIX: Only deactivate if this is still the same loop
                            # (completion handler might have activated a new loop)
                            if (hasattr(self, '_frame_accurate_loop') and 
                                self._frame_accurate_loop and 
                                self._frame_accurate_loop.get('action_id') == completed_action_id):
                                logger.info(f"🛑 Deck {self.deck_id}: Deactivating completed loop {completed_action_id}")
                                self._frame_accurate_loop['active'] = False
                                
                                # CRITICAL FIX: Continue normal playback after loop ends
                                # Don't stop the deck, just continue playing from current position
                                logger.info(f"🎵 Deck {self.deck_id}: Loop ended, continuing normal playback")
                                # Keep _user_wants_to_play = True to continue playback
                                with self._stream_lock:
                                    if not self._user_wants_to_play:
                                        logger.info(f"🎵 Deck {self.deck_id}: Restoring playback after loop completion")
                                        self._user_wants_to_play = True
                                
                            else:
                                logger.info(f"🔄 Deck {self.deck_id}: Loop {completed_action_id} completed, but new loop already active - not deactivating")
                
                # Keep display frame synchronized with actual frame in ring buffer architecture
                with self._stream_lock:
                    self._current_playback_frame_for_display = self.audio_thread_current_frame
                    
                    # Loop boundaries handled by centralized LoopManager (no duplicate detection needed)
                    
                    # Loop boundaries handled by centralized LoopManager (no duplicate detection needed)
                    
                    # Apply master isolator EQ
                    if self.master_isolator and processed_chunk is not None:
                        processed_chunk = self.master_isolator.process_block(processed_chunk)
                    
                    # Check if processed_chunk is still valid before volume
                    if processed_chunk is None:
                        logger.warning(f"Deck {self.deck_id} - processed_chunk is None after EQ - returning silence")
                        return np.zeros((out_frames, 2), dtype=np.float32)
                    
                    # CRITICAL FIX: Update volume fade before applying volume
                    self._update_volume_fade()
                    
                    # Apply volume
                    processed_chunk = processed_chunk * self._volume
                    
                    # Return requested number of frames (truncate or pad as needed)
                    if len(processed_chunk) >= out_frames:
                        result_chunk = processed_chunk[:out_frames]
                    else:
                        # Pad with zeros if not enough output
                        result_chunk = np.zeros((out_frames, 2), dtype=np.float32)
                        result_chunk[:len(processed_chunk)] = processed_chunk
                    
                    # Ensure correct stereo shape for ring buffer
                    if result_chunk.ndim == 1:
                        result_chunk = np.column_stack([result_chunk, result_chunk])
                    
                    return result_chunk.reshape(-1, 2)
                    
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - RubberBand processing error: {e}")
            import traceback
            traceback.print_exc()
            # Return silence on error to maintain consistent architecture
            return np.zeros((out_frames, 2), dtype=np.float32)

    def _init_rubberband_stretcher(self):
        """Initialize RubberBand streaming processor"""
        if not RUBBERBAND_STREAMING_AVAILABLE or self.sample_rate == 0:
            return
            
        try:
            with self._rb_lock:
                # Create RubberBand stretcher for streaming
                import rubberband_ctypes
                
                # RubberBand options for real-time processing
                options = rubberband_ctypes.RubberBandOptionProcessRealTime | \
                         rubberband_ctypes.RubberBandOptionTransientsMixed | \
                         rubberband_ctypes.RubberBandOptionPhaseLaminar | \
                         rubberband_ctypes.RubberBandOptionChannelsTogether
                
                self.rubberband_stretcher = rubberband_ctypes.RubberBand(
                    sample_rate=self.sample_rate,
                    channels=2,  # Stereo processing for better quality
                    options=options,
                    time_ratio=self.beat_manager.get_tempo_ratio(),
                    pitch_scale=1.0  # No pitch shifting for now
                )
                
                logger.info(f"Deck {self.deck_id} - RubberBand stretcher initialized (SR: {self.sample_rate}, ratio: {self.beat_manager.get_tempo_ratio()})")
                
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - Failed to initialize RubberBand: {e}")
            self.rubberband_stretcher = None

    def set_tempo_ratio(self, ratio):
        """Set tempo ratio for RubberBand processing - DEPRECATED: Use beat_manager.handle_tempo_change() instead"""
        logger.warning(f"Deck {self.deck_id} - set_tempo_ratio() is deprecated. Use beat_manager.handle_tempo_change() instead.")
        
        # Calculate new BPM from ratio
        if hasattr(self, 'original_bpm') and self.original_bpm > 0:
            new_bpm = self.original_bpm * ratio
            self.beat_manager.handle_tempo_change(new_bpm)
        
        if hasattr(self, 'rubberband_stretcher') and self.rubberband_stretcher:
            try:
                with self._rb_lock:
                    self.rubberband_stretcher.set_time_ratio(self.beat_manager.get_tempo_ratio())
                    logger.debug(f"Deck {self.deck_id} - RubberBand tempo ratio updated: {ratio}")
            except Exception as e:
                logger.error(f"Deck {self.deck_id} - Failed to set RubberBand tempo ratio: {e}")
    
    def set_pitch_ratio(self, ratio):
        """Set pitch ratio for RubberBand processing"""
        if hasattr(self, 'rubberband_stretcher') and self.rubberband_stretcher:
            try:
                with self._rb_lock:
                    self.rubberband_stretcher.set_pitch_ratio(float(ratio))
                    logger.debug(f"Deck {self.deck_id} - RubberBand pitch ratio updated: {ratio}")
            except Exception as e:
                logger.error(f"Deck {self.deck_id} - Failed to set RubberBand pitch ratio: {e}")
    
    def enable_pitch_preservation(self, enabled=True):
        """Enable or disable pitch preservation mode"""
        self.preserve_pitch_enabled = bool(enabled)
        logger.info(f"Deck {self.deck_id} - Pitch preservation {'enabled' if enabled else 'disabled'}")
        
        # Reinitialize RubberBand if switching modes
        if enabled and RUBBERBAND_STREAMING_AVAILABLE:
            self._init_rubberband_stretcher()

    def _validate_rubberband_buffer(self, input_block):
        """
        Validate buffer for RubberBand to prevent segfaults.
        Ensures proper dtype, shape, and contiguity.
        """
        # Check dtype
        if input_block.dtype != np.float32:
            input_block = input_block.astype(np.float32)
        
        # Check dimensions - must be 2D (frames, channels)
        if input_block.ndim != 2:
            if input_block.ndim == 1:
                input_block = input_block.reshape(-1, 1)
            else:
                raise ValueError(f"Invalid input shape for RubberBand: {input_block.shape}")
        
        # Ensure contiguous memory layout
        if not input_block.flags['C_CONTIGUOUS']:
            input_block = np.ascontiguousarray(input_block)
        
        # Check for invalid values
        if np.any(np.isnan(input_block)) or np.any(np.isinf(input_block)):
            logger.warning(f"Deck {self.deck_id} - Invalid values in RubberBand input, cleaning...")
            input_block = np.nan_to_num(input_block, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clamp extreme values to prevent distortion
        input_block = np.clip(input_block, -1.0, 1.0)
        
        return input_block

    def _cleanup_rubberband_on_stop(self):
        """Clean up RubberBand resources when playback stops"""
        try:
            with self._rb_lock:
                if hasattr(self, 'rubberband_stretcher') and self.rubberband_stretcher:
                    # Reset the stretcher to clean state
                    self.rubberband_stretcher.reset()
                    logger.debug(f"Deck {self.deck_id} - RubberBand stretcher reset")
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - Error cleaning up RubberBand: {e}")

    def _produce_chunk_turntable(self, target_frames):
        """Produce audio chunk without pitch preservation (turntable style)"""
        if self.audio_thread_data is None or len(self.audio_thread_data) == 0:
            return None

        # Build the stem mix with per-stem EQ (from beat viewer improvements)
        chunk = self._build_stem_mix_chunk(target_frames)
        
        if chunk is None:
            return None

        # Apply master isolator EQ
        if self.master_isolator:
            chunk = self.master_isolator.process_block(chunk)

        # Apply volume
        chunk = chunk * self._volume

        # Ensure chunk is a valid numpy array before reshaping
        if not isinstance(chunk, np.ndarray):
            chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.size == 0:
            chunk = np.zeros((target_frames, 2), dtype=np.float32)
        
        return chunk.reshape(-1, 2)  # Ensure correct stereo shape for ring buffer
    def _build_stem_mix_chunk(self, target_frames):
        """Build audio chunk from stems with per-stem processing"""
        if self.audio_thread_data is None or len(self.audio_thread_data) == 0:
            return None

        # Get current playback position
        start_frame = int(self.audio_thread_current_frame)
        end_frame = start_frame + target_frames
        
        # Store original end frame for loop detection
        original_end_frame = end_frame

        # Check bounds
        if start_frame >= len(self.audio_thread_data):
            return None

        # Get audio chunk
        if end_frame > len(self.audio_thread_data):
            # Handle end-of-track
            available_frames = len(self.audio_thread_data) - start_frame
            chunk = np.zeros(target_frames, dtype=np.float32)
            if available_frames > 0:
                chunk[:available_frames] = self.audio_thread_data[start_frame:start_frame + available_frames]
            self.audio_thread_current_frame = len(self.audio_thread_data)  # End of track
            
            # Keep display frame synchronized with actual frame in ring buffer architecture  
            with self._stream_lock:
                self._current_playback_frame_for_display = len(self.audio_thread_data)
        else:
            chunk = self.audio_thread_data[start_frame:end_frame]
            self.audio_thread_current_frame = end_frame
            
            # Keep display frame synchronized with actual frame in ring buffer architecture
            with self._stream_lock:
                self._current_playback_frame_for_display = end_frame
            
            # Loop boundaries handled by centralized LoopManager (no duplicate detection needed)
                # Loop boundaries handled by centralized LoopManager (no duplicate detection needed)
                
                # Loop start detection handled by centralized LoopManager
                
                # Check if we would cross the loop end frame with this chunk (only for active loops)
                # Loop end detection handled by centralized LoopManager
                    # Loop end detection handled by centralized LoopManager
                    # Loop repetition handling moved to centralized LoopManager
                    
                    # Loop continuation logic moved to centralized LoopManager
                    # Loop completion logic moved to centralized LoopManager

        # If stems are available, build mix from stems with per-stem EQ
        if self.stems_available and self.stem_data:
            # Initialize stereo mix chunk
            mix_chunk = np.zeros((len(chunk), 2), dtype=np.float32)
            
            # Use current audio thread frame position for stem alignment
            current_stem_frame = int(self.audio_thread_current_frame) - len(chunk)
            
            for stem_name, stem_audio in self.stem_data.items():
                if current_stem_frame + len(chunk) <= len(stem_audio) and current_stem_frame >= 0:
                    stem_chunk = stem_audio[current_stem_frame:current_stem_frame + len(chunk)]
                    
                    # Ensure stem chunk is stereo
                    if stem_chunk.ndim == 1:
                        stem_chunk = np.column_stack([stem_chunk, stem_chunk])
                    elif stem_chunk.ndim == 2 and stem_chunk.shape[1] == 1:
                        stem_chunk = np.column_stack([stem_chunk[:, 0], stem_chunk[:, 0]])
                    
                    # Apply per-stem tone EQ if enabled
                    if (stem_name in self.stem_tone_eqs and 
                        self.stem_eq_enabled.get(stem_name, False)):
                        try:
                            # Process each channel separately for EQ
                            stem_chunk_eq_left = self.stem_tone_eqs[stem_name].process_block(stem_chunk[:, 0])
                            stem_chunk_eq_right = self.stem_tone_eqs[stem_name].process_block(stem_chunk[:, 1])
                            stem_chunk = np.column_stack([stem_chunk_eq_left.flatten(), stem_chunk_eq_right.flatten()])
                        except Exception as e:
                            logger.warning(f"Deck {self.deck_id} - Stem EQ error for {stem_name}: {e}")
                    
                    # Apply stem volume
                    stem_volume = self.stem_volumes.get(stem_name, 1.0)
                    stem_chunk = stem_chunk * stem_volume
                    
                    # Add to mix
                    mix_chunk += stem_chunk
            
            chunk = mix_chunk
        else:
            # Convert mono original audio to stereo
            if chunk.ndim == 1:
                chunk = np.column_stack([chunk, chunk])

        # Ensure chunk is a valid numpy array before reshaping
        if not isinstance(chunk, np.ndarray):
            chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.size == 0:
            chunk = np.zeros((target_frames, 2), dtype=np.float32)
        
        return chunk.reshape(-1, 2)  # Ensure correct stereo shape for ring buffer

    def _sd_callback(self, outdata, frames, time_info, status_obj):
        """Hybrid ring buffer + seamless loop audio callback for zero-latency transitions"""
        # DEVELOPMENT GUARD: Never call RubberBand methods in audio callback!
        # This callback reads from ring buffer AND handles loop transitions instantly
        assert not hasattr(self, '_in_audio_callback'), "Nested audio callback detected!"
        self._in_audio_callback = True
        
        try:
            # === SIMPLE RING BUFFER PROCESSING (like beat_viewer_fixed) ===
            # NO complex logic in audio callback - this prevents hiccups!
            
            # Status monitoring only
            if status_obj:
                if status_obj.output_underflow:
                    if not hasattr(self, '_had_underflow'):
                        self._had_underflow = True
                if status_obj.output_overflow:
                    if not hasattr(self, '_had_overflow'):
                        self._had_overflow = True
            
            # Read stereo audio from ring buffer (SIMPLE)
            if self.out_ring:
                out, n = self.out_ring.read(frames)
                if n < frames:
                    # Underrun: fill tail with zeros
                    out[n:] = 0.0
                    if not hasattr(self, '_had_underrun_logged'):
                        self._had_underrun_logged = True
                        logger.warning(f"🔊 Deck {self.deck_id} - Audio underrun: requested {frames} frames, got {n}")
            else:
                # No ring buffer - output silence
                out = np.zeros((frames, 2), dtype=np.float32)
            
            # CRITICAL FIX: Apply startup fade-in to prevent pop/click
            if hasattr(self, '_startup_fade_active') and self._startup_fade_active:
                fade_progress = self._startup_fade_frames / self._startup_fade_total_frames
                fade_progress = min(fade_progress, 1.0)
                
                # Use smooth fade-in curve (sine curve for natural sound)
                import math
                fade_gain = math.sin(fade_progress * math.pi / 2)  # Smooth 0.0 to 1.0 transition
                
                # Safety check: ensure fade gain is valid
                if not (0.0 <= fade_gain <= 1.0) or math.isnan(fade_gain):
                    fade_gain = 0.0
                    logger.warning(f"🔊 Deck {self.deck_id} - Invalid fade gain detected: {fade_gain}, using 0.0")
                
                out *= fade_gain
                
                # Update fade progress
                self._startup_fade_frames += frames
                
                # Disable fade when complete
                if self._startup_fade_frames >= self._startup_fade_total_frames:
                    self._startup_fade_active = False
                    logger.info(f"🔊 Deck {self.deck_id} - Startup fade-in complete")
            
            # === STEREO OUTPUT ROUTING ===
            if self.device_output_channels == 2 and out.shape[1] == 2:
                # Perfect stereo output
                outdata[:frames, 0] = out[:, 0]  # Left channel
                outdata[:frames, 1] = out[:, 1]  # Right channel
            else:
                # Fallback handling
                outdata[:frames, 0] = out[:, 0]  # Left
                if self.device_output_channels == 2:
                    outdata[:frames, 1] = out[:, 0]  # Right (duplicate left)
                
        except Exception as e:
            # Emergency: output silence on any error
            logger.error(f"🔊 Deck {self.deck_id} - CRITICAL: Audio callback error: {e}")
            outdata[:] = 0.0
        finally:
            delattr(self, '_in_audio_callback')

    def _on_stream_finished_from_audio_thread(self):
        logger.debug(f"DEBUG: Deck {self.deck_id} AudioThread - Stream finished_callback triggered.")
        was_seek = False 
        with self._stream_lock:
            self._is_actually_playing_stream_state = False 
            # It's crucial that the audio thread's local _current_stream_in_thread is set to None
            # when this callback is for *that* stream. We also update self.playback_stream_obj.
            if self.playback_stream_obj and (self.playback_stream_obj.stopped or self.playback_stream_obj.closed):
                logger.debug(f"DEBUG: Deck {self.deck_id} finished_callback - Clearing self.playback_stream_obj.")
                self.playback_stream_obj = None

            was_seek = self.seek_in_progress_flag 
            self.seek_in_progress_flag = False    

            if not was_seek: 
                # print(f"DEBUG: Deck {self.deck_id} finished_callback: Not a seek, setting user_wants_to_play=False.")
                self._user_wants_to_play = False
            
            natural_end_condition = (
                self.audio_thread_data is not None and
                self.audio_thread_current_frame >= self.audio_thread_total_samples and
                not (hasattr(self, '_frame_accurate_loop') and self._frame_accurate_loop and self._frame_accurate_loop.get('active', False))
            )
            if not was_seek and natural_end_condition :
                logger.debug(f"DEBUG: Deck {self.deck_id} finished_callback - Track ended naturally. Resetting frames.")
                self.audio_thread_current_frame = 0 
                self._current_playback_frame_for_display = 0
        logger.debug(f"DEBUG: Deck {self.deck_id} AudioThread - Stream finished_callback processed. User wants play: {self._user_wants_to_play}")

    def is_active(self): 
        with self._stream_lock: return self._is_actually_playing_stream_state
                   
    def get_current_display_frame(self):
        with self._stream_lock: return self._current_playback_frame_for_display
        
    def shutdown(self): 
        logger.debug(f"DEBUG: Deck {self.deck_id} - Shutdown requested from external.")
        
        # Stop real-time tempo processor
        if hasattr(self, '_realtime_tempo_processor') and self._realtime_tempo_processor:
            try:
                self._realtime_tempo_processor.stop_processing()
                logger.debug(f"Deck {self.deck_id} - Real-time tempo processor stopped")
            except Exception as e:
                logger.warning(f"Deck {self.deck_id} - Error stopping tempo processor: {e}")
        
        self.audio_thread_stop_event.set()
        try: self.command_queue.put((DECK_CMD_SHUTDOWN, None), timeout=0.1) 
        except queue.Full: logger.warning(f"WARNING: Deck {self.deck_id} - Command queue full during shutdown send.")
        logger.debug(f"DEBUG: Deck {self.deck_id} - Waiting for audio thread to join...")
        self.audio_thread.join(timeout=2.0) 
        if self.audio_thread.is_alive():
            logger.warning(f"WARNING: Deck {self.deck_id} - Audio thread did not join cleanly.")
            with self._stream_lock: stream = self.playback_stream_obj 
            if stream:
                try: 
                    logger.warning(f"WARNING: Deck {self.deck_id} - Forcing abort/close on lingering stream.")
                    stream.abort(ignore_errors=True); stream.close(ignore_errors=True)
                except Exception as e_close: logger.error(f"ERROR: Deck {self.deck_id} - Exception during forced stream close: {e_close}")
        # Shutdown thread pool executor
        if hasattr(self, '_processing_executor'):
            self._processing_executor.shutdown(wait=True)
        
        # Cleanup musical timing system
        if hasattr(self, 'musical_timing_system') and self.musical_timing_system:
            try:
                self.musical_timing_system.cleanup()
                logger.debug(f"Deck {self.deck_id} - Musical timing system cleaned up")
            except Exception as e:
                logger.warning(f"Deck {self.deck_id} - Error cleaning up musical timing system: {e}")
        
        logger.debug(f"DEBUG: Deck {self.deck_id} - Shutdown complete.")

    def _apply_eq_smoothing(self):
        """Update EQ values if smoothing is active (called in audio callback)"""
        if self._eq_smoothing_active:
            if not self._eq_smoothing_frames:
                # Avoid division by zero
                self._eq_low = self._eq_smoothing_target['low']
                self._eq_mid = self._eq_smoothing_target['mid']
                self._eq_high = self._eq_smoothing_target['high']
                self._eq_smoothing_active = False
                return
            t = (self._eq_smoothing_counter + 1) / self._eq_smoothing_frames
            t = min(t, 1.0)
            self._eq_low = self._eq_smoothing_start['low'] + (self._eq_smoothing_target['low'] - self._eq_smoothing_start['low']) * t
            self._eq_mid = self._eq_smoothing_start['mid'] + (self._eq_smoothing_target['mid'] - self._eq_smoothing_start['mid']) * t
            self._eq_high = self._eq_smoothing_start['high'] + (self._eq_smoothing_target['high'] - self._eq_smoothing_start['high']) * t
            self._eq_smoothing_counter += 1
            if self._eq_smoothing_counter >= self._eq_smoothing_frames:
                self._eq_low = self._eq_smoothing_target['low']
                self._eq_mid = self._eq_smoothing_target['mid']
                self._eq_high = self._eq_smoothing_target['high']
                self._eq_smoothing_active = False

    def trigger_scratch(self, pattern, duration_seconds):
        """Start a scratch effect with the given pattern and duration (in seconds)"""
        if self.sample_rate == 0 or self.audio_thread_data is None:
            logger.warning(f"Deck {self.deck_id} - Cannot start scratch: no audio loaded or sample rate 0.")
            return
        self._scratch_active = True
        self._scratch_pattern = pattern if pattern else [100, -100]
        self._scratch_duration_frames = int(duration_seconds * self.sample_rate)
        self._scratch_start_frame = self._current_playback_frame_for_display
        self._scratch_start_time = time.time()
        self._scratch_pattern_index = 0
        # Pattern segment duration: stretch pattern to fit total duration
        if len(self._scratch_pattern) > 0:
            self._scratch_pattern_frames = int(self._scratch_duration_frames / len(self._scratch_pattern))
        else:
            self._scratch_pattern_frames = self._scratch_duration_frames
        self._scratch_elapsed_frames = 0
        self._scratch_window_frames = int(2.0 * self.sample_rate)  # 2.0s window for much larger movement
        self._scratch_pointer = float(self._scratch_start_frame)
        self._scratch_cut_remaining = 0
        # --- Vinyl mode: Pause normal playback and remember state ---
        self._was_playing_before_scratch = self.is_active()
        self._user_wants_to_play = False
        logger.info(f"Deck {self.deck_id} - Scratch started: duration={self._scratch_duration_frames} frames, pattern={self._scratch_pattern}, window={self._scratch_window_frames} frames, cut={SCRATCH_CUT_FRAMES} frames")

    def get_track_key(self):
        """Get the current track's key information"""
        return {
            'key': getattr(self, 'key', 'unknown'),
            'confidence': getattr(self, 'key_confidence', 0.0)
        }

    def load_scratch_sample(self, sample_filepath):
        """Load a scratch sample and analyze its key"""
        if not os.path.exists(sample_filepath):
            logger.error(f"Deck {self.deck_id} - Scratch sample not found: {sample_filepath}")
            return False
        
        try:
            # Analyze the sample to get its key
            sample_analysis = self.analyzer.analyze_track(sample_filepath)
            if not sample_analysis:
                logger.error(f"Deck {self.deck_id} - Failed to analyze scratch sample: {sample_filepath}")
                return False
            
            # Load the sample audio
            loader = es.MonoLoader(filename=sample_filepath)
            sample_audio = loader()
            sample_sr = int(loader.paramValue('sampleRate'))
            
            # Store sample data
            self._scratch_sample = {
                'audio': sample_audio,
                'sample_rate': sample_sr,
                'key': sample_analysis.get('key', 'unknown'),
                'key_confidence': sample_analysis.get('key_confidence', 0.0),
                'filepath': sample_filepath
            }
            
            logger.debug(f"Deck {self.deck_id} - Loaded scratch sample: {os.path.basename(sample_filepath)} (key: {self._scratch_sample['key']})")
            return True
            
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - Error loading scratch sample {sample_filepath}: {e}")
            return False






if __name__ == '__main__':
    # (Test block remains the same as the version that worked for you)
    logger.debug("--- Deck Class Standalone Test (with Looping and Cue Points) ---")
    import sys
    CURRENT_DIR_OF_DECK_PY = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT_FOR_DECK_TEST = os.path.dirname(CURRENT_DIR_OF_DECK_PY) 
    if PROJECT_ROOT_FOR_DECK_TEST not in sys.path: sys.path.append(PROJECT_ROOT_FOR_DECK_TEST)
    try:
        import config as app_config 
        from audio_engine.audio_analyzer import AudioAnalyzer 
    except ImportError as e: logger.error(f"ERROR: Could not import for test: {e}"); sys.exit(1)
    app_config.ensure_dir_exists(app_config.BEATS_CACHE_DIR)
    app_config.ensure_dir_exists(app_config.AUDIO_TRACKS_DIR)
    analyzer = AudioAnalyzer(
        cache_dir=app_config.BEATS_CACHE_DIR,
        beats_cache_file_extension=app_config.BEATS_CACHE_FILE_EXTENSION,
        beat_tracker_algo_name=app_config.DEFAULT_BEAT_TRACKER_ALGORITHM,
        bpm_estimator_algo_name=app_config.DEFAULT_BPM_ESTIMATOR_ALGORITHM
    )
    deck = Deck("testDeckA", analyzer_instance=analyzer)
    test_file_name = "starships.mp3" 
    test_audio_file = os.path.join(app_config.AUDIO_TRACKS_DIR, test_file_name)
    dummy_cue_filepath = test_audio_file + ".cue" 
    dummy_cue_data = { "intro_start": {"start_beat": 1}, "drop1": {"start_beat": 65}, "loop_point_beat_5": {"start_beat": 5} } 
    try:
        with open(dummy_cue_filepath, 'w') as f: json.dump(dummy_cue_data, f, indent=4)
        logger.debug(f"DEBUG: Test - Created/Updated dummy cue file: {dummy_cue_filepath}")
    except Exception as e_cue_write: logger.error(f"ERROR: Test - Could not create dummy cue file: {e_cue_write}")
    if not os.path.exists(test_audio_file): logger.warning(f"WARNING: Test audio file not found: {test_audio_file}.")
    else:
        if deck.load_track(test_audio_file):
            logger.info(f"\nTrack loaded: {deck.deck_id}. BPM: {deck.bpm}, SR: {deck.sample_rate}")
            if deck.bpm == 0: logger.warning("WARNING: BPM is 0, loop length calc will be incorrect.")
            logger.debug("\nPlaying from CUE 'intro_start'..."); deck.play(start_at_cue_name="intro_start")
            time.sleep(0.2); logger.debug("Playing for ~2.0s to reach near beat 5..."); time.sleep(2.0) 
            start_loop_beat = 5; loop_len_beats = 4; num_reps = 3
            logger.debug(f"\nActivating {loop_len_beats}-beat loop @ beat {start_loop_beat} for {num_reps} reps...")
            deck.activate_loop(start_beat=start_loop_beat, length_beats=loop_len_beats, repetitions=num_reps)
            loop_single_duration = (60.0 / deck.bpm * loop_len_beats if deck.bpm > 0 else 1.0)
            wait_for_loop_and_post = (loop_single_duration * num_reps) + 2.5 
            logger.debug(f"Waiting for loop ({num_reps} reps of ~{loop_single_duration:.2f}s) + ~2s post-loop (total ~{wait_for_loop_and_post:.2f}s)...")
            time.sleep(wait_for_loop_and_post) 
            logger.debug(f"\nAfter finite loop, Frame: {deck.get_current_display_frame()}, UserWantsPlay: {deck._user_wants_to_play}, Active: {deck.is_active()}")
            next_loop_start_beat = 0
            if deck.bpm > 0 and deck.sample_rate > 0 and len(deck.beat_timestamps) > 0:
                current_time_after_loop = deck.get_current_display_frame() / float(deck.sample_rate)
                current_beat_idx = np.searchsorted(deck.beat_timestamps, current_time_after_loop, side='right')
                next_loop_start_beat = current_beat_idx + 4 
                next_loop_start_beat = min(next_loop_start_beat, len(deck.beat_timestamps)) 
                if next_loop_start_beat <= 0 and len(deck.beat_timestamps) > 0 : next_loop_start_beat = len(deck.beat_timestamps) // 2 
            else: next_loop_start_beat = 25 
            logger.debug(f"\nActivating infinite loop @ beat {next_loop_start_beat} for {loop_len_beats} beats (plays 5s)...")
            if next_loop_start_beat > 0: 
                deck.activate_loop(start_beat=next_loop_start_beat, length_beats=loop_len_beats) 
                time.sleep(5)
                logger.debug(f"\nDeactivating loop, playing 2s more..."); deck.deactivate_loop()
                time.sleep(0.1) 
                if deck.is_active(): logger.debug(f"DEBUG: Test - Continuing after deactivation. Frame: {deck.get_current_display_frame()}"); time.sleep(2)
                else: logger.debug("DEBUG: Test - Not active after deactivate. Re-requesting play."); deck.play(); time.sleep(2)
            else: logger.warning("WARNING: Test - Could not determine valid start for infinite loop. Skipping.")
            logger.debug("\nStopping playback..."); deck.stop(); time.sleep(0.5) 
            logger.debug(f"\nFinal Frame: {deck.get_current_display_frame()}, UserWantsPlay: {deck._user_wants_to_play}, Active: {deck.is_active()}")
            logger.debug("\nLooping Test finished.")
        else: logger.error(f"Failed to load track: {deck.deck_id}")
    deck.shutdown() 
    if os.path.exists(dummy_cue_filepath):
        try: os.remove(dummy_cue_filepath)
        except Exception: pass
    logger.debug("--- Deck Test Complete ---")