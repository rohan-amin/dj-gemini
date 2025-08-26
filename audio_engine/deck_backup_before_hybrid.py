# dj-gemini/audio_engine/deck.py

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
from .stem_processor import create_stem_processor
from .stem_separation import create_stem_separator

# Check for RubberBand availability
try:
    import rubberband_ctypes
    RUBBERBAND_STREAMING_AVAILABLE = True
    logger.info("RubberBand streaming ctypes wrapper loaded successfully!")
except ImportError as e:
    RUBBERBAND_STREAMING_AVAILABLE = False
    logger.warning(f"RubberBand streaming not available: {e}")

try:
    import pyrubberband as pyrb
    PYRUBBERBAND_AVAILABLE = True
except ImportError:
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

PITCH_PRESERVATION_AVAILABLE = RUBBERBAND_STREAMING_AVAILABLE or PYRUBBERBAND_AVAILABLE or LIBROSA_AVAILABLE or RUBBERBAND_AVAILABLE

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

class IsolatorEQ:
    """
    DJ-style master isolator:
      - Parallel bands split by Linkwitz–Riley 24 dB/oct crossovers (LR4)
      - Per-band linear gains (0.0 = kill)
      - Persistent zi state; crossfaded coefficient updates (default 20 ms)
    """
    def __init__(self, sample_rate: int, f_lo: float = 200.0, f_hi: float = 2000.0, xfade_ms: float = 20.0):
        self.sr = int(sample_rate)
        self.f_lo = float(f_lo)
        self.f_hi = float(f_hi)
        self._xfade = int(max(1, xfade_ms * 1e-3 * self.sr))
        self._xfade_remaining = 0
        self._needs_priming = True
        
        # Current settings
        self.low_gain = 1.0
        self.mid_gain = 1.0
        self.high_gain = 1.0
        
        # Vectorized path state
        self._vectorized_filters = []
        self._vec_zi = []
        self._pending_filters = None
        self._pending_zi = None
        
        self._compute_vectorized_coefficients()

    def set_eq(self, low_gain=None, mid_gain=None, high_gain=None):
        """Set EQ gains (0.0 = kill, 1.0 = neutral, 2.0 = boost)"""
        changed = False
        if low_gain is not None and low_gain != self.low_gain:
            self.low_gain = float(low_gain)
            changed = True
        if mid_gain is not None and mid_gain != self.mid_gain:
            self.mid_gain = float(mid_gain)
            changed = True
        if high_gain is not None and high_gain != self.high_gain:
            self.high_gain = float(high_gain)
            changed = True
            
        if changed:
            self._compute_vectorized_coefficients()

    def _compute_vectorized_coefficients(self):
        """Compute vectorized filter coefficients"""
        filters = []
        
        # Convert gains to dB for filter design
        low_db = 20 * np.log10(max(self.low_gain, 1e-6))
        mid_db = 20 * np.log10(max(self.mid_gain, 1e-6)) 
        high_db = 20 * np.log10(max(self.high_gain, 1e-6))
        
        # Design filters (simplified for vectorized processing)
        if abs(low_db) > 0.1:
            b_low, a_low = signal.butter(2, self.f_lo / (self.sr / 2), 'low')
            b_low = b_low * (10 ** (low_db / 20))
            filters.append((b_low, a_low))
            
        if abs(mid_db) > 0.1:
            b_mid, a_mid = signal.butter(2, [self.f_lo / (self.sr / 2), self.f_hi / (self.sr / 2)], 'band')
            b_mid = b_mid * (10 ** (mid_db / 20))
            filters.append((b_mid, a_mid))
            
        if abs(high_db) > 0.1:
            b_high, a_high = signal.butter(2, self.f_hi / (self.sr / 2), 'high')
            b_high = b_high * (10 ** (high_db / 20))
            filters.append((b_high, a_high))

        self._schedule_vectorized_update(filters)

    def _schedule_vectorized_update(self, new_filters):
        """Schedule smooth transition to new filter coefficients"""
        new_zi = []
        for b, a in new_filters:
            try:
                zi = lfilter_zi(b, a)
                new_zi.append(zi)
            except:
                new_zi.append(np.zeros(len(a) - 1, dtype=np.float64))

        if not self._vectorized_filters:
            self._vectorized_filters = new_filters
            self._vec_zi = new_zi
            self._xfade_remaining = 0
            return

        self._pending_filters = new_filters
        self._pending_zi = new_zi
        self._xfade_remaining = self._xfade

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """Process audio block with vectorized EQ"""
        if x.size == 0:
            return x.astype(np.float32).reshape(-1, 1)

        xin = x.astype(np.float64, copy=False).reshape(-1)
        
        # Prime state on first use
        if self._needs_priming and len(xin) > 0:
            self._prime_state(xin[0])
            self._needs_priming = False

        if not self._vectorized_filters:
            return x.astype(np.float32).reshape(-1, 1)

        def run_chain(data, filters, zi_list):
            y = data
            for i, (b, a) in enumerate(filters):
                if i < len(zi_list) and zi_list[i] is not None:
                    y, zi_list[i] = signal.lfilter(b, a, y, zi=zi_list[i])
                else:
                    y = signal.lfilter(b, a, y)
            return y

        # No crossfade needed
        if self._pending_filters is None or self._xfade_remaining <= 0:
            y = run_chain(xin, self._vectorized_filters, self._vec_zi)
            return y.astype(np.float32).reshape(-1, 1)

        # Crossfade between old and new filters
        n = len(xin)
        nxf = min(self._xfade_remaining, n)

        vec_zi_tmp = [zi.copy() if zi is not None else None for zi in self._vec_zi]
        pend_zi_tmp = [zi.copy() if zi is not None else None for zi in self._pending_zi]

        y_old = run_chain(xin, self._vectorized_filters, vec_zi_tmp)
        y_new = run_chain(xin, self._pending_filters, pend_zi_tmp)

        if nxf > 0:
            w = np.linspace(0.0, 1.0, nxf, dtype=np.float64)
            y = y_old.copy()
            y[:nxf] = (1.0 - w) * y_old[:nxf] + w * y_new[:nxf]
            if nxf < n:
                y[nxf:] = y_new[nxf:]
            self._xfade_remaining -= nxf
        else:
            y = y_new

        # Commit new filters if crossfade is done
        if self._xfade_remaining <= 0:
            self._vectorized_filters = self._pending_filters
            self._vec_zi = pend_zi_tmp
            self._pending_filters = None
            self._pending_zi = None

        return y.astype(np.float32).reshape(-1, 1)

    def _prime_state(self, first_sample_value: float = 0.0):
        """Prime filter states to avoid startup transients"""
        if not self._vectorized_filters:
            return
        zis = []
        for b, a in self._vectorized_filters:
            try:
                zi = lfilter_zi(b, a) * float(first_sample_value)
            except:
                zi = np.zeros(len(a) - 1, dtype=np.float64)
            zis.append(zi)
        self._vec_zi = zis

    def reset(self):
        """Reset all filter states"""
        self._vec_zi = []
        self._pending_zi = []
        self._needs_priming = True

class Deck:
    def __init__(self, deck_id, analyzer_instance, engine_instance=None):
        self.deck_id = deck_id
        self.analyzer = analyzer_instance
        self.engine = engine_instance  # Reference to the engine for global samples
        logger.debug(f"Deck {self.deck_id} - Initializing...")

        self.filepath = None
        self.sample_rate = 0
        self.beat_timestamps = np.array([])
        self.bpm = 0.0
        self.total_frames = 0 
        self.cue_points = {} 
        
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

                # --- Loop state attributes ---
        self._loop_active = False
        self._loop_start_frame = 0
        self._loop_end_frame = 0
        self._loop_repetitions_total = None
        self._loop_repetitions_done = 0
        
        # --- Synchronized beat state ---
        self._last_synchronized_beat = 0

        # Add loop queue for handling multiple loops
        self._loop_queue = []

        # Add tempo state
        self._playback_tempo = 1.0  # 1.0 = original speed
        self._original_bpm = 0.0    # Store original BPM for calculations

        # Add tempo ramp state
        self._tempo_ramp_active = False
        self._ramp_start_beat = 0
        self._ramp_end_beat = 0
        self._ramp_start_bpm = 0
        self._ramp_end_bpm = 0
        self._ramp_curve = "linear"
        self._ramp_duration_beats = 0
        
        # Add dynamic beat tracking for ramps
        self._current_ramp_bpm = 0
        self._ramp_beat_timestamps = None  # Copy of original timestamps
        self._current_tempo_ratio = 1.0  # Current playback speed ratio

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
        self.RING_BUFFER_SIZE = 16384  # 16k frames buffer for smooth audio
        self.out_ring = None  # Will be initialized when sample rate is known
        self.master_isolator = None  # Will be initialized when sample rate is known
        
        # Producer thread for filling ring buffer
        self._producer_stop = False
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
        
        # Pitch preservation setting
        self.preserve_pitch_enabled = False  # Default to turntable mode (faster)

        self.audio_thread.start()
        logger.debug(f"Deck {self.deck_id} - Initialized and audio thread started.")

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
                self.out_ring = RingBuffer(self.RING_BUFFER_SIZE, channels=2)
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
            # Check if stems already exist
            stems_cache_dir = self._get_stems_cache_dir(audio_filepath)
            if self._check_stems_exist(audio_filepath):
                logger.info(f"Deck {self.deck_id} - Loading cached stems for {os.path.basename(audio_filepath)}")
                self._load_cached_stems(audio_filepath)
            else:
                logger.info(f"Deck {self.deck_id} - Generating stems for {os.path.basename(audio_filepath)}")
                self._generate_and_cache_stems(audio_filepath, audio_data)
                
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - Failed to load/generate stems: {e}")
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

    def _generate_and_cache_stems(self, audio_filepath, audio_data):
        """Generate stems using stem separation and cache them"""
        try:
            # Use stem separation from the existing module
            logger.info(f"Deck {self.deck_id} - Starting stem separation (this may take a while)...")
            
            # Generate stems - this returns a StemSeparationResult
            stem_separator = create_stem_separator()
            stem_result = stem_separator.separate_file(audio_filepath)
            
            if stem_result and stem_result.stems:
                # Cache the generated stems
                stems_dir = self._get_stems_cache_dir(audio_filepath)
                os.makedirs(stems_dir, exist_ok=True)
                
                self.stem_data = {}
                for stem_name, stem_data_obj in stem_result.stems.items():
                    if stem_name in self.STEM_NAMES:
                        # Extract audio data from StemData object
                        stem_audio = stem_data_obj.audio_data
                        self.stem_data[stem_name] = stem_audio
                        
                        # Cache to disk
                        stem_file = os.path.join(stems_dir, f"{stem_name}.npy")
                        np.save(stem_file, stem_audio)
                        
                        logger.debug(f"Deck {self.deck_id} - Generated and cached {stem_name}: {len(stem_audio)} samples")
                
                self.stems_available = len(self.stem_data) > 0
                logger.info(f"Deck {self.deck_id} - Successfully generated and cached {len(self.stem_data)} stems")
            else:
                logger.error(f"Deck {self.deck_id} - Stem separation failed or returned no stems")
                self.stems_available = False
                
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - Failed to generate stems: {e}")
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

        self.filepath = audio_filepath 
        self.sample_rate = int(analysis_data.get('sample_rate', 0))
        self.beat_timestamps = np.array(analysis_data.get('beat_timestamps', []))
        self.bpm = float(analysis_data.get('bpm', 0.0))
        self.cue_points = analysis_data.get('cue_points', {}) 
        # Store key information
        self.key = analysis_data.get('key', 'unknown')
        self.key_confidence = float(analysis_data.get('key_confidence', 0.0))
        logger.debug(f"Deck {self.deck_id} - Loaded cue points: {list(self.cue_points.keys())}")
        logger.debug(f"Deck {self.deck_id} - Track key: {self.key} (confidence: {self.key_confidence:.2f})")

        if self.sample_rate == 0:
            logger.error(f"Deck {self.deck_id} - Invalid sample rate from analysis for {audio_filepath}")
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
            self._loop_active = False 
        
        # Initialize real-time tempo processor
        if self._use_realtime_tempo:
            try:
                self._realtime_tempo_processor = create_realtime_tempo_processor(
                    sample_rate=self.sample_rate,
                    channels=1 if len(self.audio_thread_data.shape) == 1 else self.audio_thread_data.shape[1],
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
        """Get frame number for a specific (possibly fractional) beat, accounting for ramp adjustments"""
        beat_int = int(beat_number)
        beat_frac = beat_number - beat_int
        
        # Check if we have the integer beat position
        if beat_int in self.original_beat_positions:
            original_frame = self.original_beat_positions[beat_int]
            
            # If we have fractional beats and the next beat exists, interpolate
            if beat_frac > 0 and (beat_int + 1) in self.original_beat_positions:
                next_frame = self.original_beat_positions[beat_int + 1]
                # Linear interpolation between beats
                interpolated_frame = original_frame + (next_frame - original_frame) * beat_frac
                original_frame = int(interpolated_frame)
            
            # Calculate current tempo ratio for scaling
            if self._tempo_ramp_active and self._current_tempo_ratio != 1.0:
                # Use ramp tempo ratio
                tempo_ratio = self._current_tempo_ratio
                scaled_frame = int(original_frame / tempo_ratio)
                logger.debug(f"Deck {self.deck_id} - Beat {beat_number}: original frame {original_frame} → scaled frame {scaled_frame} (ramp ratio: {tempo_ratio:.3f})")
                return scaled_frame
            elif hasattr(self, 'original_bpm') and self.original_bpm > 0:
                # Use regular tempo ratio
                tempo_ratio = self.bpm / self.original_bpm
                if abs(tempo_ratio - 1.0) > 0.001:  # Only scale if significantly different
                    scaled_frame = int(original_frame / tempo_ratio)
                    logger.debug(f"Deck {self.deck_id} - Beat {beat_number}: original frame {original_frame} → scaled frame {scaled_frame} (tempo ratio: {tempo_ratio:.3f})")
                    return scaled_frame
            
            return original_frame
        else:
            logger.warning(f"Deck {self.deck_id} - Beat {beat_number} not found in original positions")
            return 0

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
        """Get current beat count, accounting for ramp adjustments"""
        # Add timeout to prevent infinite hangs
        import time
        start_time = time.time()
        
        with self._stream_lock:
            if self.audio_thread_data is None or self.sample_rate == 0 or len(self.beat_timestamps) == 0:
                return 0 
            
            current_time_seconds = self._current_playback_frame_for_display / float(self.sample_rate)
            
            # Use ramp-adjusted beat timestamps if in ramp
            if self._tempo_ramp_active and self._ramp_beat_timestamps is not None:
                beat_timestamps_to_use = self._ramp_beat_timestamps
            else:
                beat_timestamps_to_use = self.beat_timestamps
                
            # Add timeout check
            if time.time() - start_time > 1.0:  # 1 second timeout
                logger.error(f"Deck {self.deck_id} - Timeout in get_current_beat_count()")
                return 0
                
            try:
                # Use side='left' to get the current beat (not the next beat)
                beat_count = np.searchsorted(beat_timestamps_to_use, current_time_seconds, side='left')
                return beat_count
            except Exception as e:
                logger.error(f"Deck {self.deck_id} - Exception in searchsorted: {e}")
                return 0
    
    def get_synchronized_beat_count(self):
        """Get current beat count using the same logic as the audio callback"""
        # FIXED: Use shared state updated by audio callback for deterministic timing
        if hasattr(self, '_last_synchronized_beat'):
            return self._last_synchronized_beat
        else:
            # Fallback to direct calculation if shared state not available
            with self._stream_lock:
                if self.audio_thread_data is None or self.sample_rate == 0 or len(self.beat_timestamps) == 0:
                    return 0
                
                current_time_seconds = self._current_playback_frame_for_display / float(self.sample_rate)
                return np.searchsorted(self.beat_timestamps, current_time_seconds, side='left')

    def set_phase_offset(self, offset_beats):
        """Set a phase offset to be applied when the deck starts playing"""
        logger.debug(f"Deck {self.deck_id} - Setting phase offset: {offset_beats} beats")
        self._pending_phase_offset_beats = offset_beats
        self._phase_offset_applied = False
    
    def apply_phase_offset(self):
        """Apply the pending phase offset by seeking the appropriate number of frames"""
        if self._phase_offset_applied or self._pending_phase_offset_beats == 0.0:
            return
        
        if self.bpm <= 0:
            logger.warning(f"Deck {self.deck_id} - Cannot apply phase offset: BPM is {self.bpm}")
            return
        
        # Calculate frames to offset
        frames_per_beat = (60.0 / self.bpm) * self.sample_rate
        offset_frames = int(self._pending_phase_offset_beats * frames_per_beat)
        
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
            target_start_frame = self.get_frame_from_beat(start_at_beat)
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
            self._loop_active = False 
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
            self._loop_active = False 
        self.command_queue.put((DECK_CMD_SEEK, {'frame': valid_target_frame, 
                                               'was_playing_intent': was_playing_intent}))

    def activate_loop(self, start_beat, length_beats, repetitions=None, action_id=None):
        if self.total_frames == 0:
            logger.warning(f"Deck {self.deck_id} - Cannot activate loop: no track loaded.")
            return False
        
        if self.bpm <= 0:
            logger.warning(f"Deck {self.deck_id} - Cannot activate loop: BPM is {self.bpm}.")
            return False
        
        # Calculate loop frames - USE ORIGINAL BPM FOR CONSISTENCY
        original_bpm = self.original_bpm if hasattr(self, 'original_bpm') else self.bpm
        frames_per_beat = (60.0 / original_bpm) * self.audio_thread_sample_rate
        loop_length_frames = int(length_beats * frames_per_beat)
        
        # Get start frame - ALWAYS USE ORIGINAL BEAT POSITIONS (don't scale during ramp)
        if hasattr(self, 'original_beat_positions') and start_beat in self.original_beat_positions:
            start_frame = self.original_beat_positions[start_beat]
            # REMOVED: Don't scale frame position during ramp - let audio callback handle tempo changes
        else:
            # Fallback to get_frame_from_beat
            start_frame = self.get_frame_from_beat(start_beat)
        
        # REMOVED: Pre-roll logic that was causing the artifact
        # The loop should start exactly at the specified beat, not before it
        
        with self._stream_lock:
            current_playback_frame = self._current_playback_frame_for_display
        
        logger.info(f"[LOOP ACTIVATION] Start frame: {start_frame}, Current pointer: {current_playback_frame}")
        
        # REMOVED: Crossfade logic that was masking the real issue
        
        # Calculate loop end frame
        end_frame = start_frame + loop_length_frames
        
        # DEBUG: Log the exact beat positions for loop start and end
        end_beat = start_beat + length_beats
        logger.info(f"[LOOP FRAME CALC] Start beat: {start_beat}, End beat: {end_beat}, Length beats: {length_beats}")
        logger.info(f"[LOOP FRAME CALC] Start frame: {start_frame}, End frame: {end_frame}, Length frames: {loop_length_frames}")
        logger.info(f"[LOOP FRAME CALC] Frames per beat: {frames_per_beat:.2f}")
        logger.info(f"[LOOP FRAME CALC] BPM: {self.bpm}, Sample rate: {self.sample_rate}")
        
        # LOG: Current playback frame at loop activation
        with self._stream_lock:
            current_playback_frame = self._current_playback_frame_for_display
        logger.info(f"[LOOP ACTIVATION TIMING] At loop activation: current playback frame = {current_playback_frame}, loop start frame = {start_frame}, delta = {current_playback_frame - start_frame}")
        
        logger.debug(f"Deck {self.deck_id} - Activating loop: {length_beats} beats ({loop_length_frames} frames), {repetitions} reps, Action ID: {action_id}")
        logger.debug(f"Deck {self.deck_id} - Loop frames: {start_frame} to {end_frame}")
        
        # DISABLED: Predictive buffer to see the actual timing gap
        # IMMEDIATE LOOP STATE UPDATE: Set loop state immediately to avoid race condition
        with self._stream_lock:
            self._loop_start_frame = start_frame  # Use exact start frame
            self._loop_end_frame = end_frame
            self._loop_repetitions_total = repetitions
            self._loop_repetitions_done = 0
            self._loop_active = True
            self._current_loop_action_id = action_id
            
            # IMMEDIATE position correction if we're already past the loop end
            if self.audio_thread_current_frame >= end_frame:
                logger.info(f"Deck {self.deck_id} - IMMEDIATE jump: past loop end ({self.audio_thread_current_frame} >= {end_frame}), jumping to start {start_frame}")
                self.audio_thread_current_frame = start_frame
                self._current_playback_frame_for_display = start_frame
                # Clear the loop started flag since we're at the start
                if hasattr(self, '_loop_started'):
                    delattr(self, '_loop_started')
            
            logger.debug(f"Deck {self.deck_id} - IMMEDIATE loop state set: {action_id} (start: {start_frame}, current: {self.audio_thread_current_frame})")
        
        # Send to audio thread for playback pointer update
        self.command_queue.put((DECK_CMD_ACTIVATE_LOOP, {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'repetitions': repetitions,
            'action_id': action_id
        }))
        
        return True

    def deactivate_loop(self):
        logger.debug(f"Deck {self.deck_id} - Engine requests DEACTIVATE_LOOP.")
        self.command_queue.put((DECK_CMD_DEACTIVATE_LOOP, None))

    def stop_at_beat(self, beat_number):
        """Stop playback when reaching a specific beat"""
        logger.debug(f"Deck {self.deck_id} - Engine requests STOP_AT_BEAT. Beat: {beat_number}")
        if self.total_frames == 0:
            logger.error(f"Deck {self.deck_id} - Cannot stop at beat: track not loaded.")
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
                    self.bpm = target_bpm
                    self._playback_tempo = tempo_ratio
                    # Scale beat timestamps for UI display
                    self._scale_beat_positions(tempo_ratio)
                
                logger.info(f"Deck {self.deck_id} - Real-time tempo set to {target_bpm} BPM (ratio: {tempo_ratio:.3f})")
                return True
                
            except Exception as e:
                logger.error(f"Deck {self.deck_id} - Real-time tempo setting failed: {e}")
                # Fall back to cached method if available
        
        # Fallback to cached tempo processing (legacy method)
        logger.debug(f"Deck {self.deck_id} - Falling back to cached tempo processing")
        cache_filepath = self._get_tempo_cache_filepath(target_bpm)
        if not cache_filepath or not os.path.exists(cache_filepath):
            logger.error(f"Deck {self.deck_id} - Tempo cache not found for {target_bpm} BPM")
            logger.error(f"Expected cache file: {cache_filepath}")
            logger.error(f"Real-time processing failed and no cache available")
            return False
        
        try:
            logger.debug(f"Deck {self.deck_id} - Loading tempo cache...")
            processed_audio = np.load(cache_filepath)
            
            with self._stream_lock:
                self.audio_thread_data = processed_audio
                self.audio_thread_total_samples = len(processed_audio)
                # Update the BPM to the target BPM
                self.bpm = target_bpm
                # Scale the beat timestamps to match the new tempo
                self._scale_beat_positions(target_bpm / self.original_bpm)
            
            logger.info(f"Deck {self.deck_id} - Successfully loaded tempo {target_bpm} BPM from cache")
            return True
            
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - Failed to load tempo cache: {e}")
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
    
    def set_stem_eq(self, stem_name, low=None, mid=None, high=None):
        """Set EQ levels for specific stem (0.0 to 3.0 for each band)"""
        if stem_name not in self.STEM_NAMES:
            logger.error(f"Deck {self.deck_id} - Invalid stem name: {stem_name}")
            return False
            
        with self._stream_lock:
            # Ensure stem EQ processor exists
            if stem_name not in self.stem_tone_eqs:
                if self.sample_rate > 0:
                    self.stem_tone_eqs[stem_name] = ToneEQ3(self.sample_rate)
                else:
                    logger.warning(f"Deck {self.deck_id} - Cannot create stem EQ: sample rate is 0")
                    return False
            
            # Convert linear gains to dB for ToneEQ3
            current_gains = self.get_stem_eq(stem_name)
            low_db = self._linear_to_db(low if low is not None else current_gains['low'])
            mid_db = self._linear_to_db(mid if mid is not None else current_gains['mid']) 
            high_db = self._linear_to_db(high if high is not None else current_gains['high'])
            
            # Apply to EQ processor
            self.stem_tone_eqs[stem_name].set_params_db(low_db, mid_db, high_db)
            
            logger.debug(f"Deck {self.deck_id} - {stem_name} EQ set: L={low_db:+.1f}dB, M={mid_db:+.1f}dB, H={high_db:+.1f}dB")
            return True

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

    def _update_tempo_ramp(self, current_time=None):
        """Update tempo based on active ramp - USE TIME-BASED PROGRESS"""
        if not self._tempo_ramp_active:
            return
        # PATCH: Ensure all ramp variables are not None
        required_vars = [self._ramp_start_beat, self._ramp_end_beat, self._ramp_start_bpm, self._ramp_end_bpm, self.original_bpm]
        if any(v is None for v in required_vars):
            logger.warning(f"Deck {self.deck_id} - Tempo ramp variables not fully initialized. Skipping update.")
            return
            
        # Remove the constant "called" message
        # print(f"DEBUG: Deck {self.deck_id} - _update_tempo_ramp called")
        
        try:
            # If current_time is not provided, calculate it
            if current_time is None:
                with self._stream_lock:
                    current_time = self.audio_thread_current_frame / self.audio_thread_sample_rate
                
            # Remove constant time messages
            # print(f"DEBUG: Deck {self.deck_id} - Current time: {current_time:.3f}")
            
            # Calculate progress based on time, not beat
            ramp_start_time = self._ramp_start_beat * 60.0 / self.original_bpm  # Convert beat to time
            ramp_end_time = self._ramp_end_beat * 60.0 / self.original_bpm
            ramp_duration_time = ramp_end_time - ramp_start_time
            
            # Remove constant ramp range messages
            # print(f"DEBUG: Deck {self.deck_id} - Ramp time range: {ramp_start_time:.3f} to {ramp_end_time:.3f} (duration: {ramp_duration_time:.3f})")
            
            if current_time < ramp_start_time:
                logger.debug(f"Deck {self.deck_id} - Haven't started ramp yet (time {current_time:.3f} < {ramp_start_time:.3f})")
                return  # Haven't started ramp yet
                
            if current_time >= ramp_end_time:
                # Ramp complete
                logger.debug(f"Deck {self.deck_id} - Ramp ending, setting final BPM: {self._ramp_end_bpm}")
                
                # Set final values without calling set_tempo()
                final_tempo_ratio = self._ramp_end_bpm / self.original_bpm
                self._current_ramp_bpm = self._ramp_end_bpm
                self._current_tempo_ratio = final_tempo_ratio
                self.bpm = self._ramp_end_bpm
                
                # Keep ramp active but mark as complete
                self._tempo_ramp_active = False
                self._ramp_beat_timestamps = None
                logger.debug(f"Deck {self.deck_id} - Tempo ramp complete: {self._ramp_end_bpm} BPM (ratio: {final_tempo_ratio:.3f})")
                return
                
            # Remove constant "in ramp zone" messages
            # print(f"DEBUG: Deck {self.deck_id} - In ramp zone (time {current_time:.3f})")
            
            # Calculate progress through ramp (0.0 to 1.0) - USE TIME-BASED PROGRESS
            ramp_progress = (current_time - ramp_start_time) / ramp_duration_time
            # Only print progress every 10% or when it changes significantly
            if not hasattr(self, '_last_printed_progress') or abs(ramp_progress - self._last_printed_progress) > 0.1:
                logger.debug(f"Deck {self.deck_id} - Ramp progress: {ramp_progress:.3f}")
                self._last_printed_progress = ramp_progress
            
            # Apply curve function
            if self._ramp_curve == "linear":
                curve_progress = ramp_progress
            elif self._ramp_curve == "exponential":
                curve_progress = ramp_progress ** 2
            elif self._ramp_curve == "smooth":
                curve_progress = 3 * ramp_progress ** 2 - 2 * ramp_progress ** 3  # S-curve
            elif self._ramp_curve == "step":
                # Step every 4 beats
                step_size = 4.0 / self._ramp_duration_beats
                curve_progress = int(ramp_progress / step_size) * step_size
            else:
                curve_progress = ramp_progress
                
            # Remove constant curve progress messages
            # print(f"DEBUG: Deck {self.deck_id} - Curve progress: {curve_progress:.3f}")
                
            # Calculate current BPM
            current_bpm = self._ramp_start_bpm + (self._ramp_end_bpm - self._ramp_start_bpm) * curve_progress
            # Only print BPM when it changes significantly
            if not hasattr(self, '_last_printed_bpm') or abs(current_bpm - self._last_printed_bpm) > 1.0:
                logger.debug(f"Deck {self.deck_id} - Calculated BPM: {current_bpm:.1f}")
                self._last_printed_bpm = current_bpm
            
            # Calculate tempo ratio for playback speed
            new_tempo_ratio = current_bpm / self.original_bpm
            # Only print ratio when it changes significantly
            if not hasattr(self, '_last_printed_ratio') or abs(new_tempo_ratio - self._last_printed_ratio) > 0.01:
                logger.debug(f"Deck {self.deck_id} - New tempo ratio: {new_tempo_ratio:.3f}")
                self._last_printed_ratio = new_tempo_ratio
            
            # Update more frequently - reduce threshold for changes
            if abs(new_tempo_ratio - self._current_tempo_ratio) > 0.0001:  # Much smaller threshold
                self._current_ramp_bpm = current_bpm
                self._current_tempo_ratio = new_tempo_ratio
                self.bpm = current_bpm
                logger.debug(f"Deck {self.deck_id} - Updated BPM: {current_bpm:.1f}, Ratio: {new_tempo_ratio:.3f}")
            # else:
            #     print(f"DEBUG: Deck {self.deck_id} - No significant change in tempo ratio")
                
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - Exception in _update_tempo_ramp: {e}")
            import traceback
            traceback.print_exc()
            # Disable ramp on error
            self._tempo_ramp_active = False

    def _recalculate_beat_positions_for_ramp(self, new_bpm):
        """Recalculate beat positions when tempo changes during ramp"""
        if new_bpm <= 0 or self.original_bpm <= 0:
            return
            
        # Calculate how much the beat positions have shifted
        tempo_ratio = new_bpm / self.original_bpm
        
        # Update beat timestamps (scale by tempo ratio)
        if len(self.beat_timestamps) > 0:
            # Scale beat timestamps by tempo ratio
            self.beat_timestamps = self.beat_timestamps / tempo_ratio
            
        # Update cue points
        for cue_name, cue_info in self.cue_points.items():
            if "start_beat" in cue_info:
                # Recalculate cue position based on new tempo
                beat_number = cue_info["start_beat"]
                if beat_number in self.original_beat_positions:
                    original_frame = self.original_beat_positions[beat_number]
                    new_frame = int(original_frame / tempo_ratio)
                    # Update the cue's frame position
                    cue_info["frame"] = new_frame
        
        logger.debug(f"Deck {self.deck_id} - Recalculated beat positions for BPM {new_bpm} (ratio: {tempo_ratio:.3f})")

    def ramp_tempo(self, start_beat, end_beat, start_bpm, end_bpm, curve="linear"):
        """Start a tempo ramp from start_beat to end_beat"""
        logger.debug(f"Deck {self.deck_id} - Starting tempo ramp: {start_bpm}→{end_bpm} BPM from beat {start_beat} to {end_beat}")
        
        with self._stream_lock:
            self._tempo_ramp_active = True
            self._ramp_start_beat = start_beat
            self._ramp_end_beat = end_beat
            self._ramp_start_bpm = start_bpm
            self._ramp_end_bpm = end_bpm
            self._ramp_curve = curve
            self._ramp_duration_beats = end_beat - start_beat
            self._current_ramp_bpm = start_bpm
            
            # CRITICAL FIX: The tempo ratio should be based on the start BPM vs current playback speed
            # We want to start at the start BPM and ramp to the end BPM
            # The ratio is how much we need to speed up/slow down the current playback
            current_playback_bpm = self.bpm  # Current playback BPM
            self._current_tempo_ratio = start_bpm / current_playback_bpm
            self.bpm = start_bpm
            
            # Create copies of original data for ramp calculations
            self._ramp_beat_timestamps = self.beat_timestamps.copy()
            
            logger.debug(f"Deck {self.deck_id} - Ramp initialized with {len(self._ramp_beat_timestamps)} beats")
            logger.debug(f"Deck {self.deck_id} - Current playback BPM: {current_playback_bpm}, Start BPM: {start_bpm}, Initial ratio: {self._current_tempo_ratio:.3f}")

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
                        self._loop_active = False 
                        self._loop_repetitions_total = None
                        self._loop_repetitions_done = 0
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

                        # Reset loop parameters whenever a new PLAY command (not just resume) is initiated
                        # This ensures a seek-via-play or play-from-cue clears old loop state.
                        self._loop_active = False
                        self._loop_start_frame = 0
                        self._loop_end_frame = 0
                        self._loop_repetitions_total = None
                        self._loop_repetitions_done = 0
                        self._loop_queue.clear()  # Clear any pending loops
                        logger.debug(f"Deck {self.deck_id} AudioThread - Loop state reset for PLAY.")

                        logger.debug(f"Deck {self.deck_id} AudioThread - Creating new stream. SR: {self.audio_thread_sample_rate}, Frame: {self.audio_thread_current_frame}")
                        _current_stream_in_thread = sd.OutputStream(
                            samplerate=self.audio_thread_sample_rate, channels=2,
                            callback=self._sd_callback, 
                            finished_callback=self._on_stream_finished_from_audio_thread, 
                            blocksize=256,  # PATCH: Lower buffer size for tighter loop cueing
                            device=None  # Use default device
                        )
                        self.playback_stream_obj = _current_stream_in_thread 
                        
                        # Set device output channels for ring buffer processing
                        self.device_output_channels = 2  # Stereo output
                        
                        # Start producer thread for ring buffer-based playback
                        self._start_producer_thread()
                        
                        _current_stream_in_thread.start()
                        self._is_actually_playing_stream_state = True
                        logger.debug(f"Deck {self.deck_id} AudioThread - Stream and producer thread started for PLAY.")
                    
                elif command == DECK_CMD_PAUSE:
                    logger.debug(f"Deck {self.deck_id} AudioThread - Processing PAUSE")
                    # Stop producer thread
                    self._producer_stop = True
                    if _current_stream_in_thread and _current_stream_in_thread.active:
                        logger.debug(f"Deck {self.deck_id} AudioThread - Calling stream.stop() for PAUSE.")
                        _current_stream_in_thread.stop(ignore_errors=True) 
                        logger.debug(f"Deck {self.deck_id} AudioThread - stream.stop() called for PAUSE.")
                    else: 
                         with self._stream_lock: self._is_actually_playing_stream_state = False

                elif command == DECK_CMD_STOP:
                    logger.debug(f"Deck {self.deck_id} AudioThread - Processing STOP")
                    # Stop producer thread
                    self._producer_stop = True
                    # Clean up RubberBand resources
                    self._cleanup_rubberband_on_stop()
                    with self._stream_lock: 
                        self.audio_thread_current_frame = 0 
                        self._current_playback_frame_for_display = 0 
                        self._is_actually_playing_stream_state = False
                        self._loop_active = False 
                        self._loop_repetitions_total = None
                        self._loop_repetitions_done = 0
                        self._loop_queue.clear()  # Clear any pending loops
                    logger.debug(f"Deck {self.deck_id} AudioThread - State reset for STOP.")

                elif command == DECK_CMD_SEEK:
                    new_frame = data['frame']
                    was_playing_intent = data['was_playing_intent']
                    logger.debug(f"Deck {self.deck_id} AudioThread - Processing SEEK to frame {new_frame}, was_playing_intent: {was_playing_intent}")
                    with self._stream_lock:
                        self.audio_thread_current_frame = new_frame
                        self._current_playback_frame_for_display = new_frame 
                        self._user_wants_to_play = was_playing_intent 
                        self._loop_active = False 
                        self._loop_repetitions_total = None
                        self._loop_repetitions_done = 0
                        self._loop_queue.clear()  # Clear any pending loops
                    if was_playing_intent: 
                        logger.debug(f"Deck {self.deck_id} AudioThread - SEEK: Re-queueing PLAY command.")
                        self.command_queue.put((DECK_CMD_PLAY, {'start_frame': new_frame}))
                    else: 
                        with self._stream_lock: self._is_actually_playing_stream_state = False
                        logger.debug(f"Deck {self.deck_id} AudioThread - SEEK: Was paused, position updated.")
                
                elif command == DECK_CMD_ACTIVATE_LOOP:
                    logger.debug(f"Deck {self.deck_id} AudioThread - Processing ACTIVATE_LOOP")
                    
                    # Add to queue or replace current loop
                    loop_data = {
                        'start_frame': data['start_frame'],
                        'end_frame': data['end_frame'],
                        'repetitions': data.get('repetitions'),
                        'action_id': data.get('action_id')
                    }
                    
                    # Log loop activation info without dumping arrays
                    start_frame = data['start_frame']
                    if self.audio_thread_data is not None:
                        logger.debug(f"[LOOP ACTIVATION] Frame {start_frame}: Loop activated")

                    with self._stream_lock:
                        # Always update loop parameters, even if another loop is active
                        logger.debug(f"Deck {self.deck_id} AudioThread - Setting new loop parameters: start={data['start_frame']}, end={data['end_frame']}, reps={data.get('repetitions')}")
                        self._loop_start_frame = data['start_frame']
                        self._loop_end_frame = data['end_frame']
                        self._loop_repetitions_total = data.get('repetitions')
                        self._loop_repetitions_done = 0
                        self._loop_active = True
                        self._current_loop_action_id = data.get('action_id')
                        
                        # Clear the loop started flag for the new loop
                        if hasattr(self, '_loop_started'):
                            delattr(self, '_loop_started')
                        
                        # Clear debug flags for new loop
                        if hasattr(self, '_loop_debug_logged_activation'):
                            delattr(self, '_loop_debug_logged_activation')
                        
                        logger.info(f"Deck {self.deck_id} AudioThread - New loop activated: {data.get('action_id')} (start={data['start_frame']}, end={data['end_frame']})")
                        
                        # If we're already past the loop end, jump back to start immediately
                        if self.audio_thread_current_frame >= self._loop_end_frame:
                            logger.info(f"Deck {self.deck_id} AudioThread - Already past loop end ({self.audio_thread_current_frame} >= {self._loop_end_frame}), jumping to start")
                            self.audio_thread_current_frame = self._loop_start_frame
                            self._current_playback_frame_for_display = self._loop_start_frame
                        elif self.audio_thread_current_frame >= self._loop_start_frame:
                            logger.info(f"Deck {self.deck_id} AudioThread - Already at loop start position, loop will be active immediately")

                elif command == DECK_CMD_DEACTIVATE_LOOP:
                    logger.debug(f"Deck {self.deck_id} AudioThread - Processing DEACTIVATE_LOOP")
                    with self._stream_lock:
                        self._loop_active = False
                        self._loop_queue.clear()  # Clear any pending loops
                        logger.debug(f"Deck {self.deck_id} AudioThread - Loop deactivated and queue cleared.")

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
        self._producer_stop = False
        self._producer_error = None
        t = threading.Thread(target=self._producer_loop, daemon=True)
        t.start()
        logger.info(f"Deck {self.deck_id} - Producer thread started")
    
    def _producer_loop(self):
        """Continuously fill out_ring with processed audio."""
        logger.info(f"🎵 Deck {self.deck_id} - PRODUCER LOOP STARTED")
        TARGET_BLOCK = 8192  # frames per chunk we produce
        WATERMARK = self.sample_rate // 8 if self.sample_rate > 0 else 5512  # Fallback watermark
        
        # Ring buffer variables for partial writes
        self._pending_out = None

        while not self._producer_stop and self._user_wants_to_play:
            try:
                # Dynamic backoff for producer pacing
                available_data = self.out_ring.available_data() if self.out_ring else 0
                backoff = 0.002 if available_data > WATERMARK else 0.0
                if backoff:
                    logger.debug(f"🎵 Deck {self.deck_id} - RING BUFFER FULL: available={available_data}, watermark={WATERMARK} - backing off")
                    time.sleep(backoff)
                    continue
                
                logger.debug(f"🎵 Deck {self.deck_id} - PROCEEDING WITH CHUNK PRODUCTION: available_data={available_data}, watermark={WATERMARK}")

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
                
                # Use RubberBand if: pitch preservation is enabled AND we have active RB or pending operations
                use_rubberband = (RUBBERBAND_STREAMING_AVAILABLE and 
                                hasattr(self, 'preserve_pitch_enabled') and self.preserve_pitch_enabled and
                                (has_active_rb or has_pending_rb_ops))
                
                logger.info(f"🎵 Deck {self.deck_id} - PRODUCER PATH: use_rubberband={use_rubberband}")
                
                if use_rubberband:
                    chunk = self._produce_chunk_rubberband(TARGET_BLOCK)
                else:
                    chunk = self._produce_chunk_turntable(TARGET_BLOCK)

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
                
                # Update synchronized beat count for engine triggers
                if hasattr(self, 'beat_timestamps') and self.beat_timestamps is not None and self.audio_thread_sample_rate > 0:
                    try:
                        current_time_seconds = self.audio_thread_current_frame / self.audio_thread_sample_rate
                        beat_count = np.searchsorted(self.beat_timestamps, current_time_seconds, side='left')
                        self._last_synchronized_beat = beat_count
                    except Exception as e:
                        logger.debug(f"Deck {self.deck_id} - Error updating synchronized beat: {e}")
                
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

        logger.info(f"🎵 Deck {self.deck_id} - PRODUCER LOOP ENDED")
    def _produce_chunk_rubberband(self, out_frames):
        """Produce audio chunk using streaming RubberBand with proper locking"""
        if not RUBBERBAND_STREAMING_AVAILABLE:
            # Fall back to turntable method if RubberBand not available
            return self._produce_chunk_turntable(out_frames)
            
        try:
            # Initialize RubberBand stretcher if needed
            if not hasattr(self, 'rubberband_stretcher') or not self.rubberband_stretcher:
                self._init_rubberband_stretcher()
            
            # Get current playback position
            start_frame = int(self.audio_thread_current_frame)
            
            # Check bounds
            if start_frame >= len(self.audio_thread_data):
                return None
                
            # Calculate input chunk size based on tempo ratio
            # RubberBand streaming needs input chunks to generate output
            input_chunk_size = int(out_frames * self.current_tempo_ratio) + 1024  # Extra buffer for RB
            input_chunk_size = min(input_chunk_size, len(self.audio_thread_data) - start_frame)
            
            if input_chunk_size <= 0:
                return None
            
            # Build stem mix with per-stem EQ processing
            if self.stems_available and self.stem_data:
                input_chunk = np.zeros(input_chunk_size, dtype=np.float32)
                for stem_name, stem_audio in self.stem_data.items():
                    if start_frame + input_chunk_size <= len(stem_audio):
                        # Get raw stem chunk
                        stem_chunk = stem_audio[start_frame:start_frame + input_chunk_size]
                        
                        # Apply per-stem tone EQ if enabled
                        if (stem_name in self.stem_tone_eqs and 
                            self.stem_eq_enabled.get(stem_name, False)):
                            try:
                                stem_chunk_eq = self.stem_tone_eqs[stem_name].process_block(stem_chunk)
                                stem_chunk = stem_chunk_eq.flatten()  # Convert (N,1) back to (N,)
                            except Exception as e:
                                logger.warning(f"Deck {self.deck_id} - Stem EQ error for {stem_name}: {e}")
                        
                        # Apply stem volume
                        stem_volume = self.stem_volumes.get(stem_name, 1.0)
                        input_chunk += stem_chunk * stem_volume
            else:
                # Use main audio data
                input_chunk = self.audio_thread_data[start_frame:start_frame + input_chunk_size]
            
            # Process with RubberBand
            with self._rb_lock:
                if self.rubberband_stretcher:
                    # Convert to stereo for RubberBand (expects 2D array)
                    input_stereo = np.column_stack([input_chunk, input_chunk]).astype(np.float32)
                    
                    # Validate buffer before processing
                    input_stereo = self._validate_rubberband_buffer(input_stereo)
                    
                    # Process through RubberBand
                    processed_audio = self.rubberband_stretcher.process(input_stereo)
                    
                    # Convert back to mono (take left channel)
                    if processed_audio.ndim > 1:
                        processed_chunk = processed_audio[:, 0]
                    else:
                        processed_chunk = processed_audio
                    
                    # Update playback position based on actual input consumed
                    self.audio_thread_current_frame += input_chunk_size
                    
                    # Check for loop boundaries in RubberBand path too
                    if self._loop_active and self.audio_thread_current_frame >= self._loop_end_frame:
                        logger.debug(f"Deck {self.deck_id} - Loop end reached (RubberBand): frame {self.audio_thread_current_frame} >= {self._loop_end_frame}")
                        self._loop_repetitions_done += 1
                        logger.info(f"Deck {self.deck_id} - Loop repetition {self._loop_repetitions_done}/{self._loop_repetitions_total} completed")
                        
                        if self._loop_repetitions_total is None or self._loop_repetitions_done < self._loop_repetitions_total:
                            # Continue looping - jump back to start
                            self.audio_thread_current_frame = self._loop_start_frame
                            # Also update display frame
                            with self._stream_lock:
                                self._current_playback_frame_for_display = self._loop_start_frame
                            logger.info(f"Deck {self.deck_id} - Jumping back to loop start: frame {self._loop_start_frame}")
                        else:
                            # Loop complete - disable and continue
                            logger.info(f"Deck {self.deck_id} - Loop completed after {self._loop_repetitions_done} repetitions")
                            self._loop_active = False
                            # Clear loop started flag for next loop
                            if hasattr(self, '_loop_started'):
                                delattr(self, '_loop_started')
                            # Signal loop completion to engine
                            if hasattr(self, '_current_loop_action_id') and self._current_loop_action_id:
                                # Set flags that the engine will check (for legacy polling)
                                self._loop_just_completed = True
                                self._completed_loop_action_id = self._current_loop_action_id
                                logger.info(f"Deck {self.deck_id} - Loop completion signaled (RubberBand): action_id={self._current_loop_action_id}")
                                
                                # IMMEDIATE processing - call engine directly to eliminate gap
                                if self.engine and hasattr(self.engine, '_process_loop_completion_immediate'):
                                    self.engine._process_loop_completion_immediate(self.deck_id, self._current_loop_action_id)
                    
                    # Return requested number of frames (truncate or pad as needed)
                    if len(processed_chunk) >= out_frames:
                        return processed_chunk[:out_frames].reshape(-1, 1)
                    else:
                        # Pad with zeros if not enough output
                        padded = np.zeros(out_frames, dtype=np.float32)
                        padded[:len(processed_chunk)] = processed_chunk
                        return padded.reshape(-1, 1)
                else:
                    # RubberBand not available, fall back to turntable method
                    return self._produce_chunk_turntable(out_frames)
                    
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - RubberBand processing error: {e}")
            # Fall back to turntable method on error
            return self._produce_chunk_turntable(out_frames)

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
                         rubberband_ctypes.RubberBandOptionStretchElastic | \
                         rubberband_ctypes.RubberBandOptionPhaseLaminar
                
                self.rubberband_stretcher = rubberband_ctypes.RubberBandStretcher(
                    sample_rate=self.sample_rate,
                    channels=2,  # Stereo processing for better quality
                    options=options,
                    initial_time_ratio=self.current_tempo_ratio,
                    initial_pitch_ratio=1.0  # No pitch shifting for now
                )
                
                logger.info(f"Deck {self.deck_id} - RubberBand stretcher initialized (SR: {self.sample_rate}, ratio: {self.current_tempo_ratio})")
                
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - Failed to initialize RubberBand: {e}")
            self.rubberband_stretcher = None

    def set_tempo_ratio(self, ratio):
        """Set tempo ratio for RubberBand processing"""
        self.current_tempo_ratio = float(ratio)
        
        if hasattr(self, 'rubberband_stretcher') and self.rubberband_stretcher:
            try:
                with self._rb_lock:
                    self.rubberband_stretcher.set_time_ratio(self.current_tempo_ratio)
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
        else:
            chunk = self.audio_thread_data[start_frame:end_frame]
            self.audio_thread_current_frame = end_frame
            
            # Check for loop boundaries - only when we would actually cross them
            if self._loop_active:
                # Immediate debug for newly activated loops
                if not hasattr(self, '_loop_debug_logged_activation'):
                    logger.info(f"🔄 Deck {self.deck_id} - LOOP DETECTION ACTIVE: start={self._loop_start_frame}, end={self._loop_end_frame}, current={self.audio_thread_current_frame}")
                    logger.info(f"🔄 Deck {self.deck_id} - LOOP DETECTION: start_frame={start_frame}, end_frame={end_frame}, original_end_frame={original_end_frame}")
                    self._loop_debug_logged_activation = True
                # Debug: Log current frame vs loop boundaries every few chunks
                if hasattr(self, '_loop_debug_counter'):
                    self._loop_debug_counter += 1
                else:
                    self._loop_debug_counter = 0
                
                if self._loop_debug_counter % 50 == 0:  # Every 50 chunks for more frequent debugging
                    logger.info(f"Deck {self.deck_id} - LOOP DEBUG: current={self.audio_thread_current_frame}, start={self._loop_start_frame}, end={self._loop_end_frame}, reps={self._loop_repetitions_done}/{self._loop_repetitions_total}")
                    logger.info(f"Deck {self.deck_id} - LOOP DEBUG: Processing path: _build_stem_mix_chunk, target_frames={target_frames}")
                    logger.info(f"Deck {self.deck_id} - LOOP DEBUG: start_frame={start_frame}, end_frame={end_frame}, original_end_frame={original_end_frame}")
                    logger.info(f"Deck {self.deck_id} - LOOP DEBUG: loop_active={self._loop_active}, current_loop_action_id={getattr(self, '_current_loop_action_id', 'None')}")
                
                # Check if we've reached the loop start frame (for loops activated in the future or at current position)
                if (start_frame <= self._loop_start_frame and original_end_frame >= self._loop_start_frame and 
                    self._loop_repetitions_done == 0 and 
                    not hasattr(self, '_loop_started')):
                    logger.info(f"Deck {self.deck_id} - Loop start reached: crossing from {start_frame} to {original_end_frame}, loop start {self._loop_start_frame}")
                    self._loop_started = True  # Mark that we've entered the loop
                
                # Check if we would cross the loop end frame with this chunk
                if start_frame < self._loop_end_frame and original_end_frame >= self._loop_end_frame:
                    logger.info(f"Deck {self.deck_id} - Loop end reached: crossing from {start_frame} to {original_end_frame}, loop end {self._loop_end_frame}")
                    self._loop_repetitions_done += 1
                    logger.info(f"Deck {self.deck_id} - Loop repetition {self._loop_repetitions_done}/{self._loop_repetitions_total} completed")
                    
                    if self._loop_repetitions_total is None or self._loop_repetitions_done < self._loop_repetitions_total:
                        # Continue looping - jump back to start
                        self.audio_thread_current_frame = self._loop_start_frame
                        # Also update display frame
                        with self._stream_lock:
                            self._current_playback_frame_for_display = self._loop_start_frame
                        logger.info(f"Deck {self.deck_id} - Jumping back to loop start: frame {self._loop_start_frame}")
                        
                        # Fetch fresh audio data from the loop start position
                        loop_start_frame = int(self._loop_start_frame)
                        loop_end_frame = loop_start_frame + target_frames
                        
                        # Ensure we don't read beyond bounds
                        if loop_end_frame > len(self.audio_thread_data):
                            loop_end_frame = len(self.audio_thread_data)
                        
                        if loop_start_frame < len(self.audio_thread_data):
                            # Get fresh chunk from loop start position
                            fresh_chunk = self.audio_thread_data[loop_start_frame:loop_end_frame]
                            # Pad with zeros if needed
                            if len(fresh_chunk) < target_frames:
                                padded_chunk = np.zeros(target_frames, dtype=np.float32)
                                padded_chunk[:len(fresh_chunk)] = fresh_chunk
                                chunk = padded_chunk
                            else:
                                chunk = fresh_chunk
                            # Update current frame position
                            self.audio_thread_current_frame = loop_start_frame + len(fresh_chunk)
                        else:
                            # Fallback: silence if position is invalid
                            chunk = np.zeros(target_frames, dtype=np.float32)
                        
                        logger.debug(f"Deck {self.deck_id} - Fetched fresh audio from loop start (frames {loop_start_frame}-{loop_end_frame})")
                        
                        # Continue processing with the fresh chunk (don't return early)
                    else:
                        # Loop complete - disable and continue
                        logger.info(f"Deck {self.deck_id} - Loop completed after {self._loop_repetitions_done} repetitions")
                        self._loop_active = False
                        # Clear loop started flag for next loop
                        if hasattr(self, '_loop_started'):
                            delattr(self, '_loop_started')
                        # Signal loop completion to engine
                        if hasattr(self, '_current_loop_action_id') and self._current_loop_action_id:
                            # Set flags that the engine will check (for legacy polling)
                            self._loop_just_completed = True
                            self._completed_loop_action_id = self._current_loop_action_id
                            logger.info(f"Deck {self.deck_id} - Loop completion signaled: action_id={self._current_loop_action_id}")
                            
                            # IMMEDIATE processing - call engine directly to eliminate gap
                            if self.engine and hasattr(self.engine, '_process_loop_completion_immediate'):
                                self.engine._process_loop_completion_immediate(self.deck_id, self._current_loop_action_id)

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
        """Ring buffer-based audio callback - optimized for click-free playback"""
        # DEVELOPMENT GUARD: Never call RubberBand methods in audio callback!
        # This callback should only read from ring buffer - all processing
        # happens in the producer thread to prevent segfaults and dropouts.
        assert not hasattr(self, '_in_audio_callback'), "Nested audio callback detected!"
        self._in_audio_callback = True
        
        try:
            # Debug: count audio callback calls
            if not hasattr(self, '_audio_callback_count'):
                self._audio_callback_count = 0
            self._audio_callback_count += 1
            if self._audio_callback_count % 100 == 0:  # Log every 100 calls
                logger.debug(f"🔊 Deck {self.deck_id} - AUDIO CALLBACK: call #{self._audio_callback_count}, requesting {frames} frames")
            
            # No logging/UI here; set flags only
            if status_obj:
                if status_obj.output_underflow:
                    if not hasattr(self, '_had_underflow'):
                        self._had_underflow = True
                if status_obj.output_overflow:
                    if not hasattr(self, '_had_overflow'):
                        self._had_overflow = True
            
            # Read from ring buffer - this is the only operation in the callback
            if self.out_ring:
                out, n = self.out_ring.read(frames)
                if n < frames:
                    # underrun: fill tail with zeros
                    out[n:] = 0.0
                    if not hasattr(self, '_had_underrun_logged'):
                        self._had_underrun_logged = True
                        logger.debug(f"🔊 Deck {self.deck_id} - Ring buffer underrun: got {n}, needed {frames}")
            else:
                # No ring buffer - output silence
                out = np.zeros((frames, 2), dtype=np.float32)
            
            # Handle output channel configuration
            if self.device_output_channels == 2 and out.shape[1] == 2:
                # Stereo output
                outdata[:frames, 0] = out[:, 0]  # Left
                outdata[:frames, 1] = out[:, 1]  # Right
            else:
                # Fallback - mono or single channel
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
                not self._loop_active 
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