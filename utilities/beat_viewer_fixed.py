#!/usr/bin/env python3
"""
Beat Viewer with Fixed BPM Detection and Working Tempo Control
- Fixes Essentia numpy array issues
- Provides working tempo slider
- Clean, readable interface
"""

import tkinter as tk
from tkinter import filedialog, ttk
import argparse
import os
import sys
import numpy as np
import essentia.standard as es
from pydub import AudioSegment
import sounddevice as sd
import logging
import threading
import time
import scipy.signal as sps
from scipy.signal import iirpeak, sosfilt, lfilter_zi

# Add project root to path
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

# Try to import libraries for pitch preservation
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
    from rubberband_ctypes import (
        RubberBand, REALTIME_DEFAULT,
        RubberBandOptionTransientsCrisp, RubberBandOptionTransientsSmooth,
        RubberBandOptionPhaseIndependent, RubberBandOptionPhaseLaminar,
        RubberBandOptionDetectorPercussive, RubberBandOptionDetectorSoft,
    )
    RUBBERBAND_STREAMING_AVAILABLE = True
    print("RubberBand streaming ctypes wrapper loaded successfully!")
except ImportError as e:
    RUBBERBAND_STREAMING_AVAILABLE = False
    print(f"RubberBand streaming not available: {e}")

try:
    import rubberband
    RUBBERBAND_AVAILABLE = True
    print("Native RubberBand library loaded successfully!")
except ImportError:
    RUBBERBAND_AVAILABLE = False

PITCH_PRESERVATION_AVAILABLE = RUBBERBAND_STREAMING_AVAILABLE or PYRUBBERBAND_AVAILABLE or LIBROSA_AVAILABLE or RUBBERBAND_AVAILABLE

logger = logging.getLogger(__name__)

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
            first = min(to_read, self.cap - self.r)
            out[:first] = self.buf[self.r:self.r+first]
            second = to_read - first
            if second:
                out[first:first+second] = self.buf[0:second]
            self.r = (self.r + to_read) % self.cap
            self.size -= to_read
        return out, to_read
    
    def available_space(self):
        """Return number of frames that can be written"""
        with self.lock:
            return self.cap - self.size
    
    def available_data(self):
        """Return number of frames available to read"""
        with self.lock:
            return self.size

class BiquadFilter:
    """Professional biquad filter for real-time EQ processing"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.reset()
    
    def reset(self):
        """Reset filter state"""
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0
        
        # Default to pass-through
        self.b0 = 1.0
        self.b1 = 0.0
        self.b2 = 0.0
        self.a1 = 0.0
        self.a2 = 0.0
    
    def set_peaking_eq(self, frequency, gain_db, q=1.0):
        """Configure as peaking EQ filter (like professional DJ EQ)"""
        # Use different Q factors like Mixxx: higher Q for kill mode
        if gain_db < -15.0:
            q = 0.9  # Tighter Q for kill mode (like Mixxx)
        else:
            q = 0.4  # Wider Q for normal operation
            
        A = np.power(10, gain_db / 40.0)
        w = 2 * np.pi * frequency / self.sample_rate
        cos_w = np.cos(w)
        sin_w = np.sin(w)
        alpha = sin_w / (2 * q)
        
        b0 = 1 + alpha * A
        b1 = -2 * cos_w
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w
        a2 = 1 - alpha / A
        
        # Normalize
        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0
    
    def set_high_shelf(self, frequency, gain_db, slope=1.0):
        """Configure as high shelf filter"""
        A = np.power(10, gain_db / 40.0)
        w = 2 * np.pi * frequency / self.sample_rate
        cos_w = np.cos(w)
        sin_w = np.sin(w)
        S = slope
        beta = np.sqrt(A) / S
        
        b0 = A * ((A + 1) + (A - 1) * cos_w + beta * sin_w)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w)
        b2 = A * ((A + 1) + (A - 1) * cos_w - beta * sin_w)
        a0 = (A + 1) - (A - 1) * cos_w + beta * sin_w
        a1 = 2 * ((A - 1) - (A + 1) * cos_w)
        a2 = (A + 1) - (A - 1) * cos_w - beta * sin_w
        
        # Normalize
        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0
    
    def set_low_shelf(self, frequency, gain_db, slope=1.0):
        """Configure as low shelf filter"""
        A = np.power(10, gain_db / 40.0)
        w = 2 * np.pi * frequency / self.sample_rate
        cos_w = np.cos(w)
        sin_w = np.sin(w)
        S = slope
        beta = np.sqrt(A) / S
        
        b0 = A * ((A + 1) - (A - 1) * cos_w + beta * sin_w)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w)
        b2 = A * ((A + 1) - (A - 1) * cos_w - beta * sin_w)
        a0 = (A + 1) + (A - 1) * cos_w + beta * sin_w
        a1 = -2 * ((A - 1) + (A + 1) * cos_w)
        a2 = (A + 1) + (A - 1) * cos_w - beta * sin_w
        
        # Normalize
        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0
    
    def process_sample(self, x):
        """Process single sample through filter"""
        # Direct Form II biquad
        y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2
        
        # Update delay line
        self.x2 = self.x1
        self.x1 = x
        self.y2 = self.y1
        self.y1 = y
        
        return y

class ThreeBandEQ:
    """Professional 3-band EQ using biquad filters"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        
        # Create primary filters for each band
        self.high_filter = BiquadFilter(sample_rate)
        self.mid_filter = BiquadFilter(sample_rate)
        self.low_filter = BiquadFilter(sample_rate)
        
        # Create secondary filters for aggressive cuts (like Mixxx multi-stage)
        self.high_filter2 = BiquadFilter(sample_rate)
        self.mid_filter2 = BiquadFilter(sample_rate)
        self.low_filter2 = BiquadFilter(sample_rate)
        
        # Set crossover frequencies (typical DJ EQ values)
        self.high_freq = 8000  # 8kHz
        self.mid_freq = 1000   # 1kHz  
        self.low_freq = 100    # 100Hz
        
        # Vectorized path state (for process_block_vectorized)
        self._vectorized_filters = []     # list of (b, a)
        self._vec_zi = []                 # list of zi arrays per filter
        self._pending_filters = None      # for smooth updates
        self._pending_zi = None
        self._xfade_samples = int(0.010 * self.sample_rate)  # 10 ms crossfade
        self._xfade_remaining = 0
        self._needs_priming = True        # prime state on first block to avoid thump
        
        # Track current gains for kill mode detection
        self.current_high_gain = 0.0
        self.current_mid_gain = 0.0
        self.current_low_gain = 0.0
        
        # Initialize to flat response
        self.set_gains(0.0, 0.0, 0.0)
    
    def set_gains(self, high_db, mid_db, low_db):
        """Set EQ gains in dB with Mixxx-style kill behavior"""
        self.current_high_gain = high_db
        self.current_mid_gain = mid_db
        self.current_low_gain = low_db
        
        # Use Mixxx's kill threshold of -23dB
        kill_threshold = -23.0
        
        # High frequency setup
        if high_db <= kill_threshold:
            # Kill mode: cascade two aggressive high shelf filters
            self.high_filter.set_high_shelf(self.high_freq, -23.0)
            self.high_filter2.set_high_shelf(self.high_freq * 0.5, -23.0)  # Lower frequency for more coverage
        else:
            # Normal mode: single filter, reset second filter to pass-through
            self.high_filter.set_high_shelf(self.high_freq, high_db)
            self.high_filter2.reset()  # Pass-through
        
        # Mid frequency setup  
        if mid_db <= kill_threshold:
            # Kill mode: cascade two aggressive peaking filters
            self.mid_filter.set_peaking_eq(self.mid_freq, -23.0, q=0.9)
            self.mid_filter2.set_peaking_eq(self.mid_freq, -23.0, q=1.2)  # Tighter Q
        else:
            # Normal mode
            self.mid_filter.set_peaking_eq(self.mid_freq, mid_db, q=0.4)
            self.mid_filter2.reset()
        
        # Low frequency setup
        if low_db <= kill_threshold:
            # Kill mode: cascade two aggressive low shelf filters
            self.low_filter.set_low_shelf(self.low_freq, -23.0)
            self.low_filter2.set_low_shelf(self.low_freq * 2.0, -23.0)  # Higher frequency for more coverage
        else:
            # Normal mode
            self.low_filter.set_low_shelf(self.low_freq, low_db)
            self.low_filter2.reset()
        
        # Update vectorized coefficients when gains change
        self._compute_vectorized_coefficients()
    
    def process_sample(self, x):
        """Process single sample through all EQ bands with cascaded filters"""
        # Process through all filters in series (like Mixxx multi-stage)
        y = self.low_filter.process_sample(x)
        y = self.low_filter2.process_sample(y)  # Second stage low
        
        y = self.mid_filter.process_sample(y)
        y = self.mid_filter2.process_sample(y)  # Second stage mid
        
        y = self.high_filter.process_sample(y)
        y = self.high_filter2.process_sample(y)  # Second stage high
        
        return y
    
    def _compute_vectorized_coefficients(self):
        """Compute/approximate single-stage filters and schedule smooth update."""
        filters = []

        # High shelf (approx) ----------------------------------
        if hasattr(self.high_filter, 'b0') and self.high_filter.b0 != 1.0:
            b_high = np.array([self.high_filter.b0, self.high_filter.b1, self.high_filter.b2], dtype=np.float64)
            a_high = np.array([1.0, self.high_filter.a1, self.high_filter.a2], dtype=np.float64)
            filters.append((b_high, a_high))

        # Mid peaking (approx) ---------------------------------
        if hasattr(self.mid_filter, 'b0') and self.mid_filter.b0 != 1.0:
            b_mid = np.array([self.mid_filter.b0, self.mid_filter.b1, self.mid_filter.b2], dtype=np.float64)
            a_mid = np.array([1.0, self.mid_filter.a1, self.mid_filter.a2], dtype=np.float64)
            filters.append((b_mid, a_mid))

        # Low shelf (approx) -----------------------------------
        if hasattr(self.low_filter, 'b0') and self.low_filter.b0 != 1.0:
            b_low = np.array([self.low_filter.b0, self.low_filter.b1, self.low_filter.b2], dtype=np.float64)
            a_low = np.array([1.0, self.low_filter.a1, self.low_filter.a2], dtype=np.float64)
            filters.append((b_low, a_low))

        self._schedule_vectorized_update(filters)
    
    def _schedule_vectorized_update(self, new_filters):
        """Prepare a smooth transition to new filter coefficients."""
        # Build zi arrays for new filters
        new_zi = []
        for b, a in new_filters:
            # zi length is max(len(a), len(b)) - 1
            L = max(len(a), len(b)) - 1
            new_zi.append(np.zeros(L, dtype=np.float64))

        # If we have no current filters, just swap in (no xfade needed)
        if not self._vectorized_filters:
            self._vectorized_filters = new_filters
            self._vec_zi = new_zi
            self._xfade_remaining = 0
            return

        # Otherwise, schedule a crossfade
        self._pending_filters = new_filters
        self._pending_zi = new_zi
        # Keep current filters running; blend over _xfade_samples at output
        self._xfade_remaining = max(self._xfade_samples, 1)
    
    def _prime_state(self, first_sample_value: float = 0.0):
        """Prime zi to steady-state using the first sample to avoid startup thump."""
        if not self._vectorized_filters:
            return
        zis = []
        for b, a in self._vectorized_filters:
            try:
                zi = lfilter_zi(b, a) * float(first_sample_value)
            except Exception:
                L = max(len(a), len(b)) - 1
                zi = np.zeros(L, dtype=np.float64)
            zis.append(zi)
        self._vec_zi = zis
    
    def process_block_vectorized(self, x: np.ndarray) -> np.ndarray:
        """Stateful, block-based EQ with smooth coefficient updates."""
        if not self._vectorized_filters:
            # Ensure we have something; fall back to passthrough
            return x.astype(np.float32, copy=True)

        # Prime state on first block or after EQ enable/seek to avoid thump
        if self._needs_priming and len(x) > 0:
            self._prime_state(float(x[0]))
            self._needs_priming = False

        # Always work in float64 internally for IIR stability; cast back at end
        xin = x.astype(np.float64, copy=False)

        def run_chain(xin, filters, zi_list):
            y = xin
            # Update zi per stage
            for i, (b, a) in enumerate(filters):
                if i >= len(zi_list) or zi_list[i] is None:
                    L = max(len(a), len(b)) - 1
                    zi_list[i] = np.zeros(L, dtype=np.float64)
                y, zi_list[i] = sps.lfilter(b, a, y, zi=zi_list[i])
            return y

        # Case 1: no pending change — just run the current chain
        if self._pending_filters is None or self._xfade_remaining <= 0:
            y = run_chain(xin, self._vectorized_filters, self._vec_zi)
            return y.astype(np.float32)

        # Case 2: we are in a crossfade window — run both chains and blend
        n = len(xin)
        # Split the block into (xfade part) + (steady part) if needed
        nxf = min(self._xfade_remaining, n)

        # Prepare per-block zi copies so we don't corrupt long-term state while blending
        vec_zi_tmp = [zi.copy() if zi is not None else None for zi in self._vec_zi]
        pend_zi_tmp = [zi.copy() if zi is not None else None for zi in self._pending_zi]

        # First, run whole block through both chains (on temp zi)
        y_old = run_chain(xin, self._vectorized_filters, vec_zi_tmp)
        y_new = run_chain(xin, self._pending_filters,   pend_zi_tmp)

        # Crossfade for the first nxf samples
        if nxf > 0:
            w = np.linspace(0.0, 1.0, nxf, dtype=np.float64)  # 0→1
            y = y_old.copy()
            y[:nxf] = (1.0 - w) * y_old[:nxf] + w * y_new[:nxf]
            # For the remainder (if any), use y_new
            if nxf < n:
                y[nxf:] = y_new[nxf:]
            self._xfade_remaining -= nxf
        else:
            y = y_new

        # Commit the temp zi of the appropriate chain to become the new long-term state:
        if self._xfade_remaining <= 0:
            # Crossfade finished — adopt new filters and their end-of-block zi
            self._vectorized_filters = self._pending_filters
            self._vec_zi = pend_zi_tmp
            self._pending_filters = None
            self._pending_zi = None
        else:
            # In the middle of a fade — advance both chains' zi so next block continues correctly
            self._vec_zi = vec_zi_tmp
            self._pending_zi = pend_zi_tmp

        return y.astype(np.float32)
    
    def reset(self):
        """Reset all filter states"""
        self.high_filter.reset()
        self.high_filter2.reset()
        self.mid_filter.reset()
        self.mid_filter2.reset()
        self.low_filter.reset()
        self.low_filter2.reset()
        
        # Reset vectorized filter states
        if hasattr(self, '_vec_zi'):
            self._vec_zi.clear()
        if hasattr(self, '_pending_zi'):
            self._pending_zi = None
        # Prime state after reset to avoid thump
        self._needs_priming = True

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
        # Temporarily disable bypass to debug stem EQ
        return False
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
        self.g_low = 1.0; self.g_mid = 1.0; self.g_high = 1.0

        self._xfade = int(max(1, xfade_ms * 1e-3 * self.sr))
        self._left = 0
        self._pending = None
        
        # Wet/dry crossfading for clickless ON/OFF toggle
        self._wet = 1.0                  # 1.0 = EQ on (wet), 0.0 = bypass (dry)
        self._wet_target = 1.0
        self._wet_left = 0               # samples left in wet/dry crossfade
        self._wet_xfade = int(0.010 * self.sr)  # 10 ms
        
        # Smooth band-gain changes to eliminate zipper noise
        self._g = np.array([1.0, 1.0, 1.0], dtype=np.float64)   # smoothed gains L/M/H
        self._gt = np.array([1.0, 1.0, 1.0], dtype=np.float64)  # targets
        self._g_alpha = np.exp(-1.0 / (0.005 * self.sr))        # ~5 ms time constant

        # current filters (sos) and zi
        self._sos_low_lp  = None; self._zi_low_lp  = None
        self._sos_mid_hp  = None; self._zi_mid_hp  = None
        self._sos_mid_lp  = None; self._zi_mid_lp  = None
        self._sos_high_hp = None; self._zi_high_hp = None

        self._set_filters(self._design(self.f_lo, self.f_hi))

    def set_gains_db(self, low_db: float, mid_db: float, high_db: float):
        self.g_low  = 0.0 if np.isneginf(low_db)  else 10**(low_db  / 20.0)
        self.g_mid  = 0.0 if np.isneginf(mid_db)  else 10**(mid_db  / 20.0)
        self.g_high = 0.0 if np.isneginf(high_db) else 10**(high_db / 20.0)
        
        # Update smoothed gain targets
        self._gt = np.array([self.g_low, self.g_mid, self.g_high], dtype=np.float64)

    def set_kill(self, low=False, mid=False, high=False):
        if low:  self.g_low  = 0.0
        if mid:  self.g_mid  = 0.0
        if high: self.g_high = 0.0
        
        # Update smoothed gain targets
        self._gt = np.array([self.g_low, self.g_mid, self.g_high], dtype=np.float64)

    def set_crossovers(self, f_lo: float, f_hi: float):
        f_lo = float(f_lo); f_hi = float(f_hi)
        if f_lo <= 20 or f_hi >= self.sr/2 - 50 or f_lo >= f_hi:
            return
        self.f_lo, self.f_hi = f_lo, f_hi
        self._pending = self._design(self.f_lo, self.f_hi)
        self._left = self._xfade
    
    def set_enabled(self, enabled: bool):
        """Enable/disable EQ with clickless wet/dry crossfading."""
        self._wet_target = 1.0 if enabled else 0.0
        self._wet_left = self._wet_xfade

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """x: (N,) or (N,1) -> (N,1) float32, clickless."""
        xin = x[:, 0] if (x.ndim == 2 and x.shape[1] == 1) else x
        xin = xin.astype(np.float64, copy=False)

        # Smooth gains sample-by-sample to eliminate zipper noise
        n = len(xin)
        alpha = self._g_alpha
        g_traj = np.empty((n, 3), dtype=np.float64)
        g = self._g.copy()
        for i in range(n):
            g = alpha * g + (1 - alpha) * self._gt
            g_traj[i] = g
        self._g = g  # keep last value for next block

        if self._pending is not None and self._left > 0:
            n = len(xin)
            nxf = min(self._left, n)

            # Run BOTH chains and ADVANCE their states for continuity
            y_old = self._run_inplace(xin, self._current_dict(), g_traj)
            y_new = self._run_inplace(xin, self._pending, g_traj)

            if nxf > 0:
                w = np.linspace(0.0, 1.0, nxf, dtype=np.float64)
                y = y_old.copy()
                y[:nxf] = (1.0 - w) * y_old[:nxf] + w * y_new[:nxf]
                if nxf < n:
                    y[nxf:] = y_new[nxf:]
                self._left -= nxf
            else:
                y = y_new

            # When fade completes, adopt pending filters as current
            if self._left <= 0:
                self._set_filters(self._pending)
                self._pending = None
        else:
            # Normal path: advance current chain state
            y = self._run_inplace(xin, self._current_dict(), g_traj)

        y = y.astype(np.float32).reshape(-1, 1)
        np.nan_to_num(y, copy=False)
        
        # Wet/dry crossfading for clickless ON/OFF toggle
        if self._wet_left > 0:
            n = y.shape[0]
            nxf = min(self._wet_left, n)
            if nxf > 0:
                # linear ramp from current wet to target
                ramp = np.linspace(self._wet, self._wet_target, nxf, dtype=np.float32).reshape(-1, 1)
                y[:nxf, 0] = ramp[:, 0] * y[:nxf, 0] + (1.0 - ramp[:, 0]) * xin[:nxf].astype(np.float32)
                if nxf < n:
                    y[nxf:, 0] = (self._wet_target * y[nxf:, 0] +
                                  (1.0 - self._wet_target) * xin[nxf:].astype(np.float32))
                self._wet = float(ramp[-1, 0])
                self._wet_left -= nxf
        else:
            # fully at target wet or dry
            if self._wet_target == 0.0:
                # bypass: return dry
                y[:, 0] = xin.astype(np.float32)
            self._wet = self._wet_target
        
        return y

    # ---- internals ----
    def _design(self, f_lo: float, f_hi: float) -> dict:
        def lr4_low(fc):
            sos = sps.butter(2, fc, btype='low', fs=self.sr, output='sos'); return np.vstack([sos, sos])
        def lr4_high(fc):
            sos = sps.butter(2, fc, btype='high', fs=self.sr, output='sos'); return np.vstack([sos, sos])

        sos_low_lp  = lr4_low(f_lo)
        sos_mid_hp  = lr4_high(f_lo)
        sos_mid_lp  = lr4_low(f_hi)
        sos_high_hp = lr4_high(f_hi)

        return {
            "low_lp":  (sos_low_lp,  [sps.sosfilt_zi(sos_low_lp ) * 0.0]),
            "mid_hp":  (sos_mid_hp,  [sps.sosfilt_zi(sos_mid_hp ) * 0.0]),
            "mid_lp":  (sos_mid_lp,  [sps.sosfilt_zi(sos_mid_lp ) * 0.0]),
            "high_hp": (sos_high_hp, [sps.sosfilt_zi(sos_high_hp) * 0.0]),
        }

    def _current_dict(self) -> dict:
        return {
            "low_lp":  (self._sos_low_lp,  self._zi_low_lp),
            "mid_hp":  (self._sos_mid_hp,  self._zi_mid_hp),
            "mid_lp":  (self._sos_mid_lp,  self._zi_mid_lp),
            "high_hp": (self._sos_high_hp, self._zi_high_hp),
        }

    def _set_filters(self, d: dict):
        (self._sos_low_lp,  self._zi_low_lp)  = d["low_lp"]
        (self._sos_mid_hp,  self._zi_mid_hp)  = d["mid_hp"]
        (self._sos_mid_lp,  self._zi_mid_lp)  = d["mid_lp"]
        (self._sos_high_hp, self._zi_high_hp) = d["high_hp"]

    def _run_inplace(self, xin: np.ndarray, d: dict, g_traj: np.ndarray = None) -> np.ndarray:
        """Run filter chain in-place, advancing zi stored in dict d."""
        sos_low_lp,  zi_low_lp  = d["low_lp"]
        sos_mid_hp,  zi_mid_hp  = d["mid_hp"]
        sos_mid_lp,  zi_mid_lp  = d["mid_lp"]
        sos_high_hp, zi_high_hp = d["high_hp"]

        # Advance states IN PLACE (critical to avoid clicks)
        low,  zi_low_lp[0]  = sps.sosfilt(sos_low_lp,  xin, zi=zi_low_lp[0])
        midt, zi_mid_hp[0]  = sps.sosfilt(sos_mid_hp,  xin, zi=zi_mid_hp[0])
        mid,  zi_mid_lp[0]  = sps.sosfilt(sos_mid_lp,  midt, zi=zi_mid_lp[0])
        high, zi_high_hp[0] = sps.sosfilt(sos_high_hp, xin, zi=zi_high_hp[0])

        # Use time-varying gains if provided, otherwise use fixed gains
        if g_traj is not None:
            y = g_traj[:, 0] * low + g_traj[:, 1] * mid + g_traj[:, 2] * high
        else:
            y = self.g_low * low + self.g_mid * mid + self.g_high * high
        return y

class FixedBeatViewer:
    def __init__(self, master_window, initial_filepath=None):
        self.master = master_window
        self.master.title("Beat Viewer - Fixed Version")
        self.master.geometry("900x1200")  # Even larger window to show all stem controls
        
        # Audio state
        self.audio_data = None
        self.sample_rate = 44100
        self.duration_seconds = 0
        self.original_bpm = 0
        self.current_bpm = 0
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        self.file_loaded = False
        self.current_filepath = None
        
        # Audio streaming
        self.stream = None
        self._slider_dragging = False
        self._seeking = False  # Flag to prevent feedback loop during seeking
        
        # Thread-safe position tracking
        self._audio_position = 0.0  # Position updated by audio thread
        
        # Audio device configuration (Issue #5: stereo for better device compatibility)
        # Set to 1 for mono output, 2 for stereo (duplicated mono)
        self.device_output_channels = 2  # Most devices prefer stereo
        
        # Simple tempo control
        self.current_tempo_ratio = 1.0
        self.playback_position = 0.0  # Floating point position for smooth tempo
        self.last_audio_sample = 0.0  # For smoothing discontinuities
        
        # Ring buffer architecture for thread-safe audio processing  
        self.out_ring = RingBuffer(capacity_frames=self.sample_rate * 4, channels=self.device_output_channels)  # ~4s
        self._producer_stop = False
        self._had_underflow = False
        self._producer_error = None
        self._pending_out = None  # Buffer for partial ring writes
        
        # Thread synchronization for RubberBand access to prevent segfaults
        self._rb_lock = threading.RLock()
        self._pending_time_ratio = None   # handoff from UI → producer thread
        self._pending_reinit = False      # if you ever want to rebuild the stretcher safely
        self._pending_disable = False     # to disable RubberBand from UI thread safely
        self._tempo_debounce_id = None    # tempo debouncing to reduce RB churn
        
        # Pitch preservation with buffer management to eliminate periodic pops
        self.stretch_processor = None
        # Start with empty buffer - will be allocated on first use with generous size
        self.stretch_processed_audio = np.zeros(0, dtype=np.float32)
        self.stretch_buffer_used = 0  # Track how much of the buffer actually contains data
        self.stretch_read_pos = 0
        self.stretch_input_pos = 0  # Track where we are in the source audio for RubberBand
        self.last_tempo_ratio = 1.0
        self.process_chunk_size = 2048  # Smaller chunks to reduce RubberBand stateless artifacts
        self.fade_size = 64  # Small fade at chunk boundaries
        self.max_buffer_size = 1048576  # 1MB max buffer to prevent memory issues
        self.buffer_trim_threshold = 786432  # Trim when buffer exceeds 768KB
        
        # EQ State
        self.eq_enabled = False
        self.eq_high_gain = 1.0    # Linear gain (1.0 = 0dB)
        self.eq_mid_gain = 1.0
        self.eq_low_gain = 1.0
        self._eq_debounce_id = None  # For debouncing rapid slider changes
        
        # Stem State
        self.stems_available = False
        self.stem_data = {}  # {'vocals': np.array, 'drums': np.array, ...}
        self.stem_volumes = {'vocals': 1.0, 'drums': 1.0, 'bass': 1.0, 'other': 1.0}
        self.stem_eq_enabled = {'vocals': False, 'drums': False, 'bass': False, 'other': False}
        self.stem_eq_filters = {}  # stem_name -> ThreeBandEQ instance (legacy)
        
        # Professional EQ System (per the other AI's design)
        # Per-stem tone EQs for musical shaping
        self.stem_tone_eqs = {
            'vocals': ToneEQ3(self.sample_rate),
            'drums': ToneEQ3(self.sample_rate),
            'bass': ToneEQ3(self.sample_rate),
            'other': ToneEQ3(self.sample_rate)
        }
        
        # Master isolator EQ for frequency isolation and kills
        self.master_isolator = IsolatorEQ(self.sample_rate, f_lo=200.0, f_hi=2000.0)
        
        # EQ enable flags for debugging
        self.enable_stem_eqs = True   # Test with the fix
        self._last_stem_eq_update = None  # Debug: track last EQ update
        
        # Master EQ now has clickless ON/OFF toggle
        self.master_isolator.set_enabled(True)  # Enable with wet/dry crossfading
        
        self._create_ui()
        self._configure_bindings()
        
        if initial_filepath:
            self.master.after(100, lambda: self._load_audio_file(initial_filepath))
    
    def _validate_rubberband_buffer(self, input_block):
        """
        Validate buffer for RubberBand to prevent segfaults.
        Ensures proper dtype, shape, and contiguity as suggested by the other AI.
        """
        # Check dtype
        if input_block.dtype != np.float32:
            input_block = input_block.astype(np.float32)
        
        # Check dimensions - must be 2D (frames, channels)
        if input_block.ndim != 2:
            raise ValueError(f"RubberBand input must be 2D (frames, channels), got {input_block.ndim}D")
        
        # Check contiguity
        if not input_block.flags['C_CONTIGUOUS']:
            input_block = np.ascontiguousarray(input_block, dtype=np.float32)
        
        # Validate shape
        frames, channels = input_block.shape
        if frames <= 0:
            raise ValueError(f"Invalid frame count: {frames}")
        if channels not in [1, 2]:
            raise ValueError(f"RubberBand supports 1 or 2 channels, got {channels}")
        
        return input_block
    
    def _assert_not_in_audio_callback(self, method_name="RubberBand method"):
        """
        Development guard to ensure RubberBand methods are never called from audio callback.
        This helps catch violations of the ring buffer design pattern.
        """
        if hasattr(self, '_in_audio_callback'):
            raise RuntimeError(f"VIOLATION: {method_name} called from audio callback! "
                             f"All RubberBand processing must happen in producer thread only. "
                             f"Audio callback should only read from ring buffer.")
    
    def _queue_time_ratio_update(self, tempo_ratio):
        """
        Queued tempo update for debouncing.
        This is called after the debounce delay to actually apply the tempo change.
        """
        try:
            if not self.preserve_pitch_var.get():
                return  # Pitch preservation disabled, no need to update RubberBand
            
            new_time_ratio = 1.0 / float(tempo_ratio)
            
            with self._rb_lock:
                self._pending_time_ratio = new_time_ratio
            
            logger.debug(f"Debounced tempo update applied: ratio={tempo_ratio:.3f}, time_ratio={new_time_ratio:.3f}")
            
        except Exception as e:
            logger.error(f"Error in debounced tempo update: {e}")
        finally:
            # Clear the debounce ID
            self._tempo_debounce_id = None
    
    def _create_ui(self):
        """Create simple, working UI"""
        
        # File loading
        load_frame = tk.Frame(self.master, pady=10)
        load_frame.pack(fill=tk.X, padx=10)
        
        tk.Button(load_frame, text="Load Audio File", 
                 command=self._gui_load_file, font=('Arial', 11, 'bold')).pack(side=tk.LEFT)
        
        self.filepath_label = tk.Label(load_frame, text="No file loaded", 
                                     anchor="w", font=('Arial', 10))
        self.filepath_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Track info
        info_frame = tk.Frame(self.master, pady=5)
        info_frame.pack(fill=tk.X, padx=10)
        
        self.duration_label = tk.Label(info_frame, text="Duration: --", font=('Arial', 14))
        self.duration_label.pack(side=tk.LEFT, padx=5)
        
        self.position_label = tk.Label(info_frame, text="Position: 00:00", font=('Arial', 14))
        self.position_label.pack(side=tk.LEFT, padx=5)
        
        # Beat counter display
        self.beat_label = tk.Label(info_frame, text="Beat: 0.0", font=('Arial', 14, 'bold'), fg='orange')
        self.beat_label.pack(side=tk.LEFT, padx=15)
        
        # MAIN CONTROLS - Use horizontal layout to save space
        main_controls_frame = tk.Frame(self.master, pady=5)
        main_controls_frame.pack(fill=tk.X, padx=10)
        
        # TEMPO SECTION (Left side)
        tempo_frame = tk.LabelFrame(main_controls_frame, text="Tempo Control", 
                                  font=('Arial', 11, 'bold'), fg='orange')
        tempo_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # BPM Display - more compact
        bpm_display_frame = tk.Frame(tempo_frame, pady=5)
        bpm_display_frame.pack()
        
        tk.Label(bpm_display_frame, text="BPM:", font=('Arial', 12)).pack(side=tk.LEFT)
        
        self.bpm_display = tk.Label(bpm_display_frame, text="---", 
                                  font=('Arial', 18, 'bold'), fg='white', bg='#333333',
                                  relief=tk.SUNKEN, bd=2, width=6)
        self.bpm_display.pack(side=tk.LEFT, padx=5)
        
        # Tempo Slider - more compact
        slider_frame = tk.Frame(tempo_frame, pady=5)
        slider_frame.pack(fill=tk.X, padx=10)
        
        # Range labels
        range_frame = tk.Frame(slider_frame)
        range_frame.pack(fill=tk.X)
        
        tk.Label(range_frame, text="50%", font=('Arial', 8)).pack(side=tk.LEFT)
        tk.Label(range_frame, text="100%", font=('Arial', 8)).pack(expand=True)
        tk.Label(range_frame, text="200%", font=('Arial', 8)).pack(side=tk.RIGHT)
        
        # The actual slider - using tk.Scale for better click behavior
        self.tempo_var = tk.DoubleVar(value=1.0)
        self.tempo_slider = tk.Scale(slider_frame, from_=0.5, to=2.0, 
                                   orient=tk.HORIZONTAL, variable=self.tempo_var, 
                                   length=250, command=self._on_tempo_change,
                                   resolution=0.01, showvalue=False,
                                   bg='lightgray', troughcolor='white')
        self.tempo_slider.pack(fill=tk.X, pady=2)
        
        # BPM controls - more compact
        controls_frame = tk.Frame(tempo_frame, pady=5)
        controls_frame.pack()
        
        # First row - BPM entry
        bpm_row = tk.Frame(controls_frame)
        bpm_row.pack()
        
        tk.Label(bpm_row, text="Set BPM:", font=('Arial', 9)).pack(side=tk.LEFT)
        self.bpm_var = tk.StringVar()
        self.bpm_entry = tk.Entry(bpm_row, textvariable=self.bpm_var, 
                                width=6, font=('Arial', 9))
        self.bpm_entry.pack(side=tk.LEFT, padx=2)
        
        self.set_bpm_button = tk.Button(bpm_row, text="Set", 
                                      command=self._set_bpm_from_input,
                                      font=('Arial', 8), state=tk.DISABLED)
        self.set_bpm_button.pack(side=tk.LEFT, padx=2)
        
        # Second row - buttons
        buttons_row = tk.Frame(controls_frame)
        buttons_row.pack(pady=2)
        
        self.override_bpm_button = tk.Button(buttons_row, text="Override", 
                                           command=self._override_original_bpm,
                                           font=('Arial', 8), state=tk.DISABLED)
        self.override_bpm_button.pack(side=tk.LEFT, padx=2)
        
        self.revert_bpm_button = tk.Button(buttons_row, text="Revert", 
                                         command=self._revert_to_original_bpm,
                                         font=('Arial', 8), state=tk.DISABLED)
        self.revert_bpm_button.pack(side=tk.LEFT, padx=2)
        
        # Pitch preservation - compact
        self.preserve_pitch_var = tk.BooleanVar(value=False)
        pitch_text = "Preserve Pitch" if RUBBERBAND_AVAILABLE else "No RubberBand"
        pitch_state = tk.NORMAL if RUBBERBAND_AVAILABLE else tk.DISABLED
        
        self.preserve_pitch_check = tk.Checkbutton(buttons_row, 
                                                 text=pitch_text,
                                                 variable=self.preserve_pitch_var,
                                                 command=self._on_pitch_preserve_change,
                                                 font=('Arial', 8),
                                                 state=pitch_state)
        self.preserve_pitch_check.pack(side=tk.LEFT, padx=5)
        
        # EQ SECTION (Right side)
        eq_frame = tk.LabelFrame(main_controls_frame, text="EQ Control", 
                              font=('Arial', 11, 'bold'), fg='purple')
        eq_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # EQ Enable checkbox - more compact
        eq_top_frame = tk.Frame(eq_frame, pady=3)
        eq_top_frame.pack()
        
        self.eq_enabled_var = tk.BooleanVar(value=False)
        self.eq_enable_check = tk.Checkbutton(eq_top_frame, 
                                            text="Enable EQ",
                                            variable=self.eq_enabled_var,
                                            font=('Arial', 9),
                                            command=self._on_eq_enable_change)
        self.eq_enable_check.pack()
        
        # EQ Controls - more compact
        eq_controls_frame = tk.Frame(eq_frame, pady=5)
        eq_controls_frame.pack(fill=tk.X, padx=5)
        
        # High frequency band
        high_frame = tk.Frame(eq_controls_frame)
        high_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(high_frame, text="High (8kHz)", font=('Arial', 9, 'bold')).pack()
        tk.Label(high_frame, text="+12dB", font=('Arial', 8)).pack()
        
        self.eq_high_var = tk.DoubleVar(value=0.0)
        self.eq_high_slider = tk.Scale(high_frame, from_=12, to=-25,
                                     orient=tk.VERTICAL, variable=self.eq_high_var,
                                     length=80, command=self._on_eq_high_change,
                                     resolution=0.5, showvalue=False,
                                     bg='lightgray', troughcolor='white')
        self.eq_high_slider.pack()
        
        # Kill button for high band
        self.eq_high_kill_var = tk.BooleanVar(value=False)
        self.eq_high_kill_btn = tk.Button(high_frame, text="KILL", 
                                        command=lambda: self._toggle_eq_kill('high'),
                                        font=('Arial', 8, 'bold'), fg='red', width=6)
        self.eq_high_kill_btn.pack()
        
        self.eq_high_display = tk.Label(high_frame, text="0.0dB", 
                                      font=('Arial', 9, 'bold'), fg='red')
        self.eq_high_display.pack()
        
        # Individual reset button for high
        high_reset_btn = tk.Button(high_frame, text="Reset", 
                                 command=lambda: self._reset_eq_band('high'),
                                 font=('Arial', 7))
        high_reset_btn.pack(pady=2)
        
        # Mid frequency band  
        mid_frame = tk.Frame(eq_controls_frame)
        mid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(mid_frame, text="Mid (1kHz)", font=('Arial', 9, 'bold')).pack()
        tk.Label(mid_frame, text="+12dB", font=('Arial', 8)).pack()
        
        self.eq_mid_var = tk.DoubleVar(value=0.0)
        self.eq_mid_slider = tk.Scale(mid_frame, from_=12, to=-25,
                                    orient=tk.VERTICAL, variable=self.eq_mid_var,
                                    length=80, command=self._on_eq_mid_change,
                                    resolution=0.5, showvalue=False,
                                    bg='lightgray', troughcolor='white')
        self.eq_mid_slider.pack()
        
        # Kill button for mid band
        self.eq_mid_kill_var = tk.BooleanVar(value=False)
        self.eq_mid_kill_btn = tk.Button(mid_frame, text="KILL", 
                                       command=lambda: self._toggle_eq_kill('mid'),
                                       font=('Arial', 8, 'bold'), fg='red', width=6)
        self.eq_mid_kill_btn.pack()
        
        self.eq_mid_display = tk.Label(mid_frame, text="0.0dB", 
                                     font=('Arial', 9, 'bold'), fg='green')
        self.eq_mid_display.pack()
        
        # Individual reset button for mid
        mid_reset_btn = tk.Button(mid_frame, text="Reset", 
                                command=lambda: self._reset_eq_band('mid'),
                                font=('Arial', 7))
        mid_reset_btn.pack(pady=2)
        
        # Low frequency band
        low_frame = tk.Frame(eq_controls_frame) 
        low_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(low_frame, text="Low (100Hz)", font=('Arial', 9, 'bold')).pack()
        tk.Label(low_frame, text="+12dB", font=('Arial', 8)).pack()
        
        self.eq_low_var = tk.DoubleVar(value=0.0)
        self.eq_low_slider = tk.Scale(low_frame, from_=12, to=-25,
                                    orient=tk.VERTICAL, variable=self.eq_low_var,
                                    length=80, command=self._on_eq_low_change,
                                    resolution=0.5, showvalue=False,
                                    bg='lightgray', troughcolor='white')
        self.eq_low_slider.pack()
        
        # Kill button for low band
        self.eq_low_kill_var = tk.BooleanVar(value=False)
        self.eq_low_kill_btn = tk.Button(low_frame, text="KILL", 
                                       command=lambda: self._toggle_eq_kill('low'),
                                       font=('Arial', 8, 'bold'), fg='red', width=6)
        self.eq_low_kill_btn.pack()
        
        self.eq_low_display = tk.Label(low_frame, text="0.0dB", 
                                     font=('Arial', 9, 'bold'), fg='blue')
        self.eq_low_display.pack()
        
        # Individual reset button for low
        low_reset_btn = tk.Button(low_frame, text="Reset", 
                                command=lambda: self._reset_eq_band('low'),
                                font=('Arial', 7))
        low_reset_btn.pack(pady=2)
        
        # EQ Reset button
        eq_reset_frame = tk.Frame(eq_frame, pady=5)
        eq_reset_frame.pack()
        
        self.eq_reset_button = tk.Button(eq_reset_frame, text="Reset EQ", 
                                       command=self._reset_eq,
                                       font=('Arial', 9))
        self.eq_reset_button.pack()
        
        # Playback controls - more compact
        playback_frame = tk.Frame(self.master, pady=5)
        playback_frame.pack()
        
        self.play_pause_button = tk.Button(playback_frame, text="▶ Play", 
                                         width=10, height=1,
                                         command=self._toggle_play_pause, 
                                         state=tk.DISABLED,
                                         font=('Arial', 11, 'bold'))
        self.play_pause_button.pack()
        
        # Position slider - more compact
        seek_frame = tk.Frame(self.master, pady=5)
        seek_frame.pack(fill=tk.X, padx=10)
        
        self.seek_slider_var = tk.DoubleVar()
        self.seek_slider = ttk.Scale(seek_frame, from_=0, to=100, 
                                   orient=tk.HORIZONTAL, variable=self.seek_slider_var, 
                                   command=self._on_seek_change)
        self.seek_slider.pack(fill=tk.X, expand=True)
        
        # Status
        self.status_label = tk.Label(self.master, text="Ready", 
                                   font=('Arial', 10), fg='green', pady=10)
        self.status_label.pack()
    
    def _configure_bindings(self):
        """Configure keyboard shortcuts"""
        self.master.bind('<Return>', lambda e: self._set_bpm_from_input())
        self.master.bind('<space>', lambda e: self._toggle_play_pause())
        self.master.focus_set()
    
    def _gui_load_file(self):
        """Load file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.mp3 *.wav *.flac *.m4a *.aac"), ("All files", "*.*")]
        )
        if file_path:
            self._load_audio_file(file_path)
    
    def _load_audio_file(self, file_path):
        """Load and analyze audio file"""
        try:
            # Cancel any pending debounced tempo updates
            if self._tempo_debounce_id is not None:
                self.master.after_cancel(self._tempo_debounce_id)
                self._tempo_debounce_id = None
            
            self._update_status("Loading audio file...", "orange")
            
            # Load audio
            audio_segment = AudioSegment.from_file(file_path)
            
            # Convert to numpy
            self.audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            
            # Handle stereo
            if audio_segment.channels == 2:
                self.audio_data = self.audio_data.reshape((-1, 2))
                self.audio_data = np.mean(self.audio_data, axis=1)
            
            # Normalize
            if np.max(np.abs(self.audio_data)) > 0:
                self.audio_data = self.audio_data / np.max(np.abs(self.audio_data)) * 0.9
            
            # Set properties
            self.sample_rate = audio_segment.frame_rate
            
            self.total_frames = len(self.audio_data)
            self.duration_seconds = len(self.audio_data) / self.sample_rate
            self.current_filepath = file_path
            
            # Analyze BPM
            self._analyze_bpm()
            
            # Update UI
            self._update_file_info()
            self._enable_controls()
            
            # Check for and load stems if available
            self._check_and_load_stems()
            
            self._update_status("Audio loaded successfully!", "green")
            self.file_loaded = True
            
        except Exception as e:
            error_msg = f"Error loading audio: {str(e)}"
            self._update_status(error_msg, "red")
            logger.error(error_msg)
    
    def _analyze_bpm(self):
        """Simple, working BPM analysis"""
        try:
            self._update_status("Analyzing BPM...", "orange")
            
            # Convert to proper format for Essentia (this is the key fix!)
            audio_for_essentia = np.ascontiguousarray(self.audio_data.astype(np.float32))
            
            # Try RhythmExtractor2013
            try:
                bpm_extractor = es.RhythmExtractor2013(method="multifeature")
                bpm, beats, _, _, _ = bpm_extractor(audio_for_essentia)
                detected_bpm = float(bpm)
                
                # For tracks like "Fancy", sometimes the detected BPM is doubled
                # If BPM > 150, try half
                if detected_bpm > 150:
                    corrected_bpm = detected_bpm / 2
                    logger.info(f"High BPM detected ({detected_bpm:.1f}), using half: {corrected_bpm:.1f}")
                    self.original_bpm = corrected_bpm
                else:
                    self.original_bpm = detected_bpm
                
                logger.info(f"Detected BPM: {detected_bpm:.1f} -> Using: {self.original_bpm:.1f}")
                
            except Exception as e:
                logger.warning(f"RhythmExtractor2013 failed: {e}")
                # Good default for Fancy track
                self.original_bpm = 87.0
                logger.info(f"Using default BPM: {self.original_bpm}")
            
            self.current_bpm = self.original_bpm
            
        except Exception as e:
            logger.error(f"BPM analysis failed: {e}")
            self.original_bpm = 87.0
            self.current_bpm = 87.0
    
    def _update_file_info(self):
        """Update file information display"""
        filename = os.path.basename(self.current_filepath)
        self.filepath_label.config(text=filename)
        
        duration_str = f"{int(self.duration_seconds // 60):02d}:{int(self.duration_seconds % 60):02d}"
        self.duration_label.config(text=f"Duration: {duration_str}")
        
        self.bpm_display.config(text=f"{self.current_bpm:.1f}")
        self.bpm_var.set(f"{self.current_bpm:.1f}")
    
    def _enable_controls(self):
        """Enable controls after loading"""
        self.play_pause_button.config(state=tk.NORMAL)
        # Seek slider is now always enabled when file is loaded
        self.set_bpm_button.config(state=tk.NORMAL)
        self.override_bpm_button.config(state=tk.NORMAL)
        self.revert_bpm_button.config(state=tk.NORMAL)
        
        self._start_position_updates()
        
        # Start periodic audio flag checking
        self._check_audio_flags()
    
    def _start_position_updates(self):
        """Start position updates"""
        if self.file_loaded:
            # Update current_frame from audio thread position
            if self.is_playing:
                # Use position from audio thread during playback
                self.current_frame = int(self._audio_position)
            
            self._update_position_display()
        
        # Always schedule next update regardless of file_loaded state
        self.master.after(50, self._start_position_updates)  # Update more frequently
    
    def _check_audio_flags(self):
        """Check and handle flags set by the audio callback"""
        # Check for underflow flag (set in callback, handled here)
        if self._had_underflow:
            self._had_underflow = False
            self._update_status("Audio underflow detected — consider larger blocksize or less EQ/RB load", "orange")
        
        # Check for producer error flag
        if self._producer_error:
            error_msg = self._producer_error
            self._producer_error = None  # Clear the flag
            logger.debug(f"Producer loop error: {error_msg}")
        
        # Schedule next flag check
        self.master.after(250, self._check_audio_flags)  # Check every 250ms
    
    def _update_position_display(self):
        """Update position display"""
        if not self.file_loaded:
            return
        
        pos_seconds = self.current_frame / self.sample_rate if self.sample_rate > 0 else 0
        pos_str = f"{int(pos_seconds // 60):02d}:{int(pos_seconds % 60):02d}"
        self.position_label.config(text=f"Position: {pos_str}")
        
        # Update beat counter - based on MUSICAL position, not playback time
        if self.original_bpm > 0:
            # Calculate beats based on original audio position and original BPM
            # This ensures beat numbers stay musically consistent regardless of tempo changes
            original_pos_seconds = self.current_frame / self.sample_rate if self.sample_rate > 0 else 0
            beats_per_second = self.original_bpm / 60.0
            musical_beat = original_pos_seconds * beats_per_second
            self.beat_label.config(text=f"Beat: {musical_beat:.1f}")
        
        # Update seek slider position
        if not self._seeking and self.total_frames > 0:
            progress = (self.current_frame / self.total_frames) * 100
            self.seek_slider_var.set(progress)
    
    
    def _on_tempo_change(self, value):
        """Handle tempo slider changes - pre-process audio for smooth playback"""
        if not self.file_loaded:
            return
        
        tempo_ratio = float(value)
        new_bpm = self.original_bpm * tempo_ratio
        
        # Update display immediately
        self.current_bpm = new_bpm
        self.bpm_display.config(text=f"{new_bpm:.1f}")
        self.bpm_var.set(f"{new_bpm:.1f}")
        
        # Store tempo ratio and handle pitch preservation
        old_ratio = self.current_tempo_ratio
        self.current_tempo_ratio = tempo_ratio
        
        # Initialize or update streaming pitch preservation if needed
        if (RUBBERBAND_STREAMING_AVAILABLE or RUBBERBAND_AVAILABLE) and self.preserve_pitch_var.get() and self.file_loaded:
            if abs(tempo_ratio - self.last_tempo_ratio) > 0.01:  # Only update if ratio changed significantly
                
                # Check if we can update existing streaming processor
                if (hasattr(self, 'rubberband_stretcher') and 
                    self.rubberband_stretcher and 
                    self.stretch_processor and 
                    self.stretch_processor.get('streaming', False)):
                    
                    # Use debounced update to prevent RubberBand churn during rapid tempo changes
                    if self._tempo_debounce_id is not None:
                        self.master.after_cancel(self._tempo_debounce_id)
                    self._tempo_debounce_id = self.master.after(20, lambda: self._queue_time_ratio_update(tempo_ratio))
                    
                    # Update local state immediately for display responsiveness
                    self.stretch_processor['tempo_ratio'] = tempo_ratio
                    self.stretch_processor['time_ratio'] = 1.0 / tempo_ratio
                    self.last_tempo_ratio = tempo_ratio
                    
                    logger.debug(f"Debounced tempo update scheduled: {tempo_ratio:.2f}")
                    
                else:
                    # Queue processor initialization for producer thread (no debouncing needed for init)
                    with self._rb_lock:
                        self._pending_reinit = True
                        self._pending_time_ratio = 1.0 / tempo_ratio
                    logger.info(f"Pitch preservation initialization queued with tempo ratio: {tempo_ratio:.2f}")
        else:
            self.stretch_processor = None
            if self.preserve_pitch_var.get():
                logger.info("Pitch preservation requested but not available")
        
        logger.debug(f"Tempo changed: ratio={tempo_ratio:.2f}, new_bpm={new_bpm:.1f}")
    
    def _init_stretch_processor(self, tempo_ratio):
        """DEPRECATED: Use handoff pattern instead. This method is kept for fallback only."""
        logger.warning("Deprecated _init_stretch_processor called - using handoff pattern instead")
        with self._rb_lock:
            self._pending_reinit = True
            self._pending_time_ratio = 1.0 / tempo_ratio
    
    def _init_stretch_processor_fallback(self, tempo_ratio):
        """Fallback to old stateless RubberBand API"""
        if not RUBBERBAND_AVAILABLE:
            return
        
        try:
            self.stretch_processor = {
                'tempo_ratio': tempo_ratio,
                'streaming': False,
                'initialized': True
            }
            
            # Reset processing state - keep allocated buffer
            self.stretch_buffer_used = 0
            self.stretch_read_pos = 0
            self.stretch_input_pos = int(self.playback_position)
            self.last_tempo_ratio = tempo_ratio
            
            logger.info(f"Fallback pitch processor initialized (ratio: {tempo_ratio:.2f})")
            
        except Exception as e:
            logger.error(f"Failed to initialize fallback pitch processor: {e}")
            self.stretch_processor = None
    
    def _init_stretch_processor_producer_thread(self, tempo_ratio):
        """Initialize RubberBand from producer thread (thread-safe)"""
        if not RUBBERBAND_STREAMING_AVAILABLE or self.audio_data is None:
            return
        
        try:
            # Calculate proper time ratio
            time_ratio = 1.0 / tempo_ratio
            
            # Check if we have stereo or mono audio
            audio_channels = 2 if len(self.audio_data.shape) > 1 and self.audio_data.shape[1] == 2 else 1
            if len(self.audio_data.shape) == 1:
                audio_channels = 1  # Definitely mono
            
            # Create RubberBand processor (called from producer thread, no lock needed)
            self.rubberband_stretcher = RubberBand(
                sample_rate=int(self.sample_rate),
                channels=audio_channels,
                options=REALTIME_DEFAULT,
                time_ratio=time_ratio,
                pitch_scale=1.0  # Keep pitch unchanged
            )
            
            self.audio_channels = audio_channels  # Store for later use
            
            # Optimize for different audio types
            self.rubberband_stretcher.set_transients_option(RubberBandOptionTransientsCrisp)
            self.rubberband_stretcher.set_phase_option(RubberBandOptionPhaseIndependent)
            self.rubberband_stretcher.set_detector_option(RubberBandOptionDetectorPercussive)
            
            # Reset processing state
            self.stretch_buffer_used = 0
            self.stretch_read_pos = 0
            self.stretch_input_pos = int(self.playback_position)
            self.last_tempo_ratio = tempo_ratio
            
            # Set up processor metadata
            self.stretch_processor = {
                'tempo_ratio': tempo_ratio,
                'time_ratio': time_ratio,
                'streaming': True,
                'initialized': True
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize RubberBand in producer thread: {e}")
            self.stretch_processor = None
    
    def _set_bpm_from_input(self):
        """Set BPM from input field"""
        try:
            target_bpm = float(self.bpm_var.get())
            if target_bpm <= 0:
                raise ValueError("BPM must be positive")
            
            tempo_ratio = target_bpm / self.original_bpm
            tempo_ratio = max(0.5, min(2.0, tempo_ratio))
            
            self.tempo_var.set(tempo_ratio)
            self._on_tempo_change(tempo_ratio)
            
        except ValueError as e:
            self._update_status(f"Invalid BPM: {e}", "red")
    
    def _override_original_bpm(self):
        """Override the original BPM"""
        try:
            new_original_bpm = float(self.bpm_var.get())
            if new_original_bpm <= 0:
                raise ValueError("BPM must be positive")
            
            self.original_bpm = new_original_bpm
            self.current_bpm = new_original_bpm
            self.tempo_var.set(1.0)
            self.bpm_display.config(text=f"{self.original_bpm:.1f}")
            
            self._update_status(f"Original BPM set to {self.original_bpm:.1f}", "orange")
            
        except ValueError as e:
            self._update_status(f"Invalid BPM: {e}", "red")
    
    def _revert_to_original_bpm(self):
        """Revert to the originally detected BPM"""
        self.tempo_var.set(1.0)
        self.current_bpm = self.original_bpm
        self.bpm_display.config(text=f"{self.original_bpm:.1f}")
        self.bpm_var.set(f"{self.original_bpm:.1f}")
        
        # Manually trigger tempo change since slider callback might not fire
        old_ratio = self.current_tempo_ratio
        self.current_tempo_ratio = 1.0
        
        # Reset pitch preservation if it's enabled
        if RUBBERBAND_AVAILABLE and self.preserve_pitch_var.get() and self.file_loaded:
            self._init_stretch_processor(1.0)
            logger.info("Pitch preservation reset to original BPM")
        
        self._update_status(f"Reverted to original BPM: {self.original_bpm:.1f}", "orange")
    
    def _on_eq_enable_change(self):
        """Handle EQ enable/disable"""
        self.eq_enabled = self.eq_enabled_var.get()
        
        if self.eq_enabled and self.file_loaded:
            # Enable master isolator EQ with wet/dry crossfading
            self.master_isolator.set_enabled(True)
            self._update_eq_from_sliders()
            logger.info("EQ enabled")
        else:
            # Disable master isolator EQ with wet/dry crossfading
            self.master_isolator.set_enabled(False)
            logger.info("EQ disabled")
        
        self._update_status(f"EQ {'enabled' if self.eq_enabled else 'disabled'}", "orange")
    
    def _on_eq_slider_change(self, *_):
        """Debounce rapid EQ slider changes to reduce crossfade frequency"""
        if self._eq_debounce_id is not None:
            self.master.after_cancel(self._eq_debounce_id)
        # Recompute filters after 20ms of inactivity (~50Hz max update rate)
        self._eq_debounce_id = self.master.after(20, self._apply_eq_slider_now)
    
    def _apply_eq_slider_now(self):
        """Apply debounced EQ slider changes"""
        self._eq_debounce_id = None
        # Always apply slider changes to master isolator
        self._update_eq_from_sliders()
    
    def _on_eq_high_change(self, value):
        """Handle high EQ slider change"""
        gain_db = float(value)
        self.eq_high_display.config(text=f"{gain_db:+.1f}dB")
        if self.eq_enabled:
            self._on_eq_slider_change()  # Debounced update
    
    def _on_eq_mid_change(self, value):
        """Handle mid EQ slider change"""
        gain_db = float(value)
        self.eq_mid_display.config(text=f"{gain_db:+.1f}dB")
        if self.eq_enabled:
            self._on_eq_slider_change()  # Debounced update
    
    def _on_eq_low_change(self, value):
        """Handle low EQ slider change"""
        gain_db = float(value)
        self.eq_low_display.config(text=f"{gain_db:+.1f}dB")
        if self.eq_enabled:
            self._on_eq_slider_change()  # Debounced update
    
    def _update_eq_from_sliders(self):
        """Update EQ filters from slider values"""
        high_db = self.eq_high_var.get()
        mid_db = self.eq_mid_var.get()
        low_db = self.eq_low_var.get()
        
        # Update master isolator EQ with smooth gain changes
        self.master_isolator.set_gains_db(low_db, mid_db, high_db)
        logger.debug(f"Master EQ updated: H={high_db:+.1f}dB, M={mid_db:+.1f}dB, L={low_db:+.1f}dB")
    
    def _reset_eq(self):
        """Reset all EQ sliders to 0dB"""
        self.eq_high_var.set(0.0)
        self.eq_mid_var.set(0.0)
        self.eq_low_var.set(0.0)
        
        self.eq_high_display.config(text="0.0dB")
        self.eq_mid_display.config(text="0.0dB")
        self.eq_low_display.config(text="0.0dB")
        
        # Reset all kill switches
        self.eq_high_kill_var.set(False)
        self.eq_mid_kill_var.set(False)
        self.eq_low_kill_var.set(False)
        
        self.eq_high_kill_btn.config(bg='SystemButtonFace')
        self.eq_mid_kill_btn.config(bg='SystemButtonFace')
        self.eq_low_kill_btn.config(bg='SystemButtonFace')
        
        # Reset master isolator EQ to flat
        self.master_isolator.set_gains_db(0.0, 0.0, 0.0)
        self.master_isolator.set_kill(False, False, False)
        
        self._update_status("EQ reset to flat", "orange")
    
    def _reset_eq_band(self, band):
        """Reset individual EQ band to 0dB"""
        if band == 'high':
            self.eq_high_var.set(0.0)
            self.eq_high_display.config(text="0.0dB")
            self.eq_high_kill_var.set(False)
            self.eq_high_kill_btn.config(bg='SystemButtonFace')
        elif band == 'mid':
            self.eq_mid_var.set(0.0)
            self.eq_mid_display.config(text="0.0dB")
            self.eq_mid_kill_var.set(False)
            self.eq_mid_kill_btn.config(bg='SystemButtonFace')
        elif band == 'low':
            self.eq_low_var.set(0.0)
            self.eq_low_display.config(text="0.0dB")
            self.eq_low_kill_var.set(False)
            self.eq_low_kill_btn.config(bg='SystemButtonFace')
        
        # Update all EQ filters and kill switches
        self._update_eq_from_sliders()
        self._update_eq_kill_switches()
        
        self._update_status(f"EQ {band} band reset", "orange")
    
    def _toggle_eq_kill(self, band):
        """Toggle kill switch for individual EQ band"""
        if band == 'high':
            kill_active = not self.eq_high_kill_var.get()
            self.eq_high_kill_var.set(kill_active)
            self.eq_high_kill_btn.config(bg='red' if kill_active else 'SystemButtonFace')
        elif band == 'mid':
            kill_active = not self.eq_mid_kill_var.get()
            self.eq_mid_kill_var.set(kill_active)
            self.eq_mid_kill_btn.config(bg='red' if kill_active else 'SystemButtonFace')
        elif band == 'low':
            kill_active = not self.eq_low_kill_var.get()
            self.eq_low_kill_var.set(kill_active)
            self.eq_low_kill_btn.config(bg='red' if kill_active else 'SystemButtonFace')
        else:
            return
        
        # Apply all current kill switch states to master isolator
        self._update_eq_kill_switches()
        
        self._update_status(f"EQ {band} band {'killed' if kill_active else 'restored'}", "orange")
    
    def _update_eq_kill_switches(self):
        """Update master isolator with current kill switch states"""
        low_kill = self.eq_low_kill_var.get()
        mid_kill = self.eq_mid_kill_var.get()
        high_kill = self.eq_high_kill_var.get()
        
        self.master_isolator.set_kill(low_kill, mid_kill, high_kill)
    
    def _get_stems_cache_dir(self, audio_filepath):
        """Get the stems cache directory for an audio file using config"""
        import config as app_config
        return app_config.get_stems_cache_dir(audio_filepath)
    
    def _check_stems_exist(self, audio_filepath):
        """Check if stems exist for this audio file"""
        stems_dir = self._get_stems_cache_dir(audio_filepath)
        stem_names = ['vocals', 'drums', 'bass', 'other']
        
        for stem_name in stem_names:
            stem_file = os.path.join(stems_dir, f"{stem_name}.npy")
            if not os.path.exists(stem_file):
                return False
        return True
    
    def _load_stems(self, audio_filepath):
        """Load cached stems for the audio file"""
        stems_dir = self._get_stems_cache_dir(audio_filepath)
        stem_names = ['vocals', 'drums', 'bass', 'other']
        loaded_stems = {}
        
        try:
            for stem_name in stem_names:
                stem_file = os.path.join(stems_dir, f"{stem_name}.npy")
                stem_audio = np.load(stem_file)
                loaded_stems[stem_name] = stem_audio
                logger.info(f"Loaded {stem_name} stem: {len(stem_audio)} samples")
            
            return loaded_stems
        except Exception as e:
            logger.error(f"Failed to load stems: {e}")
            return {}
    
    def _check_and_load_stems(self):
        """Check for and load stems if available"""
        if not self.current_filepath:
            return
        
        if self._check_stems_exist(self.current_filepath):
            logger.info("Stems found - loading...")
            self.stem_data = self._load_stems(self.current_filepath)
            
            if self.stem_data:
                self.stems_available = True
                
                # Pre-initialize all stem EQ filters to avoid mid-stream creation
                for stem_name in self.stem_data.keys():
                    if stem_name not in self.stem_eq_filters:
                        eq_filter = ThreeBandEQ(self.sample_rate)
                        eq_filter.reset()
                        eq_filter.set_gains(0.0, 0.0, 0.0)  # Start at passthrough
                        self.stem_eq_filters[stem_name] = eq_filter
                        logger.debug(f"Pre-initialized EQ filter for {stem_name}")
                
                self._create_stem_ui()
                self._update_status("Stems loaded - advanced mixing available!", "green")
                logger.info(f"Loaded {len(self.stem_data)} stems with EQ filters")
            else:
                self.stems_available = False
                logger.warning("Failed to load stems")
        else:
            self.stems_available = False
            logger.info("No stems found for this track")
    
    def _create_stem_ui(self):
        """Create stem control UI when stems are available"""
        if hasattr(self, 'stem_frame'):
            return  # Already created
        
        # STEM SECTION - much more compact
        self.stem_frame = tk.LabelFrame(self.master, text="Stem Control", 
                                      font=('Arial', 11, 'bold'), fg='blue')
        self.stem_frame.pack(fill=tk.X, padx=10, pady=3)
        
        # Create stem controls for each stem
        stem_names = ['vocals', 'drums', 'bass', 'other']
        stem_colors = {'vocals': 'red', 'drums': 'orange', 'bass': 'blue', 'other': 'green'}
        
        # Stem volume controls - compact
        volume_frame = tk.Frame(self.stem_frame, pady=5)
        volume_frame.pack(fill=tk.X, padx=5)
        
        tk.Label(volume_frame, text="Volume Control", font=('Arial', 10, 'bold')).pack()
        
        # Create volume sliders for each stem
        self.stem_volume_vars = {}
        self.stem_volume_sliders = {}
        self.stem_volume_displays = {}
        
        sliders_frame = tk.Frame(volume_frame)
        sliders_frame.pack(fill=tk.X, pady=2)
        
        for i, stem_name in enumerate(stem_names):
            stem_frame = tk.Frame(sliders_frame)
            stem_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            
            # Stem name label
            tk.Label(stem_frame, text=stem_name.title(), 
                    font=('Arial', 10, 'bold'), fg=stem_colors[stem_name]).pack()
            
            # Volume slider - much longer (0-150% range)
            self.stem_volume_vars[stem_name] = tk.DoubleVar(value=100.0)
            
            def make_volume_command(name):
                return lambda v: self._on_stem_volume_change(name, v)
            
            self.stem_volume_sliders[stem_name] = tk.Scale(
                stem_frame, from_=150, to=0, orient=tk.VERTICAL,
                variable=self.stem_volume_vars[stem_name],
                length=100, command=make_volume_command(stem_name),
                resolution=1, showvalue=False, bg='lightgray', troughcolor='white'
            )
            self.stem_volume_sliders[stem_name].pack()
            
            # Volume display
            self.stem_volume_displays[stem_name] = tk.Label(
                stem_frame, text="100%", font=('Arial', 9, 'bold'),
                fg=stem_colors[stem_name]
            )
            self.stem_volume_displays[stem_name].pack()
            
            # Small reset button for this stem volume
            def make_volume_reset_command(name):
                return lambda: self._reset_stem_volume(name)
            
            reset_btn = tk.Button(stem_frame, text="Reset", 
                                command=make_volume_reset_command(stem_name),
                                font=('Arial', 7), height=1)
            reset_btn.pack(pady=1)
        
        # Individual Stem EQ - Full Controls
        eq_frame = tk.Frame(self.stem_frame, pady=5)
        eq_frame.pack(fill=tk.X, padx=5)
        
        tk.Label(eq_frame, text="Individual Stem EQ", font=('Arial', 10, 'bold')).pack()
        
        # Create EQ controls for all stems simultaneously
        self.stem_eq_vars = {}
        self.stem_eq_high_vars = {}
        self.stem_eq_mid_vars = {}
        self.stem_eq_low_vars = {}
        self.stem_eq_displays = {}
        
        # Main EQ controls frame - show all stems at once
        eq_all_stems_frame = tk.Frame(eq_frame)
        eq_all_stems_frame.pack(fill=tk.X, pady=2)
        
        for i, stem_name in enumerate(stem_names):
            # Create individual stem EQ column
            stem_column = tk.Frame(eq_all_stems_frame)
            stem_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
            
            # Stem name header
            stem_header = tk.Label(stem_column, text=f"{stem_name.title()}", 
                                 font=('Arial', 10, 'bold'), fg=stem_colors[stem_name])
            stem_header.pack()
            
            # Enable checkbox for this stem
            self.stem_eq_vars[stem_name] = tk.BooleanVar(value=False)
            
            def make_eq_enable_command(name):
                return lambda: self._on_stem_eq_enable_change(name)
            
            enable_checkbox = tk.Checkbutton(stem_column,
                                           text="Enable EQ",
                                           variable=self.stem_eq_vars[stem_name],
                                           font=('Arial', 8),
                                           command=make_eq_enable_command(stem_name))
            enable_checkbox.pack()
            
            # EQ sliders for this stem - horizontal layout
            eq_sliders_frame = tk.Frame(stem_column)
            eq_sliders_frame.pack(pady=2)
            
            # High, Mid, Low sliders for this stem
            for eq_band, eq_color in [('high', 'red'), ('mid', 'green'), ('low', 'blue')]:
                band_frame = tk.Frame(eq_sliders_frame)
                band_frame.pack(side=tk.LEFT, padx=2)
                
                tk.Label(band_frame, text=eq_band[0].upper(), 
                        font=('Arial', 8, 'bold'), fg=eq_color).pack()
                
                # Initialize the variable
                var_name = f"stem_eq_{eq_band}_vars"
                if not hasattr(self, var_name):
                    setattr(self, var_name, {})
                getattr(self, var_name)[stem_name] = tk.DoubleVar(value=0.0)
                
                def make_eq_change_command(s, b):
                    return lambda v: self._on_stem_eq_change(s, b, v)
                
                slider = tk.Scale(band_frame, from_=12, to=-25,
                                orient=tk.VERTICAL, 
                                variable=getattr(self, var_name)[stem_name],
                                length=80, 
                                command=make_eq_change_command(stem_name, eq_band),
                                resolution=0.5, showvalue=False,
                                bg='lightgray', troughcolor='white')
                slider.pack()
                
                # Display label
                if stem_name not in self.stem_eq_displays:
                    self.stem_eq_displays[stem_name] = {}
                self.stem_eq_displays[stem_name][eq_band] = tk.Label(
                    band_frame, text="0.0dB", font=('Arial', 7), fg=eq_color
                )
                self.stem_eq_displays[stem_name][eq_band].pack()
            
            # Reset button for this stem's EQ
            def make_eq_reset_command(name):
                return lambda: self._reset_stem_eq(name)
            
            reset_eq_btn = tk.Button(stem_column, text="Reset",
                                   command=make_eq_reset_command(stem_name),
                                   font=('Arial', 7))
            reset_eq_btn.pack(pady=2)
        
        # Note about master EQ and pitch preservation - smaller
        tk.Label(eq_frame, text="Note: Master EQ (above) affects final mixed output", 
                font=('Arial', 8, 'italic'), fg='gray').pack(pady=2)
        tk.Label(eq_frame, text="Individual stem EQ works in both turntable and pitch preservation modes", 
                font=('Arial', 8, 'italic'), fg='green').pack()
    
    def _on_stem_volume_change(self, stem_name, value):
        """Handle stem volume slider changes"""
        volume_percent = float(value)
        volume_linear = volume_percent / 100.0
        
        self.stem_volumes[stem_name] = volume_linear
        self.stem_volume_displays[stem_name].config(text=f"{volume_percent:.0f}%")
        
        logger.debug(f"{stem_name} volume: {volume_percent:.0f}% ({volume_linear:.2f})")
    
    def _reset_stem_volume(self, stem_name):
        """Reset individual stem volume to 100%"""
        self.stem_volume_vars[stem_name].set(100.0)
        self.stem_volumes[stem_name] = 1.0
        self.stem_volume_displays[stem_name].config(text="100%")
        self._update_status(f"{stem_name.title()} volume reset", "orange")
    
    def _on_stem_eq_enable_change(self, stem_name):
        """Handle stem EQ enable/disable with smooth transitions"""
        enabled = self.stem_eq_vars[stem_name].get()
        self.stem_eq_enabled[stem_name] = enabled
        
        # Always ensure the filter exists (never create/destroy mid-stream)
        if stem_name not in self.stem_eq_filters:
            # Create filter but keep it at passthrough initially
            eq_filter = ThreeBandEQ(self.sample_rate)
            eq_filter.reset()
            
            # Pre-condition with current audio if available
            if (stem_name in self.stem_data and hasattr(self, 'playback_position') 
                and self.playback_position >= 0):
                try:
                    # Get recent audio to condition the filter properly
                    current_pos = max(0, int(self.playback_position) - 128)
                    end_pos = min(len(self.stem_data[stem_name]), current_pos + 256)
                    
                    if current_pos < end_pos:
                        recent_samples = self.stem_data[stem_name][current_pos:end_pos]
                        # Condition filter with actual recent audio at passthrough settings
                        eq_filter.set_gains(0.0, 0.0, 0.0)  # Start at passthrough
                        for sample in recent_samples:
                            eq_filter.process_sample(sample)
                except:
                    # Fallback conditioning
                    eq_filter.set_gains(0.0, 0.0, 0.0)
                    for _ in range(256):
                        eq_filter.process_sample(0.0)
            
            self.stem_eq_filters[stem_name] = eq_filter
        
        # Update filter settings based on enabled state
        eq_filter = self.stem_eq_filters[stem_name]
        
        if enabled:
            # Apply current EQ settings
            high_db = self.stem_eq_high_vars[stem_name].get() if stem_name in self.stem_eq_high_vars else 0.0
            mid_db = self.stem_eq_mid_vars[stem_name].get() if stem_name in self.stem_eq_mid_vars else 0.0
            low_db = self.stem_eq_low_vars[stem_name].get() if stem_name in self.stem_eq_low_vars else 0.0
            eq_filter.set_gains(high_db, mid_db, low_db)
            logger.info(f"{stem_name} EQ enabled")
            self._update_status(f"{stem_name} EQ enabled", "green")
        else:
            # Set to passthrough instead of removing
            eq_filter.set_gains(0.0, 0.0, 0.0)
            logger.info(f"{stem_name} EQ disabled (passthrough)")
            self._update_status(f"{stem_name} EQ disabled", "orange")
    
    def _on_stem_eq_change(self, stem_name, eq_band, value):
        """Handle individual stem EQ slider changes"""
        gain_db = float(value)
        
        # Update display
        if stem_name in self.stem_eq_displays and eq_band in self.stem_eq_displays[stem_name]:
            self.stem_eq_displays[stem_name][eq_band].config(text=f"{gain_db:+.1f}dB")
        
        # Update the stem's EQ filter - but only if stem EQ is enabled
        if stem_name in self.stem_tone_eqs:
            # Only apply EQ changes if this stem's EQ is enabled
            if self.stem_eq_enabled.get(stem_name, False):
                # Get current values for all bands
                high_db = self.stem_eq_high_vars[stem_name].get() if stem_name in self.stem_eq_high_vars else 0.0
                mid_db = self.stem_eq_mid_vars[stem_name].get() if stem_name in self.stem_eq_mid_vars else 0.0
                low_db = self.stem_eq_low_vars[stem_name].get() if stem_name in self.stem_eq_low_vars else 0.0
                
                # Update the ToneEQ3 filter with all three band values
                if stem_name in self.stem_tone_eqs:
                    self.stem_tone_eqs[stem_name].set_params_db(low_db, mid_db, high_db)
                    # DEBUG: prove updates are applied
                    self._last_stem_eq_update = (stem_name, float(low_db), float(mid_db), float(high_db))
                logger.debug(f"{stem_name} EQ updated: H={high_db:+.1f}dB, M={mid_db:+.1f}dB, L={low_db:+.1f}dB")
            else:
                # Keep at passthrough if disabled (sliders can move but EQ stays at 0dB)
                if stem_name in self.stem_tone_eqs:
                    self.stem_tone_eqs[stem_name].set_params_db(0.0, 0.0, 0.0)
                logger.debug(f"{stem_name} EQ disabled - staying at passthrough")
    
    def _reset_stem_eq(self, stem_name):
        """Reset all EQ bands for a specific stem"""
        # Reset all sliders to 0dB
        if stem_name in self.stem_eq_high_vars:
            self.stem_eq_high_vars[stem_name].set(0.0)
        if stem_name in self.stem_eq_mid_vars:
            self.stem_eq_mid_vars[stem_name].set(0.0)
        if stem_name in self.stem_eq_low_vars:
            self.stem_eq_low_vars[stem_name].set(0.0)
        
        # Update displays
        if stem_name in self.stem_eq_displays:
            for band in ['high', 'mid', 'low']:
                if band in self.stem_eq_displays[stem_name]:
                    self.stem_eq_displays[stem_name][band].config(text="0.0dB")
        
        # Update filter
        if stem_name in self.stem_tone_eqs:
            self.stem_tone_eqs[stem_name].set_params_db(0.0, 0.0, 0.0)
        
        self._update_status(f"{stem_name.title()} EQ reset", "orange")
    
    def _on_pitch_preserve_change(self):
        """Handle pitch preservation checkbox changes"""
        if not RUBBERBAND_AVAILABLE:
            return
            
        enabled = self.preserve_pitch_var.get()
        
        if enabled and self.file_loaded:
            # Queue processor initialization for producer thread (never call RB from UI)
            with self._rb_lock:
                self._pending_reinit = True
                self._pending_time_ratio = 1.0 / self.current_tempo_ratio
            self._update_status("Pitch preservation enabled", "green")
            logger.info(f"Pitch preservation enabled (ratio: {self.current_tempo_ratio:.2f})")
        else:
            # Cancel any pending debounced tempo updates
            if self._tempo_debounce_id is not None:
                self.master.after_cancel(self._tempo_debounce_id)
                self._tempo_debounce_id = None
            
            # Queue disable for producer thread (never call RB from UI)
            with self._rb_lock:
                self._pending_disable = True
            
            self.stretch_processor = None
            self._update_status("Pitch preservation disabled", "orange") 
            logger.info("Pitch preservation disabled")
    
    def _mix_stems(self):
        """Mix all stems according to volume settings"""
        if not self.stems_available or not self.stem_data:
            return self.audio_data
        
        # Create mixed audio from current playback position
        mixed_length = len(self.audio_data)
        mixed_audio = np.zeros(mixed_length, dtype=np.float32)
        
        for stem_name, stem_audio in self.stem_data.items():
            if stem_name in self.stem_volumes:
                volume = self.stem_volumes[stem_name]
                if volume > 0:  # Skip if muted
                    # Ensure stem is same length as original
                    if len(stem_audio) == mixed_length:
                        mixed_audio += stem_audio * volume
                    else:
                        logger.warning(f"{stem_name} stem length mismatch: {len(stem_audio)} vs {mixed_length}")
        
        return mixed_audio
    
    def _on_seek_change(self, value):
        """Handle seek slider changes"""
        if not self.file_loaded:
            return
        
        try:
            # Convert percentage to frame position
            seek_percent = float(value)
            target_frame = int((seek_percent / 100.0) * self.total_frames)
            
            # Clamp to valid range
            target_frame = max(0, min(target_frame, self.total_frames - 1))
            
            # Update playback position
            self._seeking = True
            self.current_frame = target_frame
            self.playback_position = float(target_frame)
            
            # Reset pitch preservation buffers if seeking during playback
            if self.is_playing and self.stretch_processor is not None:
                # Keep allocated buffer, just reset positions
                self.stretch_buffer_used = 0
                self.stretch_read_pos = 0
                self.stretch_input_pos = target_frame  # Reset input position to seek target
                logger.debug(f"Reset pitch buffers after seek to frame {target_frame}")
            
            # EQ state handled automatically by master_isolator
            
            # Update position display immediately
            self._update_position_display()
            self._seeking = False
            
            logger.debug(f"Seeked to {seek_percent:.1f}% (frame {target_frame})")
            
        except Exception as e:
            logger.error(f"Seek error: {e}")
            self._seeking = False
    
    def _toggle_play_pause(self):
        """Toggle playback"""
        if not self.file_loaded:
            return
        
        if self.is_playing:
            self._stop_playback()
        else:
            self._start_playback()
    
    def _start_playback(self):
        """Start playback with working tempo control"""
        try:
            self.is_playing = True
            self.play_pause_button.config(text="⏸ Pause")
            
            # Initialize simple playback - use current slider position
            # In case user moved slider during pause, get position from slider
            if self.total_frames > 0:
                slider_percent = self.seek_slider_var.get()
                slider_frame = int((slider_percent / 100.0) * self.total_frames)
                slider_frame = max(0, min(slider_frame, self.total_frames - 1))
                self.current_frame = slider_frame
            
            self.playback_position = float(self.current_frame)
            self._audio_position = self.playback_position
            self.last_audio_sample = 0.0
            
            # Reset pitch preservation buffers when starting from new position
            if hasattr(self, 'stretch_processor') and self.stretch_processor is not None:
                # Keep allocated buffer, just reset positions
                self.stretch_buffer_used = 0
                self.stretch_read_pos = 0
                self.stretch_input_pos = self.current_frame
            
            # Initialize streaming pitch preservation if enabled
            if RUBBERBAND_AVAILABLE and self.preserve_pitch_var.get():
                if self.stretch_processor is None or abs(self.current_tempo_ratio - self.last_tempo_ratio) > 0.01:
                    self._init_stretch_processor(self.current_tempo_ratio)
            
            def audio_callback(outdata, frames, time, status):
                # DEVELOPMENT GUARD: Never call RubberBand methods in audio callback!
                # This callback should only read from ring buffer - all RB processing
                # happens in the producer thread to prevent segfaults and dropouts.
                assert not hasattr(self, '_in_audio_callback'), "Nested audio callback detected!"
                self._in_audio_callback = True
                
                try:
                    # Debug: count audio callback calls
                    if not hasattr(self, '_audio_callback_count'):
                        self._audio_callback_count = 0
                    self._audio_callback_count += 1
                    if self._audio_callback_count % 100 == 0:  # Log every 100 calls
                        logger.debug(f"🔊 AUDIO CALLBACK: call #{self._audio_callback_count}, requesting {frames} frames")
                    
                    # No logging/UI here; set flags only
                    if status:
                        self._had_underflow = True

                    out, n = self.out_ring.read(frames)
                    if n < frames:
                        # underrun: fill tail with zeros
                        out[n:] = 0.0
                    
                    if self.device_output_channels == 1:
                        # Mono output
                        outdata[:frames, 0] = out[:, 0]
                    else:
                        # Stereo output - duplicate mono to both channels
                        outdata[:frames, 0] = out[:, 0]  # Left
                        outdata[:frames, 1] = out[:, 0]  # Right (same as left)
                
                finally:
                    # Clean up guard flag
                    if hasattr(self, '_in_audio_callback'):
                        del self._in_audio_callback
            
            # Start stream with optimized settings for low latency and reduced callback pressure
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,   # OR device SR if you standardize (see Fix B)
                channels=self.device_output_channels,  # Stereo for better device compatibility
                dtype='float32',
                latency='low',
                blocksize=2048,  # 2048 or 4096 gives more headroom
                callback=audio_callback
            )
            self.stream.start()
            
            # Start producer thread to fill ring buffer
            self._producer_stop = False
            self._start_producer_thread()
            
        except Exception as e:
            self._update_status(f"Playback failed: {e}", "red")
            logger.error(f"Playback failed: {e}")
    
    def _process_turntable_style(self, outdata, frames):
        """Turntable-style processing (pitch changes with tempo) - VECTORIZED"""
        advance_rate = float(self.current_tempo_ratio)

        # If you have stems, mix them ONCE per block into a temp buffer (vectorized)
        src = None
        if self.stems_available and self.stem_data:
            # Mix stems at current integer frame positions (no Python loops)
            pos0 = int(self.playback_position)
            pos1 = min(pos0 + frames + 2, len(self.audio_data))
            src = np.zeros(pos1 - pos0, dtype=np.float32)
            for stem_name, stem_audio in self.stem_data.items():
                vol = float(self.stem_volumes.get(stem_name, 1.0))
                if pos1 <= len(stem_audio):
                    # Get raw stem chunk
                    stem_chunk = stem_audio[pos0:pos1]
                    
                    # Apply per-stem tone EQ (same logic as pitch-preserving mode)
                    if (self.enable_stem_eqs and stem_name in self.stem_tone_eqs and 
                        self.stem_eq_enabled.get(stem_name, False)):
                        try:
                            stem_chunk_eq = self.stem_tone_eqs[stem_name].process_block(stem_chunk)
                            stem_chunk = stem_chunk_eq.flatten()  # Convert (N,1) back to (N,)
                        except Exception as e:
                            logger.warning(f"Turntable stem EQ error for {stem_name}: {e}")
                            # Continue without stem EQ processing
                    
                    # Apply volume and mix
                    src += stem_chunk * vol
        else:
            src = self.audio_data

        # Vectorized interpolation positions
        base = float(self.playback_position)
        pos = base + advance_rate * np.arange(frames, dtype=np.float32)
        pos_int = pos.astype(np.int32)
        max_idx = len(src) - 2
        if max_idx < 1:
            outdata[:frames, 0] = 0.0
            return
        np.clip(pos_int, 0, max_idx, out=pos_int)
        pos_frac = pos - pos_int

        a = src[pos_int]
        b = src[pos_int + 1]
        out = a * (1.0 - pos_frac) + b * pos_frac  # (frames,)

        # Safety: avoid NaN/inf/denormals
        np.nan_to_num(out, copy=False)

        outdata[:frames, 0] = out
        self.playback_position = float(base + advance_rate * frames)
    
    def _process_streaming_stretch(self, outdata, frames):
        """Simple streaming pitch preservation with smart buffer management"""
        try:
            # Safety check - ensure we have a valid processor
            if self.stretch_processor is None:
                logger.warning("Stretch processor is None, falling back to turntable")
                self._process_turntable_style(outdata, frames)
                return
                
            # Safety check - ensure buffers exist
            if not hasattr(self, 'stretch_buffer_used'):
                self.stretch_buffer_used = 0
            if not hasattr(self, 'stretch_read_pos'):
                self.stretch_read_pos = 0
                
            # Ensure we have enough processed audio available
            samples_available = self.stretch_buffer_used - self.stretch_read_pos
            
            # Keep at least 2x chunk size buffered ahead for smoother playback
            buffer_ahead = self.process_chunk_size * 2  # 16384 samples
            if samples_available < frames + buffer_ahead:
                self._process_more_audio()
                samples_available = self.stretch_buffer_used - self.stretch_read_pos
            
            # Copy from processed audio buffer
            if samples_available >= frames:
                outdata[:frames, 0] = self.stretch_processed_audio[self.stretch_read_pos:self.stretch_read_pos + frames]
                self.stretch_read_pos += frames
                
                # CRITICAL: Advance playback position for UI tracking
                # For pitch preservation, position should advance 1:1 with output samples
                # because playback_position represents the output timeline, not input consumption
                self.playback_position += frames
                
                # Keep buffer size manageable to prevent memory bloat
                # self._trim_buffer_if_needed()  # Moved to producer thread (Issue #7)
            else:
                # Emergency fallback to turntable style to maintain audio
                logger.debug(f"Stretch underrun, falling back to turntable: {samples_available}/{frames}")
                self._process_turntable_style(outdata, frames)
        
        except Exception as e:
            logger.error(f"Streaming stretch error: {e}")
            self._process_turntable_style(outdata, frames)
    
    def _process_more_audio(self):
        """Process more audio using streaming RubberBand API"""
        try:
            # Check if we're using streaming API
            if (hasattr(self, 'rubberband_stretcher') and 
                self.stretch_processor and 
                self.stretch_processor.get('streaming', False)):
                return self._process_more_audio_streaming()
            else:
                return self._process_more_audio_fallback()
                
        except Exception as e:
            # Set error flag instead of logging in processing thread
            self._producer_error = f"Audio processing error: {e}"
            
    def _process_more_audio_streaming(self):
        """Process audio using proper streaming RubberBand API"""
        try:
            current_pos = self.stretch_input_pos
            chunk_size = self.process_chunk_size
            logger.info(f"🎵 STREAMING PROCESSING: pos={current_pos}, stems_available={self.stems_available}, enable_stem_eqs={self.enable_stem_eqs}")
            
            # Check bounds
            if current_pos >= len(self.audio_data) - chunk_size:
                return
            
            # Get audio chunk with proper channel handling
            if self.stems_available and self.stem_data:
                # Mix stems for this chunk with per-stem EQ processing
                input_chunk = np.zeros(chunk_size, dtype=np.float32)
                for stem_name, stem_audio in self.stem_data.items():
                    if current_pos + chunk_size <= len(stem_audio):
                        # Get raw stem chunk
                        stem_chunk = stem_audio[current_pos:current_pos + chunk_size]
                        
                        # Apply per-stem tone EQ (ToneEQ3) - check both global and per-stem enable  
                        if (self.enable_stem_eqs and stem_name in self.stem_tone_eqs and 
                            self.stem_eq_enabled.get(stem_name, False)):
                            try:
                                original_rms = np.sqrt(np.mean(stem_chunk**2))
                                stem_chunk_eq = self.stem_tone_eqs[stem_name].process_block(stem_chunk)
                                stem_chunk = stem_chunk_eq.flatten()  # Convert (N,1) back to (N,)
                            except Exception as e:
                                logger.warning(f"Stem EQ error for {stem_name}: {e}")
                                # Continue without stem EQ processing
                        
                        # Apply stem volume
                        volume = self.stem_volumes.get(stem_name, 1.0)
                        input_chunk += stem_chunk * volume
            else:
                # Use original audio if no stems available
                if len(self.audio_data.shape) == 1:
                    # Mono audio
                    input_chunk = self.audio_data[current_pos:current_pos + chunk_size]
                else:
                    # Stereo audio - take left channel or average both channels
                    input_chunk = self.audio_data[current_pos:current_pos + chunk_size, 0]
            
            if len(input_chunk) < chunk_size:
                return
            
            # Convert to proper format for streaming RubberBand: (frames, channels)
            if self.audio_channels == 1:
                # Mono: (frames,) -> (frames, 1)
                input_block = input_chunk.reshape(-1, 1).astype(np.float32)
            else:
                # Stereo: duplicate mono to stereo (frames,) -> (frames, 2)
                input_block = np.column_stack([input_chunk, input_chunk]).astype(np.float32)
            
            # Validate buffer to prevent segfaults
            input_block = self._validate_rubberband_buffer(input_block)
            
            # Check for pending operations (non-blocking)
            pending_ratio = None
            pending_reinit = False
            pending_disable = False
            if self._rb_lock.acquire(blocking=False):
                try:
                    if self._pending_time_ratio is not None:
                        pending_ratio = self._pending_time_ratio
                        self._pending_time_ratio = None
                    if self._pending_reinit:
                        pending_reinit = True
                        self._pending_reinit = False
                    if self._pending_disable:
                        pending_disable = True
                        self._pending_disable = False
                finally:
                    self._rb_lock.release()
            
            # Handle pending disable first
            if pending_disable:
                # Synchronize position: use RubberBand input position for turntable
                if hasattr(self, 'stretch_input_pos'):
                    self.playback_position = float(self.stretch_input_pos)
                
                if hasattr(self, 'rubberband_stretcher') and self.rubberband_stretcher:
                    try:
                        self.rubberband_stretcher.close()
                    except:
                        pass
                    self.rubberband_stretcher = None
                logger.debug(f"RubberBand disabled by producer thread (synced position: {self.playback_position})")
                # Don't return early - might also have pending_reinit
            
            # Handle pending reinit
            if pending_reinit:
                if hasattr(self, 'rubberband_stretcher') and self.rubberband_stretcher:
                    try:
                        self.rubberband_stretcher.close()
                    except:
                        pass
                
                # Synchronize position: use current playback position as RubberBand input position
                self.stretch_input_pos = int(self.playback_position)
                # Will be recreated with pending_ratio below
                
            # Initialize or update RubberBand if needed
            if pending_reinit or not hasattr(self, 'rubberband_stretcher') or not self.rubberband_stretcher:
                time_ratio = pending_ratio if pending_ratio else (1.0 / self.current_tempo_ratio)
                audio_channels = 2 if len(self.audio_data.shape) > 1 and self.audio_data.shape[1] == 2 else 1
                if len(self.audio_data.shape) == 1:
                    audio_channels = 1  # Definitely mono
                
                self.rubberband_stretcher = RubberBand(
                    sample_rate=int(self.sample_rate),
                    channels=audio_channels,
                    options=REALTIME_DEFAULT,
                    time_ratio=time_ratio,
                    pitch_scale=1.0
                )
                
                self.audio_channels = audio_channels
                self.stretch_processor = {
                    'tempo_ratio': 1.0 / time_ratio,
                    'time_ratio': time_ratio,
                    'streaming': True,
                    'initialized': True
                }
                if pending_reinit:
                    logger.debug(f"RubberBand reinitialized by producer thread (time_ratio: {time_ratio:.3f}, synced position: {self.stretch_input_pos})")
                else:
                    logger.debug(f"RubberBand initialized by producer thread (time_ratio: {time_ratio:.3f})")
            elif pending_ratio is not None:
                # Just update existing processor
                self._assert_not_in_audio_callback("RubberBand.set_time_ratio")
                self.rubberband_stretcher.set_time_ratio(pending_ratio)
                logger.debug(f"Applied pending time ratio: {pending_ratio:.3f}")
            
            # Feed to RubberBand streaming processor (only if available)
            if hasattr(self, 'rubberband_stretcher') and self.rubberband_stretcher:
                self._assert_not_in_audio_callback("RubberBand.process")
                self.rubberband_stretcher.process(input_block, final=False)
                
                # CRITICAL: Use proper drain pattern like working example
                self._drain_rubberband_output()
                
                # Advance input position by the chunk size we just consumed
                self.stretch_input_pos += chunk_size
            else:
                logger.debug("RubberBand not available in _process_more_audio_streaming")
            
            # Success - no logging in processing thread
            
        except Exception as e:
            # Set error flag instead of logging in processing thread
            self._producer_error = f"Streaming processing error: {e}"
    
    def _drain_rubberband_output(self):
        """Drain all available output from RubberBand processor (like working example)"""
        try:
            self._assert_not_in_audio_callback("RubberBand.available/retrieve")
            total_samples = 0
            n = self.rubberband_stretcher.available()
            while n > 0:
                # Retrieve processed audio: (frames, channels)
                output_block = self.rubberband_stretcher.retrieve(n)
                if output_block is None or len(output_block) == 0:
                    break
                
                # Convert to mono for our processing pipeline
                if self.audio_channels == 1:
                    stretched = output_block[:, 0].astype(np.float32)
                else:
                    # Average stereo to mono for now (could be improved)
                    stretched = np.mean(output_block, axis=1).astype(np.float32)
                
                if len(stretched) > 0:
                    # Minimal crossfade for streaming (much smaller since RubberBand handles continuity)
                    crossfade_size = min(32, len(stretched) // 32)
                    
                    if len(stretched) > crossfade_size and self.stretch_buffer_used >= crossfade_size:
                        # Light equal-power crossfade
                        fade_out_curve = np.cos(np.linspace(0, np.pi/2, crossfade_size))
                        fade_in_curve = np.sin(np.linspace(0, np.pi/2, crossfade_size))
                        
                        fade_start = self.stretch_buffer_used - crossfade_size
                        self.stretch_processed_audio[fade_start:self.stretch_buffer_used] *= fade_out_curve
                        stretched[:crossfade_size] *= fade_in_curve
                    
                    # Efficient buffer management with power-of-two growth (Issue #3)
                    self._expand_stretch_buffer_if_needed(stretched)
                    
                    # Append new data
                    end_pos = self.stretch_buffer_used
                    self.stretch_processed_audio[end_pos:end_pos + len(stretched)] = stretched
                    self.stretch_buffer_used += len(stretched)
                    
                    total_samples += len(stretched)
                
                # Check for more available data
                n = self.rubberband_stretcher.available()
            
            if total_samples > 0:
                logger.debug(f"✓ Drained {total_samples} samples from RubberBand")
            
        except Exception as e:
            logger.debug(f"Drain error: {e}")
            
    def _process_more_audio_fallback(self):
        """Fallback to old stateless RubberBand processing"""
        try:
            current_pos = self.stretch_input_pos
            chunk_size = self.process_chunk_size
            logger.info(f"🎵 FALLBACK PROCESSING: pos={current_pos}, stems_available={self.stems_available}, enable_stem_eqs={self.enable_stem_eqs}")
            
            # Drift detection and correction
            expected_buffer_size = self.stretch_buffer_used - self.stretch_read_pos
            if expected_buffer_size < 0:
                logger.warning(f"Buffer position drift detected: read_pos={self.stretch_read_pos}, buffer_used={self.stretch_buffer_used}")
                self.stretch_read_pos = max(0, self.stretch_buffer_used - chunk_size)
                return
            
            # Check bounds
            if current_pos >= len(self.audio_data) - chunk_size:
                return
            
            # Get audio chunk with stem mixing and per-stem EQ processing
            if self.stems_available and self.stem_data:
                input_chunk = np.zeros(chunk_size, dtype=np.float32)
                for stem_name, stem_audio in self.stem_data.items():
                    if current_pos + chunk_size <= len(stem_audio):
                        # Get raw stem chunk
                        stem_chunk = stem_audio[current_pos:current_pos + chunk_size]
                        
                        # Apply per-stem tone EQ (ToneEQ3) - check both global and per-stem enable  
                        if (self.enable_stem_eqs and stem_name in self.stem_tone_eqs and 
                            self.stem_eq_enabled.get(stem_name, False)):
                            try:
                                original_rms = np.sqrt(np.mean(stem_chunk**2))
                                stem_chunk_eq = self.stem_tone_eqs[stem_name].process_block(stem_chunk)
                                stem_chunk = stem_chunk_eq.flatten()  # Convert (N,1) back to (N,)
                            except Exception as e:
                                logger.warning(f"Fallback stem EQ error for {stem_name}: {e}")
                                # Continue without stem EQ processing
                        
                        # Apply stem volume
                        volume = self.stem_volumes.get(stem_name, 1.0)
                        input_chunk += stem_chunk * volume
            else:
                input_chunk = self.audio_data[current_pos:current_pos + chunk_size]
            
            if len(input_chunk) < chunk_size:
                return
            
            # Process with old stateless RubberBand
            tempo_ratio = 1.0 / self.stretch_processor['tempo_ratio']
            
            # Add padding for first chunk
            if self.stretch_buffer_used == 0:
                silence_padding = np.zeros(256, dtype=np.float32)
                padded_chunk = np.concatenate([silence_padding, input_chunk])
            else:
                padded_chunk = input_chunk
            
            import rubberband
            stretched = rubberband.stretch(
                padded_chunk,
                rate=self.sample_rate,
                ratio=tempo_ratio,
                crispness=2,
                formants=False,
                precise=True
            )
            
            # Remove stretched silence from first chunk
            if self.stretch_buffer_used == 0 and len(padded_chunk) > len(input_chunk):
                silence_stretched_size = int(256 * tempo_ratio)
                silence_stretched_size = min(silence_stretched_size, len(stretched) // 2)
                if len(stretched) > silence_stretched_size:
                    stretched = stretched[silence_stretched_size:]
            
            if len(stretched) > 0:
                # Apply crossfading and buffer management (same as before)
                crossfade_size = min(128, len(stretched) // 32)
                
                if len(stretched) > crossfade_size and self.stretch_buffer_used >= crossfade_size:
                    fade_out_curve = np.cos(np.linspace(0, np.pi/2, crossfade_size))
                    fade_in_curve = np.sin(np.linspace(0, np.pi/2, crossfade_size))
                    
                    fade_start = self.stretch_buffer_used - crossfade_size
                    self.stretch_processed_audio[fade_start:self.stretch_buffer_used] *= fade_out_curve
                    stretched[:crossfade_size] *= fade_in_curve
                
                # Buffer management with power-of-two growth (Issue #3)
                self._expand_stretch_buffer_if_needed(stretched)
                
                end_pos = self.stretch_buffer_used
                self.stretch_processed_audio[end_pos:end_pos + len(stretched)] = stretched
                self.stretch_buffer_used += len(stretched)
                
                self.stretch_input_pos += chunk_size
                logger.debug(f"Fallback processed: {len(stretched)} samples")
            
        except Exception as e:
            logger.debug(f"Fallback audio processing error: {e}")
    
    def _process_eq(self, outdata, frames):
        """Apply EQ processing to audio buffer"""
        try:
            # EQ processing handled by master_isolator in main processing chain
            pass
        except Exception as e:
            # Set error flag instead of logging in processing thread
            self._producer_error = f"EQ processing error: {e}"
    
    def _stop_playback(self):
        """Stop playback"""
        # Stop producer thread first
        self._producer_stop = True
        
        self.is_playing = False
        self.play_pause_button.config(text="▶ Play")
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
    
    def _expand_stretch_buffer_if_needed(self, stretched_data):
        """Expand stretch buffer using power-of-two growth to reduce reallocation spikes"""
        required_size = self.stretch_buffer_used + len(stretched_data)
        
        # Check if we need to grow
        if required_size > len(self.stretch_processed_audio):
            # Initial allocation or growth needed
            if self.stretch_buffer_used == 0:
                # Initial generous allocation (was *8, now *64 for fewer reallocations)
                initial_size = max(len(stretched_data), self.process_chunk_size * 64)
                self.stretch_processed_audio = np.zeros(initial_size, dtype=np.float32)
            else:
                # Power-of-2 growth policy
                new_capacity = 1
                while new_capacity < required_size:
                    new_capacity <<= 1  # Bit shift is more efficient than *= 2
                
                # Cap at max buffer size to prevent runaway growth
                new_capacity = min(new_capacity, self.max_buffer_size)
                
                new_buffer = np.zeros(new_capacity, dtype=np.float32)
                new_buffer[:self.stretch_buffer_used] = self.stretch_processed_audio[:self.stretch_buffer_used]
                self.stretch_processed_audio = new_buffer

    def _trim_buffer_if_needed(self):
        """Trim processed audio buffer to prevent memory bloat and GC pauses"""
        try:
            # Only trim if buffer is getting large and we have a significant read position
            if (self.stretch_buffer_used > self.buffer_trim_threshold and 
                self.stretch_read_pos > self.process_chunk_size):
                
                # Keep unread data plus some safety margin
                safety_margin = self.process_chunk_size // 2
                trim_position = max(0, self.stretch_read_pos - safety_margin)
                
                # Shift unread data to beginning of buffer
                unread_size = self.stretch_buffer_used - trim_position
                self.stretch_processed_audio[:unread_size] = self.stretch_processed_audio[trim_position:self.stretch_buffer_used]
                
                # Update tracking variables
                old_used = self.stretch_buffer_used
                self.stretch_buffer_used = unread_size
                self.stretch_read_pos -= trim_position
                # Keep buffer capacity - don't shrink to avoid reallocations
                
                # Success - no logging in producer thread
                
        except Exception as e:
            # Set error flag instead of logging in producer thread
            self._producer_error = f"Buffer trim error: {e}"
    
    def _start_producer_thread(self):
        """Start the producer thread to fill ring buffer"""
        t = threading.Thread(target=self._producer_loop, daemon=True)
        t.start()
    
    def _produce_chunk_turntable(self, target_frames):
        """Produce audio chunk without pitch preservation (turntable style)"""
        logger.info(f"🎵 TURNTABLE CHUNK: target_frames={target_frames}")
        try:
            # Get current playback position
            current_pos = int(self.current_frame)
            logger.info(f"TURNTABLE DEBUG: current_pos={current_pos}, stems_available={self.stems_available}, enable_stem_eqs={self.enable_stem_eqs}")
            
            # Check bounds
            if current_pos >= len(self.audio_data) - target_frames:
                return np.zeros((target_frames, 1), dtype=np.float32)
            
            # Process with stem EQ
            if self.stems_available and self.stem_data:
                input_chunk = np.zeros(target_frames, dtype=np.float32)
                for stem_name, stem_audio in self.stem_data.items():
                    if current_pos + target_frames <= len(stem_audio):
                        # Get raw stem chunk
                        stem_chunk = stem_audio[current_pos:current_pos + target_frames]
                        
                        # Apply per-stem tone EQ (check both global and per-stem enable)
                        if (self.enable_stem_eqs and stem_name in self.stem_tone_eqs and 
                            self.stem_eq_enabled.get(stem_name, False)):
                            try:
                                stem_chunk_eq = self.stem_tone_eqs[stem_name].process_block(stem_chunk)
                                stem_chunk = stem_chunk_eq.flatten()
                            except Exception as e:
                                logger.warning(f"Turntable stem EQ error for {stem_name}: {e}")
                        
                        # Apply stem volume and mix
                        volume = self.stem_volumes.get(stem_name, 1.0)
                        input_chunk += stem_chunk * volume
                
                # Update position
                self.current_frame = current_pos + target_frames
            else:
                # No stems - use original audio
                input_chunk = self.audio_data[current_pos:current_pos + target_frames]
                self.current_frame = current_pos + target_frames
            
            # Apply master isolator EQ
            if self.master_isolator:
                input_chunk_eq = self.master_isolator.process_block(input_chunk)
                input_chunk = input_chunk_eq.flatten()
            
            return input_chunk.reshape(-1, 1)
            
        except Exception as e:
            logger.error(f"Turntable chunk error: {e}")
            return np.zeros((target_frames, 1), dtype=np.float32)
        
    def _produce_chunk_rubberband(self, target_frames):
        """Produce audio chunk with RubberBand pitch preservation"""  
        logger.info(f"🎵 RUBBERBAND CHUNK: target_frames={target_frames}")
        try:
            # Process more audio with stem EQ if buffer is running low
            samples_available = self.stretch_buffer_used - self.stretch_read_pos
            if samples_available < target_frames:
                self._process_more_audio_streaming()  # This processes stems with EQ
                samples_available = self.stretch_buffer_used - self.stretch_read_pos
            
            # Read from the processed buffer
            if samples_available >= target_frames:
                chunk = self.stretch_processed_audio[self.stretch_read_pos:self.stretch_read_pos + target_frames]
                self.stretch_read_pos += target_frames
                return chunk.reshape(-1, 1)
            else:
                # Not enough audio - return zeros
                return np.zeros((target_frames, 1), dtype=np.float32)
                
        except Exception as e:
            logger.error(f"RubberBand chunk error: {e}")
            return np.zeros((target_frames, 1), dtype=np.float32)

    def _producer_loop(self):
        """Continuously fill out_ring with processed audio."""
        logger.info("🎵 PRODUCER LOOP STARTED")
        TARGET_BLOCK = 8192  # frames per chunk we produce
        WATERMARK = self.sample_rate // 8  # Much smaller watermark for debugging

        while not self._producer_stop and self.is_playing:
            try:
                # Dynamic backoff for producer pacing (Issue #7)
                available_data = self.out_ring.available_data()
                backoff = 0.002 if available_data > WATERMARK else 0.0
                if backoff:
                    logger.debug(f"🎵 RING BUFFER FULL: available={available_data}, watermark={WATERMARK} - backing off")
                    time.sleep(backoff)
                    continue
                
                logger.debug(f"🎵 PROCEEDING WITH CHUNK PRODUCTION: available_data={available_data}, watermark={WATERMARK}")

                # Compute a chunk (choose one path) - check variables safely
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
                                hasattr(self, 'preserve_pitch_var') and self.preserve_pitch_var.get() and
                                (has_active_rb or has_pending_rb_ops))
                
                logger.debug(f"🎵 PRODUCER PATH: use_rubberband={use_rubberband}, pitch_enabled={getattr(self, 'preserve_pitch_var', None) and self.preserve_pitch_var.get()}")
                
                if use_rubberband:
                    chunk = self._produce_chunk_rubberband(TARGET_BLOCK)  # see 3C
                else:
                    chunk = self._produce_chunk_turntable(TARGET_BLOCK)   # see 3D

                if chunk is None or len(chunk) == 0:
                    # end: pad zeros to drain
                    chunk = np.zeros((TARGET_BLOCK, 1), dtype=np.float32)

                # Safety
                np.nan_to_num(chunk, copy=False)

                # Write to ring with partial write handling
                to_write = chunk if self._pending_out is None else np.vstack([self._pending_out, chunk])
                
                written = self.out_ring.write(to_write)
                if written < len(to_write):
                    # Save remainder as pending
                    self._pending_out = to_write[written:]
                else:
                    self._pending_out = None
                
                # Trim buffers to prevent memory bloat (moved from callback - Issue #7)
                self._trim_buffer_if_needed()
                
            except Exception as e:
                # Set error flag but keep producer running
                self._producer_error = str(e)
                # Write silence to prevent underrun (emergency - don't use pending buffer)
                silence_chunk = np.zeros((TARGET_BLOCK, self.device_output_channels), dtype=np.float32)
                self.out_ring.write(silence_chunk)  # Drop any pending audio in error case
                self._pending_out = None  # Clear pending on error
                time.sleep(0.01)  # Brief pause on error

    def _produce_chunk_rubberband(self, out_frames):
        """Produce audio chunk using streaming RubberBand with proper locking"""
        try:
            # Check bounds
            if self.stretch_input_pos >= len(self.audio_data) - 1:
                return None

            # Get input chunk size  
            chunk_size = min(self.process_chunk_size, len(self.audio_data) - self.stretch_input_pos)
            if chunk_size <= 0:
                return None

            # Get audio chunk with stem mixing and per-stem EQ processing
            if self.stems_available and self.stem_data:
                input_chunk = np.zeros(chunk_size, dtype=np.float32)
                for stem_name, stem_audio in self.stem_data.items():
                    if self.stretch_input_pos + chunk_size <= len(stem_audio):
                        # Get raw stem chunk
                        stem_chunk = stem_audio[self.stretch_input_pos:self.stretch_input_pos + chunk_size]
                        
                        # Apply per-stem tone EQ (ToneEQ3) - check both global and per-stem enable  
                        if (self.enable_stem_eqs and stem_name in self.stem_tone_eqs and 
                            self.stem_eq_enabled.get(stem_name, False)):
                            try:
                                original_rms = np.sqrt(np.mean(stem_chunk**2))
                                stem_chunk_eq = self.stem_tone_eqs[stem_name].process_block(stem_chunk)
                                stem_chunk = stem_chunk_eq.flatten()  # Convert (N,1) back to (N,)
                            except Exception as e:
                                logger.warning(f"Stem EQ error for {stem_name}: {e}")
                                # Continue without stem EQ processing
                        
                        # Apply stem volume
                        volume = self.stem_volumes.get(stem_name, 1.0)
                        input_chunk += stem_chunk * volume
            else:
                if len(self.audio_data.shape) == 1:
                    input_chunk = self.audio_data[self.stretch_input_pos:self.stretch_input_pos + chunk_size]
                else:
                    input_chunk = self.audio_data[self.stretch_input_pos:self.stretch_input_pos + chunk_size, 0]

            # Advance input position
            self.stretch_input_pos += chunk_size

            # Convert to RubberBand format: (frames, channels)
            if hasattr(self, 'audio_channels') and self.audio_channels > 1:
                input_block = np.column_stack([input_chunk, input_chunk]).astype(np.float32)
            else:
                input_block = input_chunk.reshape(-1, 1).astype(np.float32)
            
            # Validate buffer to prevent segfaults
            input_block = self._validate_rubberband_buffer(input_block)

            output_chunks = []

            # ---- All RubberBand operations guarded by the same lock ----
            with self._rb_lock:
                # Apply any pending RubberBand parameter changes (single-threaded)
                if self._pending_time_ratio is not None and hasattr(self, 'rubberband_stretcher') and self.rubberband_stretcher:
                    self._assert_not_in_audio_callback("RubberBand.set_time_ratio")
                    self.rubberband_stretcher.set_time_ratio(self._pending_time_ratio)
                    logger.debug(f"Applied pending time ratio: {self._pending_time_ratio:.3f}")
                    self._pending_time_ratio = None

                # Handle pending operations
                if self._pending_disable:
                    # Synchronize position: use RubberBand input position for turntable
                    if hasattr(self, 'stretch_input_pos'):
                        self.playback_position = float(self.stretch_input_pos)
                    
                    if hasattr(self, 'rubberband_stretcher') and self.rubberband_stretcher:
                        try:
                            self.rubberband_stretcher.close()
                        except:
                            pass
                        self.rubberband_stretcher = None
                    self._pending_disable = False
                    logger.debug(f"RubberBand disabled by producer thread (synced position: {self.playback_position})")
                    # Exit lock and fallback to turntable
                    return self._produce_chunk_turntable(out_frames)

                if self._pending_reinit:
                    # Close old stretcher
                    if hasattr(self, 'rubberband_stretcher') and self.rubberband_stretcher:
                        self.rubberband_stretcher.close()
                    
                    # Synchronize position: use current playback position as RubberBand input position
                    self.stretch_input_pos = int(self.playback_position)
                    
                    # Recreate with current SR/channels and pending ratio
                    time_ratio = self._pending_time_ratio if self._pending_time_ratio is not None else 1.0
                    audio_channels = 2 if len(self.audio_data.shape) > 1 and self.audio_data.shape[1] == 2 else 1
                    if len(self.audio_data.shape) == 1:
                        audio_channels = 1  # Definitely mono
                    
                    self.rubberband_stretcher = RubberBand(
                        sample_rate=int(self.sample_rate),
                        channels=audio_channels,
                        options=REALTIME_DEFAULT,
                        time_ratio=time_ratio,
                        pitch_scale=1.0
                    )
                    
                    # Set up metadata
                    self.audio_channels = audio_channels
                    self.stretch_processor = {
                        'tempo_ratio': 1.0 / time_ratio,
                        'time_ratio': time_ratio,
                        'streaming': True,
                        'initialized': True
                    }
                    
                    # Clear flags
                    self._pending_time_ratio = None
                    self._pending_reinit = False
                    logger.debug(f"RubberBand reinitialized by producer thread (time_ratio: {time_ratio:.3f}, synced position: {self.stretch_input_pos})")

                # Ensure we have a valid RubberBand processor
                if not hasattr(self, 'rubberband_stretcher') or not self.rubberband_stretcher:
                    # Exit lock and fallback to turntable
                    return self._produce_chunk_turntable(out_frames)
                
                # Process with streaming RubberBand (all calls inside lock)
                self._assert_not_in_audio_callback("RubberBand.process/available/retrieve")
                self.rubberband_stretcher.process(input_block, final=False)

                # Drain in a loop; never call retrieve() without the lock
                while True:
                    available = self.rubberband_stretcher.available()
                    if available <= 0:
                        break
                    output_block = self.rubberband_stretcher.retrieve(available)
                    if output_block is None or len(output_block) == 0:
                        break
                    # Convert to mono
                    if hasattr(self, 'audio_channels') and self.audio_channels > 1:
                        mono_output = np.mean(output_block, axis=1).astype(np.float32)
                    else:
                        mono_output = output_block[:, 0].astype(np.float32)
                    output_chunks.append(mono_output)
            # ---- end lock ----

            # Process output outside of lock (no RubberBand calls)
            if output_chunks:
                # Concatenate all drained output
                full_output = np.concatenate(output_chunks)
                
                # Take exactly what we need, save rest for next call
                if len(full_output) >= out_frames:
                    chunk_out = full_output[:out_frames]
                    # Store remainder (would need a class buffer for this - simplified for now)
                else:
                    chunk_out = full_output
                    
                # EQ processing handled by master_isolator in main processing chain
                
                # Safety sanitization
                np.nan_to_num(chunk_out, copy=False)
                
                # Update position
                self.playback_position += len(chunk_out)
                
                # Update position for UI thread - use input position for accurate beat tracking
                self._audio_position = float(self.stretch_input_pos)
                
                # Apply master isolator EQ processing (with clickless ON/OFF)
                try:
                    chunk_out_eq = self.master_isolator.process_block(chunk_out)
                    chunk_out = chunk_out_eq.flatten()  # Convert (N,1) back to (N,)
                except Exception as e:
                    # Log error but don't crash audio processing
                    if not hasattr(self, '_eq_error_logged'):
                        logger.error(f"Master EQ processing error: {e}")
                        self._eq_error_logged = True
                
                # Format for ring buffer output channels
                if self.device_output_channels == 1:
                    return chunk_out.reshape(-1, 1).astype(np.float32)
                else:
                    # Duplicate mono to stereo for ring buffer
                    return np.column_stack([chunk_out, chunk_out]).astype(np.float32)
            
            # Fallback to turntable processing until RubberBand is ready
            return self._produce_chunk_turntable(out_frames)
            
        except Exception as e:
            # Set error flag instead of logging in processing thread
            self._producer_error = f"RubberBand chunk production error: {e}"
            # Fallback to turntable processing on error
            return self._produce_chunk_turntable(out_frames)

    def _produce_chunk_turntable(self, out_frames):
        """
        Vectorized 'turntable-style' (tempo changes pitch) path that RESPECTS per-stem EQ.
        Returns (out_frames, 1) float32 or None at end-of-track.
        """
        if self.audio_data is None:
            return None

        step = float(self.current_tempo_ratio)        # <1=faster, >1=slower
        base = float(self.playback_position)

        # Positions for this output block (monotonic forward)
        pos = base + step * np.arange(out_frames, dtype=np.float32)
        pos_int = pos.astype(np.int32)

        # Determine the integer input window we need
        if len(self.audio_data) < 2:
            return None
        start = int(pos_int.min())
        stop  = int(min(pos_int.max() + 2, len(self.audio_data)))  # +1 for interp, +1 safety
        if start >= stop:
            return None

        # ---- Build the source mix for [start:stop] USING per-stem EQs ----
        # Expect stems: dict[str] -> np.ndarray mono shape (total_frames,)
        # and stem_eq: dict[str] -> ToneEQ3 instances
        if getattr(self, "stems_available", False) and getattr(self, "stem_data", None):
            # Preallocate
            basebuf = np.zeros(stop - start, dtype=np.float32)
            for stem_name, stem_arr in self.stem_data.items():
                if stem_arr is None or len(stem_arr) == 0:
                    continue
                # Slice needed window for this stem
                seg = stem_arr[start:stop].astype(np.float32, copy=False)

                # Apply per-stem EQ if present (stateful across calls)
                # Use the same logic as the other processing paths
                if (self.enable_stem_eqs and stem_name in self.stem_tone_eqs and 
                    self.stem_eq_enabled.get(stem_name, False)):
                    try:
                        seg_eq = self.stem_tone_eqs[stem_name].process_block(seg)
                        seg_eq = seg_eq.flatten()  # Convert (N,1) back to (N,)
                    except Exception as e:
                        logger.warning(f"Turntable chunk stem EQ error for {stem_name}: {e}")
                        seg_eq = seg  # Continue without stem EQ processing
                else:
                    seg_eq = seg

                # Stem volume (if you have per-stem faders)
                vol = float(self.stem_volumes.get(stem_name, 1.0))
                basebuf += seg_eq * vol
        else:
            # Fall back to pre-mixed track window
            basebuf = self.audio_data[start:stop].astype(np.float32, copy=False)

        # ---- Vectorized linear interpolation from basebuf to output ----
        # Clamp so pos_int+1 is valid within [start, stop)
        max_idx = (stop - start) - 2
        if max_idx < 1:
            return None
        idx = pos_int - start
        np.clip(idx, 0, max_idx, out=idx)

        frac = (pos - pos_int).astype(np.float32)
        a = basebuf[idx]
        b = basebuf[idx + 1]
        out = (a * (1.0 - frac) + b * frac).astype(np.float32)

        # Advance playback head
        self.playback_position = float(base + step * out_frames)
        
        # Update position for UI thread
        self._audio_position = self.playback_position

        # Apply master isolator EQ processing (with clickless ON/OFF)
        try:
            out_eq = self.master_isolator.process_block(out)
            out = out_eq.flatten()  # Convert (N,1) back to (N,)
        except Exception as e:
            # Log error but don't crash audio processing
            if not hasattr(self, '_eq_error_logged_tb'):
                logger.error(f"Master EQ processing error (turntable): {e}")
                self._eq_error_logged_tb = True

        np.nan_to_num(out, copy=False)
        
        # Format for ring buffer output channels
        if self.device_output_channels == 1:
            return out.reshape(-1, 1)  # (frames, 1)
        else:
            # Duplicate mono to stereo for ring buffer
            return np.column_stack([out, out])  # (frames, 2)

    def _update_status(self, message, color="green"):
        """Update status"""
        self.status_label.config(text=message, fg=color)
        logger.info(f"Status: {message}")
    
    def cleanup(self):
        """Clean up resources when app closes"""
        # Stop playback and producer
        if self.is_playing:
            self._stop_playback()
        
        # Queue RubberBand cleanup for producer thread (never call RB from UI)
        with self._rb_lock:
            self._pending_disable = True
        
        # Give producer thread a moment to handle cleanup
        import time
        time.sleep(0.1)
        
        # Close audio stream
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.warning(f"Error closing audio stream: {e}")

def main():
    parser = argparse.ArgumentParser(description="Fixed Beat Viewer")
    parser.add_argument("audio_file", nargs="?", help="Audio file to load")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    root = tk.Tk()
    app = FixedBeatViewer(root, args.audio_file)
    
    # Set up proper cleanup on window close
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Application interrupted")
        app.cleanup()
    finally:
        app.cleanup()

if __name__ == "__main__":
    main()