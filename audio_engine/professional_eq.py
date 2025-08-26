#!/usr/bin/env python3
"""
Professional EQ System for DJ Audio Engine
Extracted and adapted from beat_viewer_fixed.py

Provides ToneEQ3 (musical shaping) and IsolatorEQ (kill switches) 
with full stereo support and smooth parameter updates.
"""

import numpy as np
import scipy.signal as sps
from scipy.signal import iirpeak, lfilter_zi
import logging

logger = logging.getLogger(__name__)

class ToneEQ3:
    """
    Professional 3-band tone EQ for musical shaping in stereo.
    - Serial processing: low shelf, mid peaking, high shelf
    - Stateful across blocks (per-stage zi for each channel)
    - Smooth parameter updates via short crossfade (default 10 ms)
    - Designed for per-stem musical shaping (not kill switches)
    """
    
    def __init__(self, sample_rate: int, xfade_ms: float = 10.0):
        self.sr = int(sample_rate)
        self._xfade = int(max(1, xfade_ms * 1e-3 * self.sr))
        self._left = 0
        self._cur = self._design(0.0, 0.0, 0.0, 200, 1000, 0.7, 4000)
        self._pend = None

    def set_params_db(self, low_db: float, mid_db: float, high_db: float,
                      f_low: float = 200, f_mid: float = 1000, q_mid: float = 0.7, f_high: float = 4000):
        """Set EQ parameters in dB with smooth crossfade transition"""
        self._pend = self._design(low_db, mid_db, high_db, f_low, f_mid, q_mid, f_high)
        self._left = self._xfade

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """
        Process stereo audio block with 3-band EQ
        
        Args:
            x: Audio data, shape (frames,) for mono or (frames, channels) for stereo
            
        Returns:
            Processed stereo audio, shape (frames, 2)
        """
        # Check if EQ is flat (all gains near 0dB) - bypass processing if so
        if self._is_flat():
            return self._ensure_stereo(x)
        
        # Ensure stereo input
        xin = self._ensure_stereo(x).astype(np.float64, copy=False)
        
        if self._pend is not None and self._left > 0:
            n = len(xin)
            nxf = min(self._left, n)
            y0 = self._run(xin, self._cur, copy_state=True)
            y1 = self._run(xin, self._pend, copy_state=True)
            
            if nxf > 0:
                w = np.linspace(0.0, 1.0, nxf, dtype=np.float64).reshape(-1, 1)
                y = y0.copy()
                y[:nxf] = (1.0 - w) * y0[:nxf] + w * y1[:nxf]
                if nxf < n:
                    y[nxf:] = y1[nxf:]
                self._left -= nxf
            else:
                y = y1
                
            if self._left <= 0:
                self._cur = self._pend
                self._pend = None
        else:
            y = self._run(xin, self._cur)
            
        return y.astype(np.float32)

    def _ensure_stereo(self, x: np.ndarray) -> np.ndarray:
        """Convert audio to stereo format (frames, 2)"""
        if x.ndim == 1:
            # Mono to stereo
            return np.column_stack([x, x])
        elif x.ndim == 2 and x.shape[1] == 1:
            # (N, 1) to (N, 2)
            return np.column_stack([x[:, 0], x[:, 0]])
        elif x.ndim == 2 and x.shape[1] == 2:
            # Already stereo
            return x
        else:
            raise ValueError(f"Unsupported audio shape: {x.shape}")

    def _is_flat(self) -> bool:
        """Check if current EQ settings are essentially flat (passthrough)"""
        if self._cur is None:
            return True
        return all(abs(gain) < 0.1 for gain in [self._cur.get('low_gain', 1.0) - 1.0,
                                                self._cur.get('mid_gain', 1.0) - 1.0, 
                                                self._cur.get('high_gain', 1.0) - 1.0])

    def _design(self, low_db: float, mid_db: float, high_db: float,
                f_low: float, f_mid: float, q_mid: float, f_high: float):
        """Design 3-band EQ filter coefficients for stereo processing"""
        # Convert dB to linear gain
        low_gain = 10**(low_db / 20.0)
        mid_gain = 10**(mid_db / 20.0)
        high_gain = 10**(high_db / 20.0)
        
        nyq = self.sr / 2.0
        
        # Low shelf
        if abs(low_gain - 1.0) > 1e-6:
            b_low, a_low = self._shelf_filter(f_low / nyq, low_gain, 'low')
        else:
            b_low, a_low = [1.0], [1.0]
            
        # Mid peaking
        if abs(mid_gain - 1.0) > 1e-6:
            b_mid, a_mid = iirpeak(f_mid / nyq, q_mid)
            # Scale for gain
            b_mid = b_mid * mid_gain
        else:
            b_mid, a_mid = [1.0], [1.0]
            
        # High shelf  
        if abs(high_gain - 1.0) > 1e-6:
            b_high, a_high = self._shelf_filter(f_high / nyq, high_gain, 'high')
        else:
            b_high, a_high = [1.0], [1.0]

        return {
            'filters': [(b_low, a_low), (b_mid, a_mid), (b_high, a_high)],
            'zi': [None, None, None],  # Will be initialized for stereo on first use
            'low_gain': low_gain,
            'mid_gain': mid_gain,
            'high_gain': high_gain
        }

    def _shelf_filter(self, freq, gain, shelf_type):
        """Design shelf filter (simplified implementation)"""
        if shelf_type == 'low':
            # Low shelf approximation
            if gain > 1.0:
                b = [gain, gain * (1 - freq), 0]
                a = [1, 1 - freq, 0]
            else:
                b = [1, (1 - freq), 0]
                a = [1, gain * (1 - freq), 0]
        else:  # high shelf
            if gain > 1.0:
                b = [gain * freq, gain, 0]
                a = [freq, 1, 0]
            else:
                b = [freq, 1, 0]
                a = [gain * freq, 1, 0]
                
        # Normalize
        b = np.array(b[:2])  # Keep only first 2 coefficients
        a = np.array(a[:2])
        return b, a

    def _run(self, x: np.ndarray, cfg, copy_state=False):
        """Run stereo audio through the 3-band EQ chain"""
        if cfg is None:
            return x
            
        y = x.copy()
        filters = cfg['filters']
        zi_list = cfg['zi']
        
        # Initialize zi for stereo if needed
        for i in range(len(filters)):
            b, a = filters[i]
            if zi_list[i] is None:
                if len(b) > 1 and len(a) > 1:
                    zi_mono = lfilter_zi(b, a)
                    zi_list[i] = np.column_stack([zi_mono, zi_mono])  # Stereo zi
                else:
                    zi_list[i] = np.zeros((max(len(b), len(a)) - 1, 2))
        
        # Apply each filter stage to both channels
        for i, (b, a) in enumerate(filters):
            if len(b) > 1 or len(a) > 1:  # Skip passthrough filters
                if copy_state:
                    zi = zi_list[i].copy()
                else:
                    zi = zi_list[i]
                    
                # Process left and right channels separately
                for ch in range(2):
                    y[:, ch], zi[:, ch] = sps.lfilter(b, a, y[:, ch], zi=zi[:, ch])
                    
                if not copy_state:
                    zi_list[i] = zi
                    
        return y

    def reset(self):
        """Reset all filter states"""
        if self._cur and 'zi' in self._cur:
            for i in range(len(self._cur['zi'])):
                self._cur['zi'][i] = None
        if self._pend and 'zi' in self._pend:
            for i in range(len(self._pend['zi'])):
                self._pend['zi'][i] = None


class IsolatorEQ:
    """
    DJ-style master isolator EQ for stereo processing:
    - Parallel bands split by Linkwitz-Riley 24 dB/oct crossovers (LR4)  
    - Per-band linear gains (0.0 = kill, 1.0 = full)
    - Persistent zi state; crossfaded coefficient updates (default 20 ms)
    """
    
    def __init__(self, sample_rate: int, f_lo: float = 200.0, f_hi: float = 2000.0, xfade_ms: float = 20.0):
        self.sr = int(sample_rate)
        self.f_lo = float(f_lo)
        self.f_hi = float(f_hi)
        self._xfade = int(max(1, xfade_ms * 1e-3 * self.sr))
        self._xfade_remaining = 0
        
        # Current filter configuration (for stereo)
        self._vectorized_filters = []
        self._vec_zi = []
        self._pending_filters = None
        self._pending_zi = None
        self._needs_priming = True
        
        # Gains for each band
        self.low_gain = 1.0
        self.mid_gain = 1.0  
        self.high_gain = 1.0
        self._enabled = True
        
        # Initialize with flat response
        self._update_filters()

    def set_gains(self, low: float, mid: float, high: float):
        """Set isolator gains (0.0 = kill, 1.0 = full) with smooth transition"""
        self.low_gain = max(0.0, min(1.0, float(low)))
        self.mid_gain = max(0.0, min(1.0, float(mid)))
        self.high_gain = max(0.0, min(1.0, float(high)))
        self._update_filters()

    def set_enabled(self, enabled: bool):
        """Enable/disable the isolator with smooth crossfade"""
        self._enabled = bool(enabled)
        self._update_filters()

    def _update_filters(self):
        """Update filter coefficients and trigger crossfade"""
        self._pending_filters = self._design_lr4_crossovers()
        self._pending_zi = [None] * len(self._pending_filters)
        self._xfade_remaining = self._xfade

    def _design_lr4_crossovers(self):
        """Design Linkwitz-Riley 4th order crossovers for 3-band isolation"""
        nyq = self.sr / 2.0
        
        # Design 4th order Butterworth filters (LR4 uses 4th order)
        # Low band: LPF at f_lo
        b_low, a_low = sps.butter(4, self.f_lo / nyq, btype='low')
        
        # High band: HPF at f_hi  
        b_high, a_high = sps.butter(4, self.f_hi / nyq, btype='high')
        
        # Mid band: BPF between f_lo and f_hi
        b_mid, a_mid = sps.butter(4, [self.f_lo / nyq, self.f_hi / nyq], btype='band')
        
        # Apply gains
        if not self._enabled:
            # Bypass mode - unity gain
            return [(np.array([1.0]), np.array([1.0]))]
        
        filters = []
        if self.low_gain > 0.0:
            filters.append((b_low * self.low_gain, a_low))
        if self.mid_gain > 0.0:
            filters.append((b_mid * self.mid_gain, a_mid))
        if self.high_gain > 0.0:
            filters.append((b_high * self.high_gain, a_high))
            
        if not filters:
            # All bands killed - return silence filter
            return [(np.array([0.0]), np.array([1.0]))]
            
        return filters

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """
        Process stereo audio block through isolator EQ
        
        Args:
            x: Stereo audio data, shape (frames, 2)
            
        Returns:
            Processed stereo audio, shape (frames, 2)
        """
        # Ensure stereo input
        if x.ndim == 1:
            x = np.column_stack([x, x])
        elif x.shape[1] != 2:
            raise ValueError(f"IsolatorEQ expects stereo input, got shape {x.shape}")
            
        xin = x.astype(np.float64, copy=False)
        
        # Initialize filters if needed
        if not self._vectorized_filters:
            self._vectorized_filters = self._design_lr4_crossovers()
            self._vec_zi = [None] * len(self._vectorized_filters)
            
        # Run filter chain with crossfading support
        def run_chain(audio, filters, zi_list):
            """Run audio through filter chain for stereo"""
            y = np.zeros_like(audio)
            
            for i, (b, a) in enumerate(filters):
                # Initialize zi for stereo if needed
                if zi_list[i] is None:
                    if len(b) > 1:
                        zi_mono = lfilter_zi(b, a)
                        zi_list[i] = np.column_stack([zi_mono, zi_mono])
                    else:
                        zi_list[i] = np.zeros((max(len(b), len(a)) - 1, 2))
                
                # Process each channel
                band_output = np.zeros_like(audio)
                for ch in range(2):
                    band_output[:, ch], zi_list[i][:, ch] = sps.lfilter(
                        b, a, audio[:, ch], zi=zi_list[i][:, ch]
                    )
                y += band_output
                
            return y

        # Case 1: no pending change — just run the current chain
        if self._pending_filters is None or self._xfade_remaining <= 0:
            y = run_chain(xin, self._vectorized_filters, self._vec_zi)
            return y.astype(np.float32)

        # Case 2: we are in a crossfade window — run both chains and blend
        n = len(xin)
        nxf = min(self._xfade_remaining, n)

        # Prepare per-block zi copies so we don't corrupt long-term state while blending
        vec_zi_tmp = [zi.copy() if zi is not None else None for zi in self._vec_zi]
        pend_zi_tmp = [zi.copy() if zi is not None else None for zi in self._pending_zi]

        # Run whole block through both chains (on temp zi)
        y_old = run_chain(xin, self._vectorized_filters, vec_zi_tmp)
        y_new = run_chain(xin, self._pending_filters, pend_zi_tmp)

        # Crossfade for the first nxf samples
        if nxf > 0:
            w = np.linspace(0.0, 1.0, nxf, dtype=np.float64).reshape(-1, 1)
            y = y_old.copy()
            y[:nxf] = (1.0 - w) * y_old[:nxf] + w * y_new[:nxf]
            if nxf < n:
                y[nxf:] = y_new[nxf:]
            self._xfade_remaining -= nxf
        else:
            y = y_new

        # Commit the temp zi of the appropriate chain to become the new long-term state
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
        self._vec_zi = []
        self._pending_zi = None
        self._needs_priming = True