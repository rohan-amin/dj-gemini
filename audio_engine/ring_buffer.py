#!/usr/bin/env python3
"""
Professional Ring Buffer for DJ Audio Engine
Extracted and adapted from beat_viewer_fixed.py

Provides thread-safe circular buffer for stereo audio with partial write handling
and optimized performance for real-time audio applications.
"""

import numpy as np
import threading
import logging

logger = logging.getLogger(__name__)

class RingBuffer:
    """Thread-safe ring buffer for stereo audio data"""
    
    def __init__(self, capacity_frames, channels=2):
        """
        Initialize ring buffer for stereo audio
        
        Args:
            capacity_frames: Number of frames the buffer can hold
            channels: Number of audio channels (default 2 for stereo)
        """
        self.channels = channels
        self.cap = int(capacity_frames)
        self.buf = np.zeros((self.cap, channels), dtype=np.float32)
        self.w = 0  # write index
        self.r = 0  # read index
        self.size = 0
        self.lock = threading.Lock()

    def write(self, x):
        """
        Write stereo audio data to ring buffer. Returns number of frames written.
        
        Args:
            x: Audio data as numpy array, shape (frames, channels)
            
        Returns:
            int: Number of frames actually written
        """
        if x.ndim == 1:
            # Convert mono to stereo
            x = np.column_stack([x, x])
        elif x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
            
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
        """
        Read n frames from ring buffer. Returns (data, frames_read).
        
        Args:
            n: Number of frames to read
            
        Returns:
            tuple: (stereo_data, frames_actually_read)
        """
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

    def available_write(self):
        """Return number of frames that can be written"""
        with self.lock:
            return self.cap - self.size

    def available_read(self):
        """Return number of frames available to read"""
        with self.lock:
            return self.size

    def is_full(self):
        """Return True if buffer is full"""
        with self.lock:
            return self.size >= self.cap

    def is_empty(self):
        """Return True if buffer is empty"""
        with self.lock:
            return self.size == 0

    def clear(self):
        """Clear all data from buffer"""
        with self.lock:
            self.size = 0
            self.r = 0
            self.w = 0
            self.buf.fill(0.0)

    def get_stats(self):
        """Return buffer statistics for debugging"""
        with self.lock:
            return {
                'capacity': self.cap,
                'size': self.size,
                'available_write': self.cap - self.size,
                'available_read': self.size,
                'write_index': self.w,
                'read_index': self.r,
                'channels': self.channels
            }