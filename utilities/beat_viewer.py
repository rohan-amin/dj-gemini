#!/usr/bin/env python3
"""
play_song.py - Simple audio player with BPM detection, robust seeking, and beat navigation

Dependencies:
    pip install pydub sounddevice essentia numpy
    # For MP3 support, install ffmpeg (e.g., brew install ffmpeg on macOS)
"""

import tkinter as tk
from tkinter import filedialog, ttk
import argparse
import os
import time
import threading
import numpy as np
import essentia.standard as es
from pydub import AudioSegment
import sounddevice as sd
import logging
import json

logger = logging.getLogger(__name__)

class PlaySongApp:
    def __init__(self, master_window, initial_filepath=None):
        self.master = master_window
        self.master.title("Beat Viewer - Audio Player with EQ")
        self.master.geometry("900x800")  # Increased height for tempo and stem controls
        
        # Audio state
        self.audio_segment = None
        self.sample_rate = 0
        self.duration_seconds = 0
        self.bpm = 0
        self.beat_timestamps = np.array([])
        self.current_position = 0.0
        self.is_playing = False
        self.file_loaded = False
        self.current_filepath = None
        
        # SoundDevice streaming
        self.stream = None
        self.audio_data = None
        self.sample_rate = 0
        self.current_frame = 0
        self.total_frames = 0
        self.playback_thread = None
        self.stop_playback = threading.Event()
        self.position_update_id = None
        self._playback_start_time = None
        self._playback_start_pos = 0.0
        self._slider_dragging = False
        self._stream_lock = threading.Lock()
        
        # EQ state
        self.eq_low = 1.0
        self.eq_mid = 1.0
        self.eq_high = 1.0
        self._eq_filters_initialized = False
        self._eq_low_filter = None
        self._eq_mid_filter = None
        self._eq_high_filter = None
        self._eq_low_state = None
        self._eq_mid_state = None
        self._eq_high_state = None
        
        self._create_widgets()
        self._configure_bindings()
        
        # Add tempo scaling support
        self._original_beat_timestamps = None  # Store original timestamps before scaling
        self._original_audio_data = None  # Store original audio data before tempo processing
        
        # Stem separation support
        self.stems_available = False
        self.stems_data = {}  # Dict of stem_name -> np.array
        self.stem_volumes = {'vocals': 1.0, 'drums': 1.0, 'bass': 1.0, 'other': 1.0}
        self.stem_isolated = {'vocals': False, 'drums': False, 'bass': False, 'other': False}
        
        if initial_filepath:
            self.master.after(100, lambda: self._load_audio_file(initial_filepath))
    
    def _create_widgets(self):
        load_frame = tk.Frame(self.master, padx=10, pady=10)
        load_frame.pack(fill=tk.X)
        self.load_button = tk.Button(load_frame, text="Load Audio File", command=self._gui_load_file)
        self.load_button.pack(side=tk.LEFT)
        self.filepath_label = tk.Label(load_frame, text="No file loaded", anchor="w", wraplength=400)
        self.filepath_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        info_frame = tk.Frame(self.master, padx=10, pady=5)
        info_frame.pack(fill=tk.X)
        self.bpm_label = tk.Label(info_frame, text="BPM: --", width=15, anchor="w")
        self.bpm_label.pack(side=tk.LEFT, padx=5)
        
        # BPM input field
        bpm_input_frame = tk.Frame(info_frame)
        bpm_input_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(bpm_input_frame, text="Set BPM:", font=("Arial", 10)).pack(side=tk.LEFT)
        self.bpm_var = tk.StringVar(value="")
        self.bpm_entry = tk.Entry(bpm_input_frame, textvariable=self.bpm_var, width=8, font=("Arial", 10))
        self.bpm_entry.pack(side=tk.LEFT, padx=2)
        self.set_bpm_button = tk.Button(bpm_input_frame, text="Set", command=self._set_bpm_from_input, 
                                       font=("Arial", 10), state=tk.DISABLED)
        self.set_bpm_button.pack(side=tk.LEFT, padx=2)
        self.reset_bpm_button = tk.Button(bpm_input_frame, text="Reset", command=self._reset_to_original_bpm,
                                         font=("Arial", 10), state=tk.DISABLED)
        self.reset_bpm_button.pack(side=tk.LEFT, padx=2)
        
        self.duration_label = tk.Label(info_frame, text="Duration: --", width=20, anchor="w")
        self.duration_label.pack(side=tk.LEFT, padx=5)
        self.position_label = tk.Label(info_frame, text="Position: 00:00", width=15, anchor="w")
        self.position_label.pack(side=tk.LEFT, padx=5)
        self.beat_label = tk.Label(info_frame, text="Beat: --", width=15, anchor="w")
        self.beat_label.pack(side=tk.LEFT, padx=5)
        
        # Status label for processing feedback
        status_frame = tk.Frame(self.master, padx=10, pady=2)
        status_frame.pack(fill=tk.X)
        self.status_label = tk.Label(status_frame, text="", fg="blue", font=("Arial", 9))
        self.status_label.pack()
        controls_frame = tk.Frame(self.master, pady=10)
        controls_frame.pack()
        self.play_pause_button = tk.Button(controls_frame, text="Play", width=10, 
                                         command=self._toggle_play_pause, state=tk.DISABLED)
        self.play_pause_button.pack(side=tk.LEFT, padx=5)
        slider_frame = tk.Frame(self.master, pady=10)
        slider_frame.pack(fill=tk.X, padx=10)
        self.seek_slider_var = tk.DoubleVar()
        self.seek_slider = ttk.Scale(
            slider_frame, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.seek_slider_var, state=tk.DISABLED, length=500
        )
        self.seek_slider.pack(fill=tk.X, expand=True)
        
        # EQ Controls
        eq_frame = tk.Frame(self.master, pady=10)
        eq_frame.pack(fill=tk.X, padx=10)
        
        eq_label = tk.Label(eq_frame, text="EQ Controls:", font=("Arial", 10, "bold"))
        eq_label.pack(anchor=tk.W)
        
        eq_sliders_frame = tk.Frame(eq_frame)
        eq_sliders_frame.pack(fill=tk.X, pady=5)
        
        # Low EQ
        low_frame = tk.Frame(eq_sliders_frame)
        low_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        low_label = tk.Label(low_frame, text="Low", font=("Arial", 9))
        low_label.pack()
        self.eq_low_var = tk.DoubleVar(value=1.0)
        self.eq_low_slider = ttk.Scale(
            low_frame, from_=0.0, to=2.0, orient=tk.VERTICAL,
            variable=self.eq_low_var, length=100,
            command=self._on_eq_change
        )
        self.eq_low_slider.pack()
        self.eq_low_value_label = tk.Label(low_frame, text="1.0", font=("Arial", 8))
        self.eq_low_value_label.pack()
        
        # Mid EQ
        mid_frame = tk.Frame(eq_sliders_frame)
        mid_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        mid_label = tk.Label(mid_frame, text="Mid", font=("Arial", 9))
        mid_label.pack()
        self.eq_mid_var = tk.DoubleVar(value=1.0)
        self.eq_mid_slider = ttk.Scale(
            mid_frame, from_=0.0, to=2.0, orient=tk.VERTICAL,
            variable=self.eq_mid_var, length=100,
            command=self._on_eq_change
        )
        self.eq_mid_slider.pack()
        self.eq_mid_value_label = tk.Label(mid_frame, text="1.0", font=("Arial", 8))
        self.eq_mid_value_label.pack()
        
        # High EQ
        high_frame = tk.Frame(eq_sliders_frame)
        high_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        high_label = tk.Label(high_frame, text="High", font=("Arial", 9))
        high_label.pack()
        self.eq_high_var = tk.DoubleVar(value=1.0)
        self.eq_high_slider = ttk.Scale(
            high_frame, from_=0.0, to=2.0, orient=tk.VERTICAL,
            variable=self.eq_high_var, length=100,
            command=self._on_eq_change
        )
        self.eq_high_slider.pack()
        self.eq_high_value_label = tk.Label(high_frame, text="1.0", font=("Arial", 8))
        self.eq_high_value_label.pack()
        
        # EQ Preset buttons
        preset_frame = tk.Frame(eq_frame)
        preset_frame.pack(pady=5)
        
        reset_eq_button = tk.Button(preset_frame, text="Reset EQ", command=self._reset_eq)
        reset_eq_button.pack(side=tk.LEFT, padx=5)
        
        drop_vocals_button = tk.Button(preset_frame, text="Drop Vocals", command=self._drop_vocals_preset)
        drop_vocals_button.pack(side=tk.LEFT, padx=5)
        
        boost_bass_button = tk.Button(preset_frame, text="Boost Bass", command=self._boost_bass_preset)
        boost_bass_button.pack(side=tk.LEFT, padx=5)
        
        boost_treble_button = tk.Button(preset_frame, text="Boost Treble", command=self._boost_treble_preset)
        boost_treble_button.pack(side=tk.LEFT, padx=5)
        
        # Real-time Tempo Control
        tempo_frame = tk.Frame(self.master, pady=10)
        tempo_frame.pack(fill=tk.X, padx=10)
        
        tempo_label = tk.Label(tempo_frame, text="Real-time Tempo Control:", font=("Arial", 10, "bold"))
        tempo_label.pack(anchor=tk.W)
        
        tempo_slider_frame = tk.Frame(tempo_frame)
        tempo_slider_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(tempo_slider_frame, text="Tempo:", font=("Arial", 9)).pack(side=tk.LEFT)
        self.tempo_var = tk.DoubleVar(value=120.0)
        self.tempo_slider = ttk.Scale(
            tempo_slider_frame, from_=60.0, to=180.0, orient=tk.HORIZONTAL,
            variable=self.tempo_var, length=300,
            command=self._on_tempo_change
        )
        self.tempo_slider.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.tempo_value_label = tk.Label(tempo_slider_frame, text="120.0 BPM", font=("Arial", 9))
        self.tempo_value_label.pack(side=tk.LEFT, padx=5)
        
        tempo_buttons_frame = tk.Frame(tempo_frame)
        tempo_buttons_frame.pack(pady=2)
        
        reset_tempo_button = tk.Button(tempo_buttons_frame, text="Reset Tempo", command=self._reset_tempo)
        reset_tempo_button.pack(side=tk.LEFT, padx=5)
        
        # Stem Isolation Controls
        stems_frame = tk.Frame(self.master, pady=10)
        stems_frame.pack(fill=tk.X, padx=10)
        
        stems_label = tk.Label(stems_frame, text="Stem Isolation Controls:", font=("Arial", 10, "bold"))
        stems_label.pack(anchor=tk.W)
        
        self.stems_status_label = tk.Label(stems_frame, text="Load a file to check for stems...", 
                                          font=("Arial", 9), fg="gray")
        self.stems_status_label.pack(anchor=tk.W, pady=2)
        
        stems_buttons_frame = tk.Frame(stems_frame)
        stems_buttons_frame.pack(fill=tk.X, pady=5)
        
        # Create stem toggle buttons
        self.stem_buttons = {}
        stem_names = ['Vocals', 'Drums', 'Bass', 'Other']
        stem_keys = ['vocals', 'drums', 'bass', 'other']
        
        for i, (name, key) in enumerate(zip(stem_names, stem_keys)):
            btn = tk.Button(stems_buttons_frame, text=f"Solo {name}", 
                           command=lambda k=key: self._toggle_stem_isolation(k),
                           state=tk.DISABLED, width=12)
            btn.pack(side=tk.LEFT, padx=5)
            self.stem_buttons[key] = btn
        
        # Clear isolation button
        self.clear_isolation_button = tk.Button(stems_buttons_frame, text="Clear Isolation", 
                                               command=self._clear_stem_isolation,
                                               state=tk.DISABLED, width=12)
        self.clear_isolation_button.pack(side=tk.LEFT, padx=10)
        
        instructions_frame = tk.Frame(self.master, pady=10)
        instructions_frame.pack()
        instructions = tk.Label(instructions_frame, 
                              text="Keyboard: Left/Right arrows = seek 1 beat, Space = play/pause",
                              font=("Arial", 10))
        instructions.pack()
    
    def _configure_bindings(self):
        self.seek_slider.bind("<Button-1>", self._on_slider_click)
        self.seek_slider.bind("<B1-Motion>", self._on_slider_drag)
        self.seek_slider.bind("<ButtonRelease-1>", self._on_slider_release)
        self.master.bind("<Left>", self._seek_backward_beat)
        self.master.bind("<Right>", self._seek_forward_beat)
        self.master.bind("<space>", self._toggle_play_pause)
        self.master.focus_set()
    
    def _gui_load_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=(("Audio Files", "*.wav *.mp3 *.aac *.flac"), ("All files", "*.*"))
        )
        if filepath:
            self._load_audio_file(filepath)
    
    def _load_audio_file(self, filepath):
        try:
            self.current_filepath = filepath  # Store current filepath
            self.filepath_label.config(text=f"Loading: {os.path.basename(filepath)}...")
            self.master.update_idletasks()
            self.audio_segment = AudioSegment.from_file(filepath)
            self.sample_rate = self.audio_segment.frame_rate
            self.duration_seconds = len(self.audio_segment) / 1000.0
            
            # Use same cache logic as main audio engine
            # Add project root to path to import config and analyzer
            import sys
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.append(project_root)
            
            try:
                import config as app_config
                from audio_engine.audio_analyzer import AudioAnalyzer
                
                # Use same analyzer as main engine
                analyzer = AudioAnalyzer(
                    cache_dir=app_config.CACHE_DIR,
                    beats_cache_file_extension=app_config.BEATS_CACHE_FILE_EXTENSION,
                    beat_tracker_algo_name=app_config.DEFAULT_BEAT_TRACKER_ALGORITHM,
                    bpm_estimator_algo_name=app_config.DEFAULT_BPM_ESTIMATOR_ALGORITHM
                )
                
                # Analyze track using same logic as main engine
                analysis_data = analyzer.analyze_track(filepath)
                if analysis_data:
                    self.beat_timestamps = np.array(analysis_data.get('beat_timestamps', []))
                    self.original_bpm = analysis_data.get('bpm', 0)
                    self.bpm = self.original_bpm
                    logger.info(f"Beat Viewer - Loaded cached beat data: {len(self.beat_timestamps)} beats, BPM: {self.original_bpm:.1f}")
                else:
                    logger.warning(f"Beat Viewer - Failed to analyze track with main engine analyzer")
                    self._generate_beat_timestamps()
                    
            except ImportError as e:
                logger.warning(f"Beat Viewer - Could not import main engine components: {e}")
                # Fall back to real-time beat detection
                self._generate_beat_timestamps()
            
            # Prepare audio data for sounddevice streaming
            self.audio_data = np.array(self.audio_segment.get_array_of_samples()).astype(np.float32)
            if self.audio_segment.channels > 1:
                self.audio_data = self.audio_data.reshape((-1, self.audio_segment.channels))
            # Normalize to float32 range (-1.0 to 1.0)
            if self.audio_segment.sample_width == 2:  # 16-bit
                self.audio_data /= 32767.0
            elif self.audio_segment.sample_width == 4:  # 32-bit
                self.audio_data /= 2147483647.0
            
            # Store original audio data for tempo processing
            self._original_audio_data = self.audio_data.copy()
            
            self.total_frames = len(self.audio_data)
            self.current_frame = 0
            self.current_position = 0.0
            self.is_playing = False
            self.file_loaded = True
            self._playback_start_time = None
            self._playback_start_pos = 0.0
            
            # Store original beat timestamps for tempo scaling
            self._original_beat_timestamps = self.beat_timestamps.copy()
            
            # Check for and load stems
            self._load_stems_if_available(filepath)
            
            # Set tempo slider to original BPM
            if hasattr(self, 'original_bpm'):
                self.tempo_var.set(self.original_bpm)
                self.tempo_value_label.config(text=f"{self.original_bpm:.1f} BPM")
            
            # Initialize EQ filters for this track
            self._initialize_eq_filters()
            self.filepath_label.config(text=f"File: {os.path.basename(filepath)}")
            self.bpm_label.config(text=f"BPM: {self.bpm:.1f}")
            self.duration_label.config(text=f"Duration: {time.strftime('%M:%S', time.gmtime(self.duration_seconds))}")
            self.position_label.config(text="Position: 00:00")
            self.beat_label.config(text="Beat: 1")
            self.play_pause_button.config(state=tk.NORMAL)
            self.seek_slider.config(state=tk.NORMAL, to=self.duration_seconds)
            self.seek_slider_var.set(0)
            self.set_bpm_button.config(state=tk.NORMAL)  # Enable BPM input
            self.reset_bpm_button.config(state=tk.NORMAL)  # Enable reset button
            self.bpm_var.set(f"{self.bpm:.1f}")  # Set current BPM in input field
            logger.info(f"Loaded {filepath}: BPM={self.bpm:.1f}, Duration={self.duration_seconds:.2f}s")
        except Exception as e:
            error_msg = f"Error loading file: {str(e)[:100]}"
            self.filepath_label.config(text=error_msg)
            logger.error(f"Error loading {filepath}: {e}")
            self.file_loaded = False
    
    def _toggle_play_pause(self, event=None):
        if not self.file_loaded:
            return
        if self.is_playing:
            self._pause_playback()
        else:
            self._start_playback()
    
    def _start_playback(self):
        """Start audio playback with real-time EQ"""
        if not self.file_loaded or self.is_playing:
            return
        
        self.is_playing = True
        self.play_pause_button.config(text="Pause")
        self.stop_playback.clear()
        self._playback_start_time = time.time()
        self._playback_start_pos = self.current_position
        
        # Calculate start frame
        start_frame = int(self.current_position * self.sample_rate)
        self.current_frame = start_frame
        
        # Start sounddevice stream
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.audio_segment.channels,
            callback=self._audio_callback,
            blocksize=256
        )
        self.stream.start()
        
        # Start position updates
        self._schedule_position_update()
    
    def _pause_playback(self):
        """Pause audio playback"""
        if not self.is_playing:
            return
        
        self.is_playing = False
        self.play_pause_button.config(text="Play")
        self.stop_playback.set()
        
        # Stop the stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Cancel position updates
        if self.position_update_id:
            self.master.after_cancel(self.position_update_id)
            self.position_update_id = None
    
    def _audio_callback(self, outdata, frames, time_info, status):
        """SoundDevice callback for real-time audio streaming with EQ"""
        try:
            with self._stream_lock:
                if not self.is_playing or self.audio_data is None:
                    outdata[:] = 0
                    return
                
                # Check if we have enough audio data
                if self.current_frame + frames > len(self.audio_data):
                    # End of audio reached
                    outdata[:] = 0
                    self.is_playing = False
                    return
                
                # Get audio chunk
                audio_chunk = self.audio_data[self.current_frame:self.current_frame + frames].copy()
                
                # Apply real-time EQ
                if self.eq_low != 1.0 or self.eq_mid != 1.0 or self.eq_high != 1.0:
                    audio_chunk = self._apply_eq_to_chunk(audio_chunk)
                
                # Ensure correct shape for output
                if audio_chunk.ndim == 1:
                    audio_chunk = audio_chunk.reshape(-1, 1)
                
                # Copy to output
                outdata[:] = audio_chunk
                
                # Update frame position
                self.current_frame += frames
                self.current_position = self.current_frame / self.sample_rate
                
        except Exception as e:
            logger.error(f"Audio callback error: {e}")
            outdata[:] = 0
    
    def _apply_eq_to_chunk(self, chunk):
        """Apply real-time EQ using SciPy IIR filters (same as main engine)"""
        if self.eq_low == 1.0 and self.eq_mid == 1.0 and self.eq_high == 1.0:
            return chunk  # No EQ applied
        
        # Initialize filters if not already done
        if not hasattr(self, '_eq_filters_initialized'):
            self._initialize_eq_filters()
        
        try:
            from scipy import signal
            
            # Ensure audio_chunk is 1D for processing
            if chunk.ndim == 2:
                audio_1d = chunk[:, 0]  # Take first channel
            else:
                audio_1d = chunk
            
            # Add safety checks for audio data
            if len(audio_1d) == 0:
                return chunk
            
            # Check for NaN or inf values
            if np.any(np.isnan(audio_1d)) or np.any(np.isinf(audio_1d)):
                return chunk
            
            if (hasattr(self, '_eq_low_filter') and self._eq_low_filter is not None and 
                hasattr(self, '_eq_mid_filter') and self._eq_mid_filter is not None and 
                hasattr(self, '_eq_high_filter') and self._eq_high_filter is not None):
                
                # Apply each EQ band with persistent state (same as main engine)
                try:
                    low_band, self._eq_low_state = signal.lfilter(
                        self._eq_low_filter[0], self._eq_low_filter[1], audio_1d, zi=self._eq_low_state)
                except Exception as e:
                    low_band = audio_1d.copy() * 0.5  # Fallback
                    self._eq_low_state = None
                
                try:
                    mid_band, self._eq_mid_state = signal.lfilter(
                        self._eq_mid_filter[0], self._eq_mid_filter[1], audio_1d, zi=self._eq_mid_state)
                except Exception as e:
                    mid_band = audio_1d.copy() * 0.5  # Fallback
                    self._eq_mid_state = None
                
                try:
                    high_band, self._eq_high_state = signal.lfilter(
                        self._eq_high_filter[0], self._eq_high_filter[1], audio_1d, zi=self._eq_high_state)
                except Exception as e:
                    high_band = audio_1d.copy() * 0.5  # Fallback
                    self._eq_high_state = None

                # Use direct EQ values - no smoothing (same as main engine)
                low_val = self.eq_low
                mid_val = self.eq_mid
                high_val = self.eq_high

                # DJ-style EQ mixing - direct band mixing (same as main engine)
                # Each band is mixed with its gain value
                # When gain is 0, that band contributes nothing
                eq_audio_1d = (low_band * low_val) + (mid_band * mid_val) + (high_band * high_val)
                
            else:
                # Fallback to simple EQ if filters not available
                eq_audio_1d = audio_1d * ((self.eq_low + self.eq_mid + self.eq_high) / 3.0)
            
            # Check for NaN/inf in output
            if np.any(np.isnan(eq_audio_1d)) or np.any(np.isinf(eq_audio_1d)):
                return chunk
            
            # Reshape back to original format
            if chunk.ndim == 2:
                eq_audio = eq_audio_1d.reshape(-1, 1)
            else:
                eq_audio = eq_audio_1d
            
            # Clip to prevent distortion
            eq_audio = np.clip(eq_audio, -1.0, 1.0)
            
            return eq_audio
            
        except Exception as e:
            # Fallback to simple gain if anything fails
            return chunk * ((self.eq_low + self.eq_mid + self.eq_high) / 3.0)
    
    def _initialize_eq_filters(self):
        """Initialize EQ filters using SciPy (same as main engine)"""
        if self._eq_filters_initialized or self.sample_rate == 0:
            return
        
        try:
            from scipy import signal
            
            # True DJ-style shelving filters (same as main engine)
            low_shelf_freq = 120.0   # Hz - low shelf frequency
            high_shelf_freq = 6000.0 # Hz - high shelf frequency
            mid_low_freq = 400.0     # Hz - mid band low frequency
            mid_high_freq = 2500.0   # Hz - mid band high frequency
            
            nyquist = self.sample_rate / 2.0
            
            # True DJ-style EQ filters (same as main engine)
            # Low: Low shelf filter (affects frequencies below cutoff)
            # Mid: Peak/Bell filter (affects frequencies around center)
            # High: High shelf filter (affects frequencies above cutoff)
            
            # Low band: frequencies below 120Hz
            self._eq_low_filter = signal.butter(2, low_shelf_freq/nyquist, btype='low')
            
            # Mid band: frequencies between 120-6000Hz
            self._eq_mid_filter = signal.butter(2, [low_shelf_freq/nyquist, high_shelf_freq/nyquist], btype='band')
            
            # High band: frequencies above 6000Hz
            self._eq_high_filter = signal.butter(2, high_shelf_freq/nyquist, btype='high')

            # Initialize persistent filter state for each band (same as main engine)
            if self._eq_low_filter is not None:
                self._eq_low_state = signal.lfilter_zi(self._eq_low_filter[0], self._eq_low_filter[1])
            if self._eq_mid_filter is not None:
                self._eq_mid_state = signal.lfilter_zi(self._eq_mid_filter[0], self._eq_mid_filter[1])
            if self._eq_high_filter is not None:
                self._eq_high_state = signal.lfilter_zi(self._eq_high_filter[0], self._eq_high_filter[1])
            
            self._eq_filters_initialized = True
            
        except Exception as e:
            # Fallback if SciPy not available
            self._eq_filters_initialized = True
    
    def _schedule_position_update(self):
        if self.is_playing and self.file_loaded:
            self._update_position()
            self.position_update_id = self.master.after(100, self._schedule_position_update)
    
    def _update_position(self):
        if not self.is_playing:
            return
        # Position is updated in the audio callback, just update display
        self._update_position_display()
    
    def _update_position_display(self):
        if not self.file_loaded:
            return
        position_str = time.strftime('%M:%S', time.gmtime(float(self.current_position)))
        self.position_label.config(text=f"Position: {position_str}")
        if not getattr(self, '_slider_dragging', False):
            self.seek_slider_var.set(self.current_position)
        # Calculate current beat number (same logic as main engine)
        if self.beat_timestamps.size > 0:
            # Find which beat we're currently on/after
            current_beat_idx = int(np.searchsorted(self.beat_timestamps, self.current_position, side='right'))
            # Beat numbers start at 1, not 0, for DJ use
            beat_number = current_beat_idx + 1
            # Ensure we don't exceed the total number of beats
            max_beat = len(self.beat_timestamps) + 1
            beat_number = min(beat_number, max_beat)
            self.beat_label.config(text=f"Beat: {beat_number}")
        else:
            self.beat_label.config(text="Beat: --")
    
    def _on_slider_click(self, event):
        if not self.file_loaded:
            return
        slider = event.widget
        slider_length = slider.winfo_width()
        click_x = event.x
        minval = float(slider.cget("from"))
        maxval = float(slider.cget("to"))
        value = minval + (maxval - minval) * click_x / slider_length
        self.seek_slider_var.set(value)
        self._seek_to_position(value)
    
    def _on_slider_release(self, event):
        self._slider_dragging = False
        if not self.file_loaded:
            return
        position = self.seek_slider_var.get()
        self._seek_to_position(position)
    
    def _on_slider_drag(self, event):
        if not self.file_loaded:
            return
        self._slider_dragging = True
        position = self.seek_slider_var.get()
        self.current_position = float(position)
        self._update_position_display()
    
    def _seek_to_position(self, position_seconds):
        if not self.file_loaded:
            return
        position_seconds = max(0, min(position_seconds, self.duration_seconds))
        self.current_position = float(position_seconds)
        self.current_frame = int(self.current_position * self.sample_rate)
        was_playing = self.is_playing
        self._pause_playback()
        if was_playing:
            self._start_playback()
        self._update_position_display()
        logger.info(f"Seeked to {position_seconds:.2f}s")
    
    def _seek_forward_beat(self, event):
        """Seek to the next beat position"""
        if not self.file_loaded or not self.beat_timestamps.size:
            return
        
        # Find the next beat after current position
        current_beat_idx = int(np.searchsorted(self.beat_timestamps, self.current_position, side='right'))
        if current_beat_idx < len(self.beat_timestamps):
            next_beat_time = self.beat_timestamps[current_beat_idx]
            self._seek_to_position(next_beat_time)
            logger.debug(f"Beat Viewer - Seeked forward to beat {current_beat_idx + 2}")
    
    def _seek_backward_beat(self, event):
        """Seek to the previous beat position"""
        if not self.file_loaded or not self.beat_timestamps.size:
            return
        
        # Find the previous beat before current position
        current_beat_idx = int(np.searchsorted(self.beat_timestamps, self.current_position, side='left'))
        if current_beat_idx > 0:
            prev_beat_time = self.beat_timestamps[current_beat_idx - 1]
            self._seek_to_position(prev_beat_time)
            logger.debug(f"Beat Viewer - Seeked backward to beat {current_beat_idx}")
        elif current_beat_idx == 0 and self.current_position > 0:
            # If we're at the first beat, go to the beginning
            self._seek_to_position(0.0)
            logger.debug(f"Beat Viewer - Seeked to beginning")
    
    def _on_eq_change(self, value):
        """Handle EQ slider changes in real-time"""
        # Update EQ values immediately
        self.eq_low = self.eq_low_var.get()
        self.eq_mid = self.eq_mid_var.get()
        self.eq_high = self.eq_high_var.get()
        
        # Update value labels immediately
        self.eq_low_value_label.config(text=f"{self.eq_low:.1f}")
        self.eq_mid_value_label.config(text=f"{self.eq_mid:.1f}")
        self.eq_high_value_label.config(text=f"{self.eq_high:.1f}")
        
        # No playback restart needed - EQ is applied in real-time
    
    def _reset_eq(self):
        """Reset EQ to neutral values"""
        self.eq_low_var.set(1.0)
        self.eq_mid_var.set(1.0)
        self.eq_high_var.set(1.0)
        self._on_eq_change(None)  # Apply immediately for reset
    
    def _scale_beat_timestamps(self, tempo_ratio):
        """Scale beat timestamps to match tempo changes (same logic as main engine)"""
        if self._original_beat_timestamps is None:
            return
        
        logger.debug(f"Beat Viewer - Scaling beat timestamps by tempo ratio {tempo_ratio:.3f}")
        # CRITICAL: When tempo increases, timestamps should DECREASE (audio plays faster)
        # This keeps beat numbers pointing to the same musical locations
        self.beat_timestamps = self._original_beat_timestamps / tempo_ratio
        logger.debug(f"Beat Viewer - Scaled {len(self.beat_timestamps)} beat timestamps")
    
    def _jit_process_tempo(self, target_bpm, tempo_ratio):
        """Just-in-time tempo processing using PyRubberBand"""
        try:
            import pyrubberband as pyrb
            
            if self._original_audio_data is None:
                logger.error(f"Beat Viewer - No original audio data available for JIT processing")
                return False
            
            # Disable controls during processing
            self._disable_controls_for_processing()
            self.master.update()  # Force UI update to show disabled state
            
            logger.info(f"Beat Viewer - Starting JIT tempo processing to {target_bpm} BPM (ratio: {tempo_ratio:.3f})...")
            
            # Use stored original audio data (convert to mono for PyRubberBand)
            original_audio = self._original_audio_data.copy()
            if original_audio.ndim > 1:
                # Convert to mono for processing
                original_audio = original_audio.mean(axis=1)
            
            # Ensure proper range (should already be normalized)
            
            # Process with PyRubberBand
            processed_audio = pyrb.time_stretch(original_audio, int(self.sample_rate), tempo_ratio)
            
            # Convert back to stereo if needed
            if self._original_audio_data.ndim > 1:
                processed_audio = np.column_stack([processed_audio, processed_audio])
            
            # Update audio state
            was_playing = self.is_playing
            self._pause_playback()
            
            self.audio_data = processed_audio.astype(np.float32)
            self.total_frames = len(self.audio_data)
            self.duration_seconds = self.total_frames / self.sample_rate
            
            # Scale beat timestamps to match new audio length
            self._scale_beat_timestamps(tempo_ratio)
            
            # Update BPM display
            self.bpm = target_bpm
            self.bpm_label.config(text=f"BPM: {self.bpm:.1f}")
            
            # Update duration display
            self.duration_label.config(text=f"Duration: {time.strftime('%M:%S', time.gmtime(self.duration_seconds))}")
            self.seek_slider.config(to=self.duration_seconds)
            
            if was_playing:
                self._start_playback()
            
            # Show success status briefly before clearing
            self.status_label.config(text="✅ Tempo change complete!", fg="green")
            self.master.update()
            
            # Re-enable controls after a brief delay
            self.master.after(1500, self._enable_controls_after_processing)
            
            logger.info(f"Beat Viewer - JIT tempo processing completed successfully")
            return True
            
        except ImportError:
            logger.error(f"Beat Viewer - PyRubberBand not available for JIT tempo processing")
            self._enable_controls_after_processing()
            return False
        except Exception as e:
            logger.error(f"Beat Viewer - JIT tempo processing failed: {e}")
            self._enable_controls_after_processing()
            return False
    
    def set_tempo(self, target_bpm):
        """Set playbook tempo using real-time processing only"""
        if self.original_bpm <= 0:
            logger.warning(f"Beat Viewer - Cannot set tempo: original BPM is {self.original_bpm}")
            return
        
        # Always use real-time processing - no more cached tempo files!
        self._apply_real_time_tempo(target_bpm)
    
    def _set_bpm_from_input(self):
        """Set BPM from the input field"""
        try:
            new_bpm = float(self.bpm_var.get())
            if new_bpm > 0:
                self.set_tempo(new_bpm)
                logger.info(f"Beat Viewer - BPM set to {new_bpm:.1f} from input")
            else:
                logger.warning(f"Beat Viewer - Invalid BPM value: {new_bpm}")
        except ValueError:
            logger.warning(f"Beat Viewer - Invalid BPM input: {self.bpm_var.get()}")
    
    def _reset_to_original_bpm(self):
        """Reset to original BPM and audio"""
        if self.original_bpm <= 0 or self._original_audio_data is None:
            logger.warning(f"Beat Viewer - Cannot reset: no original data available")
            return
        
        logger.info(f"Beat Viewer - Resetting to original BPM {self.original_bpm:.1f}")
        
        # Restore original audio data
        was_playing = self.is_playing
        self._pause_playback()
        
        self.audio_data = self._original_audio_data.copy()
        self.total_frames = len(self.audio_data)
        self.duration_seconds = self.total_frames / self.sample_rate
        
        # Restore original beat timestamps
        if self._original_beat_timestamps is not None:
            self.beat_timestamps = self._original_beat_timestamps.copy()
        
        # Update BPM display
        self.bpm = self.original_bpm
        self.bpm_label.config(text=f"BPM: {self.bpm:.1f}")
        self.bpm_var.set("")  # Clear input field
        
        # Update duration display
        self.duration_label.config(text=f"Duration: {time.strftime('%M:%S', time.gmtime(self.duration_seconds))}")
        self.seek_slider.config(to=self.duration_seconds)
        
        if was_playing:
            self._start_playback()
        
        logger.info(f"Beat Viewer - Reset to original BPM complete")
    
    def _disable_controls_for_processing(self):
        """Disable UI controls during processing"""
        self.play_pause_button.config(state=tk.DISABLED)
        self.set_bpm_button.config(state=tk.DISABLED)
        self.reset_bpm_button.config(state=tk.DISABLED)
        self.seek_slider.config(state=tk.DISABLED)
        self.load_button.config(state=tk.DISABLED)
        self.status_label.config(text="🔄 Processing tempo change...", fg="blue")
    
    def _enable_controls_after_processing(self):
        """Re-enable UI controls after processing"""
        if self.file_loaded:
            self.play_pause_button.config(state=tk.NORMAL)
            self.set_bpm_button.config(state=tk.NORMAL)
            self.reset_bpm_button.config(state=tk.NORMAL)
            self.seek_slider.config(state=tk.NORMAL)
        self.load_button.config(state=tk.NORMAL)
        self.status_label.config(text="", fg="blue")  # Clear status
    
    def _generate_beat_timestamps(self):
        """Generate beat timestamps using Essentia (fallback method)"""
        # Convert to mono numpy array for Essentia (for BPM detection)
        samples = np.array(self.audio_segment.get_array_of_samples()).astype(np.float32)
        if self.audio_segment.channels > 1:
            samples = samples.reshape((-1, self.audio_segment.channels)).mean(axis=1)
        samples /= np.iinfo(self.audio_segment.array_type).max
        beat_tracker = es.BeatTrackerDegara()
        self.beat_timestamps = beat_tracker(samples)
        self.original_bpm = len(self.beat_timestamps) / (self.duration_seconds / 60.0) if self.duration_seconds > 0 else 0
        self.bpm = self.original_bpm  # Current BPM (can be changed by tempo scaling)
        logger.info(f"Beat Viewer - Generated beat timestamps: {len(self.beat_timestamps)} beats, BPM: {self.original_bpm:.1f}")
    
    def _apply_eq_to_audio(self, audio_segment):
        """Apply EQ to an audio segment"""
        if self.eq_low == 1.0 and self.eq_mid == 1.0 and self.eq_high == 1.0:
            return audio_segment  # No EQ applied
        
        # Convert to numpy array for processing
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        if audio_segment.channels > 1:
            samples = samples.reshape((-1, audio_segment.channels))
        
        # Apply simple EQ using frequency domain processing
        # This is a simplified EQ - for more sophisticated EQ, you'd use proper filters
        sample_rate = audio_segment.frame_rate
        
        # Apply EQ by scaling different frequency ranges
        # Low frequencies (0-250 Hz)
        if self.eq_low != 1.0:
            # Simple low-pass filter effect
            low_gain = self.eq_low
            if audio_segment.channels == 1:
                samples = samples * low_gain
            else:
                samples = samples * low_gain
        
        # Mid frequencies (250-4000 Hz) - apply to all samples for simplicity
        if self.eq_mid != 1.0:
            mid_gain = self.eq_mid
            if audio_segment.channels == 1:
                samples = samples * mid_gain
            else:
                samples = samples * mid_gain
        
        # High frequencies (4000+ Hz) - apply to all samples for simplicity
        if self.eq_high != 1.0:
            high_gain = self.eq_high
            if audio_segment.channels == 1:
                samples = samples * high_gain
            else:
                samples = samples * high_gain
        
        # Convert back to audio segment
        if audio_segment.channels == 1:
            samples = samples.flatten()
        
        # Ensure samples are in the correct range
        samples = np.clip(samples, -1.0, 1.0)
        
        # Convert back to the original format
        if audio_segment.sample_width == 2:  # 16-bit
            samples = (samples * 32767).astype(np.int16)
        elif audio_segment.sample_width == 4:  # 32-bit
            samples = (samples * 2147483647).astype(np.int32)
        
        # Create new audio segment
        from pydub import AudioSegment
        eq_audio = AudioSegment(
            samples.tobytes(),
            frame_rate=sample_rate,
            sample_width=audio_segment.sample_width,
            channels=audio_segment.channels
        )
        
        return eq_audio
    
    def _apply_eq_to_raw_audio(self, raw_data, num_channels, bytes_per_sample, sample_rate):
        """Apply real-time EQ to raw audio data"""
        if self.eq_low == 1.0 and self.eq_mid == 1.0 and self.eq_high == 1.0:
            return raw_data  # No EQ applied
        
        # Convert raw bytes to numpy array
        if bytes_per_sample == 2:  # 16-bit
            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32767.0
        elif bytes_per_sample == 4:  # 32-bit
            samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483647.0
        else:
            return raw_data  # Unsupported format
        
        # Reshape for multi-channel
        if num_channels > 1:
            samples = samples.reshape((-1, num_channels))
        
        # Apply simple EQ (this is a simplified version - for better EQ use proper filters)
        # For now, we'll apply the EQ as a simple gain to the entire signal
        # In a real implementation, you'd use FFT and apply different gains to frequency bands
        
        # Calculate overall EQ gain (average of the three bands)
        overall_gain = (self.eq_low + self.eq_mid + self.eq_high) / 3.0
        
        # Apply gain
        if num_channels == 1:
            samples = samples * overall_gain
        else:
            samples = samples * overall_gain
        
        # Clip to prevent distortion
        samples = np.clip(samples, -1.0, 1.0)
        
        # Convert back to raw bytes
        if bytes_per_sample == 2:  # 16-bit
            samples = (samples * 32767).astype(np.int16)
        elif bytes_per_sample == 4:  # 32-bit
            samples = (samples * 2147483647).astype(np.int32)
        
        return samples.tobytes()
    
    def _drop_vocals_preset(self):
        """Apply vocal-dropping EQ preset"""
        self.eq_low_var.set(0.2)   # Reduce bass more dramatically
        self.eq_mid_var.set(0.5)   # Cut mid frequencies (vocals are often in mid range)
        self.eq_high_var.set(0.1)  # Cut high frequencies (vocals)
        self._on_eq_change(None)  # Apply immediately for presets
    
    def _boost_bass_preset(self):
        """Apply bass-boosting EQ preset"""
        self.eq_low_var.set(2.5)   # Boost bass dramatically
        self.eq_mid_var.set(1.2)   # Slight mid boost
        self.eq_high_var.set(0.6)  # Reduce high frequencies
        self._on_eq_change(None)  # Apply immediately for presets
    
    def _boost_treble_preset(self):
        """Apply treble-boosting EQ preset"""
        self.eq_low_var.set(0.4)   # Reduce bass more
        self.eq_mid_var.set(0.8)   # Slight mid cut
        self.eq_high_var.set(2.5)  # Boost high frequencies dramatically
        self._on_eq_change(None)  # Apply immediately for presets
    
    def _on_tempo_change(self, value):
        """Handle real-time tempo slider changes"""
        if not self.file_loaded:
            return
            
        target_bpm = float(value)
        self.tempo_value_label.config(text=f"{target_bpm:.1f} BPM")
        
        # Apply tempo change in real-time using PyRubberBand
        self._apply_real_time_tempo(target_bpm)
    
    def _apply_real_time_tempo(self, target_bpm):
        """Apply tempo change in real-time without caching"""
        if not hasattr(self, 'original_bpm') or self.original_bpm <= 0:
            return
            
        try:
            import pyrubberband as pyrb
            
            tempo_ratio = target_bpm / self.original_bpm
            
            # Use original audio data for processing
            if self._original_audio_data is not None:
                logger.info(f"Real-time tempo processing: {target_bpm:.1f} BPM (ratio: {tempo_ratio:.3f})")
                
                # Process tempo in real-time
                processed_audio = pyrb.time_stretch(self._original_audio_data, int(self.sample_rate), tempo_ratio)
                
                # Update current audio data
                with self._stream_lock:
                    self.audio_data = processed_audio.astype(np.float32)
                    self.total_frames = len(self.audio_data)
                
                # Update beat timestamps
                self._scale_beat_timestamps(tempo_ratio)
                
                # Update current BPM
                self.bpm = target_bpm
                self.bpm_label.config(text=f"BPM: {self.bpm:.1f}")
                
                logger.info(f"Real-time tempo applied successfully")
                
        except ImportError:
            logger.error("PyRubberBand not available for real-time tempo processing")
        except Exception as e:
            logger.error(f"Real-time tempo processing failed: {e}")
    
    def _reset_tempo(self):
        """Reset tempo to original BPM"""
        if hasattr(self, 'original_bpm'):
            self.tempo_var.set(self.original_bpm)
            self._apply_real_time_tempo(self.original_bpm)
    
    def _toggle_stem_isolation(self, stem_key):
        """Toggle isolation for specific stem"""
        if not self.stems_available:
            return
            
        # Toggle the isolation state
        self.stem_isolated[stem_key] = not self.stem_isolated[stem_key]
        
        # Update button text and state
        btn = self.stem_buttons[stem_key]
        if self.stem_isolated[stem_key]:
            btn.config(text=f"✓ {stem_key.title()}", relief=tk.SUNKEN)
        else:
            btn.config(text=f"Solo {stem_key.title()}", relief=tk.RAISED)
        
        # Apply stem isolation
        self._apply_stem_isolation()
    
    def _clear_stem_isolation(self):
        """Clear all stem isolation"""
        if not self.stems_available:
            return
            
        # Reset all isolation states
        for stem_key in self.stem_isolated:
            self.stem_isolated[stem_key] = False
            
        # Update all button states
        for stem_key, btn in self.stem_buttons.items():
            btn.config(text=f"Solo {stem_key.title()}", relief=tk.RAISED)
        
        # Apply changes
        self._apply_stem_isolation()
    
    def _apply_stem_isolation(self):
        """Apply current stem isolation settings to playback"""
        if not self.stems_available:
            return
            
        # Check if any stems are isolated
        any_isolated = any(self.stem_isolated.values())
        
        if any_isolated:
            # Mix only isolated stems
            mixed_audio = np.zeros_like(self.stems_data['vocals'])
            
            for stem_key, is_isolated in self.stem_isolated.items():
                if is_isolated and stem_key in self.stems_data:
                    mixed_audio += self.stems_data[stem_key] * self.stem_volumes[stem_key]
                    
            logger.info(f"Isolated stems: {[k for k, v in self.stem_isolated.items() if v]}")
        else:
            # Mix all stems
            mixed_audio = np.zeros_like(self.stems_data['vocals'])
            for stem_key, stem_audio in self.stems_data.items():
                mixed_audio += stem_audio * self.stem_volumes[stem_key]
                
            logger.info("Playing all stems")
        
        # Update audio data for playback
        with self._stream_lock:
            self.audio_data = mixed_audio.astype(np.float32)
            self.total_frames = len(self.audio_data)
    
    def _load_stems_if_available(self, filepath):
        """Check for and load stems if available"""
        try:
            import config as app_config
            
            # Get unified song cache directory
            song_cache_dir = app_config.get_song_cache_dir(filepath)
            stems_dir = os.path.join(song_cache_dir, "stems")
            
            # Check if stems exist
            stem_files = {
                'vocals': os.path.join(stems_dir, 'vocals.npy'),
                'drums': os.path.join(stems_dir, 'drums.npy'),
                'bass': os.path.join(stems_dir, 'bass.npy'),
                'other': os.path.join(stems_dir, 'other.npy')
            }
            
            stems_exist = all(os.path.exists(path) for path in stem_files.values())
            
            if stems_exist:
                logger.info("Loading stems for real-time isolation...")
                
                # Load all stems
                for stem_name, stem_path in stem_files.items():
                    stem_audio = np.load(stem_path)
                    self.stems_data[stem_name] = stem_audio.astype(np.float32)
                
                self.stems_available = True
                self.stems_status_label.config(text="✓ Stems available (4 stems loaded)", fg="green")
                
                # Enable stem buttons
                for btn in self.stem_buttons.values():
                    btn.config(state=tk.NORMAL)
                self.clear_isolation_button.config(state=tk.NORMAL)
                
                logger.info(f"Successfully loaded {len(self.stems_data)} stems")
                
            else:
                self.stems_available = False
                self.stems_status_label.config(text="No stems found - use preprocessing to generate stems", fg="orange")
                
                # Disable stem buttons
                for btn in self.stem_buttons.values():
                    btn.config(state=tk.DISABLED)
                self.clear_isolation_button.config(state=tk.DISABLED)
                
        except Exception as e:
            logger.error(f"Error loading stems: {e}")
            self.stems_available = False
            self.stems_status_label.config(text="Error loading stems", fg="red")
            
            # Disable stem buttons
            for btn in self.stem_buttons.values():
                btn.config(state=tk.DISABLED)
            self.clear_isolation_button.config(state=tk.DISABLED)
    
    def on_closing(self):
        """Clean up resources when closing"""
        self._pause_playback()
        self.master.destroy()

def main():
    parser = argparse.ArgumentParser(description="Simple audio player with BPM detection")
    parser.add_argument("filepath", type=str, nargs='?', default=None,
                       help="Optional. Path to the audio file to load initially.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    root = tk.Tk()
    app = PlaySongApp(root, initial_filepath=args.filepath)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main() 