#!/usr/bin/env python3
"""
play_song.py - Simple audio player with BPM detection, robust seeking, and beat navigation

Dependencies:
    pip install pydub simpleaudio essentia numpy
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
import simpleaudio as sa
import logging

logger = logging.getLogger(__name__)

class PlaySongApp:
    def __init__(self, master_window, initial_filepath=None):
        self.master = master_window
        self.master.title("Play Song - Audio Player")
        self.master.geometry("600x400")
        
        # Audio state
        self.audio_segment = None
        self.sample_rate = 0
        self.duration_seconds = 0
        self.bpm = 0
        self.beat_timestamps = np.array([])
        self.current_position = 0.0
        self.is_playing = False
        self.file_loaded = False
        self.play_obj = None
        self.playback_thread = None
        self.stop_playback = threading.Event()
        self.position_update_id = None
        self._playback_start_time = None
        self._playback_start_pos = 0.0
        self._slider_dragging = False
        
        self._create_widgets()
        self._configure_bindings()
        
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
        self.duration_label = tk.Label(info_frame, text="Duration: --", width=20, anchor="w")
        self.duration_label.pack(side=tk.LEFT, padx=5)
        self.position_label = tk.Label(info_frame, text="Position: 00:00", width=15, anchor="w")
        self.position_label.pack(side=tk.LEFT, padx=5)
        self.beat_label = tk.Label(info_frame, text="Beat: --", width=15, anchor="w")
        self.beat_label.pack(side=tk.LEFT, padx=5)
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
            self.filepath_label.config(text=f"Loading: {os.path.basename(filepath)}...")
            self.master.update_idletasks()
            self.audio_segment = AudioSegment.from_file(filepath)
            self.sample_rate = self.audio_segment.frame_rate
            self.duration_seconds = len(self.audio_segment) / 1000.0
            # Convert to mono numpy array for Essentia
            samples = np.array(self.audio_segment.get_array_of_samples()).astype(np.float32)
            if self.audio_segment.channels > 1:
                samples = samples.reshape((-1, self.audio_segment.channels)).mean(axis=1)
            samples /= np.iinfo(self.audio_segment.array_type).max
            beat_tracker = es.BeatTrackerDegara()
            self.beat_timestamps = beat_tracker(samples)
            self.bpm = len(self.beat_timestamps) / (self.duration_seconds / 60.0) if self.duration_seconds > 0 else 0
            self.current_position = 0.0
            self.is_playing = False
            self.file_loaded = True
            self._playback_start_time = None
            self._playback_start_pos = 0.0
            self.filepath_label.config(text=f"File: {os.path.basename(filepath)}")
            self.bpm_label.config(text=f"BPM: {self.bpm:.1f}")
            self.duration_label.config(text=f"Duration: {time.strftime('%M:%S', time.gmtime(self.duration_seconds))}")
            self.position_label.config(text="Position: 00:00")
            self.beat_label.config(text="Beat: 1")
            self.play_pause_button.config(state=tk.NORMAL)
            self.seek_slider.config(state=tk.NORMAL, to=self.duration_seconds)
            self.seek_slider_var.set(0)
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
        if not self.file_loaded:
            return
        self.is_playing = True
        self.play_pause_button.config(text="Pause")
        self.stop_playback.clear()
        self._playback_start_time = time.time()
        self._playback_start_pos = self.current_position
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()
        self._schedule_position_update()
    
    def _pause_playback(self):
        self.is_playing = False
        self.play_pause_button.config(text="Play")
        self.stop_playback.set()
        if self.play_obj:
            self.play_obj.stop()
        if self.position_update_id:
            self.master.after_cancel(self.position_update_id)
            self.position_update_id = None
    
    def _playback_loop(self):
        try:
            # Calculate start and end in ms
            start_ms = int(self.current_position * 1000)
            segment = self.audio_segment[start_ms:]
            # Convert to raw audio for simpleaudio
            raw_data = segment.raw_data
            num_channels = segment.channels
            bytes_per_sample = segment.sample_width
            sample_rate = segment.frame_rate
            self.play_obj = sa.play_buffer(raw_data, num_channels, bytes_per_sample, sample_rate)
            # Wait for playback to finish or be stopped
            while self.play_obj.is_playing() and not self.stop_playback.is_set():
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Playback error: {e}")
    
    def _schedule_position_update(self):
        if self.is_playing and self.file_loaded:
            self._update_position()
            self.position_update_id = self.master.after(100, self._schedule_position_update)
    
    def _update_position(self):
        if not self.is_playing:
            return
        elapsed = time.time() - self._playback_start_time if self._playback_start_time else 0
        self.current_position = min(self._playback_start_pos + elapsed, self.duration_seconds)
        self._update_position_display()
    
    def _update_position_display(self):
        if not self.file_loaded:
            return
        position_str = time.strftime('%M:%S', time.gmtime(float(self.current_position)))
        self.position_label.config(text=f"Position: {position_str}")
        if not getattr(self, '_slider_dragging', False):
            self.seek_slider_var.set(self.current_position)
        # Calculate current beat number
        if self.beat_timestamps.size > 0:
            current_beat_idx = int(np.searchsorted(self.beat_timestamps, self.current_position, side='right'))
            self.beat_label.config(text=f"Beat: {current_beat_idx + 1}")
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
        was_playing = self.is_playing
        self._pause_playback()
        if was_playing:
            self._start_playback()
        self._update_position_display()
        logger.info(f"Seeked to {position_seconds:.2f}s")
    
    def _seek_forward_beat(self, event):
        if not self.file_loaded or not self.beat_timestamps.size:
            return
        current_beat_idx = int(np.searchsorted(self.beat_timestamps, self.current_position, side='right'))
        if current_beat_idx < len(self.beat_timestamps):
            next_beat_time = self.beat_timestamps[current_beat_idx]
            self._seek_to_position(next_beat_time)
    
    def _seek_backward_beat(self, event):
        if not self.file_loaded or not self.beat_timestamps.size:
            return
        current_beat_idx = int(np.searchsorted(self.beat_timestamps, self.current_position, side='left'))
        if current_beat_idx > 0:
            prev_beat_time = self.beat_timestamps[current_beat_idx - 1]
            self._seek_to_position(prev_beat_time)
    
    def on_closing(self):
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