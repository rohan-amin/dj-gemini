# utilities/beat_viewer.py

import tkinter as tk
from tkinter import filedialog, ttk
import argparse
import os
import time
import threading
import queue 

import essentia.standard as es
import numpy as np
import sounddevice as sd

UPDATE_INTERVAL_MS = 100

CMD_LOAD_FILE = "LOAD_FILE"
CMD_PLAY = "PLAY"
CMD_PAUSE = "PAUSE"
CMD_SEEK = "SEEK" 
CMD_STOP_STREAM_FORCED = "STOP_STREAM_FORCED"
CMD_SHUTDOWN = "SHUTDOWN"

class BeatViewerApp:
    def __init__(self, master_window, initial_filepath=None):
        self.master = master_window
        self.master.title("DJ Gemini - Beat Viewer")
        self.master.geometry("550x280")

        self.audio_data_gui = None 
        self.sample_rate_gui = 0
        self.beat_timestamps_gui = []
        self.total_duration_samples_gui = 0
        self.total_duration_seconds_gui = 0.0
        
        self.audio_thread_data = None
        self.audio_thread_sample_rate = 0
        self.audio_thread_total_samples = 0
        self.audio_thread_current_frame = 0

        self.stream_lock = threading.Lock()
        self.current_playback_frame_shared = 0 
        self.is_playing_desired_state = False  
        self.audio_file_loaded = False 
        self.seek_in_progress_flag = False 

        self.audio_command_queue = queue.Queue()
        self.audio_thread_stop_event = threading.Event()
        self.audio_thread = threading.Thread(target=self._audio_management_loop, daemon=True)
        
        # self.playback_stream_obj = None # Audio thread manages its own stream reference internally

        self.update_job_id = None
        self._create_widgets()
        self._configure_bindings()

        self.audio_thread.start()

        if initial_filepath:
            self.master.after(10, lambda: self._process_load_file_request(initial_filepath))
        print("DEBUG: BeatViewerApp initialized")

    def _create_widgets(self):
        load_frame = tk.Frame(self.master, padx=10, pady=10)
        load_frame.pack(fill=tk.X)
        self.load_button = tk.Button(load_frame, text="Load Audio File", command=self._gui_select_file_and_load)
        self.load_button.pack(side=tk.LEFT)
        self.filepath_label = tk.Label(load_frame, text="No file loaded.", anchor="w", wraplength=400)
        self.filepath_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        controls_frame = tk.Frame(self.master, pady=10)
        controls_frame.pack()
        self.play_pause_button = tk.Button(controls_frame, text="Play", width=10, command=self._gui_toggle_play_pause, state=tk.DISABLED)
        self.play_pause_button.pack(side=tk.LEFT, padx=5)

        display_frame = tk.Frame(self.master, pady=10)
        display_frame.pack(fill=tk.X, padx=10)
        self.time_label = tk.Label(display_frame, text="Time: 00:00 / 00:00", width=20, anchor="w")
        self.time_label.pack(side=tk.LEFT, padx=5)
        self.beat_label = tk.Label(display_frame, text="Beat: --", width=20, anchor="w")
        self.beat_label.pack(side=tk.LEFT, padx=5)

        self.seek_slider_var = tk.DoubleVar()
        self.seek_slider = ttk.Scale(
            self.master, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.seek_slider_var, state=tk.DISABLED, length=450
        )
        self.seek_slider.pack(pady=10, fill=tk.X, padx=10, expand=True)


    def _configure_bindings(self):
        self.seek_slider.bind("<ButtonRelease-1>", self._gui_on_slider_release)
        self.seek_slider.bind("<B1-Motion>", self._gui_on_slider_drag_update_display_only)

    def _gui_select_file_and_load(self):
        print("DEBUG: GUI - _gui_select_file_and_load called")
        self._send_audio_command(CMD_STOP_STREAM_FORCED, None, "GUI - Requesting stop before file load dialog")
        with self.stream_lock: self.is_playing_desired_state = False
        if self.master.winfo_exists(): self.play_pause_button.config(text="Play")
        if self.update_job_id: 
            if self.master.winfo_exists(): self.master.after_cancel(self.update_job_id)
            self.update_job_id = None

        filepath = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=(("Audio Files", "*.wav *.mp3 *.aac *.flac"), ("All files", "*.*"))
        )
        if filepath:
            print(f"DEBUG: GUI - File selected: {filepath}")
            self._process_load_file_request(filepath)
        else:
            print("DEBUG: GUI - File selection cancelled")

    def _process_load_file_request(self, filepath):
        print(f"DEBUG: GUI - _process_load_file_request for {filepath}")
        if self.master.winfo_exists():
            self.filepath_label.config(text=f"Loading: {os.path.basename(filepath)}...")
            self.master.update_idletasks()
            self.play_pause_button.config(state=tk.DISABLED, text="Play")
            self.seek_slider.config(state=tk.DISABLED)
            self.seek_slider_var.set(0)

        try:
            loader = es.MonoLoader(filename=filepath)
            self.audio_data_gui = loader() 
            self.sample_rate_gui = loader.paramValue('sampleRate')
            if self.sample_rate_gui == 0: raise ValueError("Sample rate is 0.")

            self.total_duration_samples_gui = len(self.audio_data_gui)
            self.total_duration_seconds_gui = self.total_duration_samples_gui / float(self.sample_rate_gui)

            beat_tracker = es.BeatTrackerDegara()
            self.beat_timestamps_gui = beat_tracker(self.audio_data_gui)
            print(f"DEBUG: GUI - Loaded {filepath}, SR: {self.sample_rate_gui}, Duration: {self.total_duration_seconds_gui:.2f}s, Beats: {len(self.beat_timestamps_gui)}")

            with self.stream_lock:
                self.current_playback_frame_shared = 0
                self.is_playing_desired_state = False 
                self.audio_file_loaded = True
            
            self._send_audio_command(CMD_LOAD_FILE, {
                'audio_data': self.audio_data_gui.copy(), 
                'sample_rate': self.sample_rate_gui,
                'total_samples': self.total_duration_samples_gui
            }, "GUI - Sending LOAD_FILE command")

            if self.master.winfo_exists():
                self.filepath_label.config(text=f"File: {os.path.basename(filepath)}")
                self._update_time_display(0, self.total_duration_seconds_gui)
                self._update_beat_label(0)
                self.play_pause_button.config(state=tk.NORMAL)
                self.seek_slider.config(state=tk.NORMAL, to=self.total_duration_seconds_gui)
        except Exception as e:
            error_msg = f"Error: {str(e)[:100]}"
            if self.master.winfo_exists(): self.filepath_label.config(text=error_msg)
            print(f"ERROR in GUI - _process_load_file_request: {e}")
            self.audio_data_gui = None
            with self.stream_lock: self.audio_file_loaded = False; self.is_playing_desired_state = False
    
    def _gui_toggle_play_pause(self):
        print("DEBUG: GUI - _gui_toggle_play_pause called")
        if not self.audio_file_loaded: return

        should_play_now = False
        with self.stream_lock:
            self.is_playing_desired_state = not self.is_playing_desired_state 
            should_play_now = self.is_playing_desired_state
        
        if should_play_now:
            if self.master.winfo_exists(): self.play_pause_button.config(text="Pause")
            self._send_audio_command(CMD_PLAY, None, "GUI - Sending PLAY command")
            # _schedule_display_update will be started by audio thread's PLAY command success
        else:
            if self.master.winfo_exists(): self.play_pause_button.config(text="Play")
            self._send_audio_command(CMD_PAUSE, None, "GUI - Sending PAUSE command")
            # GUI updates are stopped by _schedule_display_update checking is_playing_desired_state

    def _gui_on_slider_drag_update_display_only(self, event=None):
        is_playing_now = False
        with self.stream_lock: is_playing_now = self.is_playing_desired_state
        if not self.audio_file_loaded or is_playing_now: return

        current_time_sec = self.seek_slider_var.get()
        self._update_time_display(current_time_sec, self.total_duration_seconds_gui)
        self._update_beat_label(current_time_sec)

    def _gui_on_slider_release(self, event=None):
        print("DEBUG: GUI - _gui_on_slider_release called")
        if not self.audio_file_loaded: return

        seek_time_seconds = self.seek_slider_var.get()
        new_frame = int(seek_time_seconds * self.sample_rate_gui) 
        new_frame = max(0, min(new_frame, self.total_duration_samples_gui)) 
        
        print(f"DEBUG: GUI - Seek to time: {seek_time_seconds:.2f}s, frame: {new_frame}")

        # Immediately update GUI display to reflect the seek target
        self._update_time_display(seek_time_seconds, self.total_duration_seconds_gui)
        self._update_beat_label(new_frame / float(self.sample_rate_gui) if self.sample_rate_gui > 0 else 0)

        was_playing_before_seek = False
        with self.stream_lock:
            was_playing_before_seek = self.is_playing_desired_state 
            self.current_playback_frame_shared = new_frame # Set new frame for audio thread
            if was_playing_before_seek:
                self.seek_in_progress_flag = True # Signal that a seek is happening
                # Desired state remains playing
        
        self._send_audio_command(CMD_SEEK, 
                                 {'frame': new_frame, 'was_playing': was_playing_before_seek}, 
                                 "GUI - Sending SEEK command")
        
        if was_playing_before_seek:
            # The audio thread (CMD_SEEK -> CMD_PLAY) will restart the audio
            # and also restart the _schedule_display_update loop.
            # GUI should reflect intent to play.
            if self.master.winfo_exists(): self.play_pause_button.config(text="Pause") 


    def _send_audio_command(self, command, data, debug_msg=""):
        data_summary = type(data).__name__ if data is not None else 'None'
        if isinstance(data, dict): data_summary = f"dict_keys({list(data.keys())})"
        print(f"DEBUG: {debug_msg if debug_msg else 'GUI - Sending command:'} {command}, Data: {data_summary}")
        self.audio_command_queue.put((command, data))

    # --- Audio Management Thread ---
    def _audio_management_loop(self):
        print("DEBUG: AudioThread - Started")
        current_managed_stream = None # Stream object managed by this thread
        
        # These are instance variables, set by CMD_LOAD_FILE, used by _sd_callback via self
        # self.audio_thread_data, self.audio_thread_sample_rate etc.

        while not self.audio_thread_stop_event.is_set():
            try:
                command, data = self.audio_command_queue.get(timeout=0.05)
                print(f"DEBUG: AudioThread - Received command: {command}")

                if command == CMD_LOAD_FILE:
                    print("DEBUG: AudioThread - Processing LOAD_FILE")
                    with self.stream_lock: 
                        if current_managed_stream: 
                            current_managed_stream.abort(ignore_errors=True)
                            current_managed_stream.close(ignore_errors=True)
                            current_managed_stream = None
                        
                        self.audio_thread_data = data['audio_data']
                        self.audio_thread_sample_rate = data['sample_rate']
                        self.audio_thread_total_samples = data['total_samples']
                        self.audio_thread_current_frame = 0 
                        self.current_playback_frame_shared = 0 
                    print("DEBUG: AudioThread - New audio data processed for LOAD_FILE")

                elif command == CMD_PLAY:
                    print("DEBUG: AudioThread - Processing PLAY")
                    with self.stream_lock: 
                        if self.audio_thread_data is None: 
                            print("DEBUG: AudioThread - No audio to play for PLAY command")
                            self.is_playing_desired_state = False 
                            self.master.after(0, lambda: self.play_pause_button.config(text="Play") if self.master.winfo_exists() else None)
                            continue 

                        self.is_playing_desired_state = True # Confirm desired state is play
                        self.audio_thread_current_frame = self.current_playback_frame_shared # Sync from shared
                        
                        if self.audio_thread_current_frame >= self.audio_thread_total_samples: 
                            self.audio_thread_current_frame = 0
                        
                        if current_managed_stream: # Clean up if not None
                            print("DEBUG: AudioThread - PLAY: Cleaning up old stream.")
                            current_managed_stream.abort(ignore_errors=True)
                            current_managed_stream.close(ignore_errors=True)
                            current_managed_stream = None

                        print(f"DEBUG: AudioThread - Creating new stream for PLAY. SR: {self.audio_thread_sample_rate}, Frame: {self.audio_thread_current_frame}")
                        current_managed_stream = sd.OutputStream(
                            samplerate=self.audio_thread_sample_rate, channels=1,
                            callback=self._sd_callback, 
                            finished_callback=lambda: self.master.after(0, self._handle_stream_finished)
                        )
                        # Store this stream reference on the instance for the audio thread to manage
                        self.playback_stream_obj = current_managed_stream
                        current_managed_stream.start()
                        print("DEBUG: AudioThread - New stream started for PLAY.")
                    
                    self.master.after(0, self._schedule_display_update) 
                    
                elif command == CMD_PAUSE:
                    print("DEBUG: AudioThread - Processing PAUSE")
                    # is_playing_desired_state already set to False by GUI thread
                    if current_managed_stream and current_managed_stream.active:
                        print("DEBUG: AudioThread - Stopping stream for PAUSE.")
                        current_managed_stream.stop(ignore_errors=True) 
                        print("DEBUG: AudioThread - Stream stopped for PAUSE.")
                    # finished_callback might be triggered by stop(). 
                    # _handle_stream_finished will check seek_flag.

                elif command == CMD_SEEK:
                    seek_info = data 
                    new_seek_frame = seek_info['frame']
                    gui_wants_to_continue_playing = seek_info['was_playing']
                    
                    print(f"DEBUG: AudioThread - Processing SEEK to frame {new_seek_frame}, gui_wants_to_continue_playing: {gui_wants_to_continue_playing}")
                    
                    if current_managed_stream:
                        print("DEBUG: AudioThread - SEEK: Aborting and closing existing stream.")
                        current_managed_stream.abort(ignore_errors=True) 
                        current_managed_stream.close(ignore_errors=True)
                        current_managed_stream = None 
                        print("DEBUG: AudioThread - SEEK: Existing stream aborted and closed.")
                    
                    with self.stream_lock:
                        self.audio_thread_current_frame = new_seek_frame
                        self.current_playback_frame_shared = new_seek_frame
                    
                    if gui_wants_to_continue_playing: 
                        print("DEBUG: AudioThread - SEEK: Was playing, queueing internal PLAY cmd to restart.")
                        with self.stream_lock: # Ensure desired state is Play before queueing PLAY
                            self.is_playing_desired_state = True 
                        self.audio_command_queue.put((CMD_PLAY, None)) 
                    else: # If not continuing play, ensure state is not playing
                        with self.stream_lock:
                            self.is_playing_desired_state = False
                        # GUI button text should be handled by _gui_on_slider_release if it wasn't playing

                    print("DEBUG: AudioThread - SEEK processed.")


                elif command == CMD_STOP_STREAM_FORCED: 
                    print("DEBUG: AudioThread - Processing CMD_STOP_STREAM_FORCED")
                    if current_managed_stream:
                        current_managed_stream.abort(ignore_errors=True)
                        current_managed_stream.close(ignore_errors=True)
                        current_managed_stream = None
                    with self.stream_lock: 
                        self.audio_thread_current_frame = 0 
                        self.current_playback_frame_shared = 0
                        self.is_playing_desired_state = False


                elif command == CMD_SHUTDOWN:
                    print("DEBUG: AudioThread - Processing SHUTDOWN")
                    if current_managed_stream:
                        current_managed_stream.abort(ignore_errors=True)
                        current_managed_stream.close(ignore_errors=True)
                        current_managed_stream = None
                    break 

                self.audio_command_queue.task_done()
            except queue.Empty:
                # This timeout allows checking self.audio_thread_stop_event regularly
                # Also, can do cleanup if stream stopped on its own.
                if current_managed_stream and not current_managed_stream.active : 
                    is_desired_playing_now = False
                    with self.stream_lock: is_desired_playing_now = self.is_playing_desired_state
                    
                    if not is_desired_playing_now and not current_managed_stream.closed : 
                        print("DEBUG: AudioThread - Stream inactive, desired state not playing, closing stream.")
                        current_managed_stream.close(ignore_errors=True)
                        current_managed_stream = None 
                continue
            except Exception as e:
                print(f"ERROR in _audio_management_loop: {e}")
                import traceback
                traceback.print_exc()
                if current_managed_stream:
                    try: current_managed_stream.abort(ignore_errors=True); current_managed_stream.close(ignore_errors=True)
                    except: pass
                    current_managed_stream = None
                with self.stream_lock: self.is_playing_desired_state = False 

        with self.stream_lock: # Update the instance ref on thread exit
            self.playback_stream_obj = current_managed_stream # This thread's stream
        print("DEBUG: AudioThread - Loop finished, thread ending.")

    def _sd_callback(self, outdata, frames, time_info, status_obj):
        if status_obj:
            if status_obj.output_underflow: print("Warning: AT_CB - Output underflow")

        with self.stream_lock: 
            if not self.is_playing_desired_state or self.audio_thread_data is None:
                outdata[:] = 0
                if not self.is_playing_desired_state and self.audio_thread_data is not None:
                    # print("DEBUG: AT_CB - is_playing_desired_state is False. Raising CallbackStop.")
                    raise sd.CallbackStop 
                return 
            
            current_frame_for_chunk = self.audio_thread_current_frame
            remaining_frames_in_track = self.audio_thread_total_samples - current_frame_for_chunk

            if remaining_frames_in_track <= 0: 
                outdata[:] = 0
                # print("DEBUG: AT_CB - End of track reached.")
                self.master.after(0, self._handle_stream_finished) 
                raise sd.CallbackStop 
            
            valid_frames_to_play = min(frames, remaining_frames_in_track)
            try:
                if current_frame_for_chunk < 0 or \
                   (current_frame_for_chunk + valid_frames_to_play > self.audio_thread_total_samples):
                     outdata[:] = 0; print(f"ERROR: AT_CB - Invalid frame range."); 
                     self.master.after(0, self._handle_stream_finished)
                     raise sd.CallbackStop
                outdata[:valid_frames_to_play] = self.audio_thread_data[current_frame_for_chunk : current_frame_for_chunk + valid_frames_to_play].reshape(-1, 1)
            except Exception as e:
                print(f"ERROR: AT_CB - Slicing error: {e}"); outdata[:] = 0; 
                self.master.after(0, self._handle_stream_finished)
                raise sd.CallbackStop
            
            if valid_frames_to_play < frames: outdata[valid_frames_to_play:] = 0
            
            self.audio_thread_current_frame += valid_frames_to_play
            self.current_playback_frame_shared = self.audio_thread_current_frame


    def _handle_stream_finished(self): 
        print("DEBUG: GUI - _handle_stream_finished called")
        
        was_seek_in_progress = False
        with self.stream_lock:
            was_seek_in_progress = self.seek_in_progress_flag
            self.seek_in_progress_flag = False 

            # Only change desired state to False if it wasn't a seek that intends to continue.
            # If it was a seek, the audio thread's CMD_PLAY will set is_playing_desired_state=True.
            if not was_seek_in_progress:
                print("DEBUG: GUI - _handle_stream_finished: Not a seek, setting desired_state=False.")
                self.is_playing_desired_state = False
            else:
                print("DEBUG: GUI - _handle_stream_finished: Seek was in progress. is_playing_desired_state not changed here.")
            
            # The audio thread is responsible for setting its current_managed_stream to None.
            # This callback just handles GUI side of stream finishing.
        
        if self.master.winfo_exists():
            current_desired_playing = False
            with self.stream_lock: current_desired_playing = self.is_playing_desired_state
            
            if not current_desired_playing: # Update button if truly stopped now
                self.play_pause_button.config(text="Play")
            
            # Stop GUI updates ONLY if we are certain playback is not going to continue immediately (e.g. after seek)
            if self.update_job_id and not current_desired_playing : 
                self.master.after_cancel(self.update_job_id)
                self.update_job_id = None
            
            final_display_frame = 0
            with self.stream_lock: final_display_frame = self.current_playback_frame_shared

            if self.audio_file_loaded:
                if not was_seek_in_progress and final_display_frame >= self.total_duration_samples_gui : 
                    final_display_frame = self.total_duration_samples_gui 
            
            current_time_sec = final_display_frame / float(self.sample_rate_gui) if self.sample_rate_gui > 0 else 0
            current_time_sec = min(current_time_sec, self.total_duration_seconds_gui if self.audio_file_loaded else 0)

            try: 
                if self.master.winfo_exists(): self.seek_slider_var.set(current_time_sec)
            except tk.TclError: pass
            self._update_time_display(current_time_sec, self.total_duration_seconds_gui if self.audio_file_loaded else 0)
            self._update_beat_label(current_time_sec)
        print("DEBUG: GUI - _handle_stream_finished finished")


    def _schedule_display_update(self):
        # print("DEBUG: GUI - _schedule_display_update called") # Can be very verbose
        is_playing_for_update = False
        with self.stream_lock: is_playing_for_update = self.is_playing_desired_state
        
        if self.update_job_id: # Always cancel previous if exists
            if self.master.winfo_exists(): self.master.after_cancel(self.update_job_id)
            self.update_job_id = None

        if is_playing_for_update and self.audio_file_loaded and self.master.winfo_exists():
            self._update_display_elements() # Call once immediately
            self.update_job_id = self.master.after(UPDATE_INTERVAL_MS, self._schedule_display_update)


    def _update_display_elements(self):
        current_frame_display = 0
        is_playing_now = False
        
        with self.stream_lock: 
            if not self.is_playing_desired_state or not self.audio_file_loaded: return
            current_frame_display = self.current_playback_frame_shared
            is_playing_now = self.is_playing_desired_state

        if not is_playing_now: return 
            
        current_time_sec = current_frame_display / float(self.sample_rate_gui) if self.sample_rate_gui > 0 else 0
        current_time_sec = min(current_time_sec, self.total_duration_seconds_gui)

        if not self.master.winfo_exists(): return
        try:
            # Only update slider if the mouse button is not currently pressed on it
            # This is tricky with ttk.Scale. A simpler way is to just always update.
            # If slider jumpiness during drag is an issue, more complex state needed.
            self.seek_slider_var.set(current_time_sec)
            self._update_time_display(current_time_sec, self.total_duration_seconds_gui) 
            self._update_beat_label(current_time_sec) 
        except tk.TclError: pass 

    def _update_time_display(self, current_seconds, total_seconds):
        try:
            if not self.master.winfo_exists(): return
            current_str = time.strftime('%M:%S', time.gmtime(current_seconds))
            total_str = time.strftime('%M:%S', time.gmtime(total_seconds))
            self.time_label.config(text=f"Time: {current_str} / {total_str}")
        except tk.TclError: pass

    def _update_beat_label(self, current_time_seconds): 
        try:
            if not self.master.winfo_exists(): return
            has_beats = False
            if self.audio_file_loaded and isinstance(self.beat_timestamps_gui, np.ndarray) and self.beat_timestamps_gui.size > 0:
                has_beats = True
            
            if has_beats:
                count = np.searchsorted(self.beat_timestamps_gui, current_time_seconds, side='right')
                self.beat_label.config(text=f"Beat: {count}")
            else: 
                self.beat_label.config(text="Beat: --")
        except tk.TclError: pass

    def on_closing(self):
        print("DEBUG: GUI - on_closing called")
        self.audio_thread_stop_event.set() 
        self._send_audio_command(CMD_SHUTDOWN, None, "GUI - Sending SHUTDOWN to audio thread")
        
        print("DEBUG: GUI - Waiting for audio thread to join...")
        self.audio_thread.join(timeout=2.0) 
        if self.audio_thread.is_alive():
            print("WARNING: GUI - Audio thread did not join in time.")
            # If thread is stuck, self.playback_stream_obj might be the audio thread's last stream
            with self.stream_lock:
                stream_to_kill = self.playback_stream_obj 
            if stream_to_kill:
                try:
                    print("WARNING: GUI - Forcing abort/close on lingering stream during unresponsive shutdown.")
                    stream_to_kill.abort(ignore_errors=True)
                    stream_to_kill.close(ignore_errors=True)
                except Exception as e_close:
                    print(f"ERROR: GUI - Exception during forced stream close: {e_close}")
        
        if self.update_job_id: 
            if self.master.winfo_exists(): self.master.after_cancel(self.update_job_id)
        
        if self.master.winfo_exists():
            self.master.destroy()
        print("DEBUG: GUI - on_closing finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Beat Viewer for audio files.")
    parser.add_argument("filepath", type=str, nargs='?', default=None,
                        help="Optional. Path to the audio file to load initially.")
    args = parser.parse_args()

    root = tk.Tk()
    app = BeatViewerApp(root, initial_filepath=args.filepath)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
    print("DEBUG: Mainloop exited.")