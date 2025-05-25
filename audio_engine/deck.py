# dj-gemini/audio_engine/deck.py

import os
import threading
import time 
import queue 
import json # <--- FIX 1: Added json import
import essentia.standard as es
import numpy as np
import sounddevice as sd

# Command constants for the Deck's internal audio thread
DECK_CMD_LOAD_AUDIO = "LOAD_AUDIO"
DECK_CMD_PLAY = "PLAY" 
DECK_CMD_PAUSE = "PAUSE"
DECK_CMD_STOP = "STOP"   
DECK_CMD_SEEK = "SEEK" 
DECK_CMD_SHUTDOWN = "SHUTDOWN"

class Deck:
    def __init__(self, deck_id, analyzer_instance):
        self.deck_id = deck_id
        self.analyzer = analyzer_instance
        print(f"DEBUG: Deck {self.deck_id} - Initializing...")

        self.filepath = None
        self.sample_rate = 0
        self.beat_timestamps = np.array([])
        self.bpm = 0.0
        self.total_frames = 0 
        self.cue_points = {} 
        
        self.audio_thread_data = None
        self.audio_thread_sample_rate = 0
        self.audio_thread_total_samples = 0
        self.audio_thread_current_frame = 0 

        self.command_queue = queue.Queue()
        self.audio_thread_stop_event = threading.Event()
        self.audio_thread = threading.Thread(target=self._audio_management_loop, daemon=True)

        self._stream_lock = threading.Lock() 
        self._current_playback_frame_for_display = 0 
        self._user_wants_to_play = False 
        self._is_actually_playing_stream_state = False 
        self.seek_in_progress_flag = False # <--- FIX 3: Initialize attribute

        self.playback_stream_obj = None 

        self.audio_thread.start()
        print(f"DEBUG: Deck {self.deck_id} - Initialized and audio thread started.")

    def load_track(self, audio_filepath):
        print(f"DEBUG: Deck {self.deck_id} - load_track requested for: {audio_filepath}")
        self.stop() 

        analysis_data = self.analyzer.analyze_track(audio_filepath)
        if not analysis_data:
            print(f"ERROR: Deck {self.deck_id} - Analysis failed for {audio_filepath}")
            self.filepath = None 
            return False

        self.filepath = audio_filepath 
        self.sample_rate = int(analysis_data.get('sample_rate', 0))
        self.beat_timestamps = np.array(analysis_data.get('beat_timestamps', []))
        self.bpm = float(analysis_data.get('bpm', 0.0))
        self.cue_points = analysis_data.get('cue_points', {}) 
        print(f"DEBUG: Deck {self.deck_id} - Loaded cue points: {list(self.cue_points.keys())}")


        if self.sample_rate == 0:
            print(f"ERROR: Deck {self.deck_id} - Invalid sample rate from analysis for {audio_filepath}")
            return False

        try:
            print(f"DEBUG: Deck {self.deck_id} - Loading audio samples with MonoLoader...")
            loader = es.MonoLoader(filename=audio_filepath, sampleRate=self.sample_rate)
            loaded_audio_samples = loader()
            current_total_frames = len(loaded_audio_samples)
            if current_total_frames == 0:
                print(f"ERROR: Deck {self.deck_id} - Loaded audio data is empty for {audio_filepath}")
                return False
            
            self.total_frames = current_total_frames 
            
            self.command_queue.put((DECK_CMD_LOAD_AUDIO, {
                'audio_data': loaded_audio_samples, 
                'sample_rate': self.sample_rate,    
                'total_frames': current_total_frames
            }))
            
        except Exception as e:
            print(f"ERROR: Deck {self.deck_id} - Failed to load audio samples for {audio_filepath}: {e}")
            return False

        with self._stream_lock:
            self._current_playback_frame_for_display = 0 
            self._user_wants_to_play = False 
        print(f"DEBUG: Deck {self.deck_id} - Track '{os.path.basename(audio_filepath)}' data sent to audio thread. BPM: {self.bpm:.2f}")
        return True

    def get_frame_from_beat(self, beat_number):
        if self.sample_rate == 0 or len(self.beat_timestamps) == 0 or self.total_frames == 0: return 0
        if beat_number <= 0: return 0
        
        actual_beat_index = min(beat_number - 1, len(self.beat_timestamps) - 1)
        beat_time = self.beat_timestamps[actual_beat_index]
        frame = int(beat_time * self.sample_rate)
        return max(0, min(frame, self.total_frames - 1))

    def get_frame_from_cue(self, cue_name):
        if not self.cue_points or cue_name not in self.cue_points:
            print(f"WARNING: Deck {self.deck_id} - Cue point '{cue_name}' not found.")
            return None

        cue_info = self.cue_points[cue_name]
        start_beat = cue_info.get("start_beat")

        if start_beat is None:
            print(f"WARNING: Deck {self.deck_id} - Cue point '{cue_name}' has no 'start_beat'.")
            return None
        
        return self.get_frame_from_beat(start_beat)


    def play(self, start_at_beat=None, start_at_cue_name=None):
        print(f"DEBUG: Deck {self.deck_id} - Engine requests PLAY. Cue: '{start_at_cue_name}', Beat: {start_at_beat}")
        
        target_start_frame = None
        operation_description = "resuming/starting from current position"

        if start_at_cue_name:
            frame_from_cue = self.get_frame_from_cue(start_at_cue_name)
            if frame_from_cue is not None:
                target_start_frame = frame_from_cue
                operation_description = f"starting from cue '{start_at_cue_name}' (frame {target_start_frame})"
            else:
                print(f"WARNING: Deck {self.deck_id} - Cue '{start_at_cue_name}' not found or invalid. Checking other options.")
        
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
                current_display_frame = self._current_playback_frame_for_display
                # If total_frames is 0 (no track loaded yet), this prevents error.
                # The audio thread will catch if audio_thread_data is None.
                is_at_end = self.total_frames > 0 and current_display_frame >= self.total_frames
                if is_at_end:
                    final_start_frame_for_command = 0 
                    self._current_playback_frame_for_display = 0
                else:
                    final_start_frame_for_command = current_display_frame
            
            print(f"DEBUG: Deck {self.deck_id} - Finalizing PLAY from frame {final_start_frame_for_command} ({operation_description})")
        
        self.command_queue.put((DECK_CMD_PLAY, {'start_frame': final_start_frame_for_command}))

    def pause(self):
        print(f"DEBUG: Deck {self.deck_id} - Engine requests PAUSE.")
        with self._stream_lock:
            self._user_wants_to_play = False
        self.command_queue.put((DECK_CMD_PAUSE, None))

    def stop(self): 
        print(f"DEBUG: Deck {self.deck_id} - Engine requests STOP.")
        with self._stream_lock:
            self._user_wants_to_play = False
            self._current_playback_frame_for_display = 0 
        self.command_queue.put((DECK_CMD_STOP, None))

    def seek(self, target_frame): 
        print(f"DEBUG: Deck {self.deck_id} - Engine requests SEEK to frame: {target_frame}")
        was_playing_intent = False
        # Ensure target_frame is valid before setting display frame
        valid_target_frame = max(0, min(target_frame, self.total_frames -1 if self.total_frames > 0 else 0))

        with self._stream_lock:
            was_playing_intent = self._user_wants_to_play 
            self._current_playback_frame_for_display = valid_target_frame
            if was_playing_intent: # If seeking while intending to play, set the flag
                self.seek_in_progress_flag = True
        self.command_queue.put((DECK_CMD_SEEK, {'frame': valid_target_frame, 
                                               'was_playing_intent': was_playing_intent}))

    # --- Audio Management Thread ---
    def _audio_management_loop(self):
        print(f"DEBUG: Deck {self.deck_id} AudioThread - Started")
        
        _current_stream_in_thread = None 
        # Instance variables self.audio_thread_data etc. are used by _sd_callback

        while not self.audio_thread_stop_event.is_set():
            try:
                command, data = self.command_queue.get(timeout=0.1)
                print(f"DEBUG: Deck {self.deck_id} AudioThread - Received command: {command}")

                if command in [DECK_CMD_LOAD_AUDIO, DECK_CMD_PLAY, DECK_CMD_SEEK, DECK_CMD_STOP, DECK_CMD_SHUTDOWN]:
                    if _current_stream_in_thread:
                        print(f"DEBUG: Deck {self.deck_id} AudioThread - Command {command} clearing existing stream.")
                        _current_stream_in_thread.abort(ignore_errors=True)
                        _current_stream_in_thread.close(ignore_errors=True)
                        _current_stream_in_thread = None
                        with self._stream_lock: self._is_actually_playing_stream_state = False
                
                if command == DECK_CMD_LOAD_AUDIO:
                    print(f"DEBUG: Deck {self.deck_id} AudioThread - Processing LOAD_AUDIO")
                    with self._stream_lock: 
                        self.audio_thread_data = data['audio_data']
                        self.audio_thread_sample_rate = data['sample_rate']
                        self.audio_thread_total_samples = data['total_frames']
                        self.audio_thread_current_frame = 0 
                        self._current_playback_frame_for_display = 0
                    print(f"DEBUG: Deck {self.deck_id} AudioThread - Audio data set internally.")

                elif command == DECK_CMD_PLAY:
                    print(f"DEBUG: Deck {self.deck_id} AudioThread - Processing PLAY")
                    with self._stream_lock: 
                        if self.audio_thread_data is None: 
                            print(f"DEBUG: Deck {self.deck_id} AudioThread - No audio data for PLAY.")
                            self._user_wants_to_play = False 
                            continue 

                        self._user_wants_to_play = True 
                        self.audio_thread_current_frame = data.get('start_frame', self._current_playback_frame_for_display)
                        
                        if self.audio_thread_current_frame >= self.audio_thread_total_samples: 
                            self.audio_thread_current_frame = 0
                        
                        self._current_playback_frame_for_display = self.audio_thread_current_frame

                        print(f"DEBUG: Deck {self.deck_id} AudioThread - Creating new stream. SR: {self.audio_thread_sample_rate}, Frame: {self.audio_thread_current_frame}")
                        _current_stream_in_thread = sd.OutputStream(
                            samplerate=self.audio_thread_sample_rate, channels=1,
                            callback=self._sd_callback, 
                            finished_callback=self._on_stream_finished_from_audio_thread 
                        )
                        self.playback_stream_obj = _current_stream_in_thread 
                        _current_stream_in_thread.start()
                        self._is_actually_playing_stream_state = True
                        print(f"DEBUG: Deck {self.deck_id} AudioThread - Stream started for PLAY.")
                    
                elif command == DECK_CMD_PAUSE:
                    print(f"DEBUG: Deck {self.deck_id} AudioThread - Processing PAUSE")
                    # _user_wants_to_play is already False
                    if _current_stream_in_thread and _current_stream_in_thread.active:
                        print(f"DEBUG: Deck {self.deck_id} AudioThread - Calling stream.stop() for PAUSE.")
                        _current_stream_in_thread.stop(ignore_errors=True) 
                        print(f"DEBUG: Deck {self.deck_id} AudioThread - stream.stop() called for PAUSE.")
                    else: 
                         with self._stream_lock: self._is_actually_playing_stream_state = False

                elif command == DECK_CMD_STOP:
                    print(f"DEBUG: Deck {self.deck_id} AudioThread - Processing STOP")
                    # Stream already stopped/closed by pre-command cleanup
                    with self._stream_lock: 
                        self.audio_thread_current_frame = 0 
                        self._current_playback_frame_for_display = 0 
                        self._is_actually_playing_stream_state = False
                    print(f"DEBUG: Deck {self.deck_id} AudioThread - State reset for STOP.")

                elif command == DECK_CMD_SEEK:
                    new_frame = data['frame']
                    was_playing_intent = data['was_playing_intent']
                    print(f"DEBUG: Deck {self.deck_id} AudioThread - Processing SEEK to frame {new_frame}, was_playing_intent: {was_playing_intent}")

                    # Stream already aborted/closed by pre-command cleanup
                    with self._stream_lock:
                        self.audio_thread_current_frame = new_frame
                        self._current_playback_frame_for_display = new_frame 
                        self._user_wants_to_play = was_playing_intent 
                        # seek_in_progress_flag is managed by GUI thread, audio thread doesn't need it directly

                    if was_playing_intent: 
                        print(f"DEBUG: Deck {self.deck_id} AudioThread - SEEK: Re-queueing PLAY command.")
                        self.command_queue.put((DECK_CMD_PLAY, {'start_frame': new_frame}))
                    else: 
                        with self._stream_lock: self._is_actually_playing_stream_state = False
                        print(f"DEBUG: Deck {self.deck_id} AudioThread - SEEK: Was paused, position updated.")

                elif command == DECK_CMD_SHUTDOWN:
                    print(f"DEBUG: Deck {self.deck_id} AudioThread - Processing SHUTDOWN")
                    break 

                self.command_queue.task_done() 
            except queue.Empty:
                with self._stream_lock:
                    stream_obj_ref = self.playback_stream_obj 
                    user_wants_play_check = self._user_wants_to_play
                    is_active_check = self._is_actually_playing_stream_state if stream_obj_ref else False 

                if stream_obj_ref and not is_active_check and not user_wants_play_check:
                    if not stream_obj_ref.closed:
                        print(f"DEBUG: Deck {self.deck_id} AudioThread - Inactive stream cleanup (timeout).")
                        stream_obj_ref.close(ignore_errors=True)
                        with self._stream_lock: 
                            if self.playback_stream_obj == stream_obj_ref: 
                                self.playback_stream_obj = None
                # Update local _current_stream_in_thread if instance one changed
                with self._stream_lock: _current_stream_in_thread = self.playback_stream_obj
                continue
            except Exception as e_loop:
                print(f"ERROR in Deck {self.deck_id} _audio_management_loop: {e_loop}")
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
        print(f"DEBUG: Deck {self.deck_id} AudioThread - Loop finished, thread ending.")

    def _sd_callback(self, outdata, frames, time_info, status_obj):
        if status_obj:
            if status_obj.output_underflow: print(f"Warning: Deck {self.deck_id} CB - Output underflow")

        with self._stream_lock: 
            if not self._user_wants_to_play or self.audio_thread_data is None:
                outdata[:] = 0
                if not self._user_wants_to_play and self.audio_thread_data is not None:
                    raise sd.CallbackStop 
                return 
            
            current_frame_for_chunk = self.audio_thread_current_frame
            remaining_frames_in_track = self.audio_thread_total_samples - current_frame_for_chunk

            if remaining_frames_in_track <= 0: 
                outdata[:] = 0
                raise sd.CallbackStop 
            
            valid_frames_to_play = min(frames, remaining_frames_in_track)
            try:
                if current_frame_for_chunk < 0 or \
                   (current_frame_for_chunk + valid_frames_to_play > self.audio_thread_total_samples):
                     outdata[:] = 0; print(f"ERROR: Deck {self.deck_id} CB - Invalid frame range.")
                     raise sd.CallbackStop
                
                chunk = self.audio_thread_data[current_frame_for_chunk : current_frame_for_chunk + valid_frames_to_play]
                outdata[:valid_frames_to_play] = chunk.reshape(-1,1)

            except Exception as e:
                print(f"ERROR: Deck {self.deck_id} CB - Slicing error: {e}"); outdata[:] = 0; 
                raise sd.CallbackStop
            
            if valid_frames_to_play < frames: outdata[valid_frames_to_play:] = 0
            
            self.audio_thread_current_frame += valid_frames_to_play
            self._current_playback_frame_for_display = self.audio_thread_current_frame


    def _on_stream_finished_from_audio_thread(self):
        print(f"DEBUG: Deck {self.deck_id} AudioThread - Stream finished_callback triggered.")
        
        was_seek_in_progress_flag = False # Local copy of the flag
        with self._stream_lock:
            self._is_actually_playing_stream_state = False 
            # The audio thread's _current_stream_in_thread will be set to None by its own logic
            # or if this finished callback corresponds to the stream it was managing.
            # We nullify the main instance playback_stream_obj to reflect the stream is done.
            self.playback_stream_obj = None 

            was_seek_in_progress_flag = self.seek_in_progress_flag
            self.seek_in_progress_flag = False # Reset flag

            # Only set user_wants_to_play to False if this wasn't a stop due to an ongoing seek
            # that intends to immediately restart playback.
            if not was_seek_in_progress_flag:
                print(f"DEBUG: Deck {self.deck_id} finished_callback: Not a seek, setting user_wants_to_play=False.")
                self._user_wants_to_play = False
            else:
                print(f"DEBUG: Deck {self.deck_id} finished_callback: Seek was in progress, user_wants_to_play not changed here by finished_callback.")
            
            # If it was a natural end of track (and not a seek that will restart)
            if not was_seek_in_progress_flag and \
               self.audio_thread_data is not None and \
               self.audio_thread_current_frame >= self.audio_thread_total_samples:
                print(f"DEBUG: Deck {self.deck_id} finished_callback - Track ended naturally. Resetting frames.")
                self.audio_thread_current_frame = 0 
                self._current_playback_frame_for_display = 0
        
        print(f"DEBUG: Deck {self.deck_id} AudioThread - Stream finished_callback processed. User wants play: {self._user_wants_to_play}")


    def is_active(self): 
        with self._stream_lock:
            return self._is_actually_playing_stream_state
                   
    def get_current_display_frame(self):
        with self._stream_lock:
            return self._current_playback_frame_for_display

    def shutdown(self): 
        print(f"DEBUG: Deck {self.deck_id} - Shutdown requested from external.")
        self.audio_thread_stop_event.set()
        try:
            self.command_queue.put((DECK_CMD_SHUTDOWN, None), timeout=0.1) 
        except queue.Full:
            print(f"WARNING: Deck {self.deck_id} - Command queue full during shutdown send.")

        print(f"DEBUG: Deck {self.deck_id} - Waiting for audio thread to join...")
        self.audio_thread.join(timeout=2.0) 
        if self.audio_thread.is_alive():
            print(f"WARNING: Deck {self.deck_id} - Audio thread did not join cleanly during shutdown.")
            with self._stream_lock: stream = self.playback_stream_obj 
            if stream: # This is self.playback_stream_obj which audio thread updates
                try: 
                    print(f"WARNING: Deck {self.deck_id} - Forcing abort/close on lingering stream during unresponsive shutdown.")
                    stream.abort(ignore_errors=True)
                    stream.close(ignore_errors=True)
                except Exception as e_close:
                    print(f"ERROR: Deck {self.deck_id} - Exception during forced stream close: {e_close}")
        print(f"DEBUG: Deck {self.deck_id} - Shutdown complete.")


# ... (rest of the Deck class code above) ...

if __name__ == '__main__':
    print("--- Deck Class Standalone Test ---")
    
    import sys
    CURRENT_DIR_OF_DECK_PY = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT_FOR_DECK_TEST = os.path.dirname(CURRENT_DIR_OF_DECK_PY) 
    
    if PROJECT_ROOT_FOR_DECK_TEST not in sys.path:
        sys.path.append(PROJECT_ROOT_FOR_DECK_TEST)
    
    try:
        import config as app_config 
        from audio_engine.audio_analyzer import AudioAnalyzer 
    except ImportError as e:
        print(f"ERROR: Could not import config or AudioAnalyzer for test: {e}")
        sys.exit(1)

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

    # --- Create a dummy .cue file for starships.mp3 for this Deck test ---
    # This allows testing Deck's cue functionality without relying on AudioAnalyzer's test to run first.
    dummy_cue_filepath_for_deck_test = test_audio_file + ".cue" 
    dummy_cue_data_for_deck_test = {
        "intro_start": {"start_beat": 1},
        "drop1": {"start_beat": 65, "comment": "First drop"},
        "main_break_loop": {"start_beat": 97, "end_beat": 105}
    }
    try:
        with open(dummy_cue_filepath_for_deck_test, 'w') as f:
            json.dump(dummy_cue_data_for_deck_test, f, indent=4)
        print(f"DEBUG: Deck Test - Created/Replaced dummy cue file: {dummy_cue_filepath_for_deck_test}")
    except Exception as e_cue_write:
        print(f"ERROR: Deck Test - Could not create dummy cue file {dummy_cue_filepath_for_deck_test}: {e_cue_write}")

    
    if not os.path.exists(test_audio_file):
        print(f"WARNING: Test audio file not found at {test_audio_file}. Test will be limited.")
    else:
        if deck.load_track(test_audio_file): # This will use the dummy cue file just created
            print(f"\nTrack loaded successfully onto {deck.deck_id}")
            
            print("\nPlaying from CUE 'drop1' for 3 seconds...")
            deck.play(start_at_cue_name="drop1")
            time.sleep(3.1) 
            print(f"After 3s, DisplayFrame: {deck.get_current_display_frame()}, UserWantsPlay: {deck._user_wants_to_play}, StreamActive: {deck.is_active()}")

            print("\nPausing for 2 seconds...")
            deck.pause() 
            time.sleep(0.5) 
            print(f"After pause, DisplayFrame: {deck.get_current_display_frame()}, UserWantsPlay: {deck._user_wants_to_play}, StreamActive: {deck.is_active()}")

            print("\nSeeking to beat 10 (while paused)...")
            seek_frame_10 = deck.get_frame_from_beat(10)
            deck.seek(seek_frame_10)
            time.sleep(0.2) 
            print(f"After seek to beat 10 (paused), DisplayFrame: {deck.get_current_display_frame()}, UserWantsPlay: {deck._user_wants_to_play}, StreamActive: {deck.is_active()}")


            print("\nResuming play (from beat 10) for 3 seconds...")
            deck.play() 
            time.sleep(3.1)
            print(f"After resume, DisplayFrame: {deck.get_current_display_frame()}, UserWantsPlay: {deck._user_wants_to_play}, StreamActive: {deck.is_active()}")

            print("\nSeeking to CUE 'main_break_loop' while playing for 2 seconds...")
            seek_frame_loop = deck.get_frame_from_cue("main_break_loop") 
            if seek_frame_loop is not None:
                deck.seek(seek_frame_loop) 
                time.sleep(2.1) 
                print(f"After seek to cue 'main_break_loop' (playing), DisplayFrame: {deck.get_current_display_frame()}, UserWantsPlay: {deck._user_wants_to_play}, StreamActive: {deck.is_active()}")
            else:
                print("ERROR: Test - Could not find cue 'main_break_loop' to seek to.")


            print("\nStopping playback...")
            deck.stop()
            time.sleep(0.5) 
            print(f"After stop, DisplayFrame: {deck.get_current_display_frame()}, UserWantsPlay: {deck._user_wants_to_play}, StreamActive: {deck.is_active()}")
            
            print("\nTest finished.")
        else:
            print(f"Failed to load track onto {deck.deck_id}")
    
    deck.shutdown() 
    
    # REMOVED/COMMENTED OUT the .cue file deletion for persistence across tests
    # if os.path.exists(dummy_cue_filepath_for_deck_test):
    #     try:
    #         os.remove(dummy_cue_filepath_for_deck_test)
    #         print(f"DEBUG: Test - Removed dummy cue file: {dummy_cue_filepath_for_deck_test}")
    #     except Exception as e_remove:
    #         print(f"ERROR: Test - Could not remove dummy_cue_file: {e_remove}")
            
    print("--- Deck Test Complete ---")