# dj-gemini/audio_engine/deck.py

import os
import threading
import time 
import queue 
import json 
import essentia.standard as es
import numpy as np
import sounddevice as sd

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
        
        # Data for Audio Thread's direct use by its callback
        self.audio_thread_data = None
        self.audio_thread_sample_rate = 0
        self.audio_thread_total_samples = 0
        self.audio_thread_current_frame = 0 # Frame counter for audio thread's playback logic

        self.command_queue = queue.Queue()
        self.audio_thread_stop_event = threading.Event()
        self.audio_thread = threading.Thread(target=self._audio_management_loop, daemon=True)

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

        # Add loop queue for handling multiple loops
        self._loop_queue = []

        self.playback_stream_obj = None 

        self.audio_thread.start()
        print(f"DEBUG: Deck {self.deck_id} - Initialized and audio thread started.")

    def load_track(self, audio_filepath):
        print(f"DEBUG: Deck {self.deck_id} - load_track requested for: {audio_filepath}")
        self.stop() # Send CMD_STOP to ensure any previous playback is fully handled

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
        if self.bpm <= 0: 
            print(f"WARNING: Deck {self.deck_id} - BPM is {self.bpm}. Beat-length loops require a positive BPM.")

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
            self._loop_active = False 
        print(f"DEBUG: Deck {self.deck_id} - Track '{os.path.basename(audio_filepath)}' data sent to audio thread. BPM: {self.bpm:.2f}")
        return True

    def get_frame_from_beat(self, beat_number):
        if self.sample_rate == 0 or len(self.beat_timestamps) == 0 or self.total_frames == 0: return 0
        try:
            beat_number_int = int(round(float(beat_number)))
        except (ValueError, TypeError):
            print(f"WARNING: Deck {self.deck_id} - Invalid beat_number type for get_frame_from_beat: {beat_number}")
            return 0
        if beat_number_int <= 0: return 0 
        
        actual_beat_index = min(beat_number_int - 1, len(self.beat_timestamps) - 1)
        if actual_beat_index < 0 : actual_beat_index = 0 
            
        beat_time = self.beat_timestamps[actual_beat_index]
        frame = int(beat_time * self.sample_rate)
        return max(0, min(frame, self.total_frames - 1 if self.total_frames > 0 else 0))

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

    def get_current_beat_count(self):
        with self._stream_lock:
            if self.audio_thread_data is None or self.sample_rate == 0 or len(self.beat_timestamps) == 0:
                return 0 
            current_time_seconds = self._current_playback_frame_for_display / float(self.sample_rate)
            beat_count = np.searchsorted(self.beat_timestamps, current_time_seconds, side='right')
            return beat_count

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
                print(f"WARNING: Deck {self.deck_id} - Cue '{start_at_cue_name}' not found/invalid. Checking other options.")
        
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
            self._loop_active = False 
        self.command_queue.put((DECK_CMD_STOP, None))

    def seek(self, target_frame): 
        print(f"DEBUG: Deck {self.deck_id} - Engine requests SEEK to frame: {target_frame}")
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

    def activate_loop(self, start_beat, length_beats, repetitions=None):
        print(f"DEBUG: Deck {self.deck_id} - Engine requests ACTIVATE_LOOP. Start Beat: {start_beat}, Length: {length_beats} beats, Reps: {repetitions}")
        if self.total_frames == 0: 
            print(f"ERROR: Deck {self.deck_id} - Cannot activate loop: track not loaded or has no length.")
            return
        if self.bpm <= 0:
            print(f"ERROR: Deck {self.deck_id} - Invalid BPM ({self.bpm}) for loop calculation.")
            return
        if self.sample_rate <= 0:
            print(f"ERROR: Deck {self.deck_id} - Invalid sample rate ({self.sample_rate}) for loop calculation.")
            return

        loop_start_frame = self.get_frame_from_beat(start_beat)
        frames_per_beat = (60.0 / self.bpm) * self.sample_rate
        loop_length_frames = int(length_beats * frames_per_beat)
        loop_end_frame = loop_start_frame + loop_length_frames
        
        loop_end_frame = min(loop_end_frame, self.total_frames) 
        if loop_start_frame >= loop_end_frame : 
            print(f"ERROR: Deck {self.deck_id} - Invalid loop parameters (start_frame {loop_start_frame} >= end_frame {loop_end_frame}).")
            return

        print(f"DEBUG: Deck {self.deck_id} - Calculated loop frames: [{loop_start_frame} - {loop_end_frame}]")
        self.command_queue.put((DECK_CMD_ACTIVATE_LOOP, {
            'start_frame': loop_start_frame,
            'end_frame': loop_end_frame,
            'repetitions': repetitions 
        }))

    def deactivate_loop(self):
        print(f"DEBUG: Deck {self.deck_id} - Engine requests DEACTIVATE_LOOP.")
        self.command_queue.put((DECK_CMD_DEACTIVATE_LOOP, None))

    def stop_at_beat(self, beat_number):
        """Stop playback when reaching a specific beat"""
        print(f"DEBUG: Deck {self.deck_id} - Engine requests STOP_AT_BEAT. Beat: {beat_number}")
        if self.total_frames == 0:
            print(f"ERROR: Deck {self.deck_id} - Cannot stop at beat: track not loaded.")
            return
        if self.bpm <= 0:
            print(f"ERROR: Deck {self.deck_id} - Invalid BPM ({self.bpm}) for beat calculation.")
            return
        
        target_frame = self.get_frame_from_beat(beat_number)
        if target_frame >= self.total_frames:
            print(f"WARNING: Deck {self.deck_id} - Beat {beat_number} is beyond track length. Stopping immediately.")
            self.stop()
            return
        
        print(f"DEBUG: Deck {self.deck_id} - Scheduling stop at beat {beat_number} (frame {target_frame})")
        self.command_queue.put((DECK_CMD_STOP_AT_BEAT, {
            'target_frame': target_frame,
            'beat_number': beat_number
        }))

    # --- Audio Management Thread ---
    def _audio_management_loop(self):
        print(f"DEBUG: Deck {self.deck_id} AudioThread - Started")
        _current_stream_in_thread = None 
        # Instance variables self.audio_thread_... are used by _sd_callback

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
                        self._loop_active = False 
                        self._loop_repetitions_total = None
                        self._loop_repetitions_done = 0
                    print(f"DEBUG: Deck {self.deck_id} AudioThread - Audio data set internally for playback.")

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

                        # Reset loop parameters whenever a new PLAY command (not just resume) is initiated
                        # This ensures a seek-via-play or play-from-cue clears old loop state.
                        self._loop_active = False
                        self._loop_start_frame = 0
                        self._loop_end_frame = 0
                        self._loop_repetitions_total = None
                        self._loop_repetitions_done = 0
                        self._loop_queue.clear()  # Clear any pending loops
                        print(f"DEBUG: Deck {self.deck_id} AudioThread - Loop state reset for PLAY.")

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
                    if _current_stream_in_thread and _current_stream_in_thread.active:
                        print(f"DEBUG: Deck {self.deck_id} AudioThread - Calling stream.stop() for PAUSE.")
                        _current_stream_in_thread.stop(ignore_errors=True) 
                        print(f"DEBUG: Deck {self.deck_id} AudioThread - stream.stop() called for PAUSE.")
                    else: 
                         with self._stream_lock: self._is_actually_playing_stream_state = False

                elif command == DECK_CMD_STOP:
                    print(f"DEBUG: Deck {self.deck_id} AudioThread - Processing STOP")
                    with self._stream_lock: 
                        self.audio_thread_current_frame = 0 
                        self._current_playback_frame_for_display = 0 
                        self._is_actually_playing_stream_state = False
                        self._loop_active = False 
                        self._loop_repetitions_total = None
                        self._loop_repetitions_done = 0
                        self._loop_queue.clear()  # Clear any pending loops
                    print(f"DEBUG: Deck {self.deck_id} AudioThread - State reset for STOP.")

                elif command == DECK_CMD_SEEK:
                    new_frame = data['frame']
                    was_playing_intent = data['was_playing_intent']
                    print(f"DEBUG: Deck {self.deck_id} AudioThread - Processing SEEK to frame {new_frame}, was_playing_intent: {was_playing_intent}")
                    with self._stream_lock:
                        self.audio_thread_current_frame = new_frame
                        self._current_playback_frame_for_display = new_frame 
                        self._user_wants_to_play = was_playing_intent 
                        self._loop_active = False 
                        self._loop_repetitions_total = None
                        self._loop_repetitions_done = 0
                        self._loop_queue.clear()  # Clear any pending loops
                    if was_playing_intent: 
                        print(f"DEBUG: Deck {self.deck_id} AudioThread - SEEK: Re-queueing PLAY command.")
                        self.command_queue.put((DECK_CMD_PLAY, {'start_frame': new_frame}))
                    else: 
                        with self._stream_lock: self._is_actually_playing_stream_state = False
                        print(f"DEBUG: Deck {self.deck_id} AudioThread - SEEK: Was paused, position updated.")
                
                elif command == DECK_CMD_ACTIVATE_LOOP:
                    print(f"DEBUG: Deck {self.deck_id} AudioThread - Processing ACTIVATE_LOOP")
                    
                    # Add to queue if there's already an active loop
                    with self._stream_lock:
                        if self._loop_active:
                            print(f"DEBUG: Deck {self.deck_id} AudioThread - Loop already active, queuing new loop")
                            self._loop_queue.append({
                                'start_frame': data['start_frame'],
                                'end_frame': data['end_frame'],
                                'repetitions': data.get('repetitions')
                            })
                        else:
                            # Activate immediately if no loop is active
                            self._loop_start_frame = data['start_frame']
                            self._loop_end_frame = data['end_frame']
                            self._loop_repetitions_total = data.get('repetitions') 
                            self._loop_repetitions_done = 0
                            self._loop_active = True
                            
                            # If currently playing, check if playhead needs to snap to loop_start_frame
                            if self._is_actually_playing_stream_state and self.audio_thread_data is not None:
                                if self.audio_thread_current_frame >= self._loop_end_frame or \
                                   self.audio_thread_current_frame < self._loop_start_frame:
                                    print(f"DEBUG: Deck {self.deck_id} AudioThread - Loop activated, playhead at {self.audio_thread_current_frame} was outside new loop [{self._loop_start_frame}-{self._loop_end_frame}]. Snapping to loop start.")
                                    self.audio_thread_current_frame = self._loop_start_frame
                                    self._current_playback_frame_for_display = self._loop_start_frame
                            print(f"DEBUG: Deck {self.deck_id} AudioThread - Loop activated: Frames [{self._loop_start_frame} - {self._loop_end_frame}], Reps: {self._loop_repetitions_total}")

                elif command == DECK_CMD_DEACTIVATE_LOOP:
                    print(f"DEBUG: Deck {self.deck_id} AudioThread - Processing DEACTIVATE_LOOP")
                    with self._stream_lock:
                        self._loop_active = False
                        self._loop_queue.clear()  # Clear any pending loops
                        print(f"DEBUG: Deck {self.deck_id} AudioThread - Loop deactivated and queue cleared.")

                elif command == DECK_CMD_STOP_AT_BEAT:
                    print(f"DEBUG: Deck {self.deck_id} AudioThread - Processing STOP_AT_BEAT")
                    target_frame = data['target_frame']
                    beat_number = data['beat_number']
                    
                    with self._stream_lock:
                        # Store the stop target for the callback to check
                        self._stop_at_frame = target_frame
                        self._stop_at_beat = beat_number
                        print(f"DEBUG: Deck {self.deck_id} AudioThread - Will stop at frame {target_frame} (beat {beat_number})")

                elif command == DECK_CMD_SHUTDOWN:
                    print(f"DEBUG: Deck {self.deck_id} AudioThread - Processing SHUTDOWN")
                    break 

                self.command_queue.task_done() 
            except queue.Empty:
                with self._stream_lock:
                    stream_obj_ref = self.playback_stream_obj 
                    user_wants_play_check = self._user_wants_to_play
                    is_active_check = self._is_actually_playing_stream_state if stream_obj_ref and hasattr(stream_obj_ref, 'active') else False # Guard access

                if stream_obj_ref and not is_active_check and not user_wants_play_check:
                    if not stream_obj_ref.closed:
                        print(f"DEBUG: Deck {self.deck_id} AudioThread - Inactive stream cleanup (timeout).")
                        stream_obj_ref.close(ignore_errors=True)
                        with self._stream_lock: 
                            if self.playback_stream_obj == stream_obj_ref: 
                                self.playback_stream_obj = None
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
        # Added more verbose debug prints here
        # print(f"DEBUG: Deck {self.deck_id} CB - ENTERING. Frames req: {frames}")
        
        if status_obj:
            if status_obj.output_underflow: print(f"Warning: Deck {self.deck_id} CB - Output underflow")
            if status_obj.output_overflow: print(f"Warning: Deck {self.deck_id} CB - Output overflow")

        with self._stream_lock: 
            if not self._user_wants_to_play or self.audio_thread_data is None:
                outdata[:] = 0
                # print(f"DEBUG: Deck {self.deck_id} CB - Condition not met (wants_play: {self._user_wants_to_play}, data: {self.audio_thread_data is not None}). Stopping.")
                if not self._user_wants_to_play and self.audio_thread_data is not None:
                    raise sd.CallbackStop 
                return 
            
            # Check for stop_at_beat BEFORE processing audio
            if hasattr(self, '_stop_at_frame') and self._stop_at_frame is not None:
                if self.audio_thread_current_frame >= self._stop_at_frame:
                    print(f"DEBUG: Deck {self.deck_id} CB - Reached stop frame {self._stop_at_frame} (beat {getattr(self, '_stop_at_beat', 'unknown')}). Stopping cleanly.")
                    self._stop_at_frame = None
                    self._stop_at_beat = None
                    self._user_wants_to_play = False
                    outdata[:] = 0  # Fill with silence
                    raise sd.CallbackStop
            
            if self._loop_active and self.audio_thread_current_frame >= self._loop_end_frame:
                self._loop_repetitions_done += 1
                print(f"DEBUG: Deck {self.deck_id} CB - Loop cycle end. Frame: {self.audio_thread_current_frame}, LoopEnd: {self._loop_end_frame}, RepsDone: {self._loop_repetitions_done}, RepsTotal: {self._loop_repetitions_total}")
                
                should_continue_looping = True
                if self._loop_repetitions_total is not None: 
                    if self._loop_repetitions_done >= self._loop_repetitions_total:
                        should_continue_looping = False
                        self._loop_active = False 
                        print(f"DEBUG: Deck {self.deck_id} CB - Loop finished {self._loop_repetitions_total} reps. Deactivating.")
                        
                        # Check if there are more loops in the queue
                        if self._loop_queue:
                            next_loop = self._loop_queue.pop(0)
                            print(f"DEBUG: Deck {self.deck_id} CB - Activating next loop from queue: {next_loop}")
                            self._loop_start_frame = next_loop['start_frame']
                            self._loop_end_frame = next_loop['end_frame']
                            self._loop_repetitions_total = next_loop.get('repetitions')
                            self._loop_repetitions_done = 0
                            self._loop_active = True
                            
                            # Snap to new loop start if needed
                            if self.audio_thread_current_frame >= self._loop_end_frame or \
                               self.audio_thread_current_frame < self._loop_start_frame:
                                self.audio_thread_current_frame = self._loop_start_frame
                                self._current_playback_frame_for_display = self._loop_start_frame
                                print(f"DEBUG: Deck {self.deck_id} CB - Snapped to new loop start: {self._loop_start_frame}")
                        else:
                            print(f"DEBUG: Deck {self.deck_id} CB - No more loops in queue, continuing normal playback.")
                
                if should_continue_looping:
                    self.audio_thread_current_frame = self._loop_start_frame
                    print(f"DEBUG: Deck {self.deck_id} CB - Jumping to loop_start_frame: {self._loop_start_frame}")
            
            current_frame_for_chunk = self.audio_thread_current_frame
            remaining_frames_in_track = self.audio_thread_total_samples - current_frame_for_chunk

            if remaining_frames_in_track <= 0: 
                outdata[:] = 0
                # print(f"DEBUG: Deck {self.deck_id} CB - End of track based on remaining_frames <= 0.")
                raise sd.CallbackStop 
            
            valid_frames_to_play = min(frames, remaining_frames_in_track)
            
            if self._loop_active: 
                frames_to_loop_end = self._loop_end_frame - current_frame_for_chunk
                if frames_to_loop_end < 0 : # Should not happen if jump logic is correct
                    print(f"WARNING: Deck {self.deck_id} CB - In active loop but current_frame ({current_frame_for_chunk}) is past loop_end ({self._loop_end_frame}). Deactivating loop.")
                    self._loop_active = False # Safety deactivation
                    frames_to_loop_end = valid_frames_to_play # Play normally
                
                valid_frames_to_play = min(valid_frames_to_play, frames_to_loop_end)
                
                if valid_frames_to_play <= 0 and current_frame_for_chunk < self._loop_end_frame: # At loop end, will jump next cycle
                     valid_frames_to_play = 0 

            try:
                if valid_frames_to_play > 0:
                    if current_frame_for_chunk < 0 or \
                       (current_frame_for_chunk + valid_frames_to_play > self.audio_thread_total_samples):
                         outdata[:] = 0; print(f"ERROR: Deck {self.deck_id} CB - Invalid frame range before slice."); 
                         raise sd.CallbackStop
                    
                    chunk = self.audio_thread_data[current_frame_for_chunk : current_frame_for_chunk + valid_frames_to_play]
                    outdata[:valid_frames_to_play] = chunk.reshape(-1,1)
                else: 
                    outdata[:] = 0

            except Exception as e:
                print(f"ERROR: Deck {self.deck_id} CB - Slicing error: {e}"); outdata[:] = 0; 
                raise sd.CallbackStop
            
            if valid_frames_to_play < frames: outdata[valid_frames_to_play:] = 0
            
            if valid_frames_to_play > 0 : 
                self.audio_thread_current_frame += valid_frames_to_play
            self._current_playback_frame_for_display = self.audio_thread_current_frame

    def _on_stream_finished_from_audio_thread(self):
        print(f"DEBUG: Deck {self.deck_id} AudioThread - Stream finished_callback triggered.")
        was_seek = False 
        with self._stream_lock:
            self._is_actually_playing_stream_state = False 
            # It's crucial that the audio thread's local _current_stream_in_thread is set to None
            # when this callback is for *that* stream. We also update self.playback_stream_obj.
            if self.playback_stream_obj and (self.playback_stream_obj.stopped or self.playback_stream_obj.closed):
                print(f"DEBUG: Deck {self.deck_id} finished_callback - Clearing self.playback_stream_obj.")
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
                print(f"DEBUG: Deck {self.deck_id} finished_callback - Track ended naturally. Resetting frames.")
                self.audio_thread_current_frame = 0 
                self._current_playback_frame_for_display = 0
        print(f"DEBUG: Deck {self.deck_id} AudioThread - Stream finished_callback processed. User wants play: {self._user_wants_to_play}")

    def is_active(self): 
        with self._stream_lock: return self._is_actually_playing_stream_state
                   
    def get_current_display_frame(self):
        with self._stream_lock: return self._current_playback_frame_for_display

    def shutdown(self): 
        print(f"DEBUG: Deck {self.deck_id} - Shutdown requested from external.")
        self.audio_thread_stop_event.set()
        try: self.command_queue.put((DECK_CMD_SHUTDOWN, None), timeout=0.1) 
        except queue.Full: print(f"WARNING: Deck {self.deck_id} - Command queue full during shutdown send.")
        print(f"DEBUG: Deck {self.deck_id} - Waiting for audio thread to join...")
        self.audio_thread.join(timeout=2.0) 
        if self.audio_thread.is_alive():
            print(f"WARNING: Deck {self.deck_id} - Audio thread did not join cleanly.")
            with self._stream_lock: stream = self.playback_stream_obj 
            if stream:
                try: 
                    print(f"WARNING: Deck {self.deck_id} - Forcing abort/close on lingering stream.")
                    stream.abort(ignore_errors=True); stream.close(ignore_errors=True)
                except Exception as e_close: print(f"ERROR: Deck {self.deck_id} - Exception during forced stream close: {e_close}")
        print(f"DEBUG: Deck {self.deck_id} - Shutdown complete.")


if __name__ == '__main__':
    # (Test block remains the same as the version that worked for you)
    print("--- Deck Class Standalone Test (with Looping and Cue Points) ---")
    import sys
    CURRENT_DIR_OF_DECK_PY = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT_FOR_DECK_TEST = os.path.dirname(CURRENT_DIR_OF_DECK_PY) 
    if PROJECT_ROOT_FOR_DECK_TEST not in sys.path: sys.path.append(PROJECT_ROOT_FOR_DECK_TEST)
    try:
        import config as app_config 
        from audio_engine.audio_analyzer import AudioAnalyzer 
    except ImportError as e: print(f"ERROR: Could not import for test: {e}"); sys.exit(1)
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
        print(f"DEBUG: Test - Created/Updated dummy cue file: {dummy_cue_filepath}")
    except Exception as e_cue_write: print(f"ERROR: Test - Could not create dummy cue file: {e_cue_write}")
    if not os.path.exists(test_audio_file): print(f"WARNING: Test audio file not found: {test_audio_file}.")
    else:
        if deck.load_track(test_audio_file):
            print(f"\nTrack loaded: {deck.deck_id}. BPM: {deck.bpm}, SR: {deck.sample_rate}")
            if deck.bpm == 0: print("WARNING: BPM is 0, loop length calc will be incorrect.")
            print("\nPlaying from CUE 'intro_start'..."); deck.play(start_at_cue_name="intro_start")
            time.sleep(0.2); print("Playing for ~2.0s to reach near beat 5..."); time.sleep(2.0) 
            start_loop_beat = 5; loop_len_beats = 4; num_reps = 3
            print(f"\nActivating {loop_len_beats}-beat loop @ beat {start_loop_beat} for {num_reps} reps...")
            deck.activate_loop(start_beat=start_loop_beat, length_beats=loop_len_beats, repetitions=num_reps)
            loop_single_duration = (60.0 / deck.bpm * loop_len_beats if deck.bpm > 0 else 1.0)
            wait_for_loop_and_post = (loop_single_duration * num_reps) + 2.5 
            print(f"Waiting for loop ({num_reps} reps of ~{loop_single_duration:.2f}s) + ~2s post-loop (total ~{wait_for_loop_and_post:.2f}s)...")
            time.sleep(wait_for_loop_and_post) 
            print(f"\nAfter finite loop, Frame: {deck.get_current_display_frame()}, UserWantsPlay: {deck._user_wants_to_play}, Active: {deck.is_active()}")
            next_loop_start_beat = 0
            if deck.bpm > 0 and deck.sample_rate > 0 and len(deck.beat_timestamps) > 0:
                current_time_after_loop = deck.get_current_display_frame() / float(deck.sample_rate)
                current_beat_idx = np.searchsorted(deck.beat_timestamps, current_time_after_loop, side='right')
                next_loop_start_beat = current_beat_idx + 4 
                next_loop_start_beat = min(next_loop_start_beat, len(deck.beat_timestamps)) 
                if next_loop_start_beat <= 0 and len(deck.beat_timestamps) > 0 : next_loop_start_beat = len(deck.beat_timestamps) // 2 
            else: next_loop_start_beat = 25 
            print(f"\nActivating infinite loop @ beat {next_loop_start_beat} for {loop_len_beats} beats (plays 5s)...")
            if next_loop_start_beat > 0: 
                deck.activate_loop(start_beat=next_loop_start_beat, length_beats=loop_len_beats) 
                time.sleep(5)
                print(f"\nDeactivating loop, playing 2s more..."); deck.deactivate_loop()
                time.sleep(0.1) 
                if deck.is_active(): print(f"DEBUG: Test - Continuing after deactivation. Frame: {deck.get_current_display_frame()}"); time.sleep(2)
                else: print("DEBUG: Test - Not active after deactivate. Re-requesting play."); deck.play(); time.sleep(2)
            else: print("WARNING: Test - Could not determine valid start for infinite loop. Skipping.")
            print("\nStopping playback..."); deck.stop(); time.sleep(0.5) 
            print(f"\nFinal Frame: {deck.get_current_display_frame()}, UserWantsPlay: {deck._user_wants_to_play}, Active: {deck.is_active()}")
            print("\nLooping Test finished.")
        else: print(f"Failed to load track: {deck.deck_id}")
    deck.shutdown() 
    if os.path.exists(dummy_cue_filepath):
        try: os.remove(dummy_cue_filepath)
        except Exception: pass
    print("--- Deck Test Complete ---")