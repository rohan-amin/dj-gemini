# dj-gemini/audio_engine/deck.py

import os
import threading
import time 
import queue 
import json 
import logging
logger = logging.getLogger(__name__)
import essentia.standard as es
import numpy as np
import sounddevice as sd
import librosa

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
DECK_CMD_SET_VOLUME = "SET_VOLUME"
DECK_CMD_FADE_VOLUME = "FADE_VOLUME"

class Deck:
    def __init__(self, deck_id, analyzer_instance):
        self.deck_id = deck_id
        self.analyzer = analyzer_instance
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

        self.playback_stream_obj = None 

        # Add to Deck class initialization
        self._tempo_cache = {}  # Cache processed audio for different tempos

        # Phase offset storage for bpm_match
        self._pending_phase_offset_beats = 0.0
        self._phase_offset_applied = False

        self.enable_hard_seek_on_loop = True  # PATCH: Toggle for hard seek buffer zeroing on loop activation
        self._just_activated_loop_flush_count = 0  # PATCH: For double-flush after loop activation

        self.audio_thread.start()
        logger.debug(f"Deck {self.deck_id} - Initialized and audio thread started.")

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
        logger.debug(f"Deck {self.deck_id} - Loaded cue points: {list(self.cue_points.keys())}")

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
        
        # Store original BPM
        self.original_bpm = self.bpm

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
            
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - Failed to load audio samples for {audio_filepath}: {e}")
            return False

        with self._stream_lock:
            self._current_playback_frame_for_display = 0 
            self._user_wants_to_play = False 
            self._loop_active = False 
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
            
            if self._tempo_ramp_active and self._current_tempo_ratio != 1.0:
                scaled_frame = int(original_frame / self._current_tempo_ratio)
                logger.debug(f"Deck {self.deck_id} - Beat {beat_number}: original frame {original_frame} → scaled frame {scaled_frame} (ratio: {self._current_tempo_ratio:.3f})")
                return scaled_frame
            else:
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
        logger.debug(f"Deck {self.deck_id} - get_current_beat_count() called")
        
        # Add timeout to prevent infinite hangs
        import time
        start_time = time.time()
        
        with self._stream_lock:
            logger.debug(f"Deck {self.deck_id} - Got stream lock")
            
            if self.audio_thread_data is None or self.sample_rate == 0 or len(self.beat_timestamps) == 0:
                logger.debug(f"Deck {self.deck_id} - Early return: data={self.audio_thread_data is None}, sr={self.sample_rate}, beats={len(self.beat_timestamps)}")
                return 0 
            
            logger.debug(f"Deck {self.deck_id} - About to calculate current time")
            current_time_seconds = self._current_playback_frame_for_display / float(self.sample_rate)
            logger.debug(f"Deck {self.deck_id} - Current time: {current_time_seconds:.3f}s")
            
            # Use ramp-adjusted beat timestamps if in ramp
            if self._tempo_ramp_active and self._ramp_beat_timestamps is not None:
                logger.debug(f"Deck {self.deck_id} - Using ramp beat timestamps")
                beat_timestamps_to_use = self._ramp_beat_timestamps
            else:
                logger.debug(f"Deck {self.deck_id} - Using original beat timestamps")
                beat_timestamps_to_use = self.beat_timestamps
                
            logger.debug(f"Deck {self.deck_id} - About to call searchsorted")
            
            # Add timeout check
            if time.time() - start_time > 1.0:  # 1 second timeout
                logger.error(f"Deck {self.deck_id} - Timeout in get_current_beat_count()")
                return 0
                
            try:
                # Use side='left' to get the current beat (not the next beat)
                beat_count = np.searchsorted(beat_timestamps_to_use, current_time_seconds, side='left')
                logger.debug(f"Deck {self.deck_id} - Beat count: {beat_count}")
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
        logger.debug(f"[LOOP FRAME CALC] Start beat: {start_beat}, End beat: {end_beat}, Length beats: {length_beats}")
        logger.debug(f"[LOOP FRAME CALC] Start frame: {start_frame}, End frame: {end_frame}, Length frames: {loop_length_frames}")
        logger.debug(f"[LOOP FRAME CALC] Frames per beat: {frames_per_beat:.2f}")
        
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
            # Don't set playback pointer here - let audio thread handle it
            logger.debug(f"Deck {self.deck_id} - IMMEDIATE loop state set: {action_id} (exact start: {start_frame})")
        
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
        """Generate cache filepath for tempo-processed audio"""
        import hashlib
        # Create a unique filename based on original file and target BPM
        # Use the analyzer's cache directory
        cache_key = f"{self.deck_id}_{target_bpm:.1f}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return os.path.join(self.analyzer.cache_dir, f"tempo_{cache_hash}.npy")

    def set_tempo(self, target_bpm):
        """Set playback tempo to exact BPM target using Rubber Band"""
        logger.info(f"Deck {self.deck_id} - Engine requests SET_TEMPO. Target BPM: {target_bpm}")
        
        if self.total_frames == 0:
            logger.error(f"Deck {self.deck_id} - Cannot set tempo: track not loaded.")
            return
        
        # Calculate tempo ratio
        tempo_ratio = target_bpm / self.original_bpm
        logger.debug(f"Deck {self.deck_id} - Calculated tempo ratio {tempo_ratio:.3f}")
        
        # Check cache first
        cache_filepath = self._get_tempo_cache_filepath(target_bpm)
        if os.path.exists(cache_filepath):
            logger.debug(f"Deck {self.deck_id} - Loading Rubber Band cache...")
            try:
                processed_audio = np.load(cache_filepath)
                with self._stream_lock:
                    self.audio_thread_data = processed_audio
                    self.audio_thread_total_samples = len(processed_audio)
                    # CRITICAL: Update the BPM to the target BPM
                    self.bpm = target_bpm
                    # Scale the beat timestamps to match the new tempo
                    self._scale_beat_positions(tempo_ratio)
                logger.debug(f"Deck {self.deck_id} - Loaded cached tempo {target_bpm}")
                return
            except Exception as e:
                logger.warning(f"Deck {self.deck_id} - Failed to load cache: {e}")
        
        # Process with Rubber Band
        logger.debug(f"Deck {self.deck_id} - Processing with Rubber Band...")
        try:
            import pyrubberband as pyrb
            
            # Debug: Check what sample rate we have
            logger.debug(f"Deck {self.deck_id} - Audio thread sample rate: {self.audio_thread_sample_rate} (type: {type(self.audio_thread_sample_rate)})")
            logger.debug(f"Deck {self.deck_id} - Main sample rate: {self.sample_rate} (type: {type(self.sample_rate)})")
            logger.debug(f"Deck {self.deck_id} - Tempo ratio: {tempo_ratio} (type: {type(tempo_ratio)})")
            logger.debug(f"Deck {self.deck_id} - Audio data shape: {self.audio_thread_data.shape if (self.audio_thread_data is not None and hasattr(self.audio_thread_data, 'shape')) else 'no shape'}")
            logger.debug(f"Deck {self.deck_id} - Audio data type: {type(self.audio_thread_data)}")
            
            # Use the audio thread sample rate to match the audio data
            # If audio_thread_sample_rate is not available, fall back to main sample_rate
            sample_rate_to_use = self.audio_thread_sample_rate if self.audio_thread_sample_rate is not None else self.sample_rate
            sample_rate_to_use = int(sample_rate_to_use)
            
            logger.debug(f"Deck {self.deck_id} - Using sample rate: {sample_rate_to_use}")
            
            # Ensure audio data is the right format for pyrubberband
            if self.audio_thread_data is None:
                logger.error(f"Deck {self.deck_id} - No audio data available for tempo processing")
                return
                
            # Convert to float32 if needed
            audio_data = self.audio_thread_data.astype(np.float32)
            
            # Let's try the correct pyrubberband function signature
            # According to the docs, it should be: pyrb.time_stretch(y, rate, sr)
            # But let's also try with explicit keyword arguments
            processed_audio = pyrb.time_stretch(
                y=audio_data,
                rate=float(tempo_ratio),
                sr=sample_rate_to_use
            )
            
            # Save to cache
            np.save(cache_filepath, processed_audio)
            
            # Update audio data and scale positions
            with self._stream_lock:
                self.audio_thread_data = processed_audio
                self.audio_thread_total_samples = len(processed_audio)
                # CRITICAL: Update the BPM to the target BPM
                self.bpm = target_bpm
                # Scale the beat timestamps to match the new tempo
                self._scale_beat_positions(tempo_ratio)
            
            logger.info(f"Deck {self.deck_id} - Rubber Band processing complete, BPM now {self.bpm}")
            
        except Exception as e:
            logger.error(f"Deck {self.deck_id} - Rubber Band processing failed: {e}")
            import traceback
            traceback.print_exc()
            return

    def _scale_beat_positions(self, tempo_ratio):
        """Scale all beat positions, cue points, and loop positions to match the new tempo"""
        logger.debug(f"Deck {self.deck_id} - Scaling positions by tempo ratio {tempo_ratio:.3f}")
        
        # Scale beat timestamps - CORRECTED: divide by tempo ratio when speeding up
        if hasattr(self, 'beat_timestamps') and len(self.beat_timestamps) > 0:
            self.beat_timestamps = self.beat_timestamps / tempo_ratio
            logger.debug(f"Deck {self.deck_id} - Scaled {len(self.beat_timestamps)} beat timestamps")
        
        # Scale cue points - CORRECTED: divide by tempo ratio
        if hasattr(self, 'cue_points') and self.cue_points:
            scaled_cue_points = {}
            for cue_name, cue_data in self.cue_points.items():
                if 'start_beat' in cue_data:
                    # Scale the beat position - CORRECTED: divide by tempo ratio
                    scaled_cue_points[cue_name] = {
                        'start_beat': cue_data['start_beat'] / tempo_ratio
                    }
            self.cue_points = scaled_cue_points
            logger.debug(f"Deck {self.deck_id} - Scaled {len(self.cue_points)} cue points")
        
        # Update original beat positions for loop calculations - CORRECTED: divide by tempo ratio
        if hasattr(self, 'original_beat_positions') and self.original_beat_positions:
            scaled_positions = {}
            for beat, frame in self.original_beat_positions.items():
                # Scale the frame position - CORRECTED: divide by tempo ratio
                scaled_positions[beat] = int(frame / tempo_ratio)
            self.original_beat_positions = scaled_positions
            logger.debug(f"Deck {self.deck_id} - Scaled {len(self.original_beat_positions)} beat positions")

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
                            samplerate=self.audio_thread_sample_rate, channels=1,
                            callback=self._sd_callback, 
                            finished_callback=self._on_stream_finished_from_audio_thread, 
                            blocksize=256  # PATCH: Lower buffer size for tighter loop cueing
                        )
                        self.playback_stream_obj = _current_stream_in_thread 
                        _current_stream_in_thread.start()
                        self._is_actually_playing_stream_state = True
                        logger.debug(f"Deck {self.deck_id} AudioThread - Stream started for PLAY.")
                    
                elif command == DECK_CMD_PAUSE:
                    logger.debug(f"Deck {self.deck_id} AudioThread - Processing PAUSE")
                    if _current_stream_in_thread and _current_stream_in_thread.active:
                        logger.debug(f"Deck {self.deck_id} AudioThread - Calling stream.stop() for PAUSE.")
                        _current_stream_in_thread.stop(ignore_errors=True) 
                        logger.debug(f"Deck {self.deck_id} AudioThread - stream.stop() called for PAUSE.")
                    else: 
                         with self._stream_lock: self._is_actually_playing_stream_state = False

                elif command == DECK_CMD_STOP:
                    logger.debug(f"Deck {self.deck_id} AudioThread - Processing STOP")
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
                    
                    # Log the first 10 sample values at the loop start for debugging
                    start_frame = data['start_frame']
                    if self.audio_thread_data is not None:
                        sample_preview = self.audio_thread_data[start_frame:start_frame+10]
                        last_sample_before = self.audio_thread_data[start_frame-1] if start_frame > 0 else None
                        first_sample = self.audio_thread_data[start_frame] if len(self.audio_thread_data) > start_frame else None
                        logger.debug(f"[LOOP ACTIVATION SAMPLES] Frame {start_frame}: {sample_preview}")
                        logger.debug(f"[LOOP ACTIVATION SAMPLES] Last sample before loop: {last_sample_before}, First sample at loop: {first_sample}")

                    with self._stream_lock:
                        # LOOP STATE IS ALREADY SET - just update playback pointer
                        if self._loop_active:
                            # Set playback pointer to loop start immediately (uses predictive start)
                            self._current_playback_frame_for_display = self._loop_start_frame
                            self.audio_thread_current_frame = self._loop_start_frame
                            logger.debug(f"Deck {self.deck_id} AudioThread - Updated playback pointer to exact loop start: {self._loop_start_frame}")
                        else:
                            # Loop state should already be set, but handle edge case
                            logger.warning(f"Deck {self.deck_id} AudioThread - Loop state not set when processing ACTIVATE_LOOP command")
                            self._loop_start_frame = data['start_frame']
                            self._loop_end_frame = data['end_frame']
                            self._loop_repetitions_total = data.get('repetitions')
                            self._loop_repetitions_done = 0
                            self._loop_active = True
                            self._current_loop_action_id = data.get('action_id')
                            self._current_playback_frame_for_display = self._loop_start_frame
                            self.audio_thread_current_frame = self._loop_start_frame

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

    def _sd_callback(self, outdata, frames, time_info, status_obj):
        # Remove the constant "Audio callback called" message
        # print(f"DEBUG: Deck {self.deck_id} CB - Audio callback called (frames: {frames})")
        
        if status_obj:
            if status_obj.output_underflow: logger.warning(f"Warning: Deck {self.deck_id} CB - Output underflow")
            if status_obj.output_overflow: logger.warning(f"Warning: Deck {self.deck_id} CB - Output overflow")

        try:
            with self._stream_lock:
                # REMOVED: Hard seek patch that was causing artifacts
                # The loop should activate naturally without aggressive frame jumping

                if not self._user_wants_to_play or self.audio_thread_data is None:
                    logger.debug(f"DEBUG: Deck {self.deck_id} CB - No play or no audio data")
                    outdata[:] = 0
                    return

                # Get current frame position
                current_frame = self._current_playback_frame_for_display
                
                # Calculate current time for tempo ramp updates
                current_time = current_frame / self.sample_rate
                
                # Check if we need to update tempo ramp - PASS CURRENT TIME INSTEAD OF BEAT
                if self._tempo_ramp_active:
                    # Remove the constant ramp update message
                    # print(f"DEBUG: Deck {self.deck_id} CB - Calling _update_tempo_ramp for time {current_time:.3f}")
                    self._update_tempo_ramp(current_time)  # Pass current time instead of beat
                
                # Check if we have enough audio data
                if current_frame + frames > len(self.audio_thread_data):
                    # End of audio reached
                    logger.debug(f"DEBUG: Deck {self.deck_id} CB - End of audio reached")
                    outdata[:] = 0
                    return
                
                # ADDED: Check for loop boundaries during ramp
                if self._loop_active and self._loop_end_frame > 0:
                    if current_frame >= self._loop_end_frame:
                        self._loop_repetitions_done += 1
                        logger.debug(f"DEBUG: Deck {self.deck_id} CB - Loop repetition {self._loop_repetitions_done}/{self._loop_repetitions_total}")
                        # PATCH: Handle None (infinite) repetitions safely
                        reps_total = self._loop_repetitions_total if self._loop_repetitions_total is not None else float('inf')
                        if self._loop_repetitions_done < reps_total:
                            # More repetitions to go - jump back to start
                            logger.debug(f"DEBUG: Deck {self.deck_id} CB - Loop boundary reached, jumping to {self._loop_start_frame}")
                            self._current_playback_frame_for_display = self._loop_start_frame
                            current_frame = self._loop_start_frame
                            outdata[:] = 0  # Prevent audio overlap
                            return  # Exit callback immediately
                        else:
                            # Loop complete - don't jump back
                            logger.debug(f"DEBUG: Deck {self.deck_id} CB - Loop complete, deactivating")
                            self._loop_active = False
                            self._loop_repetitions_done = 0
                            self._loop_repetitions_total = None
                            self._loop_just_completed = True
                            self._completed_loop_action_id = getattr(self, '_current_loop_action_id', None)
                            # If there is another loop in the queue, activate it and set its action_id
                            if self._loop_queue:
                                next_loop = self._loop_queue.pop(0)
                                self._loop_start_frame = next_loop['start_frame']
                                self._loop_end_frame = next_loop['end_frame']
                                self._loop_repetitions_total = next_loop.get('repetitions')
                                self._loop_repetitions_done = 0  # Start at 0 for new loop
                                self._loop_active = True
                                self._current_loop_action_id = next_loop.get('action_id')
                                self._loop_just_completed = False
                                self._completed_loop_action_id = None
                                self._current_playback_frame_for_display = self._loop_start_frame
                                current_frame = self._loop_start_frame
                                outdata[:] = 0  # Prevent audio overlap
                                return  # Exit callback immediately
                
                # Calculate current beat AFTER loop boundary check
                # This ensures the beat detection reflects the current position after any loop jumps
                current_time_after_loop = current_frame / self.sample_rate
                current_beat = np.searchsorted(self.beat_timestamps, current_time_after_loop, side='left')
                
                # FIXED: Update shared state for synchronized trigger checking
                self._last_synchronized_beat = current_beat
                
                # Only print beat changes, not every callback
                if not hasattr(self, '_last_printed_beat') or self._last_printed_beat != current_beat:
                    logger.debug(f"DEBUG: Deck {self.deck_id} CB - Current Beat: {current_beat} (Frame: {current_frame})")
                    self._last_printed_beat = current_beat
                
                # Get the audio chunk with tempo processing
                if self._tempo_ramp_active and self._current_tempo_ratio != 1.0:
                    # Only print tempo changes when they actually change
                    if not hasattr(self, '_last_printed_tempo_ratio') or abs(self._current_tempo_ratio - self._last_printed_tempo_ratio) > 0.01:
                        logger.debug(f"DEBUG: Deck {self.deck_id} CB - Applied tempo ratio {self._current_tempo_ratio:.3f} (BPM: {self.bpm:.1f})")
                        self._last_printed_tempo_ratio = self._current_tempo_ratio
                
                    # FIXED: Correct tempo change logic (INVERTED)
                    # When tempo_ratio > 1.0 (BPM increased), we want to speed up
                    # To speed up: read MORE input frames and compress them to output size
                    # When tempo_ratio < 1.0 (BPM decreased), we want to slow down  
                    # To slow down: read FEWER input frames and stretch them to output size
                    
                    # Calculate how many input frames we need
                    # For speed up (ratio > 1): read more frames (frames * ratio)
                    # For slow down (ratio < 1): read fewer frames (frames * ratio)
                    input_frames_needed = int(frames * self._current_tempo_ratio)
                    
                    # Ensure we don't exceed available audio
                    if current_frame + input_frames_needed > len(self.audio_thread_data):
                        input_frames_needed = len(self.audio_thread_data) - current_frame
                    
                    # Get input chunk
                    input_chunk = self.audio_thread_data[current_frame:current_frame + input_frames_needed]
                    
                    if len(input_chunk) > 0:
                        # Simple linear interpolation for tempo change
                        # Always compress/stretch the input chunk to the desired output size
                        input_indices = np.linspace(0, len(input_chunk) - 1, frames)
                        input_indices = input_indices.astype(int)
                        input_indices = np.clip(input_indices, 0, len(input_chunk) - 1)
                        audio_chunk = input_chunk[input_indices]
                        
                        # Update frame pointer - use the actual frames consumed
                        frames_consumed = input_frames_needed
                        self._current_playback_frame_for_display += frames_consumed
                    else:
                        # No audio data available
                        # PATCH: Ensure zeros are correct shape for mono
                        audio_chunk = np.zeros((frames, 1))
                else:
                    # No tempo change, use original audio
                    audio_chunk = self.audio_thread_data[current_frame:current_frame + frames]
                    self._current_playback_frame_for_display += frames
                
                # Update volume fade before applying volume
                self._update_volume_fade()
                # FIXED: Apply volume changes using the correct attribute name
                if hasattr(self, '_volume') and self._volume != 1.0:
                    audio_chunk *= self._volume
                
                # REMOVED: Crossfade logic that was masking the real issue
                
                # Ensure audio_chunk has the correct shape for output
                # outdata expects shape (frames, channels) but audio_chunk might be (frames,)
                if audio_chunk.ndim == 1:
                    # Convert 1D array to 2D array with 1 channel
                    audio_chunk = audio_chunk.reshape(-1, 1)
                
                # REMOVED: Split buffer logic that was causing artifacts
                
                # Copy to output
                outdata[:] = audio_chunk
                
        except Exception as e:
            logger.error(f"ERROR: Deck {self.deck_id} CB - Exception in audio callback: {e}")
            import traceback
            traceback.print_exc()
            outdata[:] = 0

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
        logger.debug(f"DEBUG: Deck {self.deck_id} - Shutdown complete.")


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