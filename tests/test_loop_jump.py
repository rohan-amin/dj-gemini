import sys, types, threading, pathlib
import numpy as np

# Ensure repository root is on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Stub external modules required by audio_engine.deck
essentia_module = types.ModuleType("essentia")
essentia_module.standard = types.ModuleType("standard")
sys.modules.setdefault("essentia", essentia_module)
sys.modules.setdefault("essentia.standard", essentia_module.standard)
sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice"))
sys.modules.setdefault("librosa", types.ModuleType("librosa"))

# Stub scipy modules used during import
scipy_module = types.ModuleType("scipy")
scipy_signal = types.ModuleType("signal")
def _dummy(*args, **kwargs):
    return None
scipy_signal.lfilter_zi = _dummy
scipy_signal.iirpeak = _dummy
scipy_module.signal = scipy_signal
sys.modules.setdefault("scipy", scipy_module)
sys.modules.setdefault("scipy.signal", scipy_signal)

# Minimal rubberband_ctypes stub so RUBBERBAND_STREAMING_AVAILABLE is True
class DummyRubberBand:
    def __init__(self, **kwargs):
        self.buffer = np.zeros((0, kwargs.get('channels', 2)), dtype=np.float32)
    def process(self, block, final=False):
        self.buffer = np.concatenate([self.buffer, block], axis=0)
    def available(self):
        return len(self.buffer)
    def retrieve(self, count):
        out = self.buffer[:count]
        self.buffer = self.buffer[count:]
        return out
    def reset(self):
        self.buffer = self.buffer[0:0]
    def set_time_ratio(self, ratio):
        pass
    def set_pitch_ratio(self, ratio):
        pass

rb_module = types.ModuleType("rubberband_ctypes")
rb_module.RubberBand = DummyRubberBand
rb_module.RubberBandOptionProcessRealTime = 0
rb_module.RubberBandOptionTransientsMixed = 0
rb_module.RubberBandOptionPhaseLaminar = 0
rb_module.RubberBandOptionChannelsTogether = 0
sys.modules.setdefault("rubberband_ctypes", rb_module)

import audio_engine.deck as deck

class DummyBeatManager:
    def get_tempo_ratio(self):
        return 1.0
    def update_from_frame(self, frame, sr):
        self.last = (frame, sr)

class DummyRingBuffer:
    def __init__(self):
        self.data = []
    def available_read(self):
        return len(self.data)
    def clear(self):
        self.data.clear()
    def write(self, chunk):
        self.data.append(chunk)

class DummyDeck:
    RING_BUFFER_SIZE = 4096
    def __init__(self):
        self.deck_id = 1
        self.total_frames = 1000
        self.out_ring = DummyRingBuffer()
        self._stream_lock = threading.Lock()
        self._rb_lock = threading.Lock()
        self.audio_thread_current_frame = 0
        self._current_playback_frame_for_display = 0
        self.beat_manager = DummyBeatManager()
        self._pending_out = None
        self._producer_startup_mode = False
        self.audio_thread_sample_rate = 44100
    def _reset_rubberband_for_position_jump(self, is_loop_jump):
        pass
    def _wait_for_ring_buffer_ready(self):
        pass
    def _produce_chunk_rubberband(self, chunk_size):
        # Mimic lock acquisition performed inside real method
        with self._stream_lock:
            self.audio_thread_current_frame += chunk_size
            return np.zeros((chunk_size, 2), dtype=np.float32)


def test_seamless_loop_jump_does_not_deadlock():
    deck_obj = DummyDeck()
    # Call in a separate thread and ensure it completes
    thread = threading.Thread(target=deck.Deck._perform_seamless_loop_jump_in_stream,
                              args=(deck_obj, 0))
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive(), "Loop jump stalled due to lock re-acquisition"
    # Ensure at least one chunk was written to ring buffer
    assert len(deck_obj.out_ring.data) > 0
    # Playback position should advance during priming
    assert deck_obj.audio_thread_current_frame > 0
