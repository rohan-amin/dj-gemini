"""Demonstration script for graceful deck stop.

Loads an audio file, activates a 3-iteration loop, and issues a
non-flushing stop so buffered audio can drain. Run with a local audio
file to verify all loop iterations are heard before playback stops.
"""

import sys
import time

import numpy as np
from audio_engine.deck import Deck
from audio_engine.deck import DECK_CMD_LOAD_AUDIO


class DummyAnalyzer:
    pass


class DemoEngine:
    def handle_loop_complete(self, deck_id, action_id):
        print(f"loop {action_id} complete on deck {deck_id}")


def main(audio_path: str):
    deck = Deck("demo", DummyAnalyzer(), DemoEngine())
    sr = 44100
    data, _ = None, None
    try:
        import soundfile as sf
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim == 1:
            data = np.column_stack([data, data])
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit(f"Failed to load audio file: {exc}")

    deck.command_queue.put((DECK_CMD_LOAD_AUDIO, {
        "audio_data": data,
        "sample_rate": sr,
        "total_frames": len(data),
    }))

    # configure a 3-iteration loop spanning the entire clip
    with deck._stream_lock:
        deck._loop_active = True
        deck._loop_start_frame = 0
        deck._loop_end_frame = len(data)
        deck._loop_repetitions_total = 3
        deck._loop_repetitions_done = 0
        deck._current_loop_action_id = "demo"

    deck.play()

    # Immediately request stop without flush so buffered audio drains
    deck.stop(flush=False)

    # Allow time for buffered audio to play out
    time.sleep(len(data) * 3 / sr + 0.5)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python three_loop_stop_demo.py <audio_file>")
        sys.exit(1)
    main(sys.argv[1])
