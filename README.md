# DJ Gemini

An automated DJ mixing system that creates synchronized audio performances using JSON scripts. DJ Gemini analyzes audio files for beats and BPM, then executes precise, beat-synchronized mix sequences with professional-grade tempo control.

## Features

- **Beat-accurate timing** - All actions synchronized to detected beats
- **Multi-deck mixing** - Manage multiple audio tracks simultaneously
- **Professional tempo control** - Pitch-preserving tempo changes using Rubber Band
- **Loop system** - Create precise beat-synchronized loops with repetitions
- **Stop at beat** - Clean stopping at specific beat positions
- **Cue point support** - Use predefined cue points in audio files
- **JSON-based scripting** - Define complex mix sequences in JSON format
- **Real-time monitoring** - Track deck status and script execution progress
- **Audio caching** - Pre-processed audio for instant tempo changes
- **Loop queue system** - Sequential loop execution without race conditions

## Installation

### Prerequisites

- Python 3.8+
- Conda (recommended for scientific packages)
- macOS: Homebrew for system dependencies

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd dj-gemini
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n dj-gemini-env python=3.9
   conda activate dj-gemini-env
   ```

3. **Install system dependencies (macOS)**
   ```bash
   brew install rubberband
   ```

4. **Install required packages**
   ```bash
   conda install -c conda-forge essentia numpy sounddevice
   pip install pyrubberband librosa tqdm
   ```

## Quick Start

### 1. Prepare Audio Files

Place your audio files in the `audio_tracks/` directory:

```
dj-gemini/
├── main.py                 # Main entry point
├── config.py              # Configuration settings
├── audio_engine/          # Core audio processing
│   ├── engine.py         # Main audio engine
│   ├── deck.py           # Individual deck management
│   └── audio_analyzer.py # Beat detection and analysis
├── audio_tracks/          # Audio files
├── mix_configs/           # JSON mix scripts
├── analysis_data/         # Cached beat analysis
└── utilities/             # Helper tools
```

### 2. Create a Mix Script

Create a JSON file in the `mix_configs/` directory:

```json
{
  "script_name": "My First Mix",
  "actions": [
    {
      "id": "load_song1",
      "command": "load_track",
      "deck_id": "deckA",
      "parameters": {"file_path": "song1.mp3"}
    },
    {
      "id": "set_tempo",
      "command": "set_tempo",
      "deck_id": "deckA",
      "parameters": {"target_bpm": 140}
    },
    {
      "id": "play_song1",
      "command": "play",
      "deck_id": "deckA",
      "parameters": {"start_at_beat": 1}
    },
    {
      "id": "loop_at_beat_32",
      "command": "activate_loop",
      "deck_id": "deckA",
      "trigger": {
        "type": "on_deck_beat",
        "source_deck_id": "deckA",
        "beat_number": 32
      },
      "parameters": {
        "start_at_beat": 32,
        "length_beats": 8,
        "repetitions": 3
      }
    },
    {
      "id": "stop_at_beat_64",
      "command": "stop_at_beat",
      "deck_id": "deckA",
      "parameters": {"beat_number": 64}
    }
  ]
}
```

### 3. Run the Mix

```bash
python main.py mix_configs/my_mix.json
```

## Supported Commands

### Track Management

#### `load_track`
Load an audio file onto a deck.
```json
{
  "command": "load_track",
  "deck_id": "deckA",
  "parameters": {"file_path": "song.mp3"}
}
```

#### `play`
Start playback on a deck.
```json
{
  "command": "play",
  "deck_id": "deckA",
  "parameters": {
    "start_at_beat": 1,
    "start_at_cue_name": "intro"
  }
}
```

#### `pause`
Pause playback on a deck.
```json
{
  "command": "pause",
  "deck_id": "deckA"
}
```

#### `stop`
Stop playback and reset deck position.
```json
{
  "command": "stop",
  "deck_id": "deckA"
}
```

### Tempo Control

#### `set_tempo`
Change playback tempo with pitch preservation.
```json
{
  "command": "set_tempo",
  "deck_id": "deckA",
  "parameters": {"target_bpm": 140}
}
```

**Features:**
- Pitch-preserving tempo changes using Rubber Band
- Automatic audio caching for instant tempo changes
- Beat positions, cue points, and loops scale correctly
- Works with both faster and slower tempo changes

### Loop System

#### `activate_loop`
Create a beat-synchronized loop.
```json
{
  "command": "activate_loop",
  "deck_id": "deckA",
  "trigger": {
    "type": "on_deck_beat",
    "source_deck_id": "deckA",
    "beat_number": 32
  },
  "parameters": {
    "start_at_beat": 32,
    "length_beats": 8,
    "repetitions": 3
  }
}
```

**Parameters:**
- `start_at_beat`: Beat where loop starts
- `length_beats`: Number of beats in the loop
- `repetitions`: Number of times to repeat (use `"infinite"` for endless loop)

**Loop Queue System:**
- Multiple loops are queued sequentially
- No race conditions between overlapping loops
- Each loop completes all repetitions before the next activates

#### `deactivate_loop`
Stop the current loop and continue normal playback.
```json
{
  "command": "deactivate_loop",
  "deck_id": "deckA"
}
```

### Advanced Control

#### `stop_at_beat`
Stop playback when reaching a specific beat.
```json
{
  "command": "stop_at_beat",
  "deck_id": "deckA",
  "parameters": {"beat_number": 64}
}
```

## Triggers

### `script_start`
Execute action immediately when script starts.
```json
{
  "trigger": {"type": "script_start"}
}
```

### `on_deck_beat`
Execute action when a specific deck reaches a beat.
```json
{
  "trigger": {
    "type": "on_deck_beat",
    "source_deck_id": "deckA",
    "beat_number": 32
  }
}
```

## Cue Points

Create `.cue` files next to your audio files to define cue points:

```json
{
  "intro": {"start_beat": 1},
  "verse": {"start_beat": 17},
  "chorus": {"start_beat": 33},
  "drop": {"start_beat": 65}
}
```

Then use them in play commands:
```json
{
  "command": "play",
  "deck_id": "deckA",
  "parameters": {"start_at_cue_name": "drop"}
}
```

## Advanced Usage

### Multiple Decks
```json
{
  "actions": [
    {
      "id": "load_deck_a",
      "command": "load_track",
      "deck_id": "deckA",
      "parameters": {"file_path": "song1.mp3"}
    },
    {
      "id": "load_deck_b", 
      "command": "load_track",
      "deck_id": "deckB",
      "parameters": {"file_path": "song2.mp3"}
    },
    {
      "id": "set_tempo_a",
      "command": "set_tempo",
      "deck_id": "deckA",
      "parameters": {"target_bpm": 140}
    },
    {
      "id": "play_both",
      "command": "play",
      "deck_id": "deckA",
      "parameters": {"start_at_beat": 1}
    },
    {
      "id": "play_deck_b_at_beat_32",
      "command": "play",
      "deck_id": "deckB",
      "trigger": {
        "type": "on_deck_beat",
        "source_deck_id": "deckA",
        "beat_number": 32
      },
      "parameters": {"start_at_beat": 1}
    }
  ]
}
```

### Complex Loop Sequences
```json
{
  "actions": [
    {
      "id": "first_loop",
      "command": "activate_loop",
      "deck_id": "deckA",
      "trigger": {"type": "on_deck_beat", "source_deck_id": "deckA", "beat_number": 32},
      "parameters": {"start_at_beat": 32, "length_beats": 8, "repetitions": 3}
    },
    {
      "id": "second_loop",
      "command": "activate_loop", 
      "deck_id": "deckA",
      "trigger": {"type": "on_deck_beat", "source_deck_id": "deckA", "beat_number": 56},
      "parameters": {"start_at_beat": 32, "length_beats": 4, "repetitions": 2}
    }
  ]
}
```

### Tempo Changes with Loops
```json
{
  "actions": [
    {
      "id": "load_track",
      "command": "load_track",
      "deck_id": "deckA",
      "parameters": {"file_path": "song.mp3"}
    },
    {
      "id": "set_tempo_faster",
      "command": "set_tempo",
      "deck_id": "deckA",
      "parameters": {"target_bpm": 140}
    },
    {
      "id": "play",
      "command": "play",
      "deck_id": "deckA",
      "parameters": {"start_at_beat": 100}
    },
    {
      "id": "loop_at_beat_108",
      "command": "activate_loop",
      "deck_id": "deckA",
      "trigger": {"type": "on_deck_beat", "source_deck_id": "deckA", "beat_number": 108},
      "parameters": {"start_at_beat": 108, "length_beats": 4, "repetitions": 3}
    },
    {
      "id": "stop_at_beat_120",
      "command": "stop_at_beat",
      "deck_id": "deckA",
      "parameters": {"beat_number": 120}
    }
  ]
}
```

## Troubleshooting

### Common Issues

1. **"No module named 'essentia'"**
   - Ensure you're in the correct conda environment
   - Reinstall: `conda install -c conda-forge essentia`

2. **"pyrubberband installation fails"**
   - Install system dependencies: `brew install rubberband` (macOS)
   - Ensure you have the latest pip: `pip install --upgrade pip`

3. **Audio glitches or hanging**
   - Check that audio files are valid
   - Ensure sufficient system resources

4. **Beat detection issues**
   - Check that audio files have clear rhythmic content
   - Verify BPM detection in logs

5. **Tempo change issues**
   - Ensure Rubber Band is properly installed
   - Check that target BPM is reasonable (e.g., 80-200 BPM)

### Debug Mode

Run with verbose logging:
```bash
python main.py mix_configs/my_mix.json --max_wait_after_script 3600
```

## Dependencies

### Core Libraries
- **essentia**: Audio analysis and beat detection
- **numpy**: Numerical operations
- **sounddevice**: Real-time audio playback
- **json**: Script parsing
- **threading**: Multi-threaded audio processing

### Tempo Control
- **pyrubberband**: High-quality pitch-preserving tempo changes
- **librosa**: Audio processing (fallback for tempo changes)
- **tqdm**: Progress bars for audio processing

### System Dependencies
- **Rubber Band**: Professional audio time-stretching library
- **Homebrew**: Package manager for macOS dependencies

## System Requirements

- **Python**: 3.11.11
- **OS**: macOS, Linux, Windows
- **Audio**: Working audio output device
- **Memory**: 2GB+ RAM recommended for large audio files
- **Storage**: Additional space for audio caching (varies by usage)

## Recent Changes

### v2.0 - Professional Tempo Control
- **Added Rubber Band integration** for pitch-preserving tempo changes
- **Implemented audio caching** for instant tempo changes
- **Fixed beat position scaling** for accurate timing at all tempos
- **Enhanced loop system** with queue-based execution
- **Added comprehensive tempo change support** for both faster and slower BPM
- **Improved race condition handling** in loop system

### v1.0 - Core Features
- Beat-accurate timing system
- Multi-deck mixing capabilities
- Loop system with repetitions
- Stop at beat functionality
- Cue point support
- JSON-based scripting

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

### Dependencies
- **essentia**: GPL v3 (https://github.com/MTG/essentia)
- **numpy**: BSD-3-Clause (https://numpy.org/)
- **sounddevice**: MIT (https://python-sounddevice.readthedocs.io/)
- **pyrubberband**: GPL v3 (https://github.com/bmcfee/pyrubberband)
- **librosa**: ISC License (https://librosa.org/)
- **tqdm**: MIT (https://tqdm.github.io/)
- **Python**: PSF License (https://www.python.org/)

### License Compliance
This project uses essentia and pyrubberband which are licensed under GPL v3. As a derivative work, this project is also licensed under GPL v3 to ensure compliance and maintain open source principles.

## Contributing

[Add contribution guidelines here]