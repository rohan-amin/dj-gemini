# DJ Gemini

An automated DJ mixing system that creates synchronized audio performances using JSON scripts. DJ Gemini analyzes audio files for beats and BPM, then executes precise, beat-synchronized mix sequences with professional-grade tempo control.

## Features

- **Beat-accurate timing** - All actions synchronized to detected beats
- **Multi-deck mixing** - Manage multiple audio tracks simultaneously
- **Professional tempo control** - Pitch-preserving tempo changes using Rubber Band
- **Tempo ramping** - Smooth BPM transitions over specified beat ranges
- **Loop system** - Create precise beat-synchronized loops with repetitions
- **Stop at beat** - Clean stopping at specific beat positions
- **Volume control** - Per-deck volume adjustment and crossfading
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
      "id": "play_song1",
      "command": "play",
      "deck_id": "deckA",
      "parameters": {"start_at_beat": 1}
    },
    {
      "id": "start_ramp",
      "command": "ramp_tempo",
      "deck_id": "deckA",
      "trigger": {
        "type": "on_deck_beat",
        "source_deck_id": "deckA",
        "beat_number": 64
      },
      "parameters": {
        "start_beat": 64,
        "end_beat": 96,
        "start_bpm": 128,
        "end_bpm": 135,
        "curve": "linear"
      }
    },
    {
      "id": "trigger_during_ramp",
      "command": "activate_loop",
      "deck_id": "deckA",
      "trigger": {
        "type": "on_deck_beat",
        "source_deck_id": "deckA",
        "beat_number": 80
      },
      "parameters": {
        "start_at_beat": 80,
        "length_beats": 4,
        "repetitions": 3
      }
    },
    {
      "id": "trigger_after_ramp",
      "command": "set_volume",
      "deck_id": "deckA",
      "trigger": {
        "type": "on_deck_beat",
        "source_deck_id": "deckA",
        "beat_number": 96
      },
      "parameters": {"volume": 0.3}
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

#### `ramp_tempo`
Smoothly transition BPM over a specified beat range.
```json
{
  "command": "ramp_tempo",
  "deck_id": "deckA",
  "parameters": {
    "start_beat": 64,
    "end_beat": 96,
    "start_bpm": 128,
    "end_bpm": 135,
    "curve": "linear"
  }
}
```

**Features:**
- Smooth BPM transitions from start_beat to end_beat
- Audio speed changes in real-time to match BPM
- Supports linear and exponential curves
- Triggers and loops work correctly during ramps
- Beat positions remain accurate throughout the ramp

### Volume Control

#### `set_volume`
Set deck volume (0.0 to 1.0).
```json
{
  "command": "set_volume",
  "deck_id": "deckA",
  "parameters": {"volume": 0.8}
}
```

#### `fade_volume`
Fade volume over a specified duration.
```json
{
  "command": "fade_volume",
  "deck_id": "deckA",
  "parameters": {
    "target_volume": 0.0,
    "duration_beats": 8
  }
}
```

#### `crossfade`
Crossfade between two decks.
```json
{
  "command": "crossfade",
  "deck_id": "deckA",
  "parameters": {
    "target_deck": "deckB",
    "duration_beats": 4
  }
}
```

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
- Works correctly during tempo ramps

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

### Tempo Ramping with Loops
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
      "id": "play",
      "command": "play",
      "deck_id": "deckA",
      "parameters": {"start_at_beat": 1}
    },
    {
      "id": "start_ramp",
      "command": "ramp_tempo",
      "deck_id": "deckA",
      "trigger": {
        "type": "on_deck_beat",
        "source_deck_id": "deckA",
        "beat_number": 64
      },
      "parameters": {
        "start_beat": 64,
        "end_beat": 96,
        "start_bpm": 128,
        "end_bpm": 135,
        "curve": "linear"
      }
    },
    {
      "id": "loop_during_ramp",
      "command": "activate_loop",
      "deck_id": "deckA",
      "trigger": {
        "type": "on_deck_beat",
        "source_deck_id": "deckA",
        "beat_number": 80
      },
      "parameters": {
        "start_at_beat": 80,
        "length_beats": 4,
        "repetitions": 3
      }
    },
    {
      "id": "volume_after_ramp",
      "command": "set_volume",
      "deck_id": "deckA",
      "trigger": {
        "type": "on_deck_beat",
        "source_deck_id": "deckA",
        "beat_number": 96
      },
      "parameters": {"volume": 0.3}
    }
  ]
}
```

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

6. **Tempo ramp issues**
   - Ensure start_bpm and end_bpm are reasonable values
   - Check that start_beat < end_beat
   - Verify that ramp duration is sufficient for smooth transitions

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