# DJ Gemini

An automated DJ mixing system that creates synchronized audio performances using JSON scripts. DJ Gemini analyzes audio files for beats and BPM, then executes precise, beat-synchronized mix sequences.

## Features

- **Beat-accurate timing** - All actions synchronized to detected beats
- **Multi-deck mixing** - Manage multiple audio tracks simultaneously
- **Loop system** - Create precise beat-synchronized loops with repetitions
- **Stop at beat** - Clean stopping at specific beat positions
- **Cue point support** - Use predefined cue points in audio files
- **JSON-based scripting** - Define complex mix sequences in JSON format
- **Real-time monitoring** - Track deck status and script execution progress

## Installation

### Prerequisites

- Python 3.8+
- Conda (recommended for scientific packages)

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

3. **Install required packages**
   ```bash
   conda install -c conda-forge essentia numpy sounddevice
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

## Project Structure

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

2. **Audio glitches or hanging**
   - Check that audio files are valid
   - Ensure sufficient system resources

3. **Beat detection issues**
   - Check that audio files have clear rhythmic content
   - Verify BPM detection in logs

### Debug Mode

Run with verbose logging:
```bash
python main.py mix_configs/my_mix.json --max_wait_after_script 3600
```

## Dependencies

- **essentia**: Audio analysis and beat detection
- **numpy**: Numerical operations
- **sounddevice**: Real-time audio playback
- **json**: Script parsing
- **threading**: Multi-threaded audio processing

## System Requirements

- **Python**: 3.11.11
- **OS**: macOS, Linux, Windows
- **Audio**: Working audio output device
- **Memory**: 2GB+ RAM recommended for large audio files

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

### Dependencies
- **essentia**: GPL v3 (https://github.com/MTG/essentia)
- **numpy**: BSD-3-Clause (https://numpy.org/)
- **sounddevice**: MIT (https://python-sounddevice.readthedocs.io/)
- **Python**: PSF License (https://www.python.org/)

### License Compliance
This project uses essentia which is licensed under GPL v3. As a derivative work, this project is also licensed under GPL v3 to ensure compliance and maintain open source principles.

## Contributing

[Add contribution guidelines here]