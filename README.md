# DJ Gemini

A Python-based DJ mixing system that uses JSON scripts to create automated audio mixes with precise timing, beat synchronization, and real-time audio effects.

## Overview

DJ Gemini uses a **preprocessing architecture** to separate heavy audio processing from real-time performance. The system provides two workflows:

### Preprocessing Workflow (Recommended)
1. **Run preprocessing** to analyze your mix script and prepare all audio transformations
2. **Smart BPM analysis** automatically predicts tempo needs (including `bpm_match` commands)
3. **Cache audio files** with song-based organization for easy management
4. **Execute mix** with instant playback and zero processing delays

### Just-In-Time (JIT) Workflow (Fallback)
- **Automatic fallback** when cached audio isn't available
- **Real-time processing** using PyRubberBand for missing tempo variants
- **Seamless experience** - no interruption in workflow

This hybrid approach guarantees smooth, glitch-free performance while providing flexibility for experimentation.

## Key Features

### Core DJ Features
- **Beat-accurate timing** - All actions synchronized to detected beats
- **Multi-deck mixing** - Manage multiple audio tracks simultaneously
- **Professional tempo control** - Pitch-preserving tempo changes using Rubber Band
- **Tempo ramping** - Smooth BPM transitions over specified beat ranges
- **BPM matching** - Instant tempo synchronization between decks
- **Loop system** - Create precise beat-synchronized loops with repetitions
- **Volume control** - Per-deck volume adjustment and crossfading
- **EQ control** - 3-band EQ with instant and fade modes

### Audio Processing
- **Preprocessing system** - Separate heavy processing from performance
- **Smart BPM analysis** - Automatically predicts tempo requirements
- **Song-based caching** - Organized cache structure with hash-based directories
- **JIT processing** - Real-time fallback for missing audio variants
- **Perfect beat synchronization** - Beat viewer and main engine match exactly

### Workflow Tools
- **Beat viewer utility** - Visual tool for finding precise beat positions
- **JSON-based scripting** - Define complex mix sequences in JSON format
- **Real-time monitoring** - Track deck status and script execution progress
- **Cue point support** - Use predefined cue points in audio files
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

### 1. Preprocessing Workflow (Recommended)

For the best performance, preprocess your mix scripts before execution:

```bash
# Activate conda environment
conda activate dj-gemini-env

# Run preprocessing on your mix script
python preprocess.py mix_configs/my_mix.json

# Execute the mix with instant playback
python main.py mix_configs/my_mix.json
```

**What preprocessing does:**
- Analyzes your JSON script to find all tempo/pitch requirements
- Predicts BPM needs for `bpm_match` commands using smart analysis
- Pre-processes all audio transformations using PyRubberBand
- Caches results in organized `cache/` directory structure
- Ensures zero-latency performance during mix execution

**Smart BPM Analysis Example:**
```bash
# Output shows predicted requirements
Smart analysis predicted BPMs for deckB: [124.1]
Found 2 tracks requiring preprocessing:
  Starships.mp3: 0 tempos, 0 pitches
  One More Time.mp3: 1 tempos, 0 pitches
```

### 2. Beat Viewer Utility

Use the beat viewer to find precise beat positions for your JSON scripts:

```bash
# Launch beat viewer
python utilities/beat_viewer.py

# Or load a specific file
python utilities/beat_viewer.py audio_tracks/my_song.mp3
```

**Beat Viewer Features:**
- **JIT tempo processing** - Change BPM to any value for precise timing
- **Visual beat navigation** - Find exact musical positions
- **Perfect synchronization** - Beat numbers match main engine exactly
- **Easy workflow** - Pick beats in viewer, use same numbers in JSON

### 3. Prepare Audio Files

Place your audio files in the `audio_tracks/` directory:

```
dj-gemini/
├── main.py                 # Main entry point
├── preprocess.py          # Preprocessing script  
├── config.py              # Configuration settings
├── audio_engine/          # Core audio processing
│   ├── engine.py         # Main audio engine
│   ├── deck.py           # Individual deck management
│   └── audio_analyzer.py # Beat detection and analysis
├── audio_tracks/          # Audio files
├── mix_configs/           # JSON mix scripts
├── cache/                 # Song-based cached audio processing
│   └── hash_filename/    # Individual song cache directories
│       ├── analysis.beats # Beat timestamps and BPM data
│       ├── tempo_124.1.npy # Tempo-processed audio files
│       └── pitch_+2.0.npy # Pitch-processed audio files
└── utilities/             # Helper tools
    └── beat_viewer.py    # Beat position finder tool
```

### 4. Create a Mix Script

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

### 5. Run the Mix

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

#### `seek_and_play`
Seek to a specific beat or cue point and start playback immediately on a deck.
```json
{
  "command": "seek_and_play",
  "deck_id": "deckA",
  "parameters": {
    "beat_number": 64,
    "cue_name": "drop"
  }
}
```
- `beat_number` (optional): Seek to this beat number and play from there.
- `cue_name` (optional): Seek to this cue point and play from there.
- If both are provided, `cue_name` takes precedence.
- Use to jump to a specific position and start playback in one action.

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
- **Pre-processed audio caching** - all tempo changes are processed upfront before playback
- Beat positions, cue points, and loops scale correctly
- Works with both faster and slower tempo changes
- **Instant playback** - no processing delays during performance

### Pitch Control

#### `set_pitch`
Change playback pitch with time preservation.
```json
{
  "command": "set_pitch",
  "deck_id": "deckA",
  "parameters": {"semitones": -2}
}
```

**Features:**
- Time-preserving pitch changes using Rubber Band
- **Pre-processed audio caching** - all pitch changes are processed upfront before playback
- **Instant playback** - no processing delays during performance
- Supports both upward and downward pitch shifts

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

#### `bpm_match`
Synchronize the BPM of one deck to another deck's BPM instantly.
```json
{
  "command": "bpm_match",
  "deck_id": "deckB",
  "parameters": {
    "reference_deck": "deckA",
    "phase_offset_beats": 0.5
  }
}
```
- `reference_deck`: The deck to match BPM to.
- `phase_offset_beats` (optional): Apply a fractional beat offset to the matched deck after BPM sync. Use this to fine-tune the phase alignment between decks if the beats are not perfectly aligned after matching BPM.
- For example, if you notice the downbeats are slightly off after BPM sync, you can set `phase_offset_beats` to a positive or negative value (e.g., 0.5 or -0.25) to nudge the alignment.
- Use to instantly sync the tempo of two decks for beatmatching, with optional phase correction.

#### `stop_at_beat`
Stop playback when reaching a specific beat.
```json
{
  "command": "stop_at_beat",
  "deck_id": "deckA",
  "parameters": {"beat_number": 64}
}
```
- `beat_number`: The beat at which to stop playback.
- Use for precise, beat-synchronized stopping.

#### `deactivate_loop`
Stop the current loop and continue normal playback.
```json
{
  "command": "deactivate_loop",
  "deck_id": "deckA"
}
```
- Use to exit a loop and resume normal track progression.

### EQ Control

#### `set_eq`
Set EQ bands (low, mid, high) instantly (with configurable fast smoothing to prevent clicks).

```json
{
  "command": "set_eq",
  "deck_id": "deckA",
  "parameters": {"low": 1.0, "mid": 0.5, "high": 0.2}
}
```
- The transition is smoothed over a very short window (default: 0.5ms, configurable via `EQ_SMOOTHING_MS` in `config.py`).
- Use for near-instant EQ changes (e.g., kill switches, quick boosts).

#### `fade_eq`
Fade EQ bands (low, mid, high) over a specified duration (in seconds).

```json
{
  "command": "fade_eq",
  "deck_id": "deckA",
  "parameters": {"target_low": 1.0, "target_mid": 0.0, "target_high": 0.0, "duration_seconds": 2.0}
}
```
- Use for smooth, gradual EQ transitions (e.g., filter sweeps, DJ-style fades).
- The duration is set per-action and is not affected by `EQ_SMOOTHING_MS`.

**Note:**
- `set_eq` uses the global smoothing window for click-free, fast changes.
- `fade_eq` uses the specified duration for gradual transitions.

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
   - Ensure you're in the correct conda environment: `conda activate dj-gemini-env`
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
   - Run preprocessing first: `python preprocess.py mix_configs/script.json`
   - Ensure Rubber Band is properly installed
   - Check that target BPM is reasonable (e.g., 80-200 BPM)

6. **Tempo ramp issues**
   - Ensure start_bpm and end_bpm are reasonable values
   - Check that start_beat < end_beat
   - Verify that ramp duration is sufficient for smooth transitions

### Preprocessing Issues

7. **"Tempo cache not found" errors**
   - Run preprocessing first: `python preprocess.py mix_configs/script.json`
   - Check that conda environment is activated
   - Verify PyRubberBand is installed correctly

8. **Beat viewer tempo changes don't work**
   - Ensure PyRubberBand is installed: `pip install pyrubberband`
   - Check that system dependencies are installed: `brew install rubberband`
   - Look for JIT processing messages in console output

9. **Beat numbers don't match between beat viewer and main engine**
   - This should now be fixed with the latest updates
   - Both systems use identical beat timestamp scaling logic
   - Verify both are using the same BPM setting

10. **Smart BPM analysis doesn't predict correctly**
    - Check your JSON script has `bpm_match` commands
    - Verify deck loading order and timing
    - Look for prediction output in preprocessing logs

### Debug Mode

Run with verbose logging:
```bash
python main.py mix_configs/my_mix.json --max_wait_after_script 3600
```

## New Features & Workflow

### Beat-Perfect DJ Workflow

1. **Find transition points** with beat viewer:
   ```bash
   python utilities/beat_viewer.py audio_tracks/song.mp3
   # Set BPM to match your target, navigate to find perfect transition
   # Note the beat number displayed (e.g., "Beat: 592")
   ```

2. **Use exact beat numbers** in JSON:
   ```json
   {
     "command": "play",
     "deck_id": "deckB", 
     "parameters": {"start_at_beat": 592}
   }
   ```

3. **Preprocess for performance**:
   ```bash
   python preprocess.py mix_configs/my_mix.json
   # Smart analysis automatically handles bpm_match requirements
   ```

4. **Execute with perfect synchronization**:
   ```bash
   python main.py mix_configs/my_mix.json
   # Beat 592 plays at exactly the same musical location
   ```

### Architecture Benefits

- **Separation of concerns**: Heavy processing separated from performance
- **Smart analysis**: Automatically predicts complex tempo requirements  
- **Perfect synchronization**: Beat viewer and main engine use identical logic
- **Performance optimization**: Zero-latency playback with preprocessing
- **Flexibility**: JIT fallback for experimentation and edge cases

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

## Configuration

- `EQ_SMOOTHING_MS`: Global setting (in `config.py`) for how quickly EQ changes are interpolated when using `set_eq`. Lower values make transitions faster (default: 0.5ms). Does not affect `fade_eq`.