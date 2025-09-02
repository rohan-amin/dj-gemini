# DJ Gemini Audio Pipeline Architecture

## Complete Audio Flow: From Source to Output

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AUDIO SOURCE  │    │  PREPROCESSING  │    │  REAL-TIME DJ   │
│                 │    │   (OPTIONAL)    │    │   PROCESSING    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Original Audio  │───▶│ Stem Separation │───▶│ Stem Selection  │
│   (.mp3, etc)   │    │   (Demucs/      │    │ & Volume Mix    │
│                 │    │   Spleeter)     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │                       │
                               ▼                       ▼
                    ┌─────────────────┐    ┌─────────────────┐
                    │  Cached Stems   │    │  Mixed Audio    │
                    │                 │    │   (4 stems →    │
                    │ • vocals.npy    │    │    1 signal)    │
                    │ • drums.npy     │    │                 │
                    │ • bass.npy      │    │                 │
                    │ • other.npy     │    │                 │
                    └─────────────────┘    └─────────────────┘
                                                   │
                                                   ▼
                                       ┌─────────────────┐
                                       │ TEMPO CONTROL   │
                                       │                 │
                                       │ • Turntable:    │
                                       │   pitch+tempo   │
                                       │ • Pitch Preserve│
                                       │   RubberBand    │
                                       └─────────────────┘
                                                   │
                                                   ▼
                                       ┌─────────────────┐
                                       │ EQ PROCESSING   │
                                       │                 │
                                       │ • 3-band EQ     │
                                       │ • Kill switches │
                                       │ • Cascaded      │
                                       │   biquad filters│
                                       └─────────────────┘
                                                   │
                                                   ▼
                                       ┌─────────────────┐
                                       │ AUDIO OUTPUT    │
                                       │                 │
                                       │ sounddevice →   │
                                       │ speakers/       │
                                       │ headphones      │
                                       └─────────────────┘
```

## Detailed Component Flow

### 1. PREPROCESSING STAGE (Offline)
**Purpose**: Heavy computational work done ahead of time

```python
# preprocess_stems.py
audio_file.mp3 → Demucs/Spleeter → cache/song_name/stems/
                                   ├── vocals.npy
                                   ├── drums.npy  
                                   ├── bass.npy
                                   └── other.npy
```

### 2. REAL-TIME STEM MIXING
**Purpose**: Live DJ performance with stem control

```python
# In audio callback (every 1024 samples)
def audio_callback():
    # Load cached stems
    vocals = load_stem('vocals') * vocal_volume
    drums = load_stem('drums') * drum_volume  
    bass = load_stem('bass') * bass_volume
    other = load_stem('other') * other_volume
    
    # Mix stems together
    mixed_audio = vocals + drums + bass + other
    
    # Continue to tempo processing...
```

### 3. TEMPO PROCESSING
**Purpose**: Speed changes with/without pitch preservation

```python
# Two modes:
if pitch_preserve_enabled:
    # RubberBand streaming (maintains pitch)
    audio = rubberband_stretch(mixed_audio, tempo_ratio)
else:
    # Turntable style (pitch changes with tempo)  
    audio = interpolate_samples(mixed_audio, tempo_ratio)
```

### 4. EQ PROCESSING
**Purpose**: Frequency shaping and kill switches

```python
# Cascaded biquad filters
audio = low_shelf_filter.process(audio)
audio = low_shelf_filter2.process(audio)    # Kill mode only
audio = peaking_filter.process(audio)
audio = peaking_filter2.process(audio)      # Kill mode only  
audio = high_shelf_filter.process(audio)
audio = high_shelf_filter2.process(audio)   # Kill mode only
```

### 5. OUTPUT
**Purpose**: Send to audio device

```python
# sounddevice callback
outdata[:, 0] = final_audio_samples
```

## Data Flow Summary

```
INPUT:     song.mp3 (stereo, full mix)
           ↓ (preprocessing)
STEMS:     4 × mono numpy arrays (vocals, drums, bass, other)
           ↓ (real-time mixing)
MIXED:     1 × mono signal (weighted sum of stems)
           ↓ (tempo processing) 
TEMPO:     1 × mono signal (time-stretched)
           ↓ (EQ processing)
EQ:        1 × mono signal (frequency-shaped)
           ↓ (audio output)
OUTPUT:    speakers/headphones
```

## Performance Characteristics

| Stage | Processing | Latency | CPU Usage |
|-------|------------|---------|-----------|
| Stem Separation | Offline | N/A | Very High |
| Stem Mixing | Real-time | ~2ms | Very Low |
| Tempo (Turntable) | Real-time | ~2ms | Very Low |  
| Tempo (RubberBand) | Real-time | ~10ms | Medium |
| EQ Processing | Real-time | ~1ms | Very Low |
| **Total Real-time** | | **~15ms** | **Medium** |

## Cache Directory Structure

```
cache/
└── songs/
    └── song_name_hash/
        ├── beats.json          # Beat analysis (legacy)
        └── stems/              # NEW: Stem cache
            ├── vocals.npy
            ├── drums.npy
            ├── bass.npy
            └── other.npy
```

## Integration Points

### Beat Viewer Integration
- Load stems instead of original audio
- Mix stems based on UI sliders
- Apply existing tempo + EQ pipeline

### Main Engine Integration  
- Use stem mixing in Deck class
- Add stem volume controls to audio processing
- Maintain backward compatibility with non-stem tracks