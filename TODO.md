# DJ Gemini TODO

## 🚀 High Priority
- [x] Add tempo change functionality ✅ **COMPLETED**
  - [x] Integrate Rubber Band for high-quality pitch-preserving time-stretching
  - [x] Add `set_tempo` command to JSON schema
  - [x] Update Deck class to handle tempo changes
  - [x] Implement audio caching for instant tempo changes
  - [x] Fix beat position scaling for accurate timing at all tempos
- [x] Add tempo ramping functionality ✅ **COMPLETED**
  - [x] Implement `ramp_tempo` command with start/end beats and BPMs
  - [x] Add smooth BPM transitions over specified beat ranges
  - [x] Support linear and exponential curve types
  - [x] Real-time audio speed changes to match BPM ramps
  - [x] Fix beat position recalculation during ramps
  - [x] Ensure triggers and loops work correctly during ramps
  - [x] Eliminate audio hiccups and timing issues
- [x] Add volume control per deck ✅ **COMPLETED**
  - [x] Implement `set_volume` command (0.0 to 1.0)
  - [x] Add `fade_volume` command with duration
  - [x] Implement `crossfade` between decks
  - [x] Fix volume attribute naming consistency
- [ ] Implement stem separation
  - [ ] Add Spleeter integration
  - [ ] Create stem loading commands
  - [ ] Update AudioAnalyzer for stem processing

## 🔧 Medium Priority
- [x] Add volume control per deck ✅ **COMPLETED**
- [x] Implement crossfading between decks ✅ **COMPLETED**
- [x] Add fade_eq command for EQ fade in/out effects ✅ **COMPLETED**
- [x] Add fast EQ smoothing for set_eq (configurable, click-free) ✅ **COMPLETED**
- [x] Add global EQ_SMOOTHING_MS config option ✅ **COMPLETED**
- [x] Fix bug: fade_eq not processed in audio callback ✅ **COMPLETED**
- [x] Update README and documentation for new EQ features ✅ **COMPLETED**
- [ ] Support more audio formats (WAV, FLAC, OGG)
- [ ] Add beatmatching functionality
  - [ ] Implement BPM synchronization between decks
  - [ ] Add phase alignment for beat-sync
  - [ ] Create beatmatching commands

## 💡 Low Priority / Future Ideas
- [ ] Web interface for script creation
- [ ] Real-time BPM detection
- [ ] MIDI controller support
- [ ] Audio visualization
- [ ] Export mix as single audio file
- [ ] Add more curve types for tempo ramps (sine, cosine, custom)
- [ ] Implement tempo ramping with different curves per deck
- [ ] Document, test, and improve `seek_and_play` command

## 🐛 Bugs to Fix
- [ ] Handle edge case when audio file is corrupted
- [ ] Improve error messages for invalid JSON
- [ ] Add better logging for debugging
- [ ] Optimize audio callback performance during tempo ramps

## ✅ Completed
- [x] Implement loop queue system (fixes race conditions)
- [x] Add stop_at_beat functionality (clean stopping)
- [x] Create comprehensive README with all commands
- [x] Add cue point support
- [x] Multi-deck mixing system
- [x] **Professional tempo control with Rubber Band integration**
- [x] **Tempo ramping with smooth BPM transitions**
- [x] **Volume control and crossfading system**
- [x] **Real-time audio speed changes during tempo ramps**
- [x] **Accurate loop timing during tempo changes**

## 📝 Implementation Notes
- **Tempo changes**: ✅ **COMPLETED** - Using Rubber Band (pyrubberband) for professional-grade pitch-preserving tempo changes
- **Tempo ramping**: ✅ **COMPLETED** - Smooth BPM transitions with real-time audio speed changes
- **Volume control**: ✅ **COMPLETED** - Per-deck volume with fade and crossfade support
- **Stem separation**: Spleeter provides 4-stem separation
- **Web interface**: Consider Flask for simple UI
- **Audio formats**: soundfile library for additional formats

## 🔗 Related Issues
- None yet (create GitHub issues for detailed tracking) 