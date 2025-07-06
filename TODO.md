# DJ Gemini TODO

## 🚀 High Priority
- [ ] Add tempo change functionality
  - [ ] Integrate librosa for time-stretching
  - [ ] Add `set_tempo` command to JSON schema
  - [ ] Update Deck class to handle tempo changes
- [ ] Implement stem separation
  - [ ] Add Spleeter integration
  - [ ] Create stem loading commands
  - [ ] Update AudioAnalyzer for stem processing

## 🔧 Medium Priority
- [ ] Add volume control per deck
- [ ] Implement crossfading between decks
- [ ] Add fade in/out effects
- [ ] Support more audio formats (WAV, FLAC, OGG)

## 💡 Low Priority / Future Ideas
- [ ] Web interface for script creation
- [ ] Real-time BPM detection
- [ ] MIDI controller support
- [ ] Audio visualization
- [ ] Export mix as single audio file

## 🐛 Bugs to Fix
- [ ] Handle edge case when audio file is corrupted
- [ ] Improve error messages for invalid JSON
- [ ] Add better logging for debugging

## ✅ Completed
- [x] Implement loop queue system (fixes race conditions)
- [x] Add stop_at_beat functionality (clean stopping)
- [x] Create comprehensive README with all commands
- [x] Add cue point support
- [x] Multi-deck mixing system

## 📝 Implementation Notes
- **Tempo changes**: Use librosa.effects.time_stretch
- **Stem separation**: Spleeter provides 4-stem separation
- **Web interface**: Consider Flask for simple UI
- **Audio formats**: soundfile library for additional formats

## 🔗 Related Issues
- None yet (create GitHub issues for detailed tracking) 