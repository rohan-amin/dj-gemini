# Loop Management Implementation Plan

## Phase 1: Core Loop Playback
- Add loop state to `Deck` (start/end frames, iteration counters).
- Implement `activate_loop`/`deactivate_loop` methods.
- Build loop-aware audio chunk generation that seamlessly wraps frames and logs iteration/jump/completion events.
- Ensure playback works with existing EQ and stem mixing.

## Phase 2: Engine Integration
- Extend `AudioEngine` validation and scheduling to support `activate_loop` and `on_loop_complete` triggers.
- Maintain mapping from loop IDs to completion actions.
- Provide `handle_loop_complete` for decks to notify engine.

## Phase 3: Loop Chaining & Multiple Actions
- Support sequential loop activation via `on_loop_complete` triggers.
- Allow multiple actions to fire when a loop completes.
- Verify cross-deck targeting and event ordering.

## Phase 4: Configuration & Validation
- Validate JSON parameters (start beat, length, repetitions).
- Ensure beat/frame conversions handle fractional beats accurately.
- Add bounds checking against track length.

## Phase 5: Observability & Debugging
- Implement INFO-level logging for loop lifecycle events and jumps.
- Expose loop state via engine for monitoring.
- Record timing metrics for loop accuracy.

## Phase 6: Performance & Safety
- Optimize loop processing in the producer thread to avoid audio dropouts.
- Ensure thread-safe access to loop state.
- Add tests for tight loops and edge cases.

Each phase can be developed and merged independently, enabling incremental delivery of the full loop management system.

## Status
- Phase 1 and Phase 2 implemented in current iteration.
