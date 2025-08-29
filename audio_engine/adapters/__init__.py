"""
Adapters package for the DJ Gemini audio engine.
Contains adapters that bridge clean interfaces with existing implementations.
"""

from .beat_manager_adapter import BeatManagerAdapter
from .deck_executor_adapter import DeckExecutorAdapter

__all__ = [
    'BeatManagerAdapter',
    'DeckExecutorAdapter'
]