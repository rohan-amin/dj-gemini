"""
Adapter modules for integrating the loop system with external components.

This package provides adapter classes that allow the loop management system
to integrate with existing audio engine components like beat managers,
timing systems, and audio callbacks.
"""

from .beat_manager_adapter import BeatManagerAdapter

__all__ = ['BeatManagerAdapter']