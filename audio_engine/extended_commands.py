# Extended JSON commands for stem separation and scratch effects
# New commands for advanced DJ mixing capabilities

import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class CommandDefinition:
    """Definition of a command with validation"""
    name: str
    description: str
    required_params: List[str]
    optional_params: Dict[str, Any]  # param_name -> default_value
    deck_specific: bool = True
    example: Dict[str, Any] = None

class ExtendedCommandProcessor:
    """Processor for extended JSON commands"""
    
    def __init__(self):
        self.commands = self._define_commands()
        logger.info(f"Extended command processor initialized with {len(self.commands)} commands")
    
    def _define_commands(self) -> Dict[str, CommandDefinition]:
        """Define all extended commands"""
        commands = {}
        
        # Stem control commands
        commands['set_stem_volume'] = CommandDefinition(
            name="set_stem_volume",
            description="Set volume for specific stem",
            required_params=["stem", "volume"],
            optional_params={},
            deck_specific=True,
            example={
                "command": "set_stem_volume",
                "deck_id": "deckA",
                "parameters": {
                    "stem": "vocals",
                    "volume": 0.5
                }
            }
        )
        
        commands['fade_stem_volume'] = CommandDefinition(
            name="fade_stem_volume",
            description="Fade volume for specific stem over time",
            required_params=["stem", "target_volume"],
            optional_params={"duration_seconds": 2.0},
            deck_specific=True,
            example={
                "command": "fade_stem_volume",
                "deck_id": "deckA",
                "parameters": {
                    "stem": "drums",
                    "target_volume": 0.0,
                    "duration_seconds": 4.0
                }
            }
        )
        
        commands['set_stem_eq'] = CommandDefinition(
            name="set_stem_eq",
            description="Set EQ for specific stem",
            required_params=["stem"],
            optional_params={"low": None, "mid": None, "high": None},
            deck_specific=True,
            example={
                "command": "set_stem_eq",
                "deck_id": "deckA",
                "parameters": {
                    "stem": "bass",
                    "low": 1.5,
                    "mid": 0.8,
                    "high": 0.2
                }
            }
        )
        
        commands['fade_stem_eq'] = CommandDefinition(
            name="fade_stem_eq",
            description="Fade EQ for specific stem over time",
            required_params=["stem"],
            optional_params={
                "target_low": None, "target_mid": None, "target_high": None,
                "duration_seconds": 2.0
            },
            deck_specific=True,
            example={
                "command": "fade_stem_eq",
                "deck_id": "deckA",
                "parameters": {
                    "stem": "vocals",
                    "target_low": 0.2,
                    "target_mid": 0.1,
                    "target_high": 0.0,
                    "duration_seconds": 3.0
                }
            }
        )
        
        commands['solo_stem'] = CommandDefinition(
            name="solo_stem",
            description="Solo specific stem (mute all others)",
            required_params=["stem"],
            optional_params={"solo": True},
            deck_specific=True,
            example={
                "command": "solo_stem",
                "deck_id": "deckA",
                "parameters": {
                    "stem": "drums",
                    "solo": True
                }
            }
        )
        
        commands['mute_stem'] = CommandDefinition(
            name="mute_stem",
            description="Mute/unmute specific stem",
            required_params=["stem"],
            optional_params={"muted": True},
            deck_specific=True,
            example={
                "command": "mute_stem",
                "deck_id": "deckA",
                "parameters": {
                    "stem": "vocals",
                    "muted": True
                }
            }
        )
        
        commands['isolate_stems'] = CommandDefinition(
            name="isolate_stems",
            description="Play only specified stems (mute all others)",
            required_params=["stems"],
            optional_params={},
            deck_specific=True,
            example={
                "command": "isolate_stems",
                "deck_id": "deckA",
                "parameters": {
                    "stems": ["drums", "bass"]
                }
            }
        )
        
        commands['clear_stem_isolation'] = CommandDefinition(
            name="clear_stem_isolation",
            description="Clear stem isolation (play all stems)",
            required_params=[],
            optional_params={},
            deck_specific=True,
            example={
                "command": "clear_stem_isolation",
                "deck_id": "deckA",
                "parameters": {}
            }
        )
        
        # Scratch effect commands
        commands['start_scratch'] = CommandDefinition(
            name="start_scratch",
            description="Start scratch effect with specified pattern",
            required_params=["pattern"],
            optional_params={
                "start_position": None, "max_loops": 1, "bpm": None,
                "crossfader_curve": "sharp"
            },
            deck_specific=True,
            example={
                "command": "start_scratch",
                "deck_id": "deckA",
                "parameters": {
                    "pattern": "transformer_2beat",
                    "max_loops": 3,
                    "crossfader_curve": "sharp"
                }
            }
        )
        
        commands['stop_scratch'] = CommandDefinition(
            name="stop_scratch",
            description="Stop current scratch effect",
            required_params=[],
            optional_params={},
            deck_specific=True,
            example={
                "command": "stop_scratch",
                "deck_id": "deckA",
                "parameters": {}
            }
        )
        
        commands['scratch_stem'] = CommandDefinition(
            name="scratch_stem",
            description="Apply scratch effect to specific stem only",
            required_params=["stem", "pattern"],
            optional_params={"max_loops": 1, "volume": 1.0},
            deck_specific=True,
            example={
                "command": "scratch_stem",
                "deck_id": "deckA",
                "parameters": {
                    "stem": "drums",
                    "pattern": "crab",
                    "max_loops": 2,
                    "volume": 1.2
                }
            }
        )
        
        # Multi-deck stem mixing commands
        commands['stem_crossfade'] = CommandDefinition(
            name="stem_crossfade",
            description="Crossfade specific stem between two decks",
            required_params=["from_deck", "to_deck", "stem"],
            optional_params={"position": 0.5, "duration_seconds": 0.0, "curve": "linear"},
            deck_specific=False,
            example={
                "command": "stem_crossfade",
                "parameters": {
                    "from_deck": "deckA",
                    "to_deck": "deckB", 
                    "stem": "vocals",
                    "position": 0.3,
                    "curve": "smooth"
                }
            }
        )
        
        commands['deck_crossfade'] = CommandDefinition(
            name="deck_crossfade",
            description="Crossfade between two decks",
            required_params=["from_deck", "to_deck"],
            optional_params={"duration_seconds": 4.0, "curve": "linear", "position": None},
            deck_specific=False,
            example={
                "command": "deck_crossfade",
                "parameters": {
                    "from_deck": "deckA",
                    "to_deck": "deckB",
                    "duration_seconds": 8.0,
                    "curve": "smooth"
                }
            }
        )
        
        commands['add_stem_routing'] = CommandDefinition(
            name="add_stem_routing",
            description="Route stem from one deck to another",
            required_params=["source_deck", "source_stem", "target_deck", "target_stem"],
            optional_params={"mix_ratio": 1.0},
            deck_specific=False,
            example={
                "command": "add_stem_routing",
                "parameters": {
                    "source_deck": "deckA",
                    "source_stem": "vocals",
                    "target_deck": "deckB",
                    "target_stem": "vocals",
                    "mix_ratio": 0.6
                }
            }
        )
        
        commands['remove_stem_routing'] = CommandDefinition(
            name="remove_stem_routing",
            description="Remove stem routing between decks",
            required_params=["source_deck", "source_stem", "target_deck", "target_stem"],
            optional_params={},
            deck_specific=False,
            example={
                "command": "remove_stem_routing",
                "parameters": {
                    "source_deck": "deckA",
                    "source_stem": "vocals",
                    "target_deck": "deckB", 
                    "target_stem": "vocals"
                }
            }
        )
        
        # Advanced mixing commands
        commands['harmonic_mix'] = CommandDefinition(
            name="harmonic_mix",
            description="Enable harmonic mixing between compatible stems",
            required_params=["deck_a", "deck_b"],
            optional_params={"key_match": True, "tempo_sync": True, "crossfade_duration": 8.0},
            deck_specific=False,
            example={
                "command": "harmonic_mix",
                "parameters": {
                    "deck_a": "deckA",
                    "deck_b": "deckB",
                    "key_match": True,
                    "tempo_sync": True,
                    "crossfade_duration": 12.0
                }
            }
        )
        
        commands['stem_sync'] = CommandDefinition(
            name="stem_sync",
            description="Synchronize stem timing across decks",
            required_params=["stem", "reference_deck"],
            optional_params={"sync_decks": None},  # If None, sync all decks
            deck_specific=False,
            example={
                "command": "stem_sync",
                "parameters": {
                    "stem": "drums",
                    "reference_deck": "deckA",
                    "sync_decks": ["deckB", "deckC"]
                }
            }
        )
        
        # Filter and effect commands
        commands['stem_filter_sweep'] = CommandDefinition(
            name="stem_filter_sweep",
            description="Apply filter sweep to specific stems",
            required_params=["stems", "filter_type"],
            optional_params={
                "start_freq": 20.0, "end_freq": 20000.0, 
                "duration_seconds": 4.0, "resonance": 0.7
            },
            deck_specific=True,
            example={
                "command": "stem_filter_sweep",
                "deck_id": "deckA",
                "parameters": {
                    "stems": ["vocals", "other"],
                    "filter_type": "lowpass",
                    "start_freq": 20000.0,
                    "end_freq": 200.0,
                    "duration_seconds": 6.0,
                    "resonance": 1.2
                }
            }
        )
        
        commands['add_stem_reverb'] = CommandDefinition(
            name="add_stem_reverb",
            description="Add reverb effect to specific stem",
            required_params=["stem"],
            optional_params={
                "room_size": 0.5, "damping": 0.5, 
                "wet_level": 0.3, "dry_level": 0.7
            },
            deck_specific=True,
            example={
                "command": "add_stem_reverb",
                "deck_id": "deckA",
                "parameters": {
                    "stem": "vocals",
                    "room_size": 0.8,
                    "wet_level": 0.4,
                    "dry_level": 0.6
                }
            }
        )
        
        # Stem analysis and detection commands
        commands['analyze_stem_key'] = CommandDefinition(
            name="analyze_stem_key",
            description="Analyze musical key of specific stem",
            required_params=["stem"],
            optional_params={"algorithm": "krumhansl"},
            deck_specific=True,
            example={
                "command": "analyze_stem_key",
                "deck_id": "deckA",
                "parameters": {
                    "stem": "vocals",
                    "algorithm": "krumhansl"
                }
            }
        )
        
        commands['detect_stem_energy'] = CommandDefinition(
            name="detect_stem_energy",
            description="Detect energy level changes in stem for auto-triggering",
            required_params=["stem"],
            optional_params={"threshold": 0.1, "callback_action": None},
            deck_specific=True,
            example={
                "command": "detect_stem_energy",
                "deck_id": "deckA",
                "parameters": {
                    "stem": "drums",
                    "threshold": 0.15,
                    "callback_action": "trigger_effect_on_kick"
                }
            }
        )
        
        return commands
    
    def validate_command(self, command_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate command structure and parameters"""
        try:
            command_name = command_data.get("command")
            if not command_name:
                return False, "Missing 'command' field"
            
            if command_name not in self.commands:
                return False, f"Unknown command: {command_name}"
            
            cmd_def = self.commands[command_name]
            parameters = command_data.get("parameters", {})
            
            # Check deck_id requirement
            if cmd_def.deck_specific and not command_data.get("deck_id"):
                return False, f"Command '{command_name}' requires 'deck_id'"
            
            # Check required parameters
            for param in cmd_def.required_params:
                if param not in parameters:
                    return False, f"Missing required parameter: {param}"
            
            # Validate specific parameter types/values
            validation_error = self._validate_parameters(command_name, parameters)
            if validation_error:
                return False, validation_error
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def _validate_parameters(self, command_name: str, parameters: Dict[str, Any]) -> Optional[str]:
        """Validate specific parameters for commands"""
        
        # Volume parameters
        if "volume" in parameters:
            volume = parameters["volume"]
            if not isinstance(volume, (int, float)) or volume < 0.0 or volume > 2.0:
                return "Volume must be between 0.0 and 2.0"
        
        if "target_volume" in parameters:
            volume = parameters["target_volume"]
            if not isinstance(volume, (int, float)) or volume < 0.0 or volume > 2.0:
                return "Target volume must be between 0.0 and 2.0"
        
        # EQ parameters
        for eq_param in ["low", "mid", "high", "target_low", "target_mid", "target_high"]:
            if eq_param in parameters and parameters[eq_param] is not None:
                value = parameters[eq_param]
                if not isinstance(value, (int, float)) or value < 0.0 or value > 3.0:
                    return f"{eq_param} must be between 0.0 and 3.0"
        
        # Duration parameters
        if "duration_seconds" in parameters:
            duration = parameters["duration_seconds"]
            if not isinstance(duration, (int, float)) or duration < 0.0:
                return "Duration must be positive"
        
        # Stem name validation
        if "stem" in parameters:
            stem = parameters["stem"]
            valid_stems = ["vocals", "drums", "bass", "other"]
            if stem not in valid_stems:
                return f"Invalid stem '{stem}'. Must be one of: {valid_stems}"
        
        if "stems" in parameters:
            stems = parameters["stems"]
            if not isinstance(stems, list):
                return "Stems parameter must be a list"
            valid_stems = ["vocals", "drums", "bass", "other"]
            for stem in stems:
                if stem not in valid_stems:
                    return f"Invalid stem '{stem}'. Must be one of: {valid_stems}"
        
        # Scratch pattern validation
        if "pattern" in parameters:
            pattern = parameters["pattern"]
            valid_patterns = [
                "baby", "chirp", "transformer_2beat", "crab", "flare", 
                "orbit", "tear", "scribble", "hydroplane", "drill"
            ]
            if pattern not in valid_patterns:
                return f"Invalid scratch pattern '{pattern}'. Must be one of: {valid_patterns}"
        
        # Crossfader curve validation
        if "curve" in parameters or "crossfader_curve" in parameters:
            curve = parameters.get("curve") or parameters.get("crossfader_curve")
            valid_curves = ["linear", "smooth", "sharp", "exponential"]
            if curve not in valid_curves:
                return f"Invalid curve '{curve}'. Must be one of: {valid_curves}"
        
        # Position parameters
        if "position" in parameters:
            position = parameters["position"]
            if not isinstance(position, (int, float)) or position < 0.0 or position > 1.0:
                return "Position must be between 0.0 and 1.0"
        
        # Mix ratio validation
        if "mix_ratio" in parameters:
            ratio = parameters["mix_ratio"]
            if not isinstance(ratio, (int, float)) or ratio < 0.0 or ratio > 1.0:
                return "Mix ratio must be between 0.0 and 1.0"
        
        # Loop count validation
        if "max_loops" in parameters:
            loops = parameters["max_loops"]
            if not isinstance(loops, int) or loops < 1:
                return "Max loops must be a positive integer"
        
        return None
    
    def get_command_help(self, command_name: str = None) -> Dict[str, Any]:
        """Get help information for commands"""
        if command_name:
            if command_name not in self.commands:
                return {"error": f"Unknown command: {command_name}"}
            
            cmd_def = self.commands[command_name]
            return {
                "name": cmd_def.name,
                "description": cmd_def.description,
                "required_parameters": cmd_def.required_params,
                "optional_parameters": cmd_def.optional_params,
                "deck_specific": cmd_def.deck_specific,
                "example": cmd_def.example
            }
        else:
            # Return all commands
            return {
                "total_commands": len(self.commands),
                "commands": {
                    name: {
                        "description": cmd_def.description,
                        "deck_specific": cmd_def.deck_specific,
                        "required_params": len(cmd_def.required_params),
                        "optional_params": len(cmd_def.optional_params)
                    }
                    for name, cmd_def in self.commands.items()
                }
            }
    
    def get_command_categories(self) -> Dict[str, List[str]]:
        """Get commands organized by category"""
        categories = {
            "Stem Control": [
                "set_stem_volume", "fade_stem_volume", "set_stem_eq", "fade_stem_eq",
                "solo_stem", "mute_stem", "isolate_stems", "clear_stem_isolation"
            ],
            "Scratch Effects": [
                "start_scratch", "stop_scratch", "scratch_stem"
            ],
            "Multi-Deck Mixing": [
                "stem_crossfade", "deck_crossfade", "add_stem_routing", 
                "remove_stem_routing", "harmonic_mix", "stem_sync"
            ],
            "Effects & Filters": [
                "stem_filter_sweep", "add_stem_reverb"
            ],
            "Analysis & Detection": [
                "analyze_stem_key", "detect_stem_energy"
            ]
        }
        
        return categories
    
    def generate_example_script(self) -> Dict[str, Any]:
        """Generate example script using extended commands"""
        return {
            "mix_name": "Advanced Stem Mixing Example",
            "description": "Demonstrates advanced stem separation and scratch effects",
            "actions": [
                {
                    "action_id": "load_track_a",
                    "command": "load_track",
                    "deck_id": "deckA",
                    "parameters": {
                        "filepath": "audio_tracks/song1.mp3"
                    },
                    "trigger": {"type": "script_start"}
                },
                {
                    "action_id": "load_track_b", 
                    "command": "load_track",
                    "deck_id": "deckB",
                    "parameters": {
                        "filepath": "audio_tracks/song2.mp3"
                    },
                    "trigger": {"type": "script_start"}
                },
                {
                    "action_id": "play_deck_a",
                    "command": "play",
                    "deck_id": "deckA",
                    "parameters": {"start_at_beat": 1},
                    "trigger": {"type": "script_start"}
                },
                {
                    "action_id": "isolate_drums_bass",
                    "command": "isolate_stems",
                    "deck_id": "deckA",
                    "parameters": {
                        "stems": ["drums", "bass"]
                    },
                    "trigger": {
                        "type": "on_deck_beat",
                        "source_deck_id": "deckA",
                        "beat_number": 32
                    }
                },
                {
                    "action_id": "scratch_drums",
                    "command": "scratch_stem",
                    "deck_id": "deckA",
                    "parameters": {
                        "stem": "drums",
                        "pattern": "transformer_2beat",
                        "max_loops": 2
                    },
                    "trigger": {
                        "type": "on_deck_beat",
                        "source_deck_id": "deckA", 
                        "beat_number": 64
                    }
                },
                {
                    "action_id": "start_crossfade",
                    "command": "deck_crossfade",
                    "parameters": {
                        "from_deck": "deckA",
                        "to_deck": "deckB",
                        "duration_seconds": 8.0,
                        "curve": "smooth"
                    },
                    "trigger": {
                        "type": "on_deck_beat",
                        "source_deck_id": "deckA",
                        "beat_number": 96
                    }
                },
                {
                    "action_id": "vocal_filter_sweep",
                    "command": "stem_filter_sweep",
                    "deck_id": "deckB",
                    "parameters": {
                        "stems": ["vocals"],
                        "filter_type": "lowpass",
                        "start_freq": 20000.0,
                        "end_freq": 500.0,
                        "duration_seconds": 4.0
                    },
                    "trigger": {
                        "type": "on_deck_beat",
                        "source_deck_id": "deckB",
                        "beat_number": 32
                    }
                }
            ]
        }

# Global instance
extended_command_processor = ExtendedCommandProcessor()

# Convenience functions
def validate_extended_command(command_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate extended command"""
    return extended_command_processor.validate_command(command_data)

def get_extended_command_help(command_name: str = None) -> Dict[str, Any]:
    """Get help for extended commands"""
    return extended_command_processor.get_command_help(command_name)

def get_command_categories() -> Dict[str, List[str]]:
    """Get command categories"""
    return extended_command_processor.get_command_categories()