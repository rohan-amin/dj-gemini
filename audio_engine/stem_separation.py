# Stem separation system using Demucs for dj-gemini
# Separates tracks into vocals, drums, bass, and other stems

import os
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import hashlib
import json
from dataclasses import dataclass

try:
    import demucs.separate
    import demucs.pretrained
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False
    logging.warning("Demucs not available - stem separation will be disabled")

logger = logging.getLogger(__name__)

@dataclass
class StemData:
    """Data for individual stem"""
    name: str
    audio_data: np.ndarray
    sample_rate: int
    duration_seconds: float
    rms_level: float  # RMS level for automatic gain balancing

@dataclass 
class StemSeparationResult:
    """Result of stem separation"""
    stems: Dict[str, StemData]
    original_file: str
    separation_method: str
    processing_time: float
    total_samples: int
    sample_rate: int
    success: bool = True

class StemSeparator:
    """High-quality stem separation using Demucs with Spleeter fallback"""
    
    STEM_NAMES = ['vocals', 'drums', 'bass', 'other']
    
    def __init__(self, cache_dir: str = None, model_name: str = "htdemucs"):
        # cache_dir parameter kept for compatibility but not used (uses unified song cache)
        self.model_name = model_name
        
        # Initialize Demucs
        self._demucs_model = None
        self._demucs_available = DEMUCS_AVAILABLE
        
        if self._demucs_available:
            try:
                # Initialize Demucs
                logger.info(f"Initializing Demucs with model: {model_name}")
                self._demucs_model = demucs.pretrained.get_model(model_name)
                logger.info("Demucs model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Demucs: {e}")
                self._demucs_available = False
        
        if not self._demucs_available:
            raise RuntimeError("Demucs not available. Install demucs: pip install demucs")
        
        logger.info(f"Stem separator initialized with Demucs")
    
    def separate_file(self, audio_file: str, force_reprocess: bool = False) -> Optional[StemSeparationResult]:
        """Separate audio file into stems"""
        try:
            audio_path = Path(audio_file)
            if not audio_path.exists():
                logger.error(f"Audio file not found: {audio_file}")
                return None
            
            # Check cache first
            cache_info = self._get_cache_info(audio_file)
            if not force_reprocess and self._is_cached(cache_info):
                logger.info(f"Loading stems from cache: {audio_file}")
                return self._load_cached_stems(cache_info)
            
            # Perform separation
            logger.info(f"Separating stems for: {audio_file}")
            start_time = time.time()
            
            if self._demucs_available and self._demucs_model:
                result = self._separate_with_demucs(audio_file, cache_info)
            else:
                logger.error("Demucs not available")
                return None
            
            processing_time = time.time() - start_time
            if result:
                result.processing_time = processing_time
                logger.info(f"Stem separation completed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Stem separation failed for {audio_file}: {e}")
            return None
    
    def separate_audio_data(self, audio_data: np.ndarray, sample_rate: int, 
                           source_file: str = "unknown") -> Optional[StemSeparationResult]:
        """Separate audio data directly (for real-time processing)"""
        try:
            if not DEMUCS_AVAILABLE:
                logger.error("Real-time stem separation requires Demucs")
                return None
            
            logger.info(f"Separating audio data: {len(audio_data)} samples at {sample_rate}Hz")
            start_time = time.time()
            
            # Use Demucs for real-time separation
            import torch
            import torchaudio
            
            # Convert to tensor and add batch dimension
            audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)
            if len(audio_tensor.shape) == 2:  # Add channel dimension if mono
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Separate using model
            with torch.no_grad():
                separated = demucs.separate.apply_model(self._demucs_model, audio_tensor, device='cpu')
            
            # Convert results to dictionary
            stems_dict = {}
            stem_names = ['drums', 'bass', 'other', 'vocals']  # htdemucs order
            for i, stem_name in enumerate(stem_names):
                stems_dict[stem_name] = separated[0, i].numpy()
            
            # Process results
            stems = {}
            for stem_name in self.STEM_NAMES:
                if stem_name in stems_dict:
                    stem_audio = stems_dict[stem_name]
                    
                    # Calculate RMS level
                    rms_level = np.sqrt(np.mean(stem_audio ** 2))
                    
                    stems[stem_name] = StemData(
                        name=stem_name,
                        audio_data=stem_audio,
                        sample_rate=sample_rate,
                        duration_seconds=len(stem_audio) / sample_rate,
                        rms_level=rms_level
                    )
            
            processing_time = time.time() - start_time
            
            return StemSeparationResult(
                stems=stems,
                original_file=source_file,
                separation_method="demucs_realtime",
                processing_time=processing_time,
                total_samples=len(audio_data),
                sample_rate=sample_rate
            )
            
        except Exception as e:
            logger.error(f"Real-time stem separation failed: {e}")
            return None
    
    def _separate_with_demucs(self, audio_file: str, cache_info: Dict) -> Optional[StemSeparationResult]:
        """Separate using Demucs"""
        try:
            
            # Load and separate audio file
            import torch
            import torchaudio
            
            print(f"    🎵 Loading audio file...")
            # Load audio file
            audio_tensor, audio_sample_rate = torchaudio.load(str(audio_file))
            audio_duration = len(audio_tensor[0]) / audio_sample_rate
            
            print(f"    📊 Loaded {audio_duration:.1f}s of audio at {audio_sample_rate}Hz")
            print(f"    🔄 Separating stems with Demucs (this may take 2-5 minutes)...")
            print(f"    ⏳ Processing... (grab a coffee! ☕)")
            
            # Detect best device for Apple Silicon
            if torch.backends.mps.is_available():
                device = 'mps'  # Apple Silicon GPU
                print(f"    🚀 Using Apple Silicon GPU acceleration (MPS)")
            elif torch.cuda.is_available():
                device = 'cuda'  # NVIDIA GPU
                print(f"    🚀 Using CUDA GPU acceleration")
            else:
                device = 'cpu'
                print(f"    💻 Using CPU (no GPU acceleration available)")
            
            # Move model and data to device
            self._demucs_model = self._demucs_model.to(device)
            audio_tensor = audio_tensor.to(device)
            
            # Separate using model
            start_separation = time.time()
            with torch.no_grad():
                separated = demucs.separate.apply_model(self._demucs_model, audio_tensor.unsqueeze(0), device=device)
            
            separation_time = time.time() - start_separation
            print(f"    ✅ Separation completed in {separation_time:.1f}s")
            
            # Convert results to dictionary (move back to CPU for numpy conversion)
            stems_dict = {}
            stem_names = ['drums', 'bass', 'other', 'vocals']  # htdemucs order
            for i, stem_name in enumerate(stem_names):
                stems_dict[stem_name] = separated[0, i].cpu().numpy()
            
            # Process and cache results
            print(f"    💾 Caching separated stems...")
            stems = {}
            sample_rate = audio_sample_rate
            total_samples = 0
            
            for stem_name in self.STEM_NAMES:
                print(f"    📁 Processing {stem_name} stem...")
                if stem_name in stems_dict:
                    stem_audio = stems_dict[stem_name]
                    
                    # Ensure consistent stereo format: (channels, samples)
                    if stem_audio.ndim == 1:
                        # Mono - convert to stereo by duplicating
                        stem_audio = np.stack([stem_audio, stem_audio])
                    elif stem_audio.ndim == 2:
                        if stem_audio.shape[1] == 2 and stem_audio.shape[0] > stem_audio.shape[1]:
                            # (samples, 2) -> (2, samples)
                            stem_audio = stem_audio.T
                        elif stem_audio.shape[0] == 2:
                            # Already (2, samples) - keep as is
                            pass
                        else:
                            # Other format - take first channel and duplicate for stereo
                            first_channel = stem_audio[0] if stem_audio.shape[0] < stem_audio.shape[1] else stem_audio[:, 0]
                            stem_audio = np.stack([first_channel, first_channel])
                    
                    if sample_rate is None:
                        sample_rate = stems_dict.get('sample_rate', 44100)
                        total_samples = stem_audio.shape[1] if stem_audio.ndim == 2 else len(stem_audio)
                    
                    # Calculate RMS level
                    rms_level = np.sqrt(np.mean(stem_audio ** 2))
                    
                    # Ensure cache directory exists
                    cache_info['stem_dir'].mkdir(parents=True, exist_ok=True)
                    
                    # Cache stem in (2, samples) format
                    stem_cache_path = cache_info['stem_dir'] / f"{stem_name}.npy"
                    np.save(stem_cache_path, stem_audio.astype(np.float32))
                    
                    stems[stem_name] = StemData(
                        name=stem_name,
                        audio_data=stem_audio,
                        sample_rate=sample_rate,
                        duration_seconds=len(stem_audio) / sample_rate,
                        rms_level=rms_level
                    )
            
            # Save metadata
            metadata = {
                'stems': list(stems.keys()),
                'sample_rate': sample_rate,
                'total_samples': total_samples,
                'separation_method': 'demucs',
                'model_name': self.model_name,
                'original_file': audio_file
            }
            
            metadata_path = cache_info['stem_dir'] / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"    🎉 Successfully separated {len(stems)} stems and cached to disk!")
            
            return StemSeparationResult(
                stems=stems,
                original_file=audio_file,
                separation_method="demucs",
                processing_time=separation_time,  # Use the separation time we already calculated
                total_samples=total_samples,
                sample_rate=sample_rate,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Demucs separation failed: {e}")
            return None
    
    def _get_cache_info(self, audio_file: str) -> Dict:
        """Get cache information for audio file"""
        # Use the same cache structure as the main system
        import config as app_config
        
        # Get the unified song cache directory (same as beats, tempo, pitch)
        song_cache_dir = Path(app_config.get_song_cache_dir(audio_file))
        stems_dir = song_cache_dir / "stems"
        
        file_path = Path(audio_file).resolve()
        
        return {
            'stem_dir': stems_dir,
            'original_file': str(file_path)
        }
    
    def _is_cached(self, cache_info: Dict) -> bool:
        """Check if stems are already cached"""
        stem_dir = cache_info['stem_dir']
        
        if not stem_dir.exists():
            return False
        
        # Check for metadata and all stem files
        metadata_path = stem_dir / "metadata.json"
        if not metadata_path.exists():
            return False
        
        for stem_name in self.STEM_NAMES:
            stem_path = stem_dir / f"{stem_name}.npy"
            if not stem_path.exists():
                return False
        
        return True
    
    def _load_cached_stems(self, cache_info: Dict) -> Optional[StemSeparationResult]:
        """Load stems from cache"""
        try:
            stem_dir = cache_info['stem_dir']
            
            # Load metadata
            metadata_path = stem_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load stem data
            stems = {}
            for stem_name in self.STEM_NAMES:
                stem_path = stem_dir / f"{stem_name}.npy"
                if stem_path.exists():
                    stem_audio = np.load(stem_path)
                    
                    # Calculate RMS level
                    rms_level = np.sqrt(np.mean(stem_audio ** 2))
                    
                    stems[stem_name] = StemData(
                        name=stem_name,
                        audio_data=stem_audio,
                        sample_rate=metadata['sample_rate'],
                        duration_seconds=len(stem_audio) / metadata['sample_rate'],
                        rms_level=rms_level
                    )
            
            return StemSeparationResult(
                stems=stems,
                original_file=metadata['original_file'],
                separation_method=metadata['separation_method'],
                processing_time=0.0,  # Cached, no processing time
                total_samples=metadata['total_samples'],
                sample_rate=metadata['sample_rate']
            )
            
        except Exception as e:
            logger.error(f"Failed to load cached stems: {e}")
            return None
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem safety"""
        import re
        return re.sub(r'[<>:"/\\|?*\s]', '_', filename)
    
    def get_cached_stems_info(self, audio_file: str) -> Optional[Dict]:
        """Get information about cached stems"""
        cache_info = self._get_cache_info(audio_file)
        
        if not self._is_cached(cache_info):
            return None
        
        try:
            metadata_path = cache_info['stem_dir'] / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return {
                'cached': True,
                'cache_dir': str(cache_info['stem_dir']),
                'stems': metadata['stems'],
                'separation_method': metadata['separation_method'],
                'sample_rate': metadata['sample_rate'],
                'total_samples': metadata['total_samples']
            }
            
        except Exception as e:
            logger.error(f"Failed to get cached stem info: {e}")
            return None
    
    def clear_cache(self, audio_file: str = None):
        """Clear stem cache for specific file or all files"""
        try:
            if audio_file:
                # Clear specific file stem cache
                cache_info = self._get_cache_info(audio_file)
                stem_dir = cache_info['stem_dir']
                if stem_dir.exists():
                    import shutil
                    shutil.rmtree(stem_dir)
                    logger.info(f"Cleared stem cache for: {audio_file}")
            else:
                # Clear all stem caches (search all song directories)
                import config as app_config
                cache_root = Path(app_config.CACHE_DIR)
                for song_dir in cache_root.iterdir():
                    if song_dir.is_dir():
                        stem_dir = song_dir / "stems"
                        if stem_dir.exists():
                            import shutil
                            shutil.rmtree(stem_dir)
                logger.info("Cleared all stem caches")
                
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get separator statistics"""
        import config as app_config
        return {
            'demucs_available': DEMUCS_AVAILABLE,
            'model_name': self.model_name,
            'cache_dir': str(app_config.CACHE_DIR),
            'supported_stems': self.STEM_NAMES
        }

# Factory function
def create_stem_separator(cache_dir: str = None, model: str = "htdemucs") -> StemSeparator:
    """Create stem separator with specified configuration"""
    return StemSeparator(cache_dir=cache_dir, model_name=model)