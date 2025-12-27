"""
Checkpoint Manager for ComfyUI-SAM4DBodyCapture

Handles downloading and caching of model checkpoints from HuggingFace.

Models managed:
- Depth-Anything-V2 (vitl) - depth estimation
- Diffusion-VAS Amodal Segmentation - amodal mask prediction
- Diffusion-VAS Content Completion - RGB inpainting

Usage:
    from checkpoint_manager import CheckpointManager
    
    manager = CheckpointManager()
    depth_path = manager.get_depth_model()
    vas_amodal_path = manager.get_vas_amodal_model()
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

# Try to import huggingface_hub
HF_AVAILABLE = False
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    HF_AVAILABLE = True
except ImportError:
    print("[SAM4D Checkpoints] huggingface_hub not installed. Run: pip install huggingface_hub")


# ============================================================================
# Model Specifications
# ============================================================================

@dataclass(frozen=True)
class ModelSpec:
    """Specification for a model to download."""
    name: str
    repo_id: str
    filename: Optional[str]  # None for full repo download
    rel_path: str  # Relative path in checkpoints folder
    is_dir: bool = False
    gated: bool = False


# Models we need
MODELS = {
    "depth_anything_v2": ModelSpec(
        name="Depth-Anything-V2 Large",
        repo_id="depth-anything/Depth-Anything-V2-Large",
        filename="depth_anything_v2_vitl.pth",
        rel_path="depth_anything_v2_vitl.pth",
        is_dir=False,
        gated=False,
    ),
    "vas_amodal": ModelSpec(
        name="Diffusion-VAS Amodal Segmentation",
        repo_id="kaihuac/diffusion-vas-amodal-segmentation",
        filename=None,  # Download full repo
        rel_path="diffusion-vas-amodal-segmentation",
        is_dir=True,
        gated=False,
    ),
    "vas_completion": ModelSpec(
        name="Diffusion-VAS Content Completion",
        repo_id="kaihuac/diffusion-vas-content-completion",
        filename=None,
        rel_path="diffusion-vas-content-completion",
        is_dir=True,
        gated=False,
    ),
}


# ============================================================================
# Checkpoint Manager
# ============================================================================

class CheckpointManager:
    """
    Manages model checkpoint downloads and caching.
    
    Checkpoints are stored in:
    - ComfyUI: custom_nodes/ComfyUI-SAM4DBodyCapture/checkpoints/
    - Fallback: ~/.cache/sam4d_checkpoints/
    """
    
    def __init__(self, checkpoint_root: Optional[str] = None, hf_token: Optional[str] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_root: Custom checkpoint directory. If None, auto-detect.
            hf_token: HuggingFace API token for gated models. If None, tries env/cache.
        """
        if checkpoint_root:
            self.root = Path(checkpoint_root)
        else:
            self.root = self._find_checkpoint_root()
        
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Use provided token or try to get from environment/cache
        self._token = hf_token if hf_token else self._get_hf_token()
        
        if self._token:
            print(f"[SAM4D Checkpoints] HuggingFace token: {'*' * 8}...{self._token[-4:] if len(self._token) > 4 else '****'}")
        
        print(f"[SAM4D Checkpoints] Root: {self.root}")
    
    def _find_checkpoint_root(self) -> Path:
        """Find or create checkpoint root directory."""
        # Try package directory first
        pkg_dir = Path(__file__).parent
        pkg_ckpt = pkg_dir / "checkpoints"
        
        # Check if we can write there
        try:
            pkg_ckpt.mkdir(parents=True, exist_ok=True)
            test_file = pkg_ckpt / ".write_test"
            test_file.touch()
            test_file.unlink()
            return pkg_ckpt
        except (PermissionError, OSError):
            pass
        
        # Fall back to user cache
        cache_dir = Path.home() / ".cache" / "sam4d_checkpoints"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def _get_hf_token(self) -> Optional[str]:
        """Get HuggingFace token from environment or cache."""
        # Check environment
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if token:
            return token
        
        # Check huggingface-cli login
        try:
            from huggingface_hub import HfFolder
            return HfFolder.get_token()
        except Exception:
            return None
    
    def set_token(self, token: str):
        """
        Set HuggingFace API token.
        
        Args:
            token: HuggingFace API token from https://huggingface.co/settings/tokens
        """
        self._token = token
        print(f"[SAM4D Checkpoints] Token updated")
    
    def _complete_marker(self, path: Path) -> Path:
        """Get path to completion marker file."""
        return path / ".download_complete"
    
    def _is_download_complete(self, path: Path, is_dir: bool) -> bool:
        """Check if download is complete."""
        if is_dir:
            marker = self._complete_marker(path)
            return path.exists() and path.is_dir() and marker.exists()
        else:
            return path.exists() and path.is_file() and path.stat().st_size > 0
    
    def _mark_complete(self, path: Path):
        """Mark a directory download as complete."""
        marker = self._complete_marker(path)
        marker.write_text("complete\n")
    
    def download_model(self, model_key: str, force: bool = False) -> Optional[Path]:
        """
        Download a model if not already present.
        
        Args:
            model_key: Key from MODELS dict
            force: Re-download even if exists
            
        Returns:
            Path to downloaded model, or None on failure
        """
        if not HF_AVAILABLE:
            print(f"[SAM4D Checkpoints] huggingface_hub required for downloads")
            return None
        
        if model_key not in MODELS:
            print(f"[SAM4D Checkpoints] Unknown model: {model_key}")
            return None
        
        spec = MODELS[model_key]
        local_path = self.root / spec.rel_path
        
        # Check if already downloaded
        if not force and self._is_download_complete(local_path, spec.is_dir):
            print(f"[SAM4D Checkpoints] {spec.name}: Already downloaded")
            return local_path
        
        print(f"[SAM4D Checkpoints] Downloading {spec.name}...")
        
        try:
            if spec.is_dir:
                # Download full repository
                snapshot_download(
                    repo_id=spec.repo_id,
                    local_dir=str(local_path),
                    local_dir_use_symlinks=False,
                    token=self._token,
                    resume_download=True,
                )
                self._mark_complete(local_path)
            else:
                # Download single file
                local_path.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(
                    repo_id=spec.repo_id,
                    filename=spec.filename,
                    local_dir=str(local_path.parent),
                    local_dir_use_symlinks=False,
                    token=self._token,
                )
            
            print(f"[SAM4D Checkpoints] {spec.name}: Downloaded successfully")
            return local_path
            
        except GatedRepoError:
            print(f"[SAM4D Checkpoints] {spec.name}: Requires HuggingFace login")
            print(f"  Run: huggingface-cli login")
            return None
        except (RepositoryNotFoundError, HfHubHTTPError) as e:
            print(f"[SAM4D Checkpoints] {spec.name}: Download failed - {e}")
            return None
        except Exception as e:
            print(f"[SAM4D Checkpoints] {spec.name}: Error - {e}")
            return None
    
    def get_depth_model(self, auto_download: bool = True) -> Optional[Path]:
        """Get Depth-Anything-V2 model path."""
        path = self.root / MODELS["depth_anything_v2"].rel_path
        if self._is_download_complete(path, False):
            return path
        if auto_download:
            return self.download_model("depth_anything_v2")
        return None
    
    def get_vas_amodal_model(self, auto_download: bool = True) -> Optional[Path]:
        """Get Diffusion-VAS amodal segmentation model path."""
        path = self.root / MODELS["vas_amodal"].rel_path
        if self._is_download_complete(path, True):
            return path
        if auto_download:
            return self.download_model("vas_amodal")
        return None
    
    def get_vas_completion_model(self, auto_download: bool = True) -> Optional[Path]:
        """Get Diffusion-VAS content completion model path."""
        path = self.root / MODELS["vas_completion"].rel_path
        if self._is_download_complete(path, True):
            return path
        if auto_download:
            return self.download_model("vas_completion")
        return None
    
    def download_all(self, force: bool = False) -> Dict[str, Optional[Path]]:
        """Download all models."""
        results = {}
        for key in MODELS:
            results[key] = self.download_model(key, force=force)
        return results
    
    def status(self) -> Dict[str, Tuple[bool, Optional[Path]]]:
        """Get status of all models."""
        status = {}
        for key, spec in MODELS.items():
            path = self.root / spec.rel_path
            exists = self._is_download_complete(path, spec.is_dir)
            status[key] = (exists, path if exists else None)
        return status
    
    def print_status(self):
        """Print status of all models."""
        print(f"\n[SAM4D Checkpoints] Status")
        print(f"  Root: {self.root}")
        token_status = "✅ Set" if self._token else "❌ Not set (may need for gated models)"
        print(f"  HF Token: {token_status}")
        print("-" * 60)
        for key, (exists, path) in self.status().items():
            spec = MODELS[key]
            status = "✅ Downloaded" if exists else "❌ Not downloaded"
            print(f"  {spec.name}: {status}")
        print("-" * 60)
        
        if not self._token:
            print("\n💡 Tip: Set HuggingFace token for gated models:")
            print("   - In node: paste token in 'hf_token' field")
            print("   - CLI: python checkpoint_manager.py --download all --token YOUR_TOKEN")
            print("   - Env: export HF_TOKEN=YOUR_TOKEN")
            print("   - Get token: https://huggingface.co/settings/tokens\n")


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command line interface for checkpoint management."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SAM4DBodyCapture Checkpoint Manager",
        epilog="Get HF token from: https://huggingface.co/settings/tokens"
    )
    parser.add_argument("--root", type=str, help="Custom checkpoint directory")
    parser.add_argument("--download", choices=list(MODELS.keys()) + ["all"], 
                       help="Download specific model or all")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--status", action="store_true", help="Show download status")
    parser.add_argument("--token", type=str, 
                       help="HuggingFace API token (or set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    # Use token from args or environment
    hf_token = args.token or os.getenv("HF_TOKEN")
    
    manager = CheckpointManager(args.root, hf_token=hf_token)
    
    if args.status or (not args.download):
        manager.print_status()
    
    if args.download:
        if args.download == "all":
            manager.download_all(force=args.force)
        else:
            manager.download_model(args.download, force=args.force)
        
        manager.print_status()
        
    if not args.download and not args.status:
        print("\nUsage examples:")
        print("  python checkpoint_manager.py --status")
        print("  python checkpoint_manager.py --download all")
        print("  python checkpoint_manager.py --download vas_amodal --token YOUR_HF_TOKEN")
        print("  HF_TOKEN=xxx python checkpoint_manager.py --download all")


if __name__ == "__main__":
    main()
