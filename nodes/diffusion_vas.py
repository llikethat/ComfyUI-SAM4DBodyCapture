"""
Diffusion-VAS Nodes for ComfyUI

Video Amodal Segmentation using Diffusion Priors.
Based on: https://github.com/Kaihua-Chen/diffusion-vas

Paper: "Using Diffusion Priors for Video Amodal Segmentation" (CVPR 2025)
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
from PIL import Image
import cv2

# ============================================================================
# Dependency Checks
# ============================================================================

DIFFUSERS_AVAILABLE = False
LOCAL_VAS_AVAILABLE = False

try:
    import diffusers
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("[Diffusion-VAS] diffusers not installed")

try:
    from torchvision import transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

# Try to import local VAS pipeline
try:
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _lib_dir = os.path.join(os.path.dirname(_current_dir), "lib")
    if _lib_dir not in sys.path:
        sys.path.insert(0, _lib_dir)
    
    from diffusion_vas.pipeline_diffusion_vas import DiffusionVASPipeline
    LOCAL_VAS_AVAILABLE = True
    print("[Diffusion-VAS] Local VAS pipeline loaded")
except Exception as e:
    print(f"[Diffusion-VAS] Local VAS pipeline not available: {e}")
    DiffusionVASPipeline = None

# Import checkpoint manager
try:
    from ..checkpoint_manager import CheckpointManager
    CHECKPOINT_MANAGER_AVAILABLE = True
except ImportError:
    try:
        # Direct import for testing
        _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, _pkg_dir)
        from checkpoint_manager import CheckpointManager
        CHECKPOINT_MANAGER_AVAILABLE = True
    except ImportError:
        CHECKPOINT_MANAGER_AVAILABLE = False
        CheckpointManager = None

# ============================================================================
# Model Registry
# ============================================================================

HUGGINGFACE_MODELS = {
    "amodal_segmentation": "kaihuac/diffusion-vas-amodal-segmentation",
    "content_completion": "kaihuac/diffusion-vas-content-completion",
}

# ============================================================================
# Preprocessing Utilities
# ============================================================================

def create_mask_transform(resolution: Tuple[int, int]):
    return transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

def create_rgb_transform(resolution: Tuple[int, int]):
    return transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

def preprocess_masks(masks: torch.Tensor, resolution: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Preprocess masks for pipeline. Returns [1, B, 3, H, W]."""
    B, H, W = masks.shape
    original_size = (H, W)
    transform = create_mask_transform(resolution)
    
    processed = []
    for i in range(B):
        mask_np = (masks[i].cpu().numpy() * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np, mode='L')
        binary = mask_pil.point(lambda p: 255 if p > 128 else 0)
        processed.append(transform(binary))
    
    return torch.stack(processed).unsqueeze(0), original_size

def preprocess_images(images: torch.Tensor, resolution: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Preprocess images for pipeline. Returns [1, B, 3, H, W]."""
    B, H, W, C = images.shape
    original_size = (H, W)
    transform = create_rgb_transform(resolution)
    
    processed = []
    for i in range(B):
        img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        processed.append(transform(img_pil))
    
    return torch.stack(processed).unsqueeze(0), original_size

def postprocess_masks(output: List, original_size: Tuple[int, int]) -> torch.Tensor:
    """Postprocess output masks."""
    masks_np = np.array([np.array(img) for img in output]).astype('uint8')
    if masks_np.ndim == 4 and masks_np.shape[-1] == 3:
        masks_np = (masks_np.sum(axis=-1) > 600).astype('uint8')
    
    H, W = original_size
    resized = np.array([cv2.resize(f, (W, H), interpolation=cv2.INTER_NEAREST) for f in masks_np])
    return torch.from_numpy(resized).float()

# ============================================================================
# Depth Estimation (Gradient Fallback)
# ============================================================================

def estimate_depth_gradient(images: torch.Tensor, resolution: Tuple[int, int]) -> torch.Tensor:
    """Generate gradient depth as fallback when no depth model available."""
    B, H, W, C = images.shape
    y_coords = torch.linspace(-1, 1, resolution[0]).view(1, 1, resolution[0], 1)
    y_coords = y_coords.expand(1, B, resolution[0], resolution[1])
    depth_3ch = y_coords.unsqueeze(2).repeat(1, 1, 3, 1, 1)
    return depth_3ch.to(images.device)

# ============================================================================
# Diffusion-VAS Wrapper
# ============================================================================

class DiffusionVASWrapper:
    """Wrapper for Diffusion-VAS pipelines."""
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16, hf_token: str = None):
        self.device = device
        self.dtype = dtype
        self.hf_token = hf_token
        self.amodal_pipeline = None
        self.completion_pipeline = None
        self.amodal_loaded = False
        self.completion_loaded = False
        
        # Initialize checkpoint manager with token
        self.ckpt_manager = None
        if CHECKPOINT_MANAGER_AVAILABLE:
            try:
                self.ckpt_manager = CheckpointManager(hf_token=hf_token)
            except Exception as e:
                print(f"[VAS] Checkpoint manager init failed: {e}")
    
    def load_amodal(self, model_id: str = None, auto_download: bool = True):
        """Load amodal segmentation pipeline."""
        if not LOCAL_VAS_AVAILABLE:
            print("[VAS] Local pipeline required but not available")
            print("[VAS] Ensure lib/diffusion_vas/ contains pipeline files")
            return False
        
        # Try checkpoint manager first
        local_path = None
        if self.ckpt_manager and auto_download:
            local_path = self.ckpt_manager.get_vas_amodal_model(auto_download=True)
        
        if local_path and local_path.exists():
            model_id = str(local_path)
            print(f"[VAS] Using local checkpoint: {model_id}")
        elif model_id is None:
            model_id = HUGGINGFACE_MODELS["amodal_segmentation"]
        
        try:
            print(f"[VAS] Loading amodal pipeline: {model_id}")
            self.amodal_pipeline = DiffusionVASPipeline.from_pretrained(
                model_id, torch_dtype=self.dtype
            ).to(self.device)
            self.amodal_pipeline.enable_model_cpu_offload()
            self.amodal_pipeline.set_progress_bar_config(disable=True)
            self.amodal_loaded = True
            print("[VAS] Amodal pipeline loaded")
            return True
        except Exception as e:
            print(f"[VAS] Amodal load error: {e}")
            return False
    
    def load_completion(self, model_id: str = None, auto_download: bool = True):
        """Load content completion pipeline."""
        if not LOCAL_VAS_AVAILABLE:
            return False
        
        # Try checkpoint manager first
        local_path = None
        if self.ckpt_manager and auto_download:
            local_path = self.ckpt_manager.get_vas_completion_model(auto_download=True)
        
        if local_path and local_path.exists():
            model_id = str(local_path)
            print(f"[VAS] Using local checkpoint: {model_id}")
        elif model_id is None:
            model_id = HUGGINGFACE_MODELS["content_completion"]
        
        try:
            print(f"[VAS] Loading completion pipeline: {model_id}")
            self.completion_pipeline = DiffusionVASPipeline.from_pretrained(
                model_id, torch_dtype=self.dtype
            ).to(self.device)
            self.completion_pipeline.enable_model_cpu_offload()
            self.completion_pipeline.set_progress_bar_config(disable=True)
            self.completion_loaded = True
            print("[VAS] Completion pipeline loaded")
            return True
        except Exception as e:
            print(f"[VAS] Completion load error: {e}")
            return False
    
    def amodal_segmentation(self, modal_masks, depth_maps, resolution, num_frames=25, seed=23):
        if not self.amodal_loaded:
            return None
        
        try:
            output = self.amodal_pipeline(
                modal_masks, depth_maps,
                height=resolution[0], width=resolution[1],
                num_frames=num_frames, decode_chunk_size=8,
                motion_bucket_id=127, fps=8,
                noise_aug_strength=0.02,
                min_guidance_scale=1.5, max_guidance_scale=1.5,
                generator=torch.manual_seed(seed),
            )
            return output.frames[0]
        except Exception as e:
            print(f"[VAS] Amodal error: {e}")
            return None
    
    def content_completion(self, modal_rgb, amodal_masks, resolution, num_frames=25, seed=23):
        if not self.completion_loaded:
            return None
        
        try:
            output = self.completion_pipeline(
                modal_rgb, amodal_masks,
                height=resolution[0], width=resolution[1],
                num_frames=num_frames, decode_chunk_size=8,
                motion_bucket_id=127, fps=8,
                noise_aug_strength=0.02,
                min_guidance_scale=1.5, max_guidance_scale=1.5,
                generator=torch.manual_seed(seed),
            )
            return [np.array(img) for img in output.frames[0]]
        except Exception as e:
            print(f"[VAS] Completion error: {e}")
            return None
    
    def unload(self):
        if self.amodal_pipeline:
            del self.amodal_pipeline
            self.amodal_pipeline = None
        if self.completion_pipeline:
            del self.completion_pipeline
            self.completion_pipeline = None
        self.amodal_loaded = False
        self.completion_loaded = False
        torch.cuda.empty_cache()
        print("[VAS] Models unloaded")

# ============================================================================
# ComfyUI Nodes
# ============================================================================

class DiffusionVASLoader:
    """
    Load Diffusion-VAS models.
    
    Models are downloaded to checkpoints/ folder on first use.
    Use checkpoint_manager.py CLI to pre-download models.
    
    For gated models, provide your HuggingFace API token.
    Get token from: https://huggingface.co/settings/tokens
    """
    
    RESOLUTIONS = ["512x1024", "256x512", "384x768"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": (cls.RESOLUTIONS, {"default": "256x512"}),
            },
            "optional": {
                "enable_amodal": ("BOOLEAN", {"default": True}),
                "enable_completion": ("BOOLEAN", {"default": False}),
                "auto_download": ("BOOLEAN", {"default": True, "tooltip": "Auto-download models if not present"}),
                "hf_token": ("STRING", {
                    "default": "",
                    "tooltip": "HuggingFace API token for gated models. Get from: huggingface.co/settings/tokens"
                }),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "dtype": (["float16", "float32"], {"default": "float16"}),
            }
        }
    
    RETURN_TYPES = ("VAS_PIPELINE",)
    RETURN_NAMES = ("vas_pipeline",)
    FUNCTION = "load"
    CATEGORY = "SAM4DBodyCapture/VAS"
    
    def load(self, resolution, enable_amodal=True, enable_completion=False, 
             auto_download=True, hf_token="", device="auto", dtype="float16"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        res_parts = resolution.split("x")
        res_tuple = (int(res_parts[0]), int(res_parts[1]))
        
        print(f"[VAS] Loading on {device}")
        wrapper = DiffusionVASWrapper(device=device, dtype=torch_dtype, hf_token=hf_token if hf_token else None)
        
        # Show checkpoint status
        if wrapper.ckpt_manager:
            wrapper.ckpt_manager.print_status()
        
        if enable_amodal:
            wrapper.load_amodal(auto_download=auto_download)
        if enable_completion:
            wrapper.load_completion(auto_download=auto_download)
        
        pipeline_data = {
            "wrapper": wrapper,
            "resolution": res_tuple,
            "device": device,
            "amodal_loaded": wrapper.amodal_loaded,
            "completion_loaded": wrapper.completion_loaded,
        }
        
        print(f"[VAS] Ready: amodal={wrapper.amodal_loaded}, completion={wrapper.completion_loaded}")
        return (pipeline_data,)


class DiffusionVASAmodalSegmentation:
    """
    Generate amodal masks from modal masks.
    
    Accepts optional external depth_maps from any ComfyUI depth node
    (Depth-Anything-V2, DepthCrafter, ZoeDepth, etc.)
    
    If no depth provided, uses gradient fallback.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vas_pipeline": ("VAS_PIPELINE",),
                "images": ("IMAGE",),
                "masks": ("MASK",),
            },
            "optional": {
                "depth_maps": ("IMAGE", {
                    "tooltip": "External depth maps from any depth node (Depth-Anything, DepthCrafter, etc.)"
                }),
                "num_frames": ("INT", {"default": 25, "min": 1, "max": 64}),
                "seed": ("INT", {"default": 23}),
            }
        }
    
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("amodal_masks", "depth_maps")
    FUNCTION = "segment"
    CATEGORY = "SAM4DBodyCapture/VAS"
    
    def segment(self, vas_pipeline, images, masks, depth_maps=None, num_frames=25, seed=23):
        wrapper = vas_pipeline["wrapper"]
        resolution = vas_pipeline["resolution"]
        
        # Preprocess masks
        modal_pixels, original_size = preprocess_masks(masks, resolution)
        
        # Use external depth if provided, otherwise use gradient fallback
        if depth_maps is not None:
            print("[VAS] Using external depth maps")
            # Convert external depth to pipeline format [1, B, 3, H, W]
            B = depth_maps.shape[0]
            
            # Handle different depth formats
            if depth_maps.dim() == 3:
                # [B, H, W] -> add channel dim
                depth_maps = depth_maps.unsqueeze(-1)
            
            if depth_maps.shape[-1] == 1:
                # Single channel -> repeat to 3
                depth_maps = depth_maps.repeat(1, 1, 1, 3)
            
            # Resize to pipeline resolution and normalize to [-1, 1]
            depth_resized = torch.nn.functional.interpolate(
                depth_maps.permute(0, 3, 1, 2),  # [B, C, H, W]
                size=resolution,
                mode='bilinear',
                align_corners=False
            )
            # Normalize: assume input is [0, 1], convert to [-1, 1]
            depth_pixels = (depth_resized * 2 - 1).unsqueeze(0)  # [1, B, 3, H, W]
            depth_out = depth_maps  # Keep original for output
        else:
            print("[VAS] Using gradient depth fallback")
            depth_pixels = estimate_depth_gradient(images, resolution)
            # Create depth output visualization
            depth_out = depth_pixels.squeeze(0).permute(0, 2, 3, 1)
            depth_out = (depth_out + 1) / 2  # [-1, 1] -> [0, 1]
            depth_out = depth_out[:, :, :, :1].repeat(1, 1, 1, 3)
            depth_out = torch.nn.functional.interpolate(
                depth_out.permute(0, 3, 1, 2),
                size=original_size,
                mode='bilinear'
            ).permute(0, 2, 3, 1)
        
        # Amodal segmentation
        if wrapper.amodal_loaded:
            output = wrapper.amodal_segmentation(modal_pixels, depth_pixels, resolution, num_frames, seed)
            if output:
                amodal_masks = postprocess_masks(output, original_size)
                modal_union = (masks > 0.5).float()
                amodal_masks = torch.clamp(amodal_masks + modal_union, 0, 1)
            else:
                amodal_masks = masks
        else:
            amodal_masks = masks
        
        return (amodal_masks, depth_out)


class DiffusionVASContentCompletion:
    """Complete occluded RGB content."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vas_pipeline": ("VAS_PIPELINE",),
                "images": ("IMAGE",),
                "amodal_masks": ("MASK",),
            },
            "optional": {
                "num_frames": ("INT", {"default": 25, "min": 1, "max": 64}),
                "seed": ("INT", {"default": 23}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("completed_images",)
    FUNCTION = "complete"
    CATEGORY = "SAM4DBodyCapture/VAS"
    
    def complete(self, vas_pipeline, images, amodal_masks, num_frames=25, seed=23):
        wrapper = vas_pipeline["wrapper"]
        resolution = vas_pipeline["resolution"]
        
        if not wrapper.completion_loaded:
            return (images,)
        
        rgb_pixels, original_size = preprocess_images(images, resolution)
        
        # Amodal mask tensor
        amodal_tensor = torch.where(amodal_masks > 0.5, torch.ones_like(amodal_masks), -torch.ones_like(amodal_masks))
        amodal_tensor = amodal_tensor.unsqueeze(0).unsqueeze(2).repeat(1, 1, 3, 1, 1)
        
        # Modal RGB
        modal_rgb = rgb_pixels * 2 - 1
        
        completed = wrapper.content_completion(modal_rgb, amodal_tensor, resolution, num_frames, seed)
        
        if completed is None:
            return (images,)
        
        H, W = original_size
        completed_np = np.array([cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) for f in completed])
        return (torch.from_numpy(completed_np).float() / 255.0,)


class DiffusionVASUnloader:
    """Unload VAS models."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vas_pipeline": ("VAS_PIPELINE",)}}
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    CATEGORY = "SAM4DBodyCapture/VAS"
    OUTPUT_NODE = True
    
    def unload(self, vas_pipeline):
        wrapper = vas_pipeline.get("wrapper")
        if wrapper:
            wrapper.unload()
        return ("VAS models unloaded",)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "DiffusionVASLoader": DiffusionVASLoader,
    "DiffusionVASAmodalSegmentation": DiffusionVASAmodalSegmentation,
    "DiffusionVASContentCompletion": DiffusionVASContentCompletion,
    "DiffusionVASUnloader": DiffusionVASUnloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionVASLoader": "🎭 Load Diffusion-VAS Models",
    "DiffusionVASAmodalSegmentation": "🎭 Amodal Segmentation",
    "DiffusionVASContentCompletion": "🎭 Content Completion",
    "DiffusionVASUnloader": "🎭 Unload VAS Models",
}

# ============================================================================
# Backward Compatibility Exports (for sam4d_pipeline.py)
# ============================================================================

# Alias functions with old names
preprocess_masks_for_pipeline = preprocess_masks
preprocess_images_for_pipeline = preprocess_images
postprocess_amodal_masks = postprocess_masks


class DepthEstimator:
    """
    Backward-compatible DepthEstimator class.
    Now just a wrapper that uses gradient fallback.
    For real depth, connect external depth node to depth_maps input.
    """
    
    def __init__(self, model_size: str = "vitl", device: str = "cuda"):
        self.device = device
        self.model_size = model_size
        self.loaded = False
    
    def load(self, model_path: str = None) -> bool:
        """No-op - we use external depth now."""
        print("[DepthEstimator] Using gradient fallback (connect external depth for better results)")
        return False
    
    def estimate(self, images: torch.Tensor, resolution: Tuple[int, int] = (512, 1024)) -> torch.Tensor:
        """Estimate depth using gradient fallback."""
        return estimate_depth_gradient(images, resolution)
    
    def unload(self):
        """No-op."""
        pass
