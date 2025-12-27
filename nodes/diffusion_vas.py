"""
Diffusion-VAS Nodes for ComfyUI

Video Amodal Segmentation using Diffusion Priors.
Based on: https://github.com/Kaihua-Chen/diffusion-vas

Paper: "Using Diffusion Priors for Video Amodal Segmentation" (CVPR 2025)
License: MIT

Pipeline interface discovered from demo.py:
- Amodal: pipeline(modal_masks, depth_maps, ...) -> amodal_masks
- Completion: pipeline(modal_rgb, amodal_masks, ...) -> completed_rgb
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Tuple, Any, Optional, List, Union
from PIL import Image
import cv2

# ============================================================================
# Dependency Checks
# ============================================================================

DIFFUSERS_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

try:
    import diffusers
    from diffusers import DiffusionPipeline
    DIFFUSERS_AVAILABLE = True
    DIFFUSERS_VERSION = diffusers.__version__
except ImportError:
    DIFFUSERS_VERSION = "not installed"
    print("[Diffusion-VAS] diffusers not installed. Run: pip install diffusers>=0.25.0")

try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("[Diffusion-VAS] transformers not installed. Run: pip install transformers")

try:
    from torchvision import transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("[Diffusion-VAS] torchvision not installed")

# ============================================================================
# Model Registry
# ============================================================================

HUGGINGFACE_MODELS = {
    "amodal_segmentation": "kaihuac/diffusion-vas-amodal-segmentation",
    "content_completion": "kaihuac/diffusion-vas-content-completion",
}

DEPTH_MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# ============================================================================
# Preprocessing Utilities (from demo.py)
# ============================================================================

def create_mask_transform(resolution: Tuple[int, int]):
    """Create transform for mask preprocessing."""
    return transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Grayscale to 3 channels
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)  # Normalize to [-1, 1]
    ])


def create_rgb_transform(resolution: Tuple[int, int]):
    """Create transform for RGB preprocessing."""
    return transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])


def preprocess_masks_for_pipeline(
    masks: torch.Tensor,
    resolution: Tuple[int, int] = (512, 1024)
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Preprocess masks for Diffusion-VAS pipeline.
    
    Args:
        masks: [B, H, W] tensor with values in [0, 1]
        resolution: Target resolution (height, width)
    
    Returns:
        processed: [1, B, 3, H, W] tensor normalized to [-1, 1]
        original_size: Original (H, W) for later resizing
    """
    B, H, W = masks.shape
    original_size = (H, W)
    
    transform = create_mask_transform(resolution)
    
    processed_frames = []
    for i in range(B):
        # Convert to PIL for transform
        mask_np = (masks[i].cpu().numpy() * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np, mode='L')
        
        # Binarize
        binary_pil = mask_pil.point(lambda p: 255 if p > 128 else 0)
        
        # Transform
        transformed = transform(binary_pil)
        processed_frames.append(transformed)
    
    # Stack: [B, 3, H, W] -> [1, B, 3, H, W]
    result = torch.stack(processed_frames).unsqueeze(0)
    
    return result, original_size


def preprocess_images_for_pipeline(
    images: torch.Tensor,
    resolution: Tuple[int, int] = (512, 1024)
) -> Tuple[torch.Tensor, Tuple[int, int], np.ndarray]:
    """
    Preprocess RGB images for Diffusion-VAS pipeline.
    
    Args:
        images: [B, H, W, C] tensor in [0, 1]
        resolution: Target resolution (height, width)
    
    Returns:
        processed: [1, B, 3, H, W] tensor normalized to [-1, 1]
        original_size: Original (H, W)
        raw_images: Original images as numpy array
    """
    B, H, W, C = images.shape
    original_size = (H, W)
    
    transform = create_rgb_transform(resolution)
    
    processed_frames = []
    raw_images = []
    
    for i in range(B):
        # Convert to PIL
        img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, mode='RGB')
        raw_images.append(img_np)
        
        # Transform
        transformed = transform(img_pil)
        processed_frames.append(transformed)
    
    result = torch.stack(processed_frames).unsqueeze(0)
    
    return result, original_size, np.array(raw_images)


def postprocess_amodal_masks(
    pred_masks: List[Image.Image],
    modal_masks: torch.Tensor,
    original_size: Tuple[int, int],
    threshold: int = 600
) -> torch.Tensor:
    """
    Postprocess predicted amodal masks.
    
    Args:
        pred_masks: List of PIL images from pipeline
        modal_masks: Original modal masks [1, B, 3, H, W]
        original_size: (H, W) to resize back to
        threshold: Sum threshold for binarization
    
    Returns:
        amodal_masks: [B, H, W] tensor
    """
    # Convert predictions to numpy
    pred_np = np.array([np.array(img) for img in pred_masks]).astype('uint8')
    
    # Threshold: sum RGB channels and binarize
    pred_binary = (pred_np.sum(axis=-1) > threshold).astype('uint8')
    
    # Union with modal masks
    modal_union = (modal_masks[0, :, 0, :, :].cpu().numpy() > 0).astype('uint8')
    pred_binary = np.logical_or(pred_binary, modal_union).astype('uint8')
    
    # Resize to original
    H, W = original_size
    resized = np.array([
        cv2.resize(frame, (W, H), interpolation=cv2.INTER_NEAREST)
        for frame in pred_binary
    ])
    
    return torch.from_numpy(resized).float()


def postprocess_completed_rgb(
    pred_rgb: List[Image.Image],
    original_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Postprocess completed RGB frames.
    
    Args:
        pred_rgb: List of PIL images from pipeline
        original_size: (H, W) to resize back to
    
    Returns:
        completed: [B, H, W, C] tensor in [0, 1]
    """
    pred_np = np.array([np.array(img) for img in pred_rgb]).astype('uint8')
    
    H, W = original_size
    resized = np.array([
        cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
        for frame in pred_np
    ])
    
    return torch.from_numpy(resized).float() / 255.0


# ============================================================================
# Depth Estimation
# ============================================================================

class DepthEstimator:
    """Wrapper for depth estimation using Depth Anything V2 or fallback."""
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
        self.model = None
        self.model_type = None
    
    def load_transformers_model(self, model_name: str = "Large"):
        """Try loading via HuggingFace transformers."""
        try:
            from transformers import AutoModelForDepthEstimation, AutoImageProcessor
            
            model_map = {
                "Large": "depth-anything/Depth-Anything-V2-Large",
                "Base": "depth-anything/Depth-Anything-V2-Base", 
                "Small": "depth-anything/Depth-Anything-V2-Small",
            }
            
            hf_id = model_map.get(model_name, model_map["Large"])
            
            self.model = AutoModelForDepthEstimation.from_pretrained(
                hf_id, torch_dtype=self.dtype
            ).to(self.device)
            self.processor = AutoImageProcessor.from_pretrained(hf_id)
            self.model_type = "transformers"
            
            print(f"[Depth] Loaded {model_name} via transformers")
            return True
            
        except Exception as e:
            print(f"[Depth] transformers loading failed: {e}")
            return False
    
    def estimate(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """
        Estimate depth from RGB tensor.
        
        Args:
            rgb_tensor: [1, B, 3, H, W] normalized to [-1, 1]
        
        Returns:
            depth_tensor: [1, B, 3, H, W] normalized to [-1, 1]
        """
        # Remove batch dim: [B, 3, H, W]
        rgb_images = rgb_tensor.squeeze(0)
        
        # Denormalize to [0, 255]
        rgb_images = (((rgb_images + 1.0) / 2.0) * 255)
        
        B, C, H, W = rgb_images.shape
        depth_maps = []
        
        if self.model is not None and self.model_type == "transformers":
            for i in range(B):
                # Convert to numpy HWC uint8
                rgb_np = rgb_images[i].cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
                img_pil = Image.fromarray(rgb_np)
                
                # Process
                inputs = self.processor(img_pil, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    depth = outputs.predicted_depth
                
                # Resize to match input
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
                ).squeeze()
                
                depth_maps.append(depth.cpu())
        else:
            # Fallback: gradient-based pseudo-depth
            print("[Depth] Using gradient fallback")
            for i in range(B):
                rgb_np = rgb_images[i].cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
                gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
                
                # Sobel gradients
                grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                depth = np.sqrt(grad_x**2 + grad_y**2)
                
                depth_maps.append(torch.from_numpy(depth))
        
        # Stack and normalize
        depth_stack = torch.stack(depth_maps)  # [B, H, W]
        
        # Normalize to [0, 1] then to [-1, 1]
        d_min, d_max = depth_stack.min(), depth_stack.max()
        depth_norm = (depth_stack - d_min) / (d_max - d_min + 1e-8)
        depth_norm = depth_norm * 2 - 1
        
        # Expand to 3 channels: [B, H, W] -> [1, B, 3, H, W]
        depth_3ch = depth_norm.unsqueeze(1).repeat(1, 3, 1, 1).unsqueeze(0)
        
        return depth_3ch


# ============================================================================
# Main Pipeline Wrapper
# ============================================================================

class DiffusionVASWrapper:
    """
    Wrapper for Diffusion-VAS pipelines.
    
    Pipeline signatures (from demo.py):
    - Amodal: pipeline(modal_masks, depth_maps, height, width, num_frames, ...)
    - Completion: pipeline(modal_rgb, amodal_masks, height, width, num_frames, ...)
    """
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
        
        self.amodal_pipeline = None
        self.completion_pipeline = None
        self.depth_estimator = DepthEstimator(device, dtype)
        
        self.default_resolution = (512, 1024)  # (H, W)
    
    def load_amodal_pipeline(self, model_path: str = None) -> bool:
        """Load amodal segmentation pipeline."""
        if not DIFFUSERS_AVAILABLE:
            print("[VAS] diffusers not available")
            return False
        
        model_id = model_path or HUGGINGFACE_MODELS["amodal_segmentation"]
        print(f"[VAS] Loading amodal pipeline: {model_id}")
        
        try:
            # Try loading with custom pipeline class
            # The model uses DiffusionVASPipeline which extends SVD
            self.amodal_pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                trust_remote_code=True,  # Required for custom pipeline
            )
            self.amodal_pipeline.to(self.device)
            self.amodal_pipeline.enable_model_cpu_offload()
            
            if hasattr(self.amodal_pipeline, 'set_progress_bar_config'):
                self.amodal_pipeline.set_progress_bar_config(disable=False)
            
            print("[VAS] Amodal pipeline loaded!")
            return True
            
        except Exception as e:
            print(f"[VAS] Amodal pipeline load error: {e}")
            print("[VAS] Will use placeholder outputs")
            return False
    
    def load_completion_pipeline(self, model_path: str = None) -> bool:
        """Load content completion pipeline."""
        if not DIFFUSERS_AVAILABLE:
            return False
        
        model_id = model_path or HUGGINGFACE_MODELS["content_completion"]
        print(f"[VAS] Loading completion pipeline: {model_id}")
        
        try:
            self.completion_pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            self.completion_pipeline.to(self.device)
            self.completion_pipeline.enable_model_cpu_offload()
            
            if hasattr(self.completion_pipeline, 'set_progress_bar_config'):
                self.completion_pipeline.set_progress_bar_config(disable=False)
            
            print("[VAS] Completion pipeline loaded!")
            return True
            
        except Exception as e:
            print(f"[VAS] Completion pipeline load error: {e}")
            return False
    
    def load_depth_model(self, model_name: str = "Large") -> bool:
        """Load depth estimation model."""
        return self.depth_estimator.load_transformers_model(model_name)
    
    def run_amodal_segmentation(
        self,
        images: torch.Tensor,
        modal_masks: torch.Tensor,
        resolution: Tuple[int, int] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 1.5,
        num_frames: int = 25,
        seed: int = 23,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run amodal segmentation.
        
        Args:
            images: [B, H, W, C] in [0, 1]
            modal_masks: [B, H, W] in [0, 1]
            resolution: (H, W) for processing
            
        Returns:
            amodal_masks: [B, H, W]
            depth_maps: [B, H, W]
        """
        resolution = resolution or self.default_resolution
        
        # Preprocess
        print("[VAS] Preprocessing inputs...")
        mask_tensor, orig_size = preprocess_masks_for_pipeline(modal_masks, resolution)
        rgb_tensor, _, _ = preprocess_images_for_pipeline(images, resolution)
        
        # Estimate depth
        print("[VAS] Estimating depth...")
        depth_tensor = self.depth_estimator.estimate(rgb_tensor)
        
        # Prepare depth output for visualization
        depth_vis = depth_tensor[0, :, 0, :, :].cpu()  # [B, H, W]
        depth_vis = (depth_vis + 1) / 2  # [-1,1] -> [0,1]
        
        # Resize depth to original size
        B = depth_vis.shape[0]
        H, W = orig_size
        depth_out = torch.stack([
            torch.from_numpy(cv2.resize(depth_vis[i].numpy(), (W, H)))
            for i in range(B)
        ])
        
        if self.amodal_pipeline is None:
            print("[VAS] No amodal pipeline, returning modal masks")
            return modal_masks, depth_out
        
        # Run pipeline
        print(f"[VAS] Running amodal segmentation ({num_frames} frames)...")
        
        generator = torch.manual_seed(seed)
        
        try:
            mask_tensor = mask_tensor.to(self.device, dtype=self.dtype)
            depth_tensor = depth_tensor.to(self.device, dtype=self.dtype)
            
            output = self.amodal_pipeline(
                mask_tensor,
                depth_tensor,
                height=resolution[0],
                width=resolution[1],
                num_frames=num_frames,
                decode_chunk_size=8,
                motion_bucket_id=127,
                fps=8,
                noise_aug_strength=0.02,
                min_guidance_scale=guidance_scale,
                max_guidance_scale=guidance_scale,
                generator=generator,
            )
            
            pred_masks = output.frames[0]
            amodal_masks = postprocess_amodal_masks(pred_masks, mask_tensor, orig_size)
            
            print("[VAS] Amodal segmentation complete!")
            return amodal_masks, depth_out
            
        except Exception as e:
            print(f"[VAS] Pipeline error: {e}")
            print("[VAS] Returning modal masks as fallback")
            return modal_masks, depth_out
    
    def run_content_completion(
        self,
        images: torch.Tensor,
        modal_masks: torch.Tensor,
        amodal_masks: torch.Tensor,
        resolution: Tuple[int, int] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 1.5,
        num_frames: int = 25,
        seed: int = 23,
    ) -> torch.Tensor:
        """
        Run content completion.
        
        Args:
            images: [B, H, W, C] in [0, 1]
            modal_masks: [B, H, W] in [0, 1]
            amodal_masks: [B, H, W] in [0, 1]
            
        Returns:
            completed: [B, H, W, C] in [0, 1]
        """
        resolution = resolution or self.default_resolution
        
        if self.completion_pipeline is None:
            print("[VAS] No completion pipeline, returning original")
            return images
        
        # Preprocess RGB
        rgb_tensor, orig_size, _ = preprocess_images_for_pipeline(images, resolution)
        mask_tensor, _ = preprocess_masks_for_pipeline(modal_masks, resolution)
        
        # Create modal RGB (object only, white background)
        modal_obj = (mask_tensor > 0).float()
        modal_bg = 1 - modal_obj
        rgb_normalized = (rgb_tensor + 1) / 2  # [-1,1] -> [0,1]
        modal_rgb = rgb_normalized * modal_obj + modal_bg
        modal_rgb = modal_rgb * 2 - 1  # Back to [-1,1]
        
        # Prepare amodal masks tensor: -1 for background, 1 for foreground
        amodal_tensor, _ = preprocess_masks_for_pipeline(amodal_masks, resolution)
        amodal_cond = torch.where(amodal_tensor > 0, 
                                   torch.ones_like(amodal_tensor), 
                                   -torch.ones_like(amodal_tensor))
        
        print(f"[VAS] Running content completion ({num_frames} frames)...")
        
        generator = torch.manual_seed(seed)
        
        try:
            modal_rgb = modal_rgb.to(self.device, dtype=self.dtype)
            amodal_cond = amodal_cond.to(self.device, dtype=self.dtype)
            
            output = self.completion_pipeline(
                modal_rgb,
                amodal_cond,
                height=resolution[0],
                width=resolution[1],
                num_frames=num_frames,
                decode_chunk_size=8,
                motion_bucket_id=127,
                fps=8,
                noise_aug_strength=0.02,
                min_guidance_scale=guidance_scale,
                max_guidance_scale=guidance_scale,
                generator=generator,
            )
            
            pred_rgb = output.frames[0]
            completed = postprocess_completed_rgb(pred_rgb, orig_size)
            
            print("[VAS] Content completion complete!")
            return completed
            
        except Exception as e:
            print(f"[VAS] Pipeline error: {e}")
            return images
    
    def unload(self):
        """Free GPU memory."""
        if self.amodal_pipeline is not None:
            del self.amodal_pipeline
            self.amodal_pipeline = None
        
        if self.completion_pipeline is not None:
            del self.completion_pipeline
            self.completion_pipeline = None
        
        if self.depth_estimator.model is not None:
            del self.depth_estimator.model
            self.depth_estimator.model = None
        
        torch.cuda.empty_cache()
        print("[VAS] Models unloaded")


# ============================================================================
# ComfyUI Nodes
# ============================================================================

class DiffusionVASLoader:
    """Load Diffusion-VAS models for amodal segmentation and content completion."""
    
    DEPTH_MODELS = ["Large", "Base", "Small"]
    RESOLUTIONS = ["512x1024", "256x512", "384x768"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_model": (cls.DEPTH_MODELS, {"default": "Large"}),
                "resolution": (cls.RESOLUTIONS, {"default": "512x1024"}),
                "device": (["cuda", "cpu", "auto"], {"default": "auto"}),
                "dtype": (["float16", "bfloat16", "float32"], {"default": "float16"}),
                "load_amodal": ("BOOLEAN", {"default": True}),
                "load_completion": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "amodal_model_path": ("STRING", {"default": ""}),
                "completion_model_path": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("DIFFUSION_VAS_MODEL",)
    RETURN_NAMES = ("vas_model",)
    FUNCTION = "load_models"
    CATEGORY = "SAM4DBodyCapture/Diffusion-VAS"
    
    def load_models(
        self,
        depth_model: str = "Large",
        resolution: str = "512x1024",
        device: str = "auto",
        dtype: str = "float16",
        load_amodal: bool = True,
        load_completion: bool = True,
        amodal_model_path: str = "",
        completion_model_path: str = "",
    ):
        # Parse resolution
        res_parts = resolution.split("x")
        res_tuple = (int(res_parts[0]), int(res_parts[1]))
        
        # Device
        if device == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev = device
        
        # Dtype
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dt = dtype_map.get(dtype, torch.float16)
        
        print(f"[VAS Loader] Device: {dev}, Dtype: {dtype}, Resolution: {res_tuple}")
        
        wrapper = DiffusionVASWrapper(device=dev, dtype=dt)
        wrapper.default_resolution = res_tuple
        
        # Load models
        depth_loaded = wrapper.load_depth_model(depth_model)
        amodal_loaded = wrapper.load_amodal_pipeline(amodal_model_path or None) if load_amodal else False
        completion_loaded = wrapper.load_completion_pipeline(completion_model_path or None) if load_completion else False
        
        model_data = {
            "wrapper": wrapper,
            "device": dev,
            "dtype": dt,
            "resolution": res_tuple,
            "depth_loaded": depth_loaded,
            "amodal_loaded": amodal_loaded,
            "completion_loaded": completion_loaded,
            "version": "0.1.1",
        }
        
        return (model_data,)


class DiffusionVASAmodalSegmentation:
    """Generate complete masks from occluded views using diffusion priors."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vas_model": ("DIFFUSION_VAS_MODEL",),
                "images": ("IMAGE",),
                "modal_masks": ("MASK",),
            },
            "optional": {
                "num_frames": ("INT", {"default": 25, "min": 4, "max": 64}),
                "guidance_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 5.0, "step": 0.1}),
                "seed": ("INT", {"default": 23, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("amodal_masks", "depth_maps")
    FUNCTION = "segment"
    CATEGORY = "SAM4DBodyCapture/Diffusion-VAS"
    
    def segment(
        self,
        vas_model: Dict,
        images: torch.Tensor,
        modal_masks: torch.Tensor,
        num_frames: int = 25,
        guidance_scale: float = 1.5,
        seed: int = 23,
    ):
        wrapper: DiffusionVASWrapper = vas_model["wrapper"]
        resolution = vas_model.get("resolution", (512, 1024))
        
        # Ensure mask shape
        if modal_masks.dim() == 2:
            modal_masks = modal_masks.unsqueeze(0)
        
        print(f"[VAS] Input: {images.shape[0]} frames, {images.shape[1]}x{images.shape[2]}")
        
        amodal_masks, depth_maps = wrapper.run_amodal_segmentation(
            images=images,
            modal_masks=modal_masks,
            resolution=resolution,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        
        # Convert depth to IMAGE format [B, H, W, 3]
        depth_vis = depth_maps.unsqueeze(-1).repeat(1, 1, 1, 3)
        
        return (amodal_masks, depth_vis)


class DiffusionVASContentCompletion:
    """Inpaint occluded regions using diffusion priors."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vas_model": ("DIFFUSION_VAS_MODEL",),
                "images": ("IMAGE",),
                "modal_masks": ("MASK",),
                "amodal_masks": ("MASK",),
            },
            "optional": {
                "num_frames": ("INT", {"default": 25, "min": 4, "max": 64}),
                "guidance_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 5.0, "step": 0.1}),
                "seed": ("INT", {"default": 23, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("completed_content",)
    FUNCTION = "complete"
    CATEGORY = "SAM4DBodyCapture/Diffusion-VAS"
    
    def complete(
        self,
        vas_model: Dict,
        images: torch.Tensor,
        modal_masks: torch.Tensor,
        amodal_masks: torch.Tensor,
        num_frames: int = 25,
        guidance_scale: float = 1.5,
        seed: int = 23,
    ):
        wrapper: DiffusionVASWrapper = vas_model["wrapper"]
        resolution = vas_model.get("resolution", (512, 1024))
        
        if modal_masks.dim() == 2:
            modal_masks = modal_masks.unsqueeze(0)
        if amodal_masks.dim() == 2:
            amodal_masks = amodal_masks.unsqueeze(0)
        
        completed = wrapper.run_content_completion(
            images=images,
            modal_masks=modal_masks,
            amodal_masks=amodal_masks,
            resolution=resolution,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        
        return (completed,)


class DiffusionVASUnload:
    """Unload Diffusion-VAS models to free VRAM."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vas_model": ("DIFFUSION_VAS_MODEL",)}}
    
    RETURN_TYPES = ()
    FUNCTION = "unload"
    CATEGORY = "SAM4DBodyCapture/Diffusion-VAS"
    OUTPUT_NODE = True
    
    def unload(self, vas_model: Dict):
        vas_model["wrapper"].unload()
        return ()


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "DiffusionVASLoader": DiffusionVASLoader,
    "DiffusionVASAmodalSegmentation": DiffusionVASAmodalSegmentation,
    "DiffusionVASContentCompletion": DiffusionVASContentCompletion,
    "DiffusionVASUnload": DiffusionVASUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionVASLoader": "🎭 Load Diffusion-VAS Models",
    "DiffusionVASAmodalSegmentation": "🎭 Amodal Segmentation",
    "DiffusionVASContentCompletion": "🎭 Content Completion",
    "DiffusionVASUnload": "🎭 Unload VAS Models",
}
