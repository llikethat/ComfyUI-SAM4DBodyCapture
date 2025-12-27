"""
SAM4D Pipeline Nodes for ComfyUI

Provides the main SAM-Body4D pipeline components:
- Pipeline Loader
- Occlusion Detection
- Amodal Completion
- Mesh Processing
- Temporal Fusion
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Tuple, Any, Optional, List, Union
from PIL import Image
import cv2

# Import our Diffusion-VAS module
from .diffusion_vas import (
    DiffusionVASWrapper,
    DepthEstimator,
    preprocess_masks_for_pipeline,
    preprocess_images_for_pipeline,
    postprocess_amodal_masks,
    DIFFUSERS_AVAILABLE,
)

# ============================================================================
# Data Types
# ============================================================================

class SAM4DMeshSequence:
    """Container for a sequence of 3D meshes."""
    
    def __init__(self):
        self.vertices: List[np.ndarray] = []  # Per-frame vertices [N, 3]
        self.faces: Optional[np.ndarray] = None  # Shared topology [F, 3]
        self.params: Dict[str, List] = {}  # SMPL/HMR parameters
        self.frame_count: int = 0
        self.fps: float = 30.0
        self.person_ids: List[int] = []
    
    def add_frame(self, vertices: np.ndarray, params: dict = None, person_id: int = 1):
        """Add a frame to the sequence."""
        self.vertices.append(vertices)
        if params:
            for k, v in params.items():
                if k not in self.params:
                    self.params[k] = []
                self.params[k].append(v)
        self.frame_count += 1
        if person_id not in self.person_ids:
            self.person_ids.append(person_id)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for ComfyUI transport."""
        return {
            "vertices": self.vertices,
            "faces": self.faces,
            "params": self.params,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "person_ids": self.person_ids,
            "_type": "SAM4D_MESH_SEQUENCE",
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SAM4DMeshSequence":
        """Create from dictionary."""
        seq = cls()
        seq.vertices = data.get("vertices", [])
        seq.faces = data.get("faces")
        seq.params = data.get("params", {})
        seq.frame_count = data.get("frame_count", len(seq.vertices))
        seq.fps = data.get("fps", 30.0)
        seq.person_ids = data.get("person_ids", [])
        return seq


class SAM4DOcclusionInfo:
    """Container for occlusion detection results."""
    
    def __init__(self, num_frames: int):
        self.num_frames = num_frames
        self.iou_scores: Dict[int, List[float]] = {}  # per object
        self.is_occluded: Dict[int, List[bool]] = {}  # per object
        self.occlusion_ranges: Dict[int, List[Tuple[int, int]]] = {}  # per object
    
    def add_object_results(self, obj_id: int, ious: List[float], threshold: float = 0.7):
        """Add occlusion results for an object."""
        self.iou_scores[obj_id] = ious
        self.is_occluded[obj_id] = [iou < threshold for iou in ious]
        
        # Find contiguous occlusion ranges
        ranges = []
        start = None
        for i, occ in enumerate(self.is_occluded[obj_id]):
            if occ and start is None:
                start = i
            elif not occ and start is not None:
                ranges.append((start, i - 1))
                start = None
        if start is not None:
            ranges.append((start, len(ious) - 1))
        
        self.occlusion_ranges[obj_id] = ranges
    
    def get_frames_needing_completion(self, obj_id: int, margin: int = 2) -> List[Tuple[int, int]]:
        """Get frame ranges that need amodal completion with margin."""
        expanded_ranges = []
        for start, end in self.occlusion_ranges.get(obj_id, []):
            new_start = max(0, start - margin)
            new_end = min(self.num_frames - 1, end + margin)
            expanded_ranges.append((new_start, new_end))
        return expanded_ranges
    
    def to_dict(self) -> dict:
        return {
            "num_frames": self.num_frames,
            "iou_scores": self.iou_scores,
            "is_occluded": self.is_occluded,
            "occlusion_ranges": self.occlusion_ranges,
            "_type": "SAM4D_OCCLUSION_INFO",
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SAM4DOcclusionInfo":
        info = cls(data.get("num_frames", 0))
        info.iou_scores = data.get("iou_scores", {})
        info.is_occluded = data.get("is_occluded", {})
        info.occlusion_ranges = data.get("occlusion_ranges", {})
        return info


# ============================================================================
# Utility Functions
# ============================================================================

def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    if union == 0:
        return 1.0
    return intersection / union


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component."""
    mask_uint8 = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_uint8)
    
    if num_labels <= 1:
        return np.zeros_like(mask)
    
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # Ignore background
    largest_label = counts.argmax()
    
    return (labels == largest_label).astype(mask.dtype)


def is_mask_valid(mask: np.ndarray, min_area: int = 100) -> bool:
    """Check if mask is valid (has enough foreground pixels)."""
    return (mask > 0).sum() >= min_area


# ============================================================================
# ComfyUI Nodes
# ============================================================================

class SAM4DPipelineLoader:
    """
    Load all models for the SAM4D pipeline.
    
    This loads:
    - Diffusion-VAS for amodal segmentation (optional)
    - Diffusion-VAS for content completion (optional)
    
    Depth is handled externally (connect depth_maps input) or uses gradient fallback.
    
    For gated models, provide your HuggingFace API token.
    Get token from: https://huggingface.co/settings/tokens
    """
    
    RESOLUTIONS = ["512x1024", "256x512", "384x768"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": (cls.RESOLUTIONS, {"default": "256x512"}),
                "enable_amodal": ("BOOLEAN", {"default": True}),
                "enable_completion": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "hf_token": ("STRING", {
                    "default": "",
                    "tooltip": "HuggingFace API token for model downloads. Get from: huggingface.co/settings/tokens"
                }),
                "device": (["cuda", "cpu", "auto"], {"default": "auto"}),
                "dtype": (["float16", "float32"], {"default": "float16"}),
            }
        }
    
    RETURN_TYPES = ("SAM4D_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_pipeline"
    CATEGORY = "SAM4DBodyCapture/Pipeline"
    
    def load_pipeline(
        self,
        resolution: str,
        enable_amodal: bool,
        enable_completion: bool,
        hf_token: str = "",
        device: str = "auto",
        dtype: str = "float16",
    ):
        # Parse resolution
        h, w = map(int, resolution.split("x"))
        res_tuple = (h, w)
        
        # Device
        if device == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev = device
        
        # Dtype
        dt = torch.float16 if dtype == "float16" else torch.float32
        
        print(f"[SAM4D] Loading pipeline on {dev}")
        
        # Create VAS wrapper with token
        vas_wrapper = DiffusionVASWrapper(
            device=dev, 
            dtype=dt, 
            hf_token=hf_token if hf_token else None
        )
        vas_wrapper.default_resolution = res_tuple
        
        # Depth is now external - just set flag
        depth_loaded = False  # Use external depth_maps input or gradient fallback
        
        # Optionally load amodal/completion
        amodal_loaded = False
        completion_loaded = False
        
        if enable_amodal:
            amodal_loaded = vas_wrapper.load_amodal_pipeline()
        
        if enable_completion:
            completion_loaded = vas_wrapper.load_completion_pipeline()
        
        pipeline = {
            "vas_wrapper": vas_wrapper,
            "depth_loaded": depth_loaded,
            "amodal_loaded": amodal_loaded,
            "completion_loaded": completion_loaded,
            "resolution": res_tuple,
            "device": dev,
            "dtype": dt,
            "_type": "SAM4D_PIPELINE",
        }
        
        print(f"[SAM4D] Pipeline ready: depth=external, amodal={amodal_loaded}, completion={completion_loaded}")
        
        return (pipeline,)


class SAM4DOcclusionDetector:
    """
    Detect occluded frames by comparing modal masks with predicted amodal masks.
    
    Uses Diffusion-VAS amodal segmentation to predict complete masks,
    then computes IoU to identify frames where the object is partially hidden.
    
    Accepts optional external depth_maps from any ComfyUI depth node
    (Depth-Anything-V2, DepthCrafter, ZoeDepth, etc.)
    
    Compatible with both SAM4D_PIPELINE and VAS_PIPELINE loaders.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
            },
            "optional": {
                "pipeline": ("SAM4D_PIPELINE,VAS_PIPELINE",),
                "depth_maps": ("IMAGE", {
                    "tooltip": "External depth maps from any depth node (Depth-Anything, DepthCrafter, etc.)"
                }),
                "iou_threshold": ("FLOAT", {"default": 0.7, "min": 0.3, "max": 0.95, "step": 0.05}),
                "object_ids": ("STRING", {"default": "1", "tooltip": "Comma-separated object IDs"}),
                "num_frames": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 256,
                    "tooltip": "0 = auto (use actual frame count). Set manually only if needed."
                }),
                "chunk_size": ("INT", {
                    "default": 12,
                    "min": 8,
                    "max": 64,
                    "tooltip": "Max frames per chunk. Lower if OOM, higher if you have VRAM."
                }),
                "overlap": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 16,
                    "tooltip": "Overlap frames between chunks for smooth transitions. 0 = no blending."
                }),
                "seed": ("INT", {"default": 23}),
            }
        }
    
    RETURN_TYPES = ("SAM4D_OCCLUSION_INFO", "IMAGE", "MASK")
    RETURN_NAMES = ("occlusion_info", "depth_maps", "amodal_masks")
    FUNCTION = "detect"
    CATEGORY = "SAM4DBodyCapture/Pipeline"
    
    def detect(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        pipeline: dict = None,
        depth_maps: torch.Tensor = None,
        iou_threshold: float = 0.7,
        object_ids: str = "1",
        num_frames: int = 0,
        chunk_size: int = 12,
        overlap: int = 4,
        seed: int = 23,
    ):
        # Handle both pipeline types
        if pipeline is None:
            # No pipeline - use gradient depth fallback, no amodal
            vas_wrapper = None
            resolution = (256, 512)
            amodal_available = False
        elif pipeline.get("_type") == "SAM4D_PIPELINE":
            vas_wrapper = pipeline["vas_wrapper"]
            resolution = pipeline["resolution"]
            amodal_available = pipeline.get("amodal_loaded", False)
        else:
            # VAS_PIPELINE format
            vas_wrapper = pipeline.get("wrapper")
            resolution = pipeline.get("resolution", (256, 512))
            amodal_available = pipeline.get("amodal_loaded", False)
        
        # Parse object IDs
        obj_ids = [int(x.strip()) for x in object_ids.split(",")]
        
        B = images.shape[0]
        H, W = images.shape[1:3]
        
        # num_frames=0 means auto (use actual frame count) - SAM-Body4D approach
        actual_num_frames = B if num_frames == 0 else num_frames
        
        print(f"[SAM4D] Detecting occlusions for {B} frames, objects: {obj_ids}, num_frames={actual_num_frames}")
        
        # Ensure masks shape
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
        
        # Create occlusion info
        occ_info = SAM4DOcclusionInfo(B)
        
        # Use external depth if provided, otherwise use gradient fallback
        if depth_maps is not None:
            print("[SAM4D] Using external depth maps")
            # Ensure depth is in correct format [B, H, W, C]
            if depth_maps.dim() == 3:
                depth_maps = depth_maps.unsqueeze(-1).repeat(1, 1, 1, 3)
            elif depth_maps.shape[-1] == 1:
                depth_maps = depth_maps.repeat(1, 1, 1, 3)
            depth_out = depth_maps
        elif vas_wrapper is not None:
            print("[SAM4D] Estimating depth (using gradient fallback)...")
            _, depth_out = vas_wrapper.run_amodal_segmentation(
                images=images,
                modal_masks=masks,
                resolution=resolution,
                num_frames=actual_num_frames,
                seed=seed,
                max_chunk_size=chunk_size,
                overlap=overlap,
            )
            depth_out = depth_out.unsqueeze(-1).repeat(1, 1, 1, 3) if depth_out.dim() == 3 else depth_out
        else:
            # No pipeline, no external depth - create gradient fallback
            print("[SAM4D] No pipeline - using gradient depth fallback")
            depth_out = torch.zeros(B, H, W, 3, device=images.device)
            for i in range(B):
                y_grad = torch.linspace(0, 1, H).view(H, 1).expand(H, W)
                depth_out[i] = y_grad.unsqueeze(-1).repeat(1, 1, 3)
        
        # If no amodal pipeline, return empty occlusion info
        if not amodal_available or vas_wrapper is None:
            print("[SAM4D] No amodal pipeline - assuming no occlusions")
            for obj_id in obj_ids:
                occ_info.add_object_results(obj_id, [1.0] * B, iou_threshold)
            
            return (occ_info.to_dict(), depth_out, masks)
        
        # Run amodal segmentation for each object
        all_amodal_masks = torch.zeros_like(masks)
        
        for obj_id in obj_ids:
            # Extract masks for this object
            obj_masks = (masks == obj_id).float() if masks.max() > 1 else masks.float()
            
            # Run amodal segmentation with external depth if available
            amodal_masks, _ = vas_wrapper.run_amodal_segmentation(
                images=images,
                modal_masks=obj_masks,
                resolution=resolution,
                num_frames=actual_num_frames,
                seed=seed,
                depth_maps=depth_maps,  # Pass external depth
                max_chunk_size=chunk_size,
                overlap=overlap,
            )
            
            # Compute IoU per frame
            ious = []
            for i in range(B):
                modal = obj_masks[i].cpu().numpy()
                amodal = amodal_masks[i].cpu().numpy()
                
                # Keep largest component to avoid noise
                amodal = keep_largest_component(amodal)
                
                # Modal area > amodal area means bad prediction
                if modal.sum() > amodal.sum():
                    ious.append(1.0)  # Trust modal mask
                else:
                    iou = compute_mask_iou(modal, amodal)
                    ious.append(iou)
            
            occ_info.add_object_results(obj_id, ious, iou_threshold)
            
            # Combine amodal masks
            all_amodal_masks = torch.maximum(all_amodal_masks, amodal_masks.to(all_amodal_masks.device))
            
            print(f"[SAM4D] Object {obj_id}: {sum(occ_info.is_occluded[obj_id])} occluded frames")
        
        return (occ_info.to_dict(), depth_out, all_amodal_masks)


class SAM4DAmodalCompletion:
    """
    Complete occluded masks and optionally RGB content.
    
    Only processes frames identified as occluded by SAM4DOcclusionDetector.
    
    Compatible with both SAM4D_PIPELINE and VAS_PIPELINE loaders.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "occlusion_info": ("SAM4D_OCCLUSION_INFO",),
            },
            "optional": {
                "pipeline": ("SAM4D_PIPELINE,VAS_PIPELINE",),
                "complete_rgb": ("BOOLEAN", {"default": False}),
                "num_frames": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 256,
                    "tooltip": "0 = auto (use actual frame count). Set manually only if needed."
                }),
                "chunk_size": ("INT", {
                    "default": 12,
                    "min": 8,
                    "max": 64,
                    "tooltip": "Max frames per chunk. Lower if OOM, higher if you have VRAM."
                }),
                "overlap": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 16,
                    "tooltip": "Overlap frames between chunks for smooth transitions. 0 = no blending."
                }),
                "seed": ("INT", {"default": 23}),
            }
        }
    
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("completed_masks", "completed_images")
    FUNCTION = "complete"
    CATEGORY = "SAM4DBodyCapture/Pipeline"
    
    def complete(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        occlusion_info: dict,
        pipeline: dict = None,
        complete_rgb: bool = False,
        num_frames: int = 0,
        chunk_size: int = 12,
        overlap: int = 4,
        seed: int = 23,
    ):
        # Handle both pipeline types
        if pipeline is None:
            print("[SAM4D] No pipeline provided - returning original masks/images")
            return (masks, images)
        elif pipeline.get("_type") == "SAM4D_PIPELINE":
            vas_wrapper = pipeline["vas_wrapper"]
            resolution = pipeline["resolution"]
            completion_available = pipeline.get("completion_loaded", False)
        else:
            # VAS_PIPELINE format
            vas_wrapper = pipeline.get("wrapper")
            resolution = pipeline.get("resolution", (256, 512))
            completion_available = pipeline.get("completion_loaded", False)
        
        if vas_wrapper is None:
            print("[SAM4D] No pipeline wrapper - returning original masks/images")
            return (masks, images)
        
        occ_info = SAM4DOcclusionInfo.from_dict(occlusion_info)
        
        B = images.shape[0]
        completed_masks = masks.clone()
        completed_images = images.clone()
        
        # Process each object
        for obj_id in occ_info.occlusion_ranges.keys():
            ranges = occ_info.get_frames_needing_completion(obj_id, margin=2)
            
            if not ranges:
                print(f"[SAM4D] Object {obj_id}: No occlusions to complete")
                continue
            
            print(f"[SAM4D] Object {obj_id}: Completing {len(ranges)} ranges")
            
            for start, end in ranges:
                # Extract frame range
                frame_images = images[start:end+1]
                frame_masks = masks[start:end+1]
                
                obj_masks = (frame_masks == obj_id).float() if masks.max() > 1 else frame_masks.float()
                
                # num_frames=0 means auto - use range size
                range_size = end - start + 1
                actual_num_frames = range_size if num_frames == 0 else min(num_frames, range_size)
                
                # Run amodal segmentation
                amodal_masks, _ = vas_wrapper.run_amodal_segmentation(
                    images=frame_images,
                    modal_masks=obj_masks,
                    resolution=resolution,
                    num_frames=actual_num_frames,
                    seed=seed,
                    max_chunk_size=chunk_size,
                    overlap=overlap,
                )
                
                # Update completed masks
                completed_masks[start:end+1] = torch.maximum(
                    completed_masks[start:end+1],
                    amodal_masks.to(completed_masks.device)
                )
                
                # Optionally complete RGB
                if complete_rgb and completion_available:
                    completed_rgb = vas_wrapper.run_content_completion(
                        images=frame_images,
                        amodal_masks=amodal_masks,
                        resolution=resolution,
                        num_frames=actual_num_frames,
                        seed=seed,
                    )
                    completed_images[start:end+1] = completed_rgb.to(completed_images.device)
        
        return (completed_masks, completed_images)


class SAM4DPipelineUnload:
    """Unload SAM4D/VAS pipeline models to free VRAM."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipeline": ("SAM4D_PIPELINE,VAS_PIPELINE",)}}
    
    RETURN_TYPES = ()
    FUNCTION = "unload"
    CATEGORY = "SAM4DBodyCapture/Pipeline"
    OUTPUT_NODE = True
    
    def unload(self, pipeline: dict):
        # Handle both pipeline types
        vas_wrapper = pipeline.get("vas_wrapper") or pipeline.get("wrapper")
        if vas_wrapper:
            vas_wrapper.unload()
        return ()


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "SAM4DPipelineLoader": SAM4DPipelineLoader,
    "SAM4DOcclusionDetector": SAM4DOcclusionDetector,
    "SAM4DAmodalCompletion": SAM4DAmodalCompletion,
    "SAM4DPipelineUnload": SAM4DPipelineUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM4DPipelineLoader": "🎬 Load SAM4D Pipeline",
    "SAM4DOcclusionDetector": "🔍 Detect Occlusions",
    "SAM4DAmodalCompletion": "🎭 Complete Occluded Regions",
    "SAM4DPipelineUnload": "🗑️ Unload SAM4D Pipeline",
}
