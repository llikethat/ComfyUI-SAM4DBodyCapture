# Copyright (c) 2025 SAM4DBodyCapture
# SPDX-License-Identifier: MIT
"""
SAM3DBody Integration for SAM4DBodyCapture.

Provides integrated SAM3DBody nodes with:
- BFloat16 → Float16 fix for sparse CUDA operations
- Batch video processing (per-frame mesh generation)
- Temporal smoothing for smooth animation sequences
"""

import os
import sys
import tempfile
import torch
import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
import folder_paths

# Global cache for model
_MODEL_CACHE = {}


class SAM4DBodyLoader:
    """
    Load SAM3DBody model with dtype fix.
    
    This loader fixes the BFloat16 sparse matrix error by forcing Float16 mode.
    The original SAM3DBody uses bfloat16 which doesn't work with sparse CUDA ops.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": os.path.join(folder_paths.models_dir, "sam3dbody"),
                    "tooltip": "Path to SAM3DBody model folder"
                }),
            },
            "optional": {
                "hf_token": ("STRING", {
                    "default": "",
                    "tooltip": "HuggingFace token for automatic download"
                }),
                "force_fp16": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Force Float16 instead of BFloat16 (fixes sparse matrix error)"
                }),
            }
        }
    
    RETURN_TYPES = ("SAM4D_BODY_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM4DBodyCapture/Model"
    
    def load_model(self, model_path: str, hf_token: str = "", force_fp16: bool = True):
        """Load SAM3DBody model with dtype fix."""
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = os.path.abspath(model_path)
        
        # Check cache
        cache_key = f"{model_path}_{device}_{force_fp16}"
        if cache_key in _MODEL_CACHE:
            print(f"[SAM4DBodyCapture] Using cached SAM3DBody model")
            return (_MODEL_CACHE[cache_key],)
        
        # Check model files
        ckpt_path = os.path.join(model_path, "model.ckpt")
        mhr_path = os.path.join(model_path, "assets", "mhr_model.pt")
        config_path = os.path.join(model_path, "model_config.yaml")
        
        if not os.path.exists(ckpt_path) or not os.path.exists(mhr_path):
            if hf_token:
                # Try downloading
                os.environ["HF_TOKEN"] = hf_token
                try:
                    from huggingface_hub import snapshot_download
                    os.makedirs(model_path, exist_ok=True)
                    snapshot_download(
                        repo_id="facebook/sam-3d-body-dinov3",
                        local_dir=model_path
                    )
                except Exception as e:
                    raise RuntimeError(f"Download failed: {e}")
            else:
                raise RuntimeError(
                    f"\n[SAM4DBodyCapture] SAM3DBody model not found.\n\n"
                    f"Please place model files at: {model_path}/\n"
                    f"  ├── model.ckpt\n"
                    f"  ├── model_config.yaml\n"
                    f"  └── assets/mhr_model.pt\n\n"
                    f"Or provide HuggingFace token for automatic download."
                )
        
        # Import SAM3DBody components
        try:
            # Add SAM3DBody to path if needed
            sam3d_path = None
            for path in sys.path:
                if 'SAM3DBody' in path or 'sam3dbody' in path.lower():
                    sam3d_path = path
                    break
            
            # Try importing
            from sam_3d_body.models.meta_arch import SAM3DBody
            from sam_3d_body.utils.config import get_config
            from sam_3d_body.utils.checkpoint import load_state_dict
            
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import sam_3d_body. Make sure ComfyUI-SAM3DBody is installed.\n"
                f"Error: {e}"
            )
        
        # Load config
        if os.path.exists(config_path):
            model_cfg = get_config(config_path)
        else:
            # Use bundled config
            bundled_config = os.path.join(
                os.path.dirname(__file__), "..", "sam_3d_body", "configs", "model_config.yaml"
            )
            if os.path.exists(bundled_config):
                model_cfg = get_config(bundled_config)
            else:
                # Look in SAM3DBody package
                import sam_3d_body
                pkg_config = os.path.join(
                    os.path.dirname(sam_3d_body.__file__), "configs", "model_config.yaml"
                )
                model_cfg = get_config(pkg_config)
        
        # === THE FIX: Override bfloat16 → float16 ===
        if force_fp16:
            model_cfg.defrost()
            model_cfg.TRAIN.FP16_TYPE = "float16"  # Fix sparse matrix error!
            model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path
            model_cfg.freeze()
            print(f"[SAM4DBodyCapture] Forced FP16_TYPE=float16 (fixes BFloat16 sparse error)")
        else:
            model_cfg.defrost()
            model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path
            model_cfg.freeze()
        
        # Initialize model
        print(f"[SAM4DBodyCapture] Loading SAM3DBody model...")
        model = SAM3DBody(model_cfg)
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        load_state_dict(model, state_dict, strict=False)
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        # Create model dict
        model_dict = {
            "model": model,
            "model_cfg": model_cfg,
            "device": device,
            "model_path": model_path,
            "mhr_path": mhr_path,
            "faces": model.head_pose.faces.cpu().numpy(),
        }
        
        # Cache it
        _MODEL_CACHE[cache_key] = model_dict
        print(f"[SAM4DBodyCapture] SAM3DBody model loaded successfully on {device}")
        
        return (model_dict,)


class SAM4DBodyBatchProcess:
    """
    Process video frames through SAM3DBody to generate mesh sequence.
    
    This node processes each frame individually and collects all outputs
    for temporal smoothing and FBX export.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAM4D_BODY_MODEL", {
                    "tooltip": "SAM3DBody model from SAM4DBodyLoader"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Video frames (batch of images)"
                }),
                "masks": ("MASK", {
                    "tooltip": "Segmentation masks for each frame"
                }),
            },
            "optional": {
                "camera_intrinsics": ("CAMERA_INTRINSICS", {
                    "tooltip": "Camera intrinsics from MoGe2 or manual input"
                }),
                "inference_type": (["full", "body"], {
                    "default": "body",
                    "tooltip": "full: body+hands (slower), body: body only (faster)"
                }),
                "person_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "tooltip": "Which person to track (0=first detected)"
                }),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE", "IMAGE")
    RETURN_NAMES = ("mesh_sequence", "debug_images")
    FUNCTION = "process_batch"
    CATEGORY = "SAM4DBodyCapture/Processing"
    
    def _compute_bbox_from_mask(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Compute bounding box from binary mask."""
        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)
        
        if not rows.any() or not cols.any():
            return None
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return np.array([[cmin, rmin, cmax, rmax]], dtype=np.float32)
    
    def process_batch(
        self,
        model: Dict,
        images: torch.Tensor,
        masks: torch.Tensor,
        camera_intrinsics: Optional[Dict] = None,
        inference_type: str = "body",
        person_index: int = 0
    ):
        """Process all video frames through SAM3DBody."""
        
        from sam_3d_body import SAM3DBodyEstimator
        
        # Extract model components
        sam_model = model["model"]
        model_cfg = model["model_cfg"]
        device = model["device"]
        faces = model["faces"]
        
        # Create estimator
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=sam_model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )
        
        # Get number of frames
        num_frames = images.shape[0]
        print(f"[SAM4DBodyCapture] Processing {num_frames} frames through SAM3DBody...")
        
        # Process camera intrinsics
        cam_int = None
        if camera_intrinsics is not None:
            # Build camera intrinsic matrix
            focal = camera_intrinsics.get("focal_length", 1000.0)
            cx = camera_intrinsics.get("cx", images.shape[2] / 2)
            cy = camera_intrinsics.get("cy", images.shape[1] / 2)
            
            # Per-frame focal lengths if available
            per_frame_focal = camera_intrinsics.get("per_frame_focal", None)
        else:
            per_frame_focal = None
        
        # Collect all frame outputs
        mesh_sequence = {
            "frames": [],
            "faces": faces,
            "num_frames": num_frames,
            "fps": 30.0,  # Default, can be overridden
        }
        
        debug_images = []
        
        # Process each frame
        for frame_idx in range(num_frames):
            print(f"  Frame {frame_idx + 1}/{num_frames}...", end="\r")
            
            # Get frame image (ComfyUI format: [B, H, W, C] float 0-1)
            img_tensor = images[frame_idx]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Get frame mask
            mask_tensor = masks[frame_idx] if masks.dim() == 3 else masks[frame_idx, 0]
            mask_np = mask_tensor.cpu().numpy()
            if mask_np.max() <= 1.0:
                mask_np = (mask_np * 255).astype(np.uint8)
            
            # Compute bbox from mask
            bbox = self._compute_bbox_from_mask(mask_np)
            if bbox is None:
                print(f"\n  Warning: No person detected in frame {frame_idx}, using previous")
                if len(mesh_sequence["frames"]) > 0:
                    mesh_sequence["frames"].append(mesh_sequence["frames"][-1].copy())
                continue
            
            # Build camera intrinsics matrix for this frame
            # SAM3DBody expects cam_int as a [3, 3] intrinsic matrix tensor
            cam_int_tensor = None
            if camera_intrinsics is not None:
                # Get focal length for this frame
                if per_frame_focal is not None and frame_idx < len(per_frame_focal):
                    frame_focal = per_frame_focal[frame_idx]
                else:
                    frame_focal = camera_intrinsics.get("focal_length", 1000.0)
                
                # Get principal point
                cx = camera_intrinsics.get("cx", img_np.shape[1] / 2)
                cy = camera_intrinsics.get("cy", img_np.shape[0] / 2)
                
                # Build 3x3 intrinsic matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                # For simplicity, assume fx = fy (square pixels)
                cam_int_matrix = np.array([
                    [frame_focal, 0, cx],
                    [0, frame_focal, cy],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                cam_int_tensor = torch.from_numpy(cam_int_matrix).to(device)
            else:
                frame_focal = None
            
            # Save to temp file (SAM3DBody requires file path)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, img_bgr)
                tmp_path = tmp.name
            
            try:
                # Process frame with camera intrinsics
                outputs = estimator.process_one_image(
                    tmp_path,
                    bboxes=bbox,
                    masks=mask_np,
                    cam_int=cam_int_tensor,  # Pass camera intrinsics!
                    bbox_thr=0.5,
                    use_mask=True,
                    inference_type=inference_type,
                )
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
            if not outputs or len(outputs) == 0:
                print(f"\n  Warning: SAM3DBody failed on frame {frame_idx}")
                if len(mesh_sequence["frames"]) > 0:
                    mesh_sequence["frames"].append(mesh_sequence["frames"][-1].copy())
                continue
            
            # Get the requested person (or first if not enough detected)
            output_idx = min(person_index, len(outputs) - 1)
            output = outputs[output_idx]
            
            # Extract frame data
            frame_data = {
                "frame_idx": frame_idx,
                "vertices": output.get("pred_vertices", None),
                "joints_3d": output.get("pred_keypoints_3d", None),
                "joint_coords": output.get("pred_joint_coords", None),
                "joint_rotations": output.get("pred_global_rots", None),
                "camera_t": output.get("pred_cam_t", None),
                "focal_length": output.get("focal_length", frame_focal),
                "bbox": output.get("bbox", None),
                "body_pose": output.get("body_pose_params", None),
                "hand_pose": output.get("hand_pose_params", None),
                "global_rot": output.get("global_rot", None),
                "shape": output.get("shape_params", None),
                "scale": output.get("scale_params", None),
            }
            
            mesh_sequence["frames"].append(frame_data)
            
            # Debug image (just the input for now)
            debug_images.append(img_np)
        
        print(f"\n[SAM4DBodyCapture] Processed {len(mesh_sequence['frames'])} frames successfully")
        
        # Stack debug images
        debug_tensor = torch.from_numpy(
            np.stack(debug_images, axis=0)
        ).float() / 255.0
        
        return (mesh_sequence, debug_tensor)


class SAM4DTemporalSmoothing:
    """
    Apply temporal smoothing to mesh sequence.
    
    Smooths vertices, joints, and rotations across time to reduce jitter
    and create smooth animation sequences.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE", {
                    "tooltip": "Mesh sequence from SAM4DBodyBatchProcess"
                }),
            },
            "optional": {
                "smoothing_window": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 30,
                    "tooltip": "Temporal smoothing window size (frames)"
                }),
                "vertex_smoothing": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Vertex position smoothing strength"
                }),
                "joint_smoothing": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Joint position smoothing strength"
                }),
                "rotation_smoothing": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Rotation smoothing strength"
                }),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE",)
    RETURN_NAMES = ("smoothed_sequence",)
    FUNCTION = "smooth_sequence"
    CATEGORY = "SAM4DBodyCapture/Processing"
    
    def _smooth_array(self, data: List[np.ndarray], window: int, strength: float) -> List[np.ndarray]:
        """Apply temporal smoothing to array sequence."""
        if strength == 0 or len(data) < 2:
            return data
        
        smoothed = []
        half_window = window // 2
        
        for i in range(len(data)):
            # Get window indices
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            
            # Calculate weighted average
            window_data = data[start:end]
            if len(window_data) > 0 and window_data[0] is not None:
                # Simple average
                avg = np.mean(np.stack(window_data, axis=0), axis=0)
                # Blend with original based on strength
                smoothed_val = data[i] * (1 - strength) + avg * strength
                smoothed.append(smoothed_val)
            else:
                smoothed.append(data[i])
        
        return smoothed
    
    def smooth_sequence(
        self,
        mesh_sequence: Dict,
        smoothing_window: int = 5,
        vertex_smoothing: float = 0.5,
        joint_smoothing: float = 0.7,
        rotation_smoothing: float = 0.8
    ):
        """Apply temporal smoothing to mesh sequence."""
        
        frames = mesh_sequence["frames"]
        if len(frames) < 2:
            print("[SAM4DBodyCapture] Not enough frames for temporal smoothing")
            return (mesh_sequence,)
        
        print(f"[SAM4DBodyCapture] Applying temporal smoothing (window={smoothing_window})...")
        
        # Extract arrays for smoothing
        vertices_list = [f.get("vertices") for f in frames]
        joints_list = [f.get("joint_coords") for f in frames]
        rotations_list = [f.get("joint_rotations") for f in frames]
        
        # Apply smoothing
        if vertex_smoothing > 0 and vertices_list[0] is not None:
            vertices_list = self._smooth_array(vertices_list, smoothing_window, vertex_smoothing)
        
        if joint_smoothing > 0 and joints_list[0] is not None:
            joints_list = self._smooth_array(joints_list, smoothing_window, joint_smoothing)
        
        # For rotations, we need special handling (slerp would be better, but average works for small changes)
        if rotation_smoothing > 0 and rotations_list[0] is not None:
            rotations_list = self._smooth_array(rotations_list, smoothing_window, rotation_smoothing)
        
        # Update frames with smoothed values
        smoothed_frames = []
        for i, frame in enumerate(frames):
            smoothed_frame = frame.copy()
            if vertices_list[i] is not None:
                smoothed_frame["vertices"] = vertices_list[i]
            if joints_list[i] is not None:
                smoothed_frame["joint_coords"] = joints_list[i]
            if rotations_list[i] is not None:
                smoothed_frame["joint_rotations"] = rotations_list[i]
            smoothed_frames.append(smoothed_frame)
        
        # Create smoothed sequence
        smoothed_sequence = mesh_sequence.copy()
        smoothed_sequence["frames"] = smoothed_frames
        
        print(f"[SAM4DBodyCapture] Temporal smoothing applied to {len(smoothed_frames)} frames")
        
        return (smoothed_sequence,)


class SAM4DMeshSequenceInfo:
    """
    Display information about a mesh sequence.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE", {
                    "tooltip": "Mesh sequence to inspect"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "get_info"
    CATEGORY = "SAM4DBodyCapture/Utils"
    OUTPUT_NODE = True
    
    def get_info(self, mesh_sequence: Dict):
        """Get mesh sequence information."""
        
        frames = mesh_sequence.get("frames", [])
        faces = mesh_sequence.get("faces", None)
        
        info_lines = [
            "=== Mesh Sequence Info ===",
            f"Number of frames: {len(frames)}",
            f"FPS: {mesh_sequence.get('fps', 30.0)}",
        ]
        
        if faces is not None:
            info_lines.append(f"Faces: {faces.shape[0]} triangles")
        
        if len(frames) > 0:
            first_frame = frames[0]
            if first_frame.get("vertices") is not None:
                info_lines.append(f"Vertices per frame: {first_frame['vertices'].shape[0]}")
            if first_frame.get("joint_coords") is not None:
                info_lines.append(f"Joints per frame: {first_frame['joint_coords'].shape[0]}")
            if first_frame.get("focal_length") is not None:
                info_lines.append(f"Focal length (frame 0): {first_frame['focal_length']:.1f}")
        
        info = "\n".join(info_lines)
        print(info)
        
        return (info,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SAM4DBodyLoader": SAM4DBodyLoader,
    "SAM4DBodyBatchProcess": SAM4DBodyBatchProcess,
    "SAM4DTemporalSmoothing": SAM4DTemporalSmoothing,
    "SAM4DMeshSequenceInfo": SAM4DMeshSequenceInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM4DBodyLoader": "SAM4D: Load SAM3DBody Model (Fixed)",
    "SAM4DBodyBatchProcess": "SAM4D: Batch Process Video",
    "SAM4DTemporalSmoothing": "SAM4D: Temporal Smoothing",
    "SAM4DMeshSequenceInfo": "SAM4D: Mesh Sequence Info",
}
