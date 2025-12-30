"""
MoGe2 Camera Intrinsics Node for ComfyUI-SAM4DBodyCapture

Uses MoGe2 (Monocular Geometry Estimation) to extract camera intrinsics
(focal length, FOV) from video frames - following SAM-Body4D approach.

MoGe2 outputs:
- Point map (3D points per pixel)
- Depth map  
- Camera FOV (fov_x in degrees)
- Intrinsics matrix (3x3 normalized)

The key formula for converting FOV to focal length:
    focal_px = (image_width / 2) / tan(fov_x * pi / 180 / 2)

This is used by SAM-3D-Body for proper 3D mesh projection.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, List
from datetime import datetime, timezone, timedelta
import os
import math

# IST timezone (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

def get_timestamp():
    """Get current timestamp in IST format."""
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")

# Try to import MoGe2
MOGE2_AVAILABLE = False
MoGeModel = None
try:
    # MoGe2 is typically installed via: pip install moge
    # or from: https://github.com/microsoft/MoGe
    from moge.model.v2 import MoGeModel as _MoGeModel
    MoGeModel = _MoGeModel
    MOGE2_AVAILABLE = True
    print(f"[{get_timestamp()}] [MoGe2] MoGe2 v2 model available")
except ImportError:
    try:
        from moge.model import MoGeModel as _MoGeModel
        MoGeModel = _MoGeModel
        MOGE2_AVAILABLE = True
        print(f"[{get_timestamp()}] [MoGe2] MoGe model available")
    except ImportError:
        pass


def fov_to_focal_length(fov_degrees: float, image_width: int) -> float:
    """
    Convert horizontal FOV in degrees to focal length in pixels.
    
    Formula: focal_px = (image_width / 2) / tan(fov_x * pi / 180 / 2)
    
    This is the standard pinhole camera model formula used by SAM-3D-Body.
    """
    fov_rad = fov_degrees * math.pi / 180.0
    focal_px = (image_width / 2.0) / math.tan(fov_rad / 2.0)
    return focal_px


def focal_length_to_fov(focal_px: float, image_width: int) -> float:
    """
    Convert focal length in pixels to horizontal FOV in degrees.
    
    Formula: fov_x = 2 * arctan(image_width / (2 * focal_px)) * 180 / pi
    """
    fov_rad = 2.0 * math.atan(image_width / (2.0 * focal_px))
    fov_degrees = fov_rad * 180.0 / math.pi
    return fov_degrees


def calculate_film_offsets(cx: float, cy: float, width: int, height: int, sensor_width_mm: float = 36.0) -> dict:
    """
    Calculate film offset X and Y in millimeters.
    
    Film offset is the displacement of the principal point (cx, cy) from the
    image center, expressed in film/sensor units (mm).
    
    In Maya:
    - Film Offset X: horizontal displacement in inches (we return mm, convert in Maya)
    - Film Offset Y: vertical displacement in inches
    
    Args:
        cx: Principal point X in pixels
        cy: Principal point Y in pixels  
        width: Image width in pixels
        height: Image height in pixels
        sensor_width_mm: Sensor/film width in mm (default 36mm for full-frame)
    
    Returns:
        dict with film_offset_x, film_offset_y in mm and inches
    """
    # Calculate sensor height based on aspect ratio
    aspect_ratio = height / width
    sensor_height_mm = sensor_width_mm * aspect_ratio
    
    # Calculate offset from center in pixels
    offset_x_px = cx - (width / 2.0)
    offset_y_px = cy - (height / 2.0)
    
    # Convert to mm (proportional to sensor size)
    # offset_mm = offset_px / image_size * sensor_size
    film_offset_x_mm = (offset_x_px / width) * sensor_width_mm
    film_offset_y_mm = (offset_y_px / height) * sensor_height_mm
    
    # Convert to inches (Maya uses inches for film offset)
    mm_to_inch = 0.0393701
    film_offset_x_inch = film_offset_x_mm * mm_to_inch
    film_offset_y_inch = film_offset_y_mm * mm_to_inch
    
    return {
        "film_offset_x_mm": film_offset_x_mm,
        "film_offset_y_mm": film_offset_y_mm,
        "film_offset_x_inch": film_offset_x_inch,
        "film_offset_y_inch": film_offset_y_inch,
        "sensor_width_mm": sensor_width_mm,
        "sensor_height_mm": sensor_height_mm,
    }


def focal_px_to_focal_mm(focal_px: float, image_width: int, sensor_width_mm: float = 36.0) -> float:
    """
    Convert focal length from pixels to millimeters.
    
    Formula: focal_mm = focal_px * sensor_width_mm / image_width
    """
    return focal_px * sensor_width_mm / image_width


class MoGe2CameraIntrinsics:
    """
    Extract camera intrinsics (focal length, FOV) from images using MoGe2.
    
    Following SAM-Body4D approach: use MoGe2 for accurate monocular
    geometry estimation including camera parameters.
    
    MoGe2 estimates the horizontal field of view (fov_x), which is
    converted to focal length using the standard pinhole camera formula.
    
    Outputs camera intrinsics that can be used for:
    - Proper 3D mesh projection onto 2D images
    - FBX camera animation export
    - Depth-aware rendering
    """
    
    MODELS = ["Ruicheng/moge-2-vitl-normal", "Ruicheng/moge-vitl"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "model_name": (cls.MODELS, {"default": "Ruicheng/moge-2-vitl-normal"}),
                "use_fp16": ("BOOLEAN", {"default": True, "tooltip": "Use FP16 for faster inference"}),
                "known_fov": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 1.0,
                    "tooltip": "If camera FOV is known, provide it here in degrees (0 = estimate automatically)"
                }),
            }
        }
    
    RETURN_TYPES = ("CAMERA_INTRINSICS", "IMAGE", "FLOAT")
    RETURN_NAMES = ("camera_intrinsics", "depth_maps", "focal_length")
    FUNCTION = "extract_intrinsics"
    CATEGORY = "SAM4DBodyCapture/Camera"
    
    def __init__(self):
        self.model = None
        self.model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self, model_name: str, use_fp16: bool = True):
        """Load MoGe2 model."""
        if self.model is not None and self.model_name == model_name:
            return True
        
        if not MOGE2_AVAILABLE or MoGeModel is None:
            print(f"[{get_timestamp()}] [MoGe2] MoGe2 not available.")
            print(f"[{get_timestamp()}] [MoGe2] Install with: pip install moge")
            print(f"[{get_timestamp()}] [MoGe2] Or clone from: https://github.com/microsoft/MoGe")
            return False
        
        try:
            print(f"[MoGe2] Loading model: {model_name}")
            
            # Load from HuggingFace
            self.model = MoGeModel.from_pretrained(model_name)
            
            if use_fp16 and self.device == "cuda":
                self.model = self.model.half()
            
            self.model = self.model.to(self.device)
            self.model.eval()
            self.model_name = model_name
            
            print(f"[MoGe2] Model loaded on {self.device}")
            return True
            
        except Exception as e:
            print(f"[MoGe2] Load error: {e}")
            return False
    
    def extract_intrinsics(
        self,
        images: torch.Tensor,
        model_name: str = "Ruicheng/moge-2-vitl-normal",
        use_fp16: bool = True,
        known_fov: float = 0.0,
    ) -> Tuple[Dict, torch.Tensor, float]:
        """
        Extract camera intrinsics from images using MoGe2.
        
        Args:
            images: Input images [B, H, W, C] in range [0, 1]
            model_name: MoGe2 model variant
            use_fp16: Use FP16 precision
            known_fov: If camera FOV is known in degrees (0 = estimate)
        
        Returns:
            (camera_intrinsics, depth_maps, focal_length)
        """
        B, H, W, C = images.shape
        
        # Default intrinsics (fallback if MoGe2 not available)
        # Using a reasonable default FOV of 60 degrees
        default_fov = 60.0 if known_fov <= 0 else known_fov
        default_focal = fov_to_focal_length(default_fov, W)
        
        # Principal point at center
        cx = W / 2.0
        cy = H / 2.0
        
        # Default sensor width 36mm (full-frame)
        default_sensor_width = 36.0
        film_offsets = calculate_film_offsets(cx, cy, W, H, default_sensor_width)
        focal_mm = focal_px_to_focal_mm(default_focal, W, default_sensor_width)
        
        intrinsics = {
            "focal_length": default_focal,
            "focal_length_mm": focal_mm,
            "fov_x": default_fov,
            "fov_y": focal_length_to_fov(default_focal, H),
            "cx": cx,
            "cy": cy,
            "width": W,
            "height": H,
            "per_frame_focal": [default_focal] * B,
            "per_frame_fov": [default_fov] * B,
            # Film offset data (for Maya/DCC)
            "film_offset_x_mm": film_offsets["film_offset_x_mm"],
            "film_offset_y_mm": film_offsets["film_offset_y_mm"],
            "film_offset_x_inch": film_offsets["film_offset_x_inch"],
            "film_offset_y_inch": film_offsets["film_offset_y_inch"],
            "sensor_width_mm": film_offsets["sensor_width_mm"],
            "sensor_height_mm": film_offsets["sensor_height_mm"],
        }
        
        # Default depth maps (gradient fallback for visualization)
        depth_maps = torch.zeros(B, H, W, 3, device=images.device)
        for i in range(B):
            y_grad = torch.linspace(0.3, 1.0, H, device=images.device).view(H, 1).expand(H, W)
            depth_maps[i] = y_grad.unsqueeze(-1).repeat(1, 1, 3)
        
        # If known FOV is provided, use it directly
        if known_fov > 0:
            print(f"[MoGe2] Using provided FOV: {known_fov:.1f}° → Focal: {default_focal:.1f}px ({focal_mm:.1f}mm)")
            print(f"[MoGe2] Principal Point: cx={cx:.1f}, cy={cy:.1f}")
            print(f"[MoGe2] Film Offset: X={film_offsets['film_offset_x_mm']:.3f}mm, Y={film_offsets['film_offset_y_mm']:.3f}mm")
            return (intrinsics, depth_maps, float(default_focal))
        
        # Try to load MoGe2
        if not self.load_model(model_name, use_fp16):
            print(f"[{get_timestamp()}] [MoGe2] Using fallback intrinsics (default FOV=60°)")
            print(f"[MoGe2] Principal Point: cx={cx:.1f}, cy={cy:.1f}")
            print(f"[MoGe2] Film Offset: X={film_offsets['film_offset_x_mm']:.3f}mm, Y={film_offsets['film_offset_y_mm']:.3f}mm")
            return (intrinsics, depth_maps, float(default_focal))
        
        # Process with MoGe2
        try:
            print(f"[MoGe2] Processing {B} frames...")
            
            per_frame_focal = []
            per_frame_fov = []
            all_depths = []
            
            for i in range(B):
                # MoGe2 expects a torch tensor, not numpy array
                img_tensor = images[i]  # [H, W, C] in range [0, 1]
                
                # Convert to [C, H, W] format and scale to [0, 255] range
                # MoGe2 v2 expects tensor in [C, H, W] format
                if img_tensor.dim() == 3 and img_tensor.shape[-1] == 3:
                    # [H, W, C] -> [C, H, W]
                    img_tensor = img_tensor.permute(2, 0, 1)
                
                # Scale to 0-255 if needed
                if img_tensor.max() <= 1.0:
                    img_tensor = img_tensor * 255.0
                
                # Move to correct device and dtype
                img_tensor = img_tensor.to(self.device)
                if hasattr(self.model, 'dtype'):
                    img_tensor = img_tensor.to(self.model.dtype)
                else:
                    img_tensor = img_tensor.half() if self.device == "cuda" else img_tensor.float()
                
                # Run MoGe2 inference
                with torch.no_grad():
                    output = self.model.infer(img_tensor)
                
                # Extract FOV from output
                # MoGe2 outputs: points, depth, mask, fov_x, intrinsics
                fov_x = output.get('fov_x', default_fov)
                if isinstance(fov_x, torch.Tensor):
                    fov_x = fov_x.item()
                
                # Calculate focal length from FOV using the standard formula
                focal = fov_to_focal_length(fov_x, W)
                
                per_frame_focal.append(focal)
                per_frame_fov.append(fov_x)
                
                # Process depth map
                depth = output.get('depth', None)
                if depth is not None:
                    if isinstance(depth, torch.Tensor):
                        depth = depth.cpu().numpy()
                    # Normalize to [0, 1]
                    depth_min, depth_max = depth.min(), depth.max()
                    if depth_max - depth_min > 1e-6:
                        depth = (depth - depth_min) / (depth_max - depth_min)
                    else:
                        depth = np.zeros_like(depth)
                    depth_rgb = np.stack([depth, depth, depth], axis=-1)
                    all_depths.append(depth_rgb.astype(np.float32))
                else:
                    # Fallback gradient depth
                    y_grad = np.linspace(0.3, 1.0, H).reshape(H, 1)
                    y_grad = np.tile(y_grad, (1, W))
                    all_depths.append(np.stack([y_grad, y_grad, y_grad], axis=-1).astype(np.float32))
                
                if i == 0:
                    print(f"[MoGe2] Frame 0: FOV={fov_x:.1f}°, Focal={focal:.1f}px")
            
            # Average focal length across frames for consistency
            avg_focal = np.mean(per_frame_focal)
            avg_fov = np.mean(per_frame_fov)
            
            # Principal point (cx, cy) - default to image center
            # MoGe2 currently doesn't provide off-center principal point
            cx = W / 2.0
            cy = H / 2.0
            
            # Calculate film offsets (for Maya camera)
            # Default sensor width 36mm (full-frame), can be overridden in export
            default_sensor_width = 36.0
            film_offsets = calculate_film_offsets(cx, cy, W, H, default_sensor_width)
            focal_mm = focal_px_to_focal_mm(avg_focal, W, default_sensor_width)
            
            # Update intrinsics
            intrinsics = {
                "focal_length": float(avg_focal),
                "focal_length_mm": float(focal_mm),
                "fov_x": float(avg_fov),
                "fov_y": float(focal_length_to_fov(avg_focal, H)),
                "cx": cx,
                "cy": cy,
                "width": W,
                "height": H,
                "per_frame_focal": per_frame_focal,
                "per_frame_fov": per_frame_fov,
                # Film offset data (for Maya/DCC)
                "film_offset_x_mm": film_offsets["film_offset_x_mm"],
                "film_offset_y_mm": film_offsets["film_offset_y_mm"],
                "film_offset_x_inch": film_offsets["film_offset_x_inch"],
                "film_offset_y_inch": film_offsets["film_offset_y_inch"],
                "sensor_width_mm": film_offsets["sensor_width_mm"],
                "sensor_height_mm": film_offsets["sensor_height_mm"],
            }
            
            # Stack depth maps
            depth_maps = torch.from_numpy(np.stack(all_depths, axis=0)).float()
            depth_maps = depth_maps.to(images.device)
            
            # Debug logging
            print(f"[MoGe2] Average: FOV={avg_fov:.1f}°, Focal={avg_focal:.1f}px ({focal_mm:.1f}mm @ {default_sensor_width}mm sensor)")
            print(f"[MoGe2] Principal Point: cx={cx:.1f}, cy={cy:.1f}")
            print(f"[MoGe2] Film Offset: X={film_offsets['film_offset_x_mm']:.3f}mm ({film_offsets['film_offset_x_inch']:.4f}in), Y={film_offsets['film_offset_y_mm']:.3f}mm ({film_offsets['film_offset_y_inch']:.4f}in)")
            
            return (intrinsics, depth_maps, float(avg_focal))
            
        except Exception as e:
            print(f"[MoGe2] Processing error: {e}")
            import traceback
            traceback.print_exc()
            return (intrinsics, depth_maps, float(default_focal))


class CameraIntrinsicsFromFOV:
    """
    Create camera intrinsics from known FOV.
    
    Use this if you know your camera's field of view.
    Common values:
    - iPhone: ~70° horizontal
    - GoPro wide: ~120° horizontal
    - Standard lens: ~45-55° horizontal
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fov_horizontal": ("FLOAT", {
                    "default": 60.0,
                    "min": 10.0,
                    "max": 180.0,
                    "tooltip": "Horizontal field of view in degrees"
                }),
            }
        }
    
    RETURN_TYPES = ("CAMERA_INTRINSICS", "FLOAT")
    RETURN_NAMES = ("camera_intrinsics", "focal_length")
    FUNCTION = "create_intrinsics"
    CATEGORY = "SAM4DBodyCapture/Camera"
    
    def create_intrinsics(
        self,
        images: torch.Tensor,
        fov_horizontal: float = 60.0,
    ) -> Tuple[Dict, float]:
        """Create camera intrinsics from FOV."""
        B, H, W, C = images.shape
        
        # Calculate focal length from FOV
        # f = (W/2) / tan(fov/2)
        focal_length = (W / 2.0) / np.tan(np.radians(fov_horizontal) / 2.0)
        fov_vertical = 2 * np.degrees(np.arctan((H / 2.0) / focal_length))
        
        # Principal point at center
        cx = W / 2.0
        cy = H / 2.0
        
        # Default sensor width 36mm (full-frame)
        default_sensor_width = 36.0
        film_offsets = calculate_film_offsets(cx, cy, W, H, default_sensor_width)
        focal_mm = focal_px_to_focal_mm(focal_length, W, default_sensor_width)
        
        intrinsics = {
            "focal_length": float(focal_length),
            "focal_length_mm": float(focal_mm),
            "fov_x": float(fov_horizontal),
            "fov_y": float(fov_vertical),
            "cx": cx,
            "cy": cy,
            "width": W,
            "height": H,
            "per_frame_focal": [focal_length] * B,
            "per_frame_fov": [fov_horizontal] * B,
            # Film offset data (for Maya/DCC)
            "film_offset_x_mm": film_offsets["film_offset_x_mm"],
            "film_offset_y_mm": film_offsets["film_offset_y_mm"],
            "film_offset_x_inch": film_offsets["film_offset_x_inch"],
            "film_offset_y_inch": film_offsets["film_offset_y_inch"],
            "sensor_width_mm": film_offsets["sensor_width_mm"],
            "sensor_height_mm": film_offsets["sensor_height_mm"],
        }
        
        print(f"[Camera] FOV: {fov_horizontal:.1f}° → Focal: {focal_length:.1f}px ({focal_mm:.1f}mm)")
        print(f"[Camera] Principal Point: cx={cx:.1f}, cy={cy:.1f}")
        print(f"[Camera] Film Offset: X={film_offsets['film_offset_x_mm']:.3f}mm, Y={film_offsets['film_offset_y_mm']:.3f}mm")
        
        return (intrinsics, float(focal_length))


class CameraIntrinsicsInfo:
    """Display camera intrinsics information."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_intrinsics": ("CAMERA_INTRINSICS",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "get_info"
    CATEGORY = "SAM4DBodyCapture/Camera"
    OUTPUT_NODE = True
    
    def get_info(self, camera_intrinsics: Dict) -> Tuple[str]:
        """Get formatted camera intrinsics info."""
        info_lines = [
            "=== Camera Intrinsics ===",
            f"Image Size: {camera_intrinsics.get('width', 'N/A')} x {camera_intrinsics.get('height', 'N/A')}",
            f"Focal Length: {camera_intrinsics.get('focal_length', 'N/A'):.2f} px",
        ]
        
        # Add focal in mm if available
        if 'focal_length_mm' in camera_intrinsics:
            info_lines.append(f"Focal Length: {camera_intrinsics['focal_length_mm']:.2f} mm")
        
        info_lines.extend([
            f"FOV Horizontal: {camera_intrinsics.get('fov_x', 'N/A'):.2f}°",
            f"FOV Vertical: {camera_intrinsics.get('fov_y', 'N/A'):.2f}°",
            f"Principal Point: cx={camera_intrinsics.get('cx', 'N/A'):.1f}, cy={camera_intrinsics.get('cy', 'N/A'):.1f}",
        ])
        
        # Film offset info if available
        if 'film_offset_x_mm' in camera_intrinsics:
            info_lines.append("--- Film Offset (for Maya) ---")
            info_lines.append(f"Film Offset X: {camera_intrinsics['film_offset_x_mm']:.3f} mm ({camera_intrinsics.get('film_offset_x_inch', 0):.4f} in)")
            info_lines.append(f"Film Offset Y: {camera_intrinsics['film_offset_y_mm']:.3f} mm ({camera_intrinsics.get('film_offset_y_inch', 0):.4f} in)")
        
        if 'sensor_width_mm' in camera_intrinsics:
            info_lines.append(f"Sensor: {camera_intrinsics['sensor_width_mm']:.1f} x {camera_intrinsics.get('sensor_height_mm', 0):.1f} mm")
        
        # Per-frame info if available
        per_frame_focal = camera_intrinsics.get('per_frame_focal', [])
        if len(per_frame_focal) > 1:
            info_lines.append(f"Frames: {len(per_frame_focal)}")
            info_lines.append(f"Focal Range: {min(per_frame_focal):.1f} - {max(per_frame_focal):.1f} px")
        
        info = "\n".join(info_lines)
        print(info)
        
        return (info,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "MoGe2CameraIntrinsics": MoGe2CameraIntrinsics,
    "CameraIntrinsicsFromFOV": CameraIntrinsicsFromFOV,
    "CameraIntrinsicsInfo": CameraIntrinsicsInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoGe2CameraIntrinsics": "MoGe2 Camera Intrinsics",
    "CameraIntrinsicsFromFOV": "Camera Intrinsics from FOV",
    "CameraIntrinsicsInfo": "Camera Intrinsics Info",
}
