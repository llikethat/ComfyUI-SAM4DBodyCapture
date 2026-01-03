"""
Joint Debug Overlay Node - Simple diagnostic visualization

Overlays both joint data sources on footage with different colors:
- pred_keypoints_2d (MHR 70-joint): BLUE dots
- joint_coords projected (SMPLH 127-joint): RED dots

Supports two input modes:
1. mesh_sequence - uses data from SAM4D mesh sequence
2. Direct inputs - pred_keypoints_2d, joint_coords, pred_cam_t, focal_length
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Optional


def to_numpy(x):
    """Convert tensor to numpy array."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def project_to_2d(points_3d, focal, cam_t, img_w, img_h):
    """Project 3D points to 2D using SAM3DBody camera model."""
    points_3d = np.array(points_3d).copy()
    cam_t = np.array(cam_t).flatten()
    
    cx, cy = img_w / 2.0, img_h / 2.0
    
    if len(cam_t) < 3:
        return np.column_stack([np.full(len(points_3d), cx), np.full(len(points_3d), cy)])
    
    # Apply camera translation
    X = points_3d[:, 0] + cam_t[0]
    Y = points_3d[:, 1] + cam_t[1]
    Z = points_3d[:, 2] + cam_t[2]
    
    # Apply 180° rotation around X axis
    Y = -Y
    Z = -Z
    
    # Avoid division by zero
    Z = np.where(np.abs(Z) < 0.1, 0.1, Z)
    
    # Perspective projection
    x_2d = focal * X / Z + cx
    y_2d = focal * Y / Z + cy
    
    return np.stack([x_2d, y_2d], axis=1)


class JointDebugOverlay:
    """
    Debug node to visualize joint positions from different sources.
    
    BLUE dots = pred_keypoints_2d (raw values)
    RED dots = joint_coords (projected to 2D)
    
    Two input modes:
    1. Connect mesh_sequence (from SAM4D pipeline)
    2. Connect direct outputs from SAM3DBody node
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "mesh_sequence": ("SAM4D_MESH_SEQUENCE",),
                "pred_keypoints_2d": ("KEYPOINTS_2D",),
                "joint_coords": ("JOINT_COORDS",),
                "pred_cam_t": ("CAM_T",),
                "focal_length": ("FLOAT", {"default": 1000.0, "min": 100.0, "max": 5000.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("debug_overlay",)
    FUNCTION = "create_overlay"
    CATEGORY = "SAM4D/Debug"
    
    def create_overlay(
        self, 
        images: torch.Tensor, 
        mesh_sequence: Dict = None,
        pred_keypoints_2d = None,
        joint_coords = None,
        pred_cam_t = None,
        focal_length: float = 1000.0,
    ):
        print(f"\n[JointDebugOverlay] ========== JOINT DEBUG ==========")
        
        num_frames = images.shape[0]
        _, H, W, _ = images.shape
        
        # Determine data source
        if mesh_sequence is not None:
            print(f"[JointDebugOverlay] Using mesh_sequence input")
            params = mesh_sequence.get("params", {})
            keypoints_2d_list = params.get("keypoints_2d", [])
            joint_coords_list = params.get("joint_coords", [])
            camera_t_list = params.get("camera_t", [])
            focal_list = params.get("focal_length", [])
        else:
            print(f"[JointDebugOverlay] Using direct SAM3DBody outputs")
            # Convert direct inputs to lists
            keypoints_2d_list = self._to_list(pred_keypoints_2d, num_frames)
            joint_coords_list = self._to_list(joint_coords, num_frames)
            camera_t_list = self._to_list(pred_cam_t, num_frames)
            focal_list = [focal_length] * num_frames
        
        print(f"[JointDebugOverlay] Frames: {num_frames}, Size: {W}x{H}")
        print(f"[JointDebugOverlay] pred_keypoints_2d: {len(keypoints_2d_list) > 0 and keypoints_2d_list[0] is not None}")
        print(f"[JointDebugOverlay] joint_coords: {len(joint_coords_list) > 0 and joint_coords_list[0] is not None}")
        
        # Colors (BGR)
        BLUE = (255, 100, 100)  # pred_keypoints_2d
        RED = (100, 100, 255)   # joint_coords
        
        output_frames = []
        
        for i in range(num_frames):
            frame = (images[i].cpu().numpy() * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            focal = focal_list[i] if i < len(focal_list) and focal_list[i] is not None else 1000.0
            cam_t = to_numpy(camera_t_list[i]) if i < len(camera_t_list) and camera_t_list[i] is not None else np.array([0, 0, 5])
            if cam_t is not None and cam_t.ndim > 1:
                cam_t = cam_t.flatten()
            
            # ===== BLUE: pred_keypoints_2d (raw) =====
            if i < len(keypoints_2d_list) and keypoints_2d_list[i] is not None:
                kp2d = to_numpy(keypoints_2d_list[i])
                if kp2d.ndim == 3:
                    kp2d = kp2d.squeeze(0)
                
                if i == 0:
                    print(f"[JointDebugOverlay] BLUE pred_keypoints_2d shape: {kp2d.shape}")
                    print(f"[JointDebugOverlay] BLUE pred_keypoints_2d[0]: ({kp2d[0,0]:.1f}, {kp2d[0,1]:.1f})")
                
                for j in range(min(22, len(kp2d))):
                    x, y = int(kp2d[j, 0]), int(kp2d[j, 1])
                    if 0 <= x < W and 0 <= y < H:
                        cv2.circle(frame, (x, y), 6, BLUE, -1)
                        cv2.circle(frame, (x, y), 6, (0, 0, 0), 1)
            
            # ===== RED: joint_coords (projected) =====
            if i < len(joint_coords_list) and joint_coords_list[i] is not None:
                jc = to_numpy(joint_coords_list[i])
                if jc.ndim == 3:
                    jc = jc.squeeze(0)
                
                if i == 0:
                    print(f"[JointDebugOverlay] RED joint_coords shape: {jc.shape}")
                    print(f"[JointDebugOverlay] RED joint_coords[16] (head 3D): ({jc[16,0]:.3f}, {jc[16,1]:.3f}, {jc[16,2]:.3f})")
                    print(f"[JointDebugOverlay] cam_t: [{cam_t[0]:.3f}, {cam_t[1]:.3f}, {cam_t[2]:.3f}]")
                    print(f"[JointDebugOverlay] focal: {focal:.1f}px")
                
                jc_2d = project_to_2d(jc, focal, cam_t, W, H)
                
                if i == 0:
                    print(f"[JointDebugOverlay] RED joint_coords[16] projected: ({jc_2d[16,0]:.1f}, {jc_2d[16,1]:.1f})")
                
                for j in range(min(22, len(jc_2d))):
                    x, y = int(jc_2d[j, 0]), int(jc_2d[j, 1])
                    if 0 <= x < W and 0 <= y < H:
                        cv2.circle(frame, (x, y), 6, RED, -1)
                        cv2.circle(frame, (x, y), 6, (0, 0, 0), 1)
            
            # Legend
            cv2.putText(frame, "BLUE: pred_keypoints_2d", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLUE, 2)
            cv2.putText(frame, "RED: joint_coords projected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_frames.append(frame)
        
        output_tensor = torch.from_numpy(np.stack(output_frames, axis=0)).float() / 255.0
        
        print(f"[JointDebugOverlay] ====================================")
        
        return (output_tensor,)
    
    def _to_list(self, data, num_frames):
        """Convert tensor/array to list of per-frame data."""
        if data is None:
            return [None] * num_frames
        
        data = to_numpy(data)
        if data is None:
            return [None] * num_frames
        
        # If it's a batch (first dim = num_frames), split it
        if data.ndim >= 2 and len(data) == num_frames:
            return [data[i] for i in range(num_frames)]
        else:
            # Single frame data, replicate
            return [data] * num_frames


NODE_CLASS_MAPPINGS = {
    "JointDebugOverlay": JointDebugOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JointDebugOverlay": "Joint Debug Overlay",
}
