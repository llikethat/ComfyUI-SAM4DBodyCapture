"""
Mesh Overlay Visualization Node for ComfyUI-SAM4DBodyCapture

Uses SAM3DBody's pyrender-based Renderer for proper mesh visualization.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict
import cv2


class SAM4DMeshSequenceOverlay:
    """
    Overlay mesh sequence on video frames using SAM3DBody's Renderer.
    
    Uses the same visualization approach as SAM-Body4D/SAM3DBody.
    
    Input: SAM4D_MESH_SEQUENCE from SAM4DBodyBatchProcess or SAM4DTemporalSmoothing
    Output: IMAGE with mesh rendered on top
    """
    
    RENDER_MODES = ["overlay", "mesh_only", "side_by_side"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Original video frames"
                }),
                "mesh_sequence": ("SAM4D_MESH_SEQUENCE", {
                    "tooltip": "Mesh sequence from SAM4DBodyBatchProcess or SAM4DTemporalSmoothing"
                }),
            },
            "optional": {
                "camera_intrinsics": ("CAMERA_INTRINSICS", {
                    "tooltip": "Camera intrinsics from MoGe2 (uses sequence params if not provided)"
                }),
                "render_mode": (cls.RENDER_MODES, {"default": "overlay"}),
                "mesh_color_r": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.1}),
                "mesh_color_g": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.1}),
                "mesh_color_b": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rendered_images",)
    FUNCTION = "render"
    CATEGORY = "SAM4DBodyCapture/Visualization"
    
    def render(
        self,
        images: torch.Tensor,
        mesh_sequence: Dict,
        camera_intrinsics: Dict = None,
        render_mode: str = "overlay",
        mesh_color_r: float = 0.9,
        mesh_color_g: float = 0.9,
        mesh_color_b: float = 0.7,
    ) -> Tuple[torch.Tensor]:
        """Render mesh sequence overlay using SAM3DBody's Renderer."""
        
        B, H, W, C = images.shape
        mesh_color = (mesh_color_r, mesh_color_g, mesh_color_b)
        
        # Extract mesh data from SAM4D_MESH_SEQUENCE format
        vertices_list = mesh_sequence.get("vertices", [])
        faces = mesh_sequence.get("faces", None)
        params = mesh_sequence.get("params", {})
        
        if not vertices_list:
            print("[MeshOverlay] No vertices in mesh sequence")
            return (images,)
        
        if faces is None:
            print("[MeshOverlay] No faces in mesh sequence")
            return (images,)
        
        # Ensure faces is numpy array
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        faces = np.asarray(faces)
        
        # Get camera translation and focal length from params
        cam_t_list = params.get("camera_t", [])
        focal_list = params.get("focal_length", [])
        
        # Get camera parameters from intrinsics or defaults
        if camera_intrinsics is not None:
            default_focal = camera_intrinsics.get("focal_length", max(H, W))
            cx = camera_intrinsics.get("cx", W / 2.0)
            cy = camera_intrinsics.get("cy", H / 2.0)
            per_frame_focal = camera_intrinsics.get("per_frame_focal", None)
            print(f"[MeshOverlay] Using camera intrinsics: focal={default_focal:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        else:
            default_focal = max(H, W)
            cx = W / 2.0
            cy = H / 2.0
            per_frame_focal = None
            print(f"[MeshOverlay] Using default camera: focal={default_focal:.1f}")
        
        # Try to import SAM3DBody's Renderer
        try:
            from sam_3d_body.visualization.renderer import Renderer
            USE_PYRENDER = True
            print("[MeshOverlay] Using SAM3DBody pyrender Renderer")
        except ImportError:
            USE_PYRENDER = False
            print("[MeshOverlay] SAM3DBody Renderer not available, using simple wireframe")
        
        result_frames = []
        
        for i in range(B):
            # Get frame image (convert to numpy uint8)
            img = images[i].cpu().numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            
            # Get vertices for this frame
            if i < len(vertices_list) and vertices_list[i] is not None:
                verts = vertices_list[i]
                if isinstance(verts, torch.Tensor):
                    verts = verts.cpu().numpy()
                verts = np.asarray(verts)
            elif len(vertices_list) > 0 and vertices_list[0] is not None:
                verts = np.asarray(vertices_list[0])
            else:
                result_frames.append(img)
                continue
            
            # Get camera translation for this frame
            if i < len(cam_t_list) and cam_t_list[i] is not None:
                cam_t = cam_t_list[i]
                if isinstance(cam_t, torch.Tensor):
                    cam_t = cam_t.cpu().numpy()
                cam_t = np.asarray(cam_t).flatten()
            else:
                # Default camera translation
                cam_t = np.array([0, 0, 2.5])
            
            # Get focal length for this frame
            if per_frame_focal is not None and i < len(per_frame_focal):
                focal = float(per_frame_focal[i])
            elif i < len(focal_list) and focal_list[i] is not None:
                focal = focal_list[i]
                if isinstance(focal, (list, np.ndarray, torch.Tensor)):
                    focal = float(np.asarray(focal).flatten()[0])
                else:
                    focal = float(focal)
            else:
                focal = default_focal
            
            if USE_PYRENDER:
                try:
                    renderer = Renderer(focal_length=focal, faces=faces)
                    
                    # SAM3DBody renderer expects camera_center
                    camera_center = [cx, cy]
                    
                    if render_mode == "overlay":
                        rendered = renderer(
                            vertices=verts,
                            cam_t=cam_t,
                            image=img.astype(np.float32),
                            mesh_base_color=mesh_color,
                            camera_center=camera_center,
                        )
                        rendered = (rendered * 255).astype(np.uint8)
                    elif render_mode == "mesh_only":
                        rendered = renderer(
                            vertices=verts,
                            cam_t=cam_t,
                            image=np.zeros_like(img, dtype=np.float32),
                            mesh_base_color=mesh_color,
                            scene_bg_color=(0, 0, 0),
                            camera_center=camera_center,
                        )
                        rendered = (rendered * 255).astype(np.uint8)
                    else:  # side_by_side
                        overlay = renderer(
                            vertices=verts,
                            cam_t=cam_t,
                            image=img.astype(np.float32),
                            mesh_base_color=mesh_color,
                            camera_center=camera_center,
                        )
                        overlay = (overlay * 255).astype(np.uint8)
                        rendered = np.hstack([img, overlay])
                    
                    result_frames.append(rendered)
                    
                except Exception as e:
                    print(f"[MeshOverlay] Renderer error frame {i}: {e}")
                    # Fallback to simple wireframe
                    result_frames.append(self._draw_wireframe(img, verts, faces, cam_t, focal, cx, cy, mesh_color))
            else:
                # Simple wireframe fallback
                result_frames.append(self._draw_wireframe(img, verts, faces, cam_t, focal, cx, cy, mesh_color))
            
            if (i + 1) % 10 == 0:
                print(f"[MeshOverlay] Rendered frame {i + 1}/{B}")
        
        # Stack results
        result_tensor = torch.from_numpy(np.stack(result_frames, axis=0)).float() / 255.0
        result_tensor = result_tensor.to(images.device)
        
        print(f"[MeshOverlay] Rendered {len(result_frames)} frames in '{render_mode}' mode")
        
        return (result_tensor,)
    
    def _draw_wireframe(self, img, verts, faces, cam_t, focal, cx, cy, mesh_color):
        """Simple wireframe rendering fallback."""
        result = img.copy()
        H, W = img.shape[:2]
        
        # Apply camera translation and flip for rendering coordinate system
        verts_cam = verts.copy()
        verts_cam[:, 0] = -verts_cam[:, 0]  # Flip X (matches SAM3DBody renderer)
        verts_cam = verts_cam + cam_t
        
        # Perspective projection
        z = verts_cam[:, 2:3]
        z = np.maximum(z, 0.1)
        
        x_2d = focal * verts_cam[:, 0:1] / z + cx
        y_2d = focal * verts_cam[:, 1:2] / z + cy
        
        pts_2d = np.concatenate([x_2d, y_2d], axis=1).astype(np.int32)
        
        # Draw wireframe edges
        color_bgr = (int(mesh_color[2] * 255), int(mesh_color[1] * 255), int(mesh_color[0] * 255))
        
        for face in faces:
            for j in range(3):
                p1 = pts_2d[face[j]]
                p2 = pts_2d[face[(j + 1) % 3]]
                
                # Check if both points are within bounds
                if (0 <= p1[0] < W and 0 <= p1[1] < H and
                    0 <= p2[0] < W and 0 <= p2[1] < H):
                    cv2.line(result, tuple(p1), tuple(p2), color_bgr, 1, cv2.LINE_AA)
        
        return result


class SAM4DDepthOverlay:
    """
    Overlay depth map on original images for visualization.
    
    Note: With alpha=1.0 you see ONLY depth. Use alpha=0.5 to see both!
    """
    
    COLORMAPS = ["viridis", "plasma", "magma", "inferno", "jet", "gray"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Original video frames"
                }),
                "depth_maps": ("IMAGE", {
                    "tooltip": "Depth maps (grayscale or RGB)"
                }),
            },
            "optional": {
                "colormap": (cls.COLORMAPS, {"default": "viridis"}),
                "alpha": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.1,
                    "tooltip": "Blend alpha: 0=original only, 1=depth only, 0.5=50/50 blend"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("overlaid_images",)
    FUNCTION = "overlay"
    CATEGORY = "SAM4DBodyCapture/Visualization"
    
    def overlay(
        self,
        images: torch.Tensor,
        depth_maps: torch.Tensor,
        colormap: str = "viridis",
        alpha: float = 0.5,
    ) -> Tuple[torch.Tensor]:
        """Overlay colored depth map on images."""
        B, H, W, C = images.shape
        
        cmap_dict = {
            "viridis": cv2.COLORMAP_VIRIDIS,
            "plasma": cv2.COLORMAP_PLASMA,
            "magma": cv2.COLORMAP_MAGMA,
            "inferno": cv2.COLORMAP_INFERNO,
            "jet": cv2.COLORMAP_JET,
            "gray": None,
        }
        cmap = cmap_dict.get(colormap, cv2.COLORMAP_VIRIDIS)
        
        print(f"[DepthOverlay] Blending {B} frames with alpha={alpha} ({colormap})")
        
        result_frames = []
        
        for i in range(B):
            # Get original image
            img = images[i].cpu().numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            
            # Get depth map
            if i < len(depth_maps):
                depth = depth_maps[i].cpu().numpy()
            else:
                depth = depth_maps[0].cpu().numpy()
            
            # Convert to single channel if needed
            if depth.ndim == 3:
                depth = depth.mean(axis=2)
            
            # Normalize depth to 0-255
            depth_min, depth_max = depth.min(), depth.max()
            if depth_max - depth_min > 1e-6:
                depth_norm = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth, dtype=np.uint8)
            
            # Apply colormap
            if cmap is not None:
                depth_color = cv2.applyColorMap(depth_norm, cmap)
            else:
                depth_color = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)
            
            # Resize depth to match image if needed
            if depth_color.shape[:2] != (H, W):
                depth_color = cv2.resize(depth_color, (W, H))
            
            # Blend: result = (1-alpha)*original + alpha*depth
            result = cv2.addWeighted(img, 1.0 - alpha, depth_color, alpha, 0)
            result_frames.append(result)
        
        result_tensor = torch.from_numpy(np.stack(result_frames, axis=0)).float() / 255.0
        result_tensor = result_tensor.to(images.device)
        
        return (result_tensor,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "SAM4DMeshSequenceOverlay": SAM4DMeshSequenceOverlay,
    "SAM4DDepthOverlay": SAM4DDepthOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM4DMeshSequenceOverlay": "👁️ Mesh Sequence Overlay",
    "SAM4DDepthOverlay": "👁️ Depth Overlay Preview",
}
