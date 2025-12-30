"""
Mesh Overlay Visualization Node for ComfyUI-SAM4DBodyCapture

Uses SAM3DBody's pyrender-based Renderer for proper mesh visualization.
Follows the same approach as SAM3DBody's process_multiple.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# IST timezone (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

def get_timestamp():
    """Get current timestamp in IST format."""
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")

# Set up pyrender for headless rendering BEFORE importing
if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import torch
import numpy as np
from typing import Tuple, Optional, Dict
import cv2


def _setup_sam3dbody_path():
    """Add SAM3DBody to path if available."""
    # Try common locations
    possible_paths = [
        Path(__file__).parent.parent.parent / "ComfyUI-SAM3DBody-main",
        Path(__file__).parent.parent.parent / "ComfyUI-SAM3DBody",
        Path(__file__).parent.parent.parent.parent / "ComfyUI-SAM3DBody-main",
        Path(__file__).parent.parent.parent.parent / "ComfyUI-SAM3DBody",
        Path.home() / "ComfyUI" / "custom_nodes" / "ComfyUI-SAM3DBody-main",
        Path.home() / "ComfyUI" / "custom_nodes" / "ComfyUI-SAM3DBody",
    ]
    
    for p in possible_paths:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
            print(f"[MeshOverlay] Added SAM3DBody path: {p}")
            return True
    return False

# Try to set up path on module load
_setup_sam3dbody_path()


class SAM4DMeshSequenceOverlay:
    """
    Overlay mesh sequence on video frames using SAM3DBody's Renderer.
    
    Uses the same visualization approach as SAM3DBody's _create_multi_person_preview.
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
    
    def _render_frame_pyrender(self, img_bgr, verts, faces, cam_t, focal, mesh_color, render_mode):
        """Render single frame using SAM3DBody's Renderer - matches their approach exactly."""
        from sam_3d_body.visualization.renderer import Renderer
        
        h, w = img_bgr.shape[:2]
        
        # Create renderer (same as SAM3DBody process_multiple.py)
        renderer = Renderer(
            focal_length=focal,
            faces=faces,
        )
        
        if render_mode == "mesh_only":
            # Render on black background
            render_result = renderer.render_rgba_multiple(
                [verts],
                [cam_t],
                render_res=(w, h),
                mesh_base_color=mesh_color,
            )
            
            if render_result is not None:
                # render_rgba_multiple returns float [0-1] with RGBA
                if render_result.shape[-1] == 4:
                    # Convert RGBA to BGR
                    render_rgb = (render_result[:, :, :3] * 255).astype(np.uint8)
                    render_bgr = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)
                    return render_bgr
            
            return np.zeros_like(img_bgr)
        
        else:  # overlay
            # Use render_rgba_multiple like SAM3DBody does
            render_result = renderer.render_rgba_multiple(
                [verts],
                [cam_t],
                render_res=(w, h),
                mesh_base_color=mesh_color,
            )
            
            if render_result is not None:
                # Composite onto original (same as SAM3DBody)
                if render_result.shape[-1] == 4:
                    # Alpha compositing
                    alpha = render_result[:, :, 3:4]  # Already 0-1 from render_rgba_multiple
                    render_rgb = (render_result[:, :, :3] * 255).astype(np.uint8)
                    render_bgr = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Blend: result = img * (1 - alpha) + render * alpha
                    result = (img_bgr.astype(np.float32) * (1 - alpha) + 
                             render_bgr.astype(np.float32) * alpha).astype(np.uint8)
                    return result
            
            return img_bgr
    
    def _render_frame_fallback(self, img_bgr, verts, faces, cam_t, focal, mesh_color):
        """Fallback: draw solid triangles with depth sorting (better than just points)."""
        result = img_bgr.copy()
        h, w = img_bgr.shape[:2]
        
        # Apply camera translation
        verts_world = verts.copy()
        verts_world = verts_world + cam_t
        
        # Apply 180° rotation around X (same as renderer.py line 209)
        verts_world[:, 1] *= -1
        verts_world[:, 2] *= -1
        
        # Perspective projection
        z = verts_world[:, 2:3]
        z = np.maximum(z, 0.1)
        
        x_2d = focal * verts_world[:, 0:1] / z + w / 2
        y_2d = focal * verts_world[:, 1:2] / z + h / 2
        
        pts_2d = np.concatenate([x_2d, y_2d], axis=1)
        
        # Calculate face depths for sorting (use average Z of vertices)
        face_depths = []
        for face in faces:
            avg_z = (verts_world[face[0], 2] + verts_world[face[1], 2] + verts_world[face[2], 2]) / 3
            face_depths.append(avg_z)
        
        # Sort faces by depth (far to near for painter's algorithm)
        sorted_indices = np.argsort(face_depths)[::-1]
        
        # Draw solid triangles
        color_bgr = (int(mesh_color[2] * 255), int(mesh_color[1] * 255), int(mesh_color[0] * 255))
        
        # Create overlay for blending
        overlay = result.copy()
        
        for idx in sorted_indices:
            face = faces[idx]
            pts = pts_2d[face].astype(np.int32)
            
            # Check if triangle is visible (all points roughly in frame)
            if (np.all(pts[:, 0] >= -w) and np.all(pts[:, 0] < 2*w) and
                np.all(pts[:, 1] >= -h) and np.all(pts[:, 1] < 2*h)):
                
                # Calculate face normal for basic shading
                v0, v1, v2 = verts_world[face[0]], verts_world[face[1]], verts_world[face[2]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                normal_len = np.linalg.norm(normal)
                
                if normal_len > 1e-6:
                    normal = normal / normal_len
                    # Simple lighting: dot product with view direction (0, 0, -1)
                    light_intensity = max(0.3, abs(normal[2]))
                else:
                    light_intensity = 0.5
                
                # Apply shading to color
                shaded_color = tuple(int(c * light_intensity) for c in color_bgr)
                
                # Draw filled triangle
                cv2.fillPoly(overlay, [pts], shaded_color)
        
        # Blend overlay with original (semi-transparent mesh)
        alpha = 0.7
        result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
        
        return result
    
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
            print(f"[{get_timestamp()}] [MeshOverlay] No vertices in mesh sequence")
            return (images,)
        
        if faces is None:
            print(f"[{get_timestamp()}] [MeshOverlay] No faces in mesh sequence")
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
            default_focal = camera_intrinsics.get("focal_length", 5000.0)
            per_frame_focal = camera_intrinsics.get("per_frame_focal", None)
            print(f"[MeshOverlay] Using camera intrinsics: focal={default_focal:.1f}")
        else:
            default_focal = 5000.0  # SAM3DBody default
            per_frame_focal = None
            print(f"[MeshOverlay] Using default focal: {default_focal:.1f}")
        
        # Check if pyrender renderer is available
        USE_PYRENDER = False
        pyrender_error = None
        try:
            # Try to set up path again
            _setup_sam3dbody_path()
            from sam_3d_body.visualization.renderer import Renderer
            
            # Also check pyrender itself
            import pyrender
            import trimesh
            
            USE_PYRENDER = True
            print(f"[{get_timestamp()}] [MeshOverlay] Using SAM3DBody pyrender Renderer")
        except ImportError as e:
            pyrender_error = str(e)
            print(f"[MeshOverlay] Pyrender not available: {pyrender_error}")
            print(f"[{get_timestamp()}] [MeshOverlay] Using solid mesh fallback (depth-sorted triangles)")
        except Exception as e:
            pyrender_error = str(e)
            print(f"[MeshOverlay] Renderer error: {pyrender_error}")
            print(f"[{get_timestamp()}] [MeshOverlay] Using solid mesh fallback")
        
        result_frames = []
        
        for i in range(B):
            # Get frame image as BGR (SAM3DBody uses BGR internally)
            img = images[i].cpu().numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Get vertices for this frame
            if i < len(vertices_list) and vertices_list[i] is not None:
                verts = vertices_list[i]
                if isinstance(verts, torch.Tensor):
                    verts = verts.cpu().numpy()
                verts = np.asarray(verts).copy()
            elif len(vertices_list) > 0 and vertices_list[0] is not None:
                verts = np.asarray(vertices_list[0]).copy()
            else:
                result_frames.append(img)  # Return RGB
                continue
            
            # Get camera translation for this frame
            if i < len(cam_t_list) and cam_t_list[i] is not None:
                cam_t = cam_t_list[i]
                if isinstance(cam_t, torch.Tensor):
                    cam_t = cam_t.cpu().numpy()
                cam_t = np.asarray(cam_t).flatten().copy()
            else:
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
            
            # Render frame
            if USE_PYRENDER:
                try:
                    rendered_bgr = self._render_frame_pyrender(
                        img_bgr, verts, faces, cam_t, focal, mesh_color, render_mode
                    )
                except Exception as e:
                    print(f"[MeshOverlay] Pyrender error frame {i}: {e}")
                    rendered_bgr = self._render_frame_fallback(
                        img_bgr, verts, faces, cam_t, focal, mesh_color
                    )
            else:
                rendered_bgr = self._render_frame_fallback(
                    img_bgr, verts, faces, cam_t, focal, mesh_color
                )
            
            # Convert back to RGB for ComfyUI
            if render_mode == "side_by_side":
                rendered_rgb = np.hstack([img, cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)])
            else:
                rendered_rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)
            
            result_frames.append(rendered_rgb)
            
            if (i + 1) % 10 == 0:
                print(f"[MeshOverlay] Rendered frame {i + 1}/{B}")
        
        # Stack results
        result_tensor = torch.from_numpy(np.stack(result_frames, axis=0)).float() / 255.0
        result_tensor = result_tensor.to(images.device)
        
        print(f"[MeshOverlay] Rendered {len(result_frames)} frames in '{render_mode}' mode")
        
        return (result_tensor,)


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
                depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)  # Convert to RGB
            else:
                depth_color = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2RGB)
            
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
