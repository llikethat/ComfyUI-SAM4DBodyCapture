"""
Mesh Overlay Visualization Node for ComfyUI-SAM4DBodyCapture

Renders 3D body mesh overlaid on original background images for verification.
Uses camera intrinsics (from MoGe2 or manual) for proper projection.

This allows you to verify that:
1. The 3D mesh aligns with the person in the video
2. Camera intrinsics are correct
3. The reconstruction quality is acceptable before FBX export
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, List
import cv2

# Try to import trimesh for mesh handling
TRIMESH_AVAILABLE = False
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    pass


def project_3d_to_2d(
    points_3d: np.ndarray,
    focal_length: float,
    cx: float,
    cy: float,
    rotation: np.ndarray = None,
    translation: np.ndarray = None,
) -> np.ndarray:
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points_3d: 3D points [N, 3]
        focal_length: Camera focal length in pixels
        cx, cy: Principal point
        rotation: Optional rotation matrix [3, 3]
        translation: Optional translation vector [3]
    
    Returns:
        2D points [N, 2]
    """
    points = points_3d.copy()
    
    # Apply rotation and translation if provided
    if rotation is not None:
        points = points @ rotation.T
    if translation is not None:
        points = points + translation
    
    # Perspective projection
    # x' = f * X / Z + cx
    # y' = f * Y / Z + cy
    z = points[:, 2:3]
    z = np.maximum(z, 1e-6)  # Avoid division by zero
    
    x_2d = focal_length * points[:, 0:1] / z + cx
    y_2d = focal_length * points[:, 1:2] / z + cy
    
    return np.concatenate([x_2d, y_2d], axis=1)


def draw_mesh_wireframe(
    image: np.ndarray,
    vertices_2d: np.ndarray,
    faces: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Draw mesh wireframe on image.
    
    Args:
        image: Background image [H, W, 3]
        vertices_2d: 2D vertex positions [V, 2]
        faces: Face indices [F, 3]
        color: Line color (B, G, R)
        thickness: Line thickness
        alpha: Overlay transparency
    
    Returns:
        Image with wireframe overlay
    """
    overlay = image.copy()
    H, W = image.shape[:2]
    
    for face in faces:
        pts = vertices_2d[face].astype(np.int32)
        
        # Check if all points are within image bounds
        if np.all((pts >= 0) & (pts[:, 0] < W) & (pts[:, 1] < H)):
            # Draw triangle edges
            for i in range(3):
                p1 = tuple(pts[i])
                p2 = tuple(pts[(i + 1) % 3])
                cv2.line(overlay, p1, p2, color, thickness)
    
    # Blend with original
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return result


def draw_mesh_solid(
    image: np.ndarray,
    vertices_2d: np.ndarray,
    faces: np.ndarray,
    depths: np.ndarray,
    color: Tuple[int, int, int] = (100, 200, 100),
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Draw solid mesh with depth-based shading.
    
    Args:
        image: Background image [H, W, 3]
        vertices_2d: 2D vertex positions [V, 2]
        faces: Face indices [F, 3]
        depths: Depth values per vertex [V]
        color: Base color (B, G, R)
        alpha: Overlay transparency
    
    Returns:
        Image with solid mesh overlay
    """
    overlay = np.zeros_like(image)
    H, W = image.shape[:2]
    
    # Sort faces by depth (back to front for painter's algorithm)
    face_depths = depths[faces].mean(axis=1)
    sorted_indices = np.argsort(-face_depths)  # Far to near
    
    for idx in sorted_indices:
        face = faces[idx]
        pts = vertices_2d[face].astype(np.int32)
        
        # Check bounds
        if np.all((pts >= 0) & (pts[:, 0] < W) & (pts[:, 1] < H)):
            # Depth-based shading
            depth = face_depths[idx]
            depth_norm = (depth - depths.min()) / (depths.max() - depths.min() + 1e-6)
            shade = 0.5 + 0.5 * (1 - depth_norm)  # Closer = brighter
            
            face_color = tuple(int(c * shade) for c in color)
            cv2.fillPoly(overlay, [pts], face_color)
    
    # Blend with original
    mask = (overlay.sum(axis=2) > 0).astype(np.float32)
    mask = mask[:, :, np.newaxis]
    result = image * (1 - mask * alpha) + overlay * mask * alpha
    return result.astype(np.uint8)


def draw_skeleton(
    image: np.ndarray,
    joints_2d: np.ndarray,
    skeleton_edges: List[Tuple[int, int]] = None,
    joint_color: Tuple[int, int, int] = (0, 0, 255),
    bone_color: Tuple[int, int, int] = (255, 0, 0),
    joint_radius: int = 4,
    bone_thickness: int = 2,
) -> np.ndarray:
    """
    Draw skeleton joints and bones on image.
    
    Args:
        image: Background image [H, W, 3]
        joints_2d: 2D joint positions [J, 2]
        skeleton_edges: List of (parent, child) joint indices
        joint_color: Joint circle color
        bone_color: Bone line color
        joint_radius: Joint circle radius
        bone_thickness: Bone line thickness
    
    Returns:
        Image with skeleton overlay
    """
    result = image.copy()
    H, W = image.shape[:2]
    
    # Default skeleton edges (SMPL-like)
    if skeleton_edges is None:
        skeleton_edges = [
            (0, 1), (0, 2), (0, 3),  # Pelvis to legs and spine
            (1, 4), (2, 5),  # Hips to knees
            (4, 7), (5, 8),  # Knees to ankles
            (3, 6), (6, 9),  # Spine
            (9, 12), (9, 13), (9, 14),  # Spine to shoulders and head
            (12, 15),  # Neck to head
            (13, 16), (14, 17),  # Shoulders to elbows
            (16, 18), (17, 19),  # Elbows to wrists
            (18, 20), (19, 21),  # Wrists to hands
        ]
    
    # Draw bones
    for parent, child in skeleton_edges:
        if parent < len(joints_2d) and child < len(joints_2d):
            p1 = joints_2d[parent].astype(np.int32)
            p2 = joints_2d[child].astype(np.int32)
            
            # Check bounds
            if (0 <= p1[0] < W and 0 <= p1[1] < H and
                0 <= p2[0] < W and 0 <= p2[1] < H):
                cv2.line(result, tuple(p1), tuple(p2), bone_color, bone_thickness)
    
    # Draw joints
    for joint in joints_2d:
        pt = joint.astype(np.int32)
        if 0 <= pt[0] < W and 0 <= pt[1] < H:
            cv2.circle(result, tuple(pt), joint_radius, joint_color, -1)
    
    return result


class SAM4DMeshOverlay:
    """
    Overlay 3D mesh on background images for visualization.
    
    Takes:
    - Background images
    - 3D mesh vertices (from SAM3DBody or similar)
    - Camera intrinsics (from MoGe2 or manual)
    
    Outputs overlaid images for verification.
    """
    
    RENDER_MODES = ["wireframe", "solid", "skeleton", "wireframe+skeleton"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "vertices": ("VERTICES",),  # [B, V, 3] or [V, 3]
            },
            "optional": {
                "camera_intrinsics": ("CAMERA_INTRINSICS",),
                "faces": ("FACES",),  # [F, 3]
                "joints": ("JOINTS",),  # [B, J, 3] or [J, 3]
                "render_mode": (cls.RENDER_MODES, {"default": "wireframe"}),
                "color_r": ("INT", {"default": 0, "min": 0, "max": 255}),
                "color_g": ("INT", {"default": 255, "min": 0, "max": 255}),
                "color_b": ("INT", {"default": 0, "min": 0, "max": 255}),
                "alpha": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "line_thickness": ("INT", {"default": 1, "min": 1, "max": 5}),
                "camera_distance": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.5,
                    "max": 20.0,
                    "tooltip": "Distance from camera to subject center (meters)"
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
        vertices: torch.Tensor,
        camera_intrinsics: Dict = None,
        faces: torch.Tensor = None,
        joints: torch.Tensor = None,
        render_mode: str = "wireframe",
        color_r: int = 0,
        color_g: int = 255,
        color_b: int = 0,
        alpha: float = 0.5,
        line_thickness: int = 1,
        camera_distance: float = 3.0,
    ) -> Tuple[torch.Tensor]:
        """
        Overlay 3D mesh on images.
        """
        B, H, W, C = images.shape
        color = (color_b, color_g, color_r)  # OpenCV uses BGR
        
        # Get camera parameters
        if camera_intrinsics is not None:
            focal_length = camera_intrinsics.get("focal_length", max(H, W))
            cx = camera_intrinsics.get("cx", W / 2.0)
            cy = camera_intrinsics.get("cy", H / 2.0)
        else:
            # Default camera (approximate)
            focal_length = max(H, W)
            cx = W / 2.0
            cy = H / 2.0
        
        print(f"[Overlay] Focal: {focal_length:.1f}px, Principal: ({cx:.1f}, {cy:.1f})")
        
        # Ensure vertices are per-frame
        vertices_np = vertices.cpu().numpy()
        if vertices_np.ndim == 2:
            # Single mesh for all frames
            vertices_np = np.tile(vertices_np[np.newaxis], (B, 1, 1))
        
        # Get faces
        if faces is not None:
            faces_np = faces.cpu().numpy().astype(np.int32)
        else:
            # Generate default faces if not provided (assumes SMPL-like topology)
            faces_np = None
        
        # Get joints if provided
        joints_np = None
        if joints is not None:
            joints_np = joints.cpu().numpy()
            if joints_np.ndim == 2:
                joints_np = np.tile(joints_np[np.newaxis], (B, 1, 1))
        
        # Process each frame
        result_frames = []
        
        for i in range(B):
            # Get frame image
            img = images[i].cpu().numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            
            # Get frame vertices
            verts = vertices_np[i] if i < len(vertices_np) else vertices_np[0]
            
            # Center and scale mesh
            verts_centered = verts - verts.mean(axis=0, keepdims=True)
            
            # Position mesh at camera_distance in front of camera
            translation = np.array([0, 0, camera_distance])
            
            # Project to 2D
            verts_2d = project_3d_to_2d(
                verts_centered,
                focal_length=focal_length,
                cx=cx,
                cy=cy,
                translation=translation,
            )
            
            # Render based on mode
            if render_mode == "wireframe" and faces_np is not None:
                result = draw_mesh_wireframe(
                    img, verts_2d, faces_np,
                    color=color, thickness=line_thickness, alpha=alpha
                )
            elif render_mode == "solid" and faces_np is not None:
                depths = verts_centered[:, 2]
                result = draw_mesh_solid(
                    img, verts_2d, faces_np, depths,
                    color=color, alpha=alpha
                )
            elif render_mode == "skeleton" and joints_np is not None:
                jnts = joints_np[i] if i < len(joints_np) else joints_np[0]
                jnts_centered = jnts - jnts.mean(axis=0, keepdims=True)
                jnts_2d = project_3d_to_2d(
                    jnts_centered,
                    focal_length=focal_length,
                    cx=cx,
                    cy=cy,
                    translation=translation,
                )
                result = draw_skeleton(
                    img, jnts_2d,
                    joint_color=(color_b, color_g, color_r),
                    bone_color=(color_r, color_g, color_b),
                )
            elif render_mode == "wireframe+skeleton":
                result = img.copy()
                if faces_np is not None:
                    result = draw_mesh_wireframe(
                        result, verts_2d, faces_np,
                        color=color, thickness=line_thickness, alpha=alpha * 0.5
                    )
                if joints_np is not None:
                    jnts = joints_np[i] if i < len(joints_np) else joints_np[0]
                    jnts_centered = jnts - jnts.mean(axis=0, keepdims=True)
                    jnts_2d = project_3d_to_2d(
                        jnts_centered,
                        focal_length=focal_length,
                        cx=cx,
                        cy=cy,
                        translation=translation,
                    )
                    result = draw_skeleton(result, jnts_2d)
            else:
                # Just draw vertices as points
                result = img.copy()
                for pt in verts_2d:
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= x < W and 0 <= y < H:
                        cv2.circle(result, (x, y), 2, color, -1)
            
            result_frames.append(result)
        
        # Stack results
        result_tensor = torch.from_numpy(np.stack(result_frames, axis=0)).float() / 255.0
        result_tensor = result_tensor.to(images.device)
        
        print(f"[Overlay] Rendered {B} frames in '{render_mode}' mode")
        
        return (result_tensor,)


class SAM4DDepthOverlay:
    """
    Overlay depth map on original images for visualization.
    
    Useful for verifying depth estimation quality.
    """
    
    COLORMAPS = ["viridis", "plasma", "magma", "inferno", "jet", "gray"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "depth_maps": ("IMAGE",),  # Depth as grayscale or RGB
            },
            "optional": {
                "colormap": (cls.COLORMAPS, {"default": "viridis"}),
                "alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
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
        
        # Get colormap
        cmap_dict = {
            "viridis": cv2.COLORMAP_VIRIDIS,
            "plasma": cv2.COLORMAP_PLASMA,
            "magma": cv2.COLORMAP_MAGMA,
            "inferno": cv2.COLORMAP_INFERNO,
            "jet": cv2.COLORMAP_JET,
            "gray": None,
        }
        cmap = cmap_dict.get(colormap, cv2.COLORMAP_VIRIDIS)
        
        result_frames = []
        
        for i in range(B):
            # Get image
            img = images[i].cpu().numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            # Get depth
            depth = depth_maps[i].cpu().numpy()
            if depth.ndim == 3:
                depth = depth.mean(axis=2)  # Convert RGB to grayscale
            
            # Normalize depth to 0-255
            depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
            
            # Apply colormap
            if cmap is not None:
                depth_color = cv2.applyColorMap(depth_norm, cmap)
            else:
                depth_color = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)
            
            # Blend
            result = cv2.addWeighted(img, 1 - alpha, depth_color, alpha, 0)
            result_frames.append(result)
        
        result_tensor = torch.from_numpy(np.stack(result_frames, axis=0)).float() / 255.0
        result_tensor = result_tensor.to(images.device)
        
        return (result_tensor,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "SAM4DMeshOverlay": SAM4DMeshOverlay,
    "SAM4DDepthOverlay": SAM4DDepthOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM4DMeshOverlay": "SAM4D Mesh Overlay",
    "SAM4DDepthOverlay": "SAM4D Depth Overlay",
}
