"""
SAM4D Temporal Fusion and Mesh Processing Nodes

Provides:
- Temporal Fusion (Kalman/EMA smoothing across frames)
- Mesh Sequence Export (FBX/Alembic/OBJ)
"""

import os
import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional, List
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict

# Import our pipeline types
from .sam4d_pipeline import SAM4DMeshSequence


# ============================================================================
# Temporal Smoothing Functions
# ============================================================================

def ema_smooth_1d(signal: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Exponential Moving Average smoothing.
    
    Args:
        signal: 1D array to smooth
        alpha: Smoothing factor (0-1), lower = more smooth
    
    Returns:
        Smoothed signal
    """
    result = np.zeros_like(signal)
    result[0] = signal[0]
    
    for i in range(1, len(signal)):
        result[i] = alpha * signal[i] + (1 - alpha) * result[i - 1]
    
    return result


def bidirectional_ema(signal: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Bidirectional EMA (forward + backward averaged) for better smoothing.
    """
    forward = ema_smooth_1d(signal, alpha)
    backward = ema_smooth_1d(signal[::-1], alpha)[::-1]
    return (forward + backward) / 2


def smooth_vertices_sequence(
    vertices_list: List[np.ndarray],
    method: str = "gaussian",
    sigma: float = 1.0,
    alpha: float = 0.3,
) -> List[np.ndarray]:
    """
    Smooth a sequence of mesh vertices temporally.
    
    Args:
        vertices_list: List of [N, 3] vertex arrays
        method: "gaussian" or "ema"
        sigma: Gaussian sigma (for gaussian method)
        alpha: EMA alpha (for ema method)
    
    Returns:
        List of smoothed vertex arrays
    """
    if len(vertices_list) < 3:
        return vertices_list
    
    # Stack into [T, N, 3]
    vertices_stack = np.stack(vertices_list, axis=0)
    T, N, D = vertices_stack.shape
    
    # Smooth along time axis for each vertex coordinate
    smoothed = np.zeros_like(vertices_stack)
    
    for v in range(N):
        for d in range(D):
            signal = vertices_stack[:, v, d]
            
            if method == "gaussian":
                smoothed[:, v, d] = gaussian_filter1d(signal, sigma=sigma)
            elif method == "ema":
                smoothed[:, v, d] = bidirectional_ema(signal, alpha=alpha)
            else:
                smoothed[:, v, d] = signal
    
    return [smoothed[t] for t in range(T)]


def smooth_params_sequence(
    params: Dict[str, List],
    method: str = "gaussian",
    sigma: float = 1.0,
    alpha: float = 0.3,
) -> Dict[str, List]:
    """
    Smooth SMPL/HMR parameters temporally.
    
    Handles:
    - global_orient: [T, 3] rotation
    - body_pose: [T, 69] body joint rotations
    - betas: [T, 10] shape parameters (usually constant)
    - transl: [T, 3] translation
    """
    smoothed_params = {}
    
    for key, values in params.items():
        if not values:
            smoothed_params[key] = values
            continue
        
        # Convert to numpy array
        if isinstance(values[0], torch.Tensor):
            arr = torch.stack(values).cpu().numpy()
        else:
            arr = np.array(values)
        
        T = arr.shape[0]
        
        if T < 3:
            smoothed_params[key] = values
            continue
        
        # Don't smooth shape parameters (betas) - they should be consistent
        if key == "betas":
            # Average across frames instead
            avg_betas = arr.mean(axis=0, keepdims=True)
            smoothed_params[key] = [avg_betas[0]] * T
            continue
        
        # Smooth other parameters
        if arr.ndim == 1:
            if method == "gaussian":
                smoothed = gaussian_filter1d(arr, sigma=sigma)
            else:
                smoothed = bidirectional_ema(arr, alpha=alpha)
            smoothed_params[key] = list(smoothed)
        else:
            smoothed_arr = np.zeros_like(arr)
            for d in range(arr.shape[1]):
                signal = arr[:, d]
                if method == "gaussian":
                    smoothed_arr[:, d] = gaussian_filter1d(signal, sigma=sigma)
                else:
                    smoothed_arr[:, d] = bidirectional_ema(signal, alpha=alpha)
            smoothed_params[key] = [smoothed_arr[t] for t in range(T)]
    
    return smoothed_params


# ============================================================================
# Export Functions
# ============================================================================

def export_obj_sequence(
    mesh_sequence: SAM4DMeshSequence,
    output_dir: str,
    prefix: str = "frame",
) -> List[str]:
    """
    Export mesh sequence as individual OBJ files.
    
    Args:
        mesh_sequence: SAM4DMeshSequence object
        output_dir: Directory to save OBJ files
        prefix: Filename prefix
    
    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    faces = mesh_sequence.faces
    if faces is None:
        print("[Export] Warning: No faces available, exporting point clouds only")
    
    for i, vertices in enumerate(mesh_sequence.vertices):
        filepath = os.path.join(output_dir, f"{prefix}_{i:06d}.obj")
        
        with open(filepath, 'w') as f:
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces (1-indexed)
            if faces is not None:
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        saved_paths.append(filepath)
    
    return saved_paths


def export_vertices_npz(
    mesh_sequence: SAM4DMeshSequence,
    filepath: str,
) -> str:
    """
    Export mesh sequence as compressed NPZ file.
    
    This is efficient for large sequences and preserves all data.
    """
    data = {
        "vertices": np.array(mesh_sequence.vertices),  # [T, N, 3]
        "faces": mesh_sequence.faces,
        "fps": mesh_sequence.fps,
        "frame_count": mesh_sequence.frame_count,
        "person_ids": mesh_sequence.person_ids,
    }
    
    # Add parameters if available
    for key, values in mesh_sequence.params.items():
        if values:
            data[f"params_{key}"] = np.array(values)
    
    np.savez_compressed(filepath, **data)
    return filepath


# ============================================================================
# ComfyUI Nodes
# ============================================================================

class SAM4DTemporalFusion:
    """
    Apply temporal smoothing to mesh sequences.
    
    Reduces jitter and creates smoother motion by filtering
    vertex positions and parameters across frames.
    """
    
    METHODS = ["gaussian", "ema", "none"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("SAM4D_MESH_SEQUENCE",),
            },
            "optional": {
                "method": (cls.METHODS, {"default": "gaussian"}),
                "smoothing_strength": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.1, 
                    "max": 5.0, 
                    "step": 0.1,
                    "tooltip": "Gaussian sigma or 1/EMA alpha"
                }),
                "smooth_vertices": ("BOOLEAN", {"default": True}),
                "smooth_params": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("SAM4D_MESH_SEQUENCE",)
    RETURN_NAMES = ("smoothed_sequence",)
    FUNCTION = "smooth"
    CATEGORY = "SAM4DBodyCapture/Mesh"
    
    def smooth(
        self,
        mesh_sequence: dict,
        method: str = "gaussian",
        smoothing_strength: float = 1.0,
        smooth_vertices: bool = True,
        smooth_params: bool = True,
    ):
        # Convert from dict
        seq = SAM4DMeshSequence.from_dict(mesh_sequence)
        
        if method == "none":
            return (seq.to_dict(),)
        
        print(f"[SAM4D] Applying {method} smoothing (strength={smoothing_strength})")
        
        # Convert strength to method-specific parameter
        if method == "gaussian":
            sigma = smoothing_strength
            alpha = 0.3  # unused
        else:
            sigma = 1.0  # unused
            alpha = 1.0 / smoothing_strength  # Higher strength = lower alpha
            alpha = max(0.1, min(0.9, alpha))
        
        # Smooth vertices
        if smooth_vertices and seq.vertices:
            seq.vertices = smooth_vertices_sequence(
                seq.vertices, method=method, sigma=sigma, alpha=alpha
            )
            print(f"[SAM4D] Smoothed {len(seq.vertices)} frames of vertices")
        
        # Smooth parameters
        if smooth_params and seq.params:
            seq.params = smooth_params_sequence(
                seq.params, method=method, sigma=sigma, alpha=alpha
            )
            print(f"[SAM4D] Smoothed {len(seq.params)} parameter sets")
        
        return (seq.to_dict(),)


class SAM4DExportMeshSequence:
    """
    Export mesh sequence to various formats.
    
    Supported formats:
    - OBJ sequence: Individual OBJ files per frame
    - NPZ: Compressed numpy archive (efficient)
    - FBX: Animated mesh (requires Blender) [future]
    - Alembic: Animated point cache [future]
    """
    
    FORMATS = ["obj_sequence", "npz"]  # "fbx", "alembic" for future
    COORD_SYSTEMS = ["Y-up (Maya/Blender)", "Z-up (Unreal)", "Unity"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("SAM4D_MESH_SEQUENCE",),
                "output_format": (cls.FORMATS, {"default": "npz"}),
                "output_path": ("STRING", {
                    "default": "outputs/sam4d_mesh",
                    "tooltip": "Output path (without extension)"
                }),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0}),
                "coordinate_system": (cls.COORD_SYSTEMS, {"default": "Y-up (Maya/Blender)"}),
                "person_id": ("INT", {"default": 0, "tooltip": "0 = all persons"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_filepath",)
    FUNCTION = "export"
    CATEGORY = "SAM4DBodyCapture/Export"
    OUTPUT_NODE = True
    
    def export(
        self,
        mesh_sequence: dict,
        output_format: str,
        output_path: str,
        fps: float = 30.0,
        coordinate_system: str = "Y-up (Maya/Blender)",
        person_id: int = 0,
    ):
        seq = SAM4DMeshSequence.from_dict(mesh_sequence)
        seq.fps = fps
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Apply coordinate system transform if needed
        if coordinate_system == "Z-up (Unreal)":
            # Swap Y and Z, negate new Y
            for i, verts in enumerate(seq.vertices):
                new_verts = verts.copy()
                new_verts[:, 1] = verts[:, 2]
                new_verts[:, 2] = -verts[:, 1]
                seq.vertices[i] = new_verts
        elif coordinate_system == "Unity":
            # Unity is left-handed, negate X
            for i, verts in enumerate(seq.vertices):
                seq.vertices[i][:, 0] *= -1
        
        print(f"[SAM4D Export] Format: {output_format}, Frames: {seq.frame_count}")
        
        if output_format == "obj_sequence":
            saved_paths = export_obj_sequence(seq, output_path, prefix="mesh")
            result_path = output_path
            print(f"[SAM4D Export] Saved {len(saved_paths)} OBJ files to {output_path}")
            
        elif output_format == "npz":
            filepath = output_path + ".npz"
            export_vertices_npz(seq, filepath)
            result_path = filepath
            print(f"[SAM4D Export] Saved to {filepath}")
            
        else:
            print(f"[SAM4D Export] Format {output_format} not yet implemented")
            result_path = ""
        
        return (result_path,)


class SAM4DCreateMeshSequence:
    """
    Create a mesh sequence from individual frames.
    
    Use this to collect per-frame mesh outputs from SAM3DBody
    into a temporal sequence for smoothing and export.
    
    Accepts:
    - mesh_data: Direct output from SAM3DBody nodes (MESH_DATA type)
    - vertices/faces: Raw numpy arrays
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "mesh_data": ("MESH_DATA",),  # Direct from SAM3DBody
                "vertices": ("NUMPY",),  # [N, 3] or [B, N, 3]
                "faces": ("NUMPY",),  # [F, 3]
                "existing_sequence": ("SAM4D_MESH_SEQUENCE",),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0}),
                "person_id": ("INT", {"default": 1, "min": 1, "max": 10}),
            }
        }
    
    RETURN_TYPES = ("SAM4D_MESH_SEQUENCE",)
    RETURN_NAMES = ("mesh_sequence",)
    FUNCTION = "create"
    CATEGORY = "SAM4DBodyCapture/Mesh"
    
    def extract_from_mesh_data(self, mesh_data):
        """
        Extract vertices, faces, and params from SAM3DBody mesh_data.
        
        mesh_data can be:
        - dict with 'vertices', 'faces', etc.
        - tuple/list of (vertices, faces, ...)
        - object with .vertices, .faces attributes
        """
        vertices = None
        faces = None
        params = {}
        
        if mesh_data is None:
            return None, None, {}
        
        # Handle dict
        if isinstance(mesh_data, dict):
            vertices = mesh_data.get('vertices', mesh_data.get('verts', None))
            faces = mesh_data.get('faces', mesh_data.get('f', None))
            
            # Extract SMPL params if available
            for key in ['body_pose', 'global_orient', 'betas', 'transl', 'pose', 'shape']:
                if key in mesh_data:
                    params[key] = mesh_data[key]
        
        # Handle tuple/list
        elif isinstance(mesh_data, (tuple, list)):
            if len(mesh_data) >= 1:
                vertices = mesh_data[0]
            if len(mesh_data) >= 2:
                faces = mesh_data[1]
        
        # Handle object with attributes
        elif hasattr(mesh_data, 'vertices'):
            vertices = mesh_data.vertices
            if hasattr(mesh_data, 'faces'):
                faces = mesh_data.faces
            elif hasattr(mesh_data, 'f'):
                faces = mesh_data.f
        
        # Convert tensors to numpy
        if vertices is not None and isinstance(vertices, torch.Tensor):
            vertices = vertices.detach().cpu().numpy()
        if faces is not None and isinstance(faces, torch.Tensor):
            faces = faces.detach().cpu().numpy()
        
        # Handle batched vertices [B, N, 3] - take first if single batch
        if vertices is not None and vertices.ndim == 3 and vertices.shape[0] == 1:
            vertices = vertices[0]
        
        return vertices, faces, params
    
    def create(
        self,
        mesh_data=None,
        vertices: np.ndarray = None,
        faces: np.ndarray = None,
        existing_sequence: dict = None,
        fps: float = 30.0,
        person_id: int = 1,
    ):
        # Get or create sequence
        if existing_sequence is not None:
            seq = SAM4DMeshSequence.from_dict(existing_sequence)
        else:
            seq = SAM4DMeshSequence()
            seq.fps = fps
        
        # Extract from mesh_data if provided (SAM3DBody output)
        extracted_verts = None
        extracted_faces = None
        extracted_params = {}
        
        if mesh_data is not None:
            extracted_verts, extracted_faces, extracted_params = self.extract_from_mesh_data(mesh_data)
        
        # Use extracted or provided data
        final_vertices = extracted_verts if extracted_verts is not None else vertices
        final_faces = extracted_faces if extracted_faces is not None else faces
        
        # Set faces if provided and not already set
        if final_faces is not None and seq.faces is None:
            seq.faces = final_faces
        
        # Add frame(s)
        if final_vertices is not None:
            if final_vertices.ndim == 2:
                # Single frame [N, 3]
                seq.add_frame(final_vertices, params=extracted_params, person_id=person_id)
            elif final_vertices.ndim == 3:
                # Batched [B, N, 3]
                for i in range(final_vertices.shape[0]):
                    frame_params = {k: v[i] if hasattr(v, '__getitem__') and len(v) > i else v 
                                   for k, v in extracted_params.items()}
                    seq.add_frame(final_vertices[i], params=frame_params, person_id=person_id)
        else:
            print("[SAM4D] Warning: No vertex data provided to CreateMeshSequence")
        
        return (seq.to_dict(),)


class SAM4DVisualizeMeshSequence:
    """
    Render mesh sequence to video frames for preview.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("SAM4D_MESH_SEQUENCE",),
            },
            "optional": {
                "background_images": ("IMAGE",),
                "render_size": ("INT", {"default": 512, "min": 256, "max": 2048}),
                "show_wireframe": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rendered_frames",)
    FUNCTION = "visualize"
    CATEGORY = "SAM4DBodyCapture/Mesh"
    
    def visualize(
        self,
        mesh_sequence: dict,
        background_images: torch.Tensor = None,
        render_size: int = 512,
        show_wireframe: bool = False,
    ):
        seq = SAM4DMeshSequence.from_dict(mesh_sequence)
        
        # Simple orthographic projection visualization
        # For full rendering, users should export and use external tools
        
        T = seq.frame_count
        frames = []
        
        for i, vertices in enumerate(seq.vertices):
            # Create simple visualization
            img = np.ones((render_size, render_size, 3), dtype=np.float32) * 0.8
            
            if vertices is not None and len(vertices) > 0:
                # Normalize vertices to image space
                v_min = vertices.min(axis=0)
                v_max = vertices.max(axis=0)
                v_range = (v_max - v_min).max()
                
                if v_range > 0:
                    v_norm = (vertices - v_min) / v_range
                    v_norm = v_norm * 0.8 + 0.1  # Add margin
                    
                    # Project to 2D (XY plane)
                    px = (v_norm[:, 0] * render_size).astype(int)
                    py = ((1 - v_norm[:, 1]) * render_size).astype(int)  # Flip Y
                    
                    # Draw points
                    for x, y in zip(px, py):
                        if 0 <= x < render_size and 0 <= y < render_size:
                            img[max(0,y-1):min(render_size,y+2), 
                                max(0,x-1):min(render_size,x+2)] = [0.2, 0.5, 0.8]
            
            frames.append(img)
        
        # Stack frames
        result = torch.from_numpy(np.stack(frames, axis=0))
        
        return (result,)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "SAM4DTemporalFusion": SAM4DTemporalFusion,
    "SAM4DExportMeshSequence": SAM4DExportMeshSequence,
    "SAM4DCreateMeshSequence": SAM4DCreateMeshSequence,
    "SAM4DVisualizeMeshSequence": SAM4DVisualizeMeshSequence,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM4DTemporalFusion": "🔄 Temporal Fusion",
    "SAM4DExportMeshSequence": "📦 Export Mesh Sequence",
    "SAM4DCreateMeshSequence": "✨ Create Mesh Sequence",
    "SAM4DVisualizeMeshSequence": "👁️ Visualize Mesh Sequence",
}
