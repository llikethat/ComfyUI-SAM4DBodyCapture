"""
SAM4D Export Nodes for ComfyUI

Provides export capabilities for:
- Character meshes (FBX, Alembic, OBJ sequence)
- Camera data (FBX, Alembic)

Uses Blender from SAM3DBody for proper animated FBX export.
"""

import os
import json
import subprocess
import shutil
import glob
import tempfile
import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional, List
from datetime import datetime

# Try to import folder_paths for ComfyUI output directory
try:
    import folder_paths
    COMFYUI_OUTPUT = folder_paths.get_output_directory()
except ImportError:
    COMFYUI_OUTPUT = "outputs"

# Import our pipeline types
from .sam4d_pipeline import SAM4DMeshSequence

# ============================================================================
# Blender Path Finding
# ============================================================================

BLENDER_TIMEOUT = 600
_BLENDER_PATH = None

def find_blender() -> Optional[str]:
    """Find Blender executable, prioritizing SAM3DBody bundled version."""
    global _BLENDER_PATH
    
    if _BLENDER_PATH is not None:
        return _BLENDER_PATH
    
    locations = []
    
    # Check SAM3DBody bundled Blender first (recommended)
    try:
        custom_nodes = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        patterns = [
            os.path.join(custom_nodes, "ComfyUI-SAM3DBody", "lib", "blender", "blender-*-linux-x64", "blender"),
            os.path.join(custom_nodes, "ComfyUI-SAM3DBody", "lib", "blender", "*", "blender"),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            locations.extend(matches)
    except Exception:
        pass
    
    # Also check workspace path (RunPod/cloud)
    workspace_patterns = [
        "/workspace/ComfyUI/custom_nodes/ComfyUI-SAM3DBody/lib/blender/blender-*-linux-x64/blender",
    ]
    for pattern in workspace_patterns:
        matches = glob.glob(pattern)
        locations.extend(matches)
    
    # System Blender as fallback
    locations.extend([
        shutil.which("blender"),
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ])
    
    # Windows
    for ver in ["4.2", "4.1", "4.0", "3.6"]:
        locations.append(f"C:\\Program Files\\Blender Foundation\\Blender {ver}\\blender.exe")
    
    for loc in locations:
        if loc and os.path.exists(loc):
            _BLENDER_PATH = loc
            print(f"[SAM4D Export] Found Blender: {loc}")
            return loc
    
    return None


# Get the Blender script path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir = os.path.join(os.path.dirname(_current_dir), "lib")
BLENDER_SCRIPT = os.path.join(_lib_dir, "blender_animated_fbx.py")

# ============================================================================
# Coordinate System Transforms
# ============================================================================

COORD_SYSTEMS = {
    "Y-up (Maya/Blender)": {
        "matrix": np.eye(3),
        "scale": 1.0,
    },
    "Z-up (Unreal)": {
        "matrix": np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float32),
        "scale": 100.0,  # Unreal uses cm
    },
}


def transform_vertices(vertices: np.ndarray, coord_system: str) -> np.ndarray:
    """Apply coordinate system transformation to vertices."""
    config = COORD_SYSTEMS.get(coord_system, COORD_SYSTEMS["Y-up (Maya/Blender)"])
    matrix = config["matrix"]
    scale = config["scale"]
    
    # Apply rotation and scale
    transformed = (vertices @ matrix.T) * scale
    return transformed


def transform_camera(position: np.ndarray, rotation: np.ndarray, coord_system: str) -> Tuple[np.ndarray, np.ndarray]:
    """Apply coordinate system transformation to camera."""
    config = COORD_SYSTEMS.get(coord_system, COORD_SYSTEMS["Y-up (Maya/Blender)"])
    matrix = config["matrix"]
    scale = config["scale"]
    
    transformed_pos = (position @ matrix.T) * scale
    # Rotation needs special handling based on representation
    transformed_rot = rotation  # TODO: Proper rotation transform
    
    return transformed_pos, transformed_rot


# ============================================================================
# OBJ Export (Always Available)
# ============================================================================

def export_obj_file(
    filepath: str,
    vertices: np.ndarray,
    faces: np.ndarray = None,
    normals: np.ndarray = None,
    uvs: np.ndarray = None,
) -> str:
    """
    Export single mesh as OBJ file.
    
    Args:
        filepath: Output path
        vertices: [N, 3] vertex positions
        faces: [F, 3] face indices (0-indexed)
        normals: [N, 3] vertex normals (optional)
        uvs: [N, 2] UV coordinates (optional)
    """
    with open(filepath, 'w') as f:
        f.write(f"# SAM4DBodyCapture OBJ Export\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        if faces is not None:
            f.write(f"# Faces: {len(faces)}\n")
        f.write("\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write normals
        if normals is not None:
            f.write("\n")
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        
        # Write UVs
        if uvs is not None:
            f.write("\n")
            for uv in uvs:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        
        # Write faces (1-indexed)
        if faces is not None:
            f.write("\n")
            for face in faces:
                if normals is not None and uvs is not None:
                    f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                           f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                           f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")
                elif normals is not None:
                    f.write(f"f {face[0]+1}//{face[0]+1} "
                           f"{face[1]+1}//{face[1]+1} "
                           f"{face[2]+1}//{face[2]+1}\n")
                else:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    return filepath


def export_obj_sequence(
    output_dir: str,
    mesh_sequence: SAM4DMeshSequence,
    prefix: str = "mesh",
    coord_system: str = "Y-up (Maya/Blender)",
) -> List[str]:
    """Export mesh sequence as OBJ files."""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for i, vertices in enumerate(mesh_sequence.vertices):
        filepath = os.path.join(output_dir, f"{prefix}_{i:06d}.obj")
        
        # Transform coordinates
        transformed = transform_vertices(vertices, coord_system)
        
        export_obj_file(filepath, transformed, mesh_sequence.faces)
        saved_paths.append(filepath)
    
    return saved_paths


# ============================================================================
# FBX Export (ASCII Format - No Dependencies)
# ============================================================================

def export_fbx_ascii(
    filepath: str,
    mesh_sequence: SAM4DMeshSequence,
    coord_system: str = "Y-up (Maya/Blender)",
    include_animation: bool = True,
) -> str:
    """
    Export mesh sequence as ASCII FBX file.
    
    This creates a basic FBX 7.4 ASCII file that can be imported
    into Maya, Blender, and most 3D software.
    """
    fps = mesh_sequence.fps
    frame_count = mesh_sequence.frame_count
    
    # Get first frame for base mesh
    base_vertices = mesh_sequence.vertices[0] if mesh_sequence.vertices else np.zeros((1, 3))
    base_vertices = transform_vertices(base_vertices, coord_system)
    faces = mesh_sequence.faces
    
    with open(filepath, 'w') as f:
        # FBX Header
        f.write("; FBX 7.4.0 project file\n")
        f.write("; Created by SAM4DBodyCapture\n")
        f.write("; ----------------------------------------------------\n\n")
        
        f.write("FBXHeaderExtension:  {\n")
        f.write("    FBXHeaderVersion: 1003\n")
        f.write("    FBXVersion: 7400\n")
        f.write(f"    CreationTimeStamp:  {{\n")
        now = datetime.now()
        f.write(f"        Version: 1000\n")
        f.write(f"        Year: {now.year}\n")
        f.write(f"        Month: {now.month}\n")
        f.write(f"        Day: {now.day}\n")
        f.write(f"        Hour: {now.hour}\n")
        f.write(f"        Minute: {now.minute}\n")
        f.write(f"        Second: {now.second}\n")
        f.write(f"        Millisecond: 0\n")
        f.write(f"    }}\n")
        f.write(f"    Creator: \"SAM4DBodyCapture v0.3.0\"\n")
        f.write("}\n\n")
        
        # Global Settings
        f.write("GlobalSettings:  {\n")
        f.write("    Version: 1000\n")
        f.write("    Properties70:  {\n")
        f.write(f"        P: \"UpAxis\", \"int\", \"Integer\", \"\", 1\n")
        f.write(f"        P: \"UpAxisSign\", \"int\", \"Integer\", \"\", 1\n")
        f.write(f"        P: \"FrontAxis\", \"int\", \"Integer\", \"\", 2\n")
        f.write(f"        P: \"FrontAxisSign\", \"int\", \"Integer\", \"\", 1\n")
        f.write(f"        P: \"CoordAxis\", \"int\", \"Integer\", \"\", 0\n")
        f.write(f"        P: \"CoordAxisSign\", \"int\", \"Integer\", \"\", 1\n")
        f.write(f"        P: \"UnitScaleFactor\", \"double\", \"Number\", \"\", 1\n")
        f.write(f"        P: \"TimeSpanStart\", \"KTime\", \"Time\", \"\", 0\n")
        f.write(f"        P: \"TimeSpanStop\", \"KTime\", \"Time\", \"\", {int(frame_count * 46186158000 / fps)}\n")
        f.write(f"        P: \"CustomFrameRate\", \"double\", \"Number\", \"\", {fps}\n")
        f.write("    }\n")
        f.write("}\n\n")
        
        # Objects section
        f.write("Objects:  {\n")
        
        # Geometry (Mesh)
        num_vertices = len(base_vertices)
        num_faces = len(faces) if faces is not None else 0
        
        f.write(f"    Geometry: 1000, \"Geometry::SAM4D_Mesh\", \"Mesh\" {{\n")
        f.write(f"        Properties70:  {{\n")
        f.write(f"        }}\n")
        
        # Vertices
        f.write(f"        Vertices: *{num_vertices * 3} {{\n")
        f.write(f"            a: ")
        vertex_strs = []
        for v in base_vertices:
            vertex_strs.extend([f"{v[0]:.6f}", f"{v[1]:.6f}", f"{v[2]:.6f}"])
        f.write(",".join(vertex_strs))
        f.write("\n        }\n")
        
        # Faces (polygon vertex indices)
        if faces is not None:
            f.write(f"        PolygonVertexIndex: *{num_faces * 3} {{\n")
            f.write(f"            a: ")
            face_strs = []
            for face in faces:
                # FBX uses negative index to mark end of polygon
                face_strs.extend([str(face[0]), str(face[1]), str(-(face[2] + 1))])
            f.write(",".join(face_strs))
            f.write("\n        }\n")
        
        f.write(f"        GeometryVersion: 124\n")
        f.write(f"    }}\n")
        
        # Model (Mesh Node)
        f.write(f"    Model: 2000, \"Model::SAM4D_Body\", \"Mesh\" {{\n")
        f.write(f"        Version: 232\n")
        f.write(f"        Properties70:  {{\n")
        f.write(f"            P: \"DefaultAttributeIndex\", \"int\", \"Integer\", \"\", 0\n")
        f.write(f"        }}\n")
        f.write(f"        Shading: T\n")
        f.write(f"        Culling: \"CullingOff\"\n")
        f.write(f"    }}\n")
        
        f.write("}\n\n")
        
        # Connections
        f.write("Connections:  {\n")
        f.write("    C: \"OO\", 1000, 2000\n")  # Geometry to Model
        f.write("    C: \"OO\", 2000, 0\n")     # Model to Root
        f.write("}\n\n")
        
        # Animation (if enabled and we have multiple frames)
        if include_animation and frame_count > 1:
            f.write("; Animation data stored as shape keys / blend shapes\n")
            f.write("; For full vertex animation, import OBJ sequence instead\n")
    
    return filepath


# ============================================================================
# Alembic Export (Requires PyAlembic)
# ============================================================================

def export_alembic(
    filepath: str,
    mesh_sequence: SAM4DMeshSequence,
    coord_system: str = "Y-up (Maya/Blender)",
) -> str:
    """
    Export mesh sequence as Alembic file.
    
    Requires: pip install alembic
    """
    if not ALEMBIC_AVAILABLE:
        raise RuntimeError(
            "Alembic export requires PyAlembic. "
            "Install with: pip install alembic"
        )
    
    fps = mesh_sequence.fps
    frame_count = mesh_sequence.frame_count
    faces = mesh_sequence.faces
    
    # Create Alembic archive
    archive = Abc.OArchive(filepath)
    top = archive.getTop()
    
    # Create time sampling
    time_per_frame = 1.0 / fps
    time_sampling = AbcGeom.TimeSampling(time_per_frame, 0.0)
    ts_idx = archive.addTimeSampling(time_sampling)
    
    # Create mesh object
    mesh_obj = AbcGeom.OPolyMesh(top, "SAM4D_Body", ts_idx)
    mesh_schema = mesh_obj.getSchema()
    
    # Write each frame
    for i, vertices in enumerate(mesh_sequence.vertices):
        # Transform coordinates
        transformed = transform_vertices(vertices, coord_system)
        
        # Create sample
        if i == 0 and faces is not None:
            # First frame includes topology
            face_counts = np.full(len(faces), 3, dtype=np.int32)
            face_indices = faces.flatten().astype(np.int32)
            
            sample = AbcGeom.OPolyMeshSchemaSample(
                transformed.flatten().astype(np.float32),
                face_indices,
                face_counts
            )
        else:
            # Subsequent frames only update positions
            sample = AbcGeom.OPolyMeshSchemaSample(
                transformed.flatten().astype(np.float32)
            )
        
        mesh_schema.set(sample)
    
    return filepath


# ============================================================================
# Camera Export
# ============================================================================

def export_camera_fbx(
    filepath: str,
    positions: List[np.ndarray],
    rotations: List[np.ndarray],
    focal_lengths: List[float] = None,
    fps: float = 30.0,
    coord_system: str = "Y-up (Maya/Blender)",
) -> str:
    """
    Export camera animation as FBX.
    """
    frame_count = len(positions)
    
    with open(filepath, 'w') as f:
        # FBX Header
        f.write("; FBX 7.4.0 project file\n")
        f.write("; Camera Export by SAM4DBodyCapture\n\n")
        
        f.write("FBXHeaderExtension:  {\n")
        f.write("    FBXHeaderVersion: 1003\n")
        f.write("    FBXVersion: 7400\n")
        f.write(f"    Creator: \"SAM4DBodyCapture v0.3.0\"\n")
        f.write("}\n\n")
        
        # Global Settings
        f.write("GlobalSettings:  {\n")
        f.write("    Version: 1000\n")
        f.write("    Properties70:  {\n")
        f.write(f"        P: \"CustomFrameRate\", \"double\", \"Number\", \"\", {fps}\n")
        f.write("    }\n")
        f.write("}\n\n")
        
        # Objects
        f.write("Objects:  {\n")
        
        # Camera Node
        f.write(f"    NodeAttribute: 1000, \"NodeAttribute::SAM4D_Camera\", \"Camera\" {{\n")
        f.write(f"        Properties70:  {{\n")
        if focal_lengths and len(focal_lengths) > 0:
            f.write(f"            P: \"FocalLength\", \"double\", \"Number\", \"\", {focal_lengths[0]}\n")
        f.write(f"        }}\n")
        f.write(f"        TypeFlags: \"Camera\"\n")
        f.write(f"    }}\n")
        
        # Camera Model
        pos = transform_vertices(positions[0].reshape(1, 3), coord_system)[0]
        f.write(f"    Model: 2000, \"Model::SAM4D_Camera\", \"Camera\" {{\n")
        f.write(f"        Version: 232\n")
        f.write(f"        Properties70:  {{\n")
        f.write(f"            P: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\", {pos[0]}, {pos[1]}, {pos[2]}\n")
        f.write(f"        }}\n")
        f.write(f"    }}\n")
        
        f.write("}\n\n")
        
        # Connections
        f.write("Connections:  {\n")
        f.write("    C: \"OO\", 1000, 2000\n")
        f.write("    C: \"OO\", 2000, 0\n")
        f.write("}\n")
    
    return filepath


def export_camera_json(
    filepath: str,
    positions: List[np.ndarray],
    rotations: List[np.ndarray],
    focal_lengths: List[float] = None,
    sensor_width: float = 36.0,
    fps: float = 30.0,
    coord_system: str = "Y-up (Maya/Blender)",
) -> str:
    """
    Export camera data as JSON (universal format).
    """
    frames = []
    
    for i, (pos, rot) in enumerate(zip(positions, rotations)):
        pos_t, rot_t = transform_camera(pos, rot, coord_system)
        
        frame_data = {
            "frame": i,
            "time": i / fps,
            "position": pos_t.tolist() if isinstance(pos_t, np.ndarray) else pos_t,
            "rotation": rot_t.tolist() if isinstance(rot_t, np.ndarray) else rot_t,
        }
        
        if focal_lengths and i < len(focal_lengths):
            frame_data["focal_length"] = focal_lengths[i]
        
        frames.append(frame_data)
    
    data = {
        "version": "1.0",
        "generator": "SAM4DBodyCapture v0.3.0",
        "fps": fps,
        "frame_count": len(frames),
        "sensor_width": sensor_width,
        "coord_system": coord_system,
        "frames": frames,
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return filepath


# ============================================================================
# ComfyUI Nodes
# ============================================================================

class SAM4DExportCharacterFBX:
    """
    Export mesh sequence as animated FBX file using Blender.
    
    Creates an animated FBX with:
    - Mesh with shape keys (vertex animation)
    - Skeleton with keyframes
    - Camera with intrinsics (optional)
    
    Camera and character are exported in the SAME file for easy handling.
    Compatible with Maya, Blender, Unreal, Unity, 3ds Max.
    """
    
    COORD_SYSTEMS = list(COORD_SYSTEMS.keys())
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("SAM4D_MESH_SEQUENCE",),
                "filename": ("STRING", {
                    "default": "sam4d_animation",
                    "tooltip": "Output filename (without extension)"
                }),
            },
            "optional": {
                "camera_intrinsics": ("CAMERA_INTRINSICS", {
                    "tooltip": "Camera intrinsics from MoGe2 or manual input"
                }),
                "coordinate_system": (cls.COORD_SYSTEMS, {"default": "Y-up (Maya/Blender)"}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0}),
                "flip_x": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Mirror mesh on X axis (usually needed for correct left/right orientation)"
                }),
                "include_camera": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include camera in FBX (recommended)"
                }),
                "sensor_width": ("FLOAT", {
                    "default": 36.0,
                    "min": 1.0,
                    "max": 100.0,
                    "tooltip": "Camera sensor width in mm (35mm = 36mm)"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_filepath",)
    FUNCTION = "export"
    CATEGORY = "SAM4DBodyCapture/Export"
    OUTPUT_NODE = True
    
    def _get_incremental_filepath(self, output_dir: str, filename: str, extension: str) -> str:
        """
        Generate incremental filepath to avoid overwriting existing files.
        
        Examples:
            sam4d_animation.fbx
            sam4d_animation_0001.fbx
            sam4d_animation_0002.fbx
        """
        base_path = os.path.join(output_dir, f"{filename}{extension}")
        
        # If file doesn't exist, use the base name
        if not os.path.exists(base_path):
            return base_path
        
        # Find next available number
        counter = 1
        while True:
            numbered_path = os.path.join(output_dir, f"{filename}_{counter:04d}{extension}")
            if not os.path.exists(numbered_path):
                return numbered_path
            counter += 1
            
            # Safety limit to avoid infinite loop
            if counter > 9999:
                # Fallback with timestamp
                import time
                timestamp = int(time.time())
                return os.path.join(output_dir, f"{filename}_{timestamp}{extension}")
    
    def export(
        self,
        mesh_sequence: dict,
        filename: str = "sam4d_animation",
        camera_intrinsics: dict = None,
        coordinate_system: str = "Y-up (Maya/Blender)",
        fps: float = 30.0,
        flip_x: bool = True,
        include_camera: bool = True,
        sensor_width: float = 36.0,
    ):
        seq = SAM4DMeshSequence.from_dict(mesh_sequence)
        
        if seq.frame_count == 0:
            print("[SAM4D Export] No frames in sequence")
            return ("",)
        
        print(f"[SAM4D Export] flip_x={flip_x}")
        
        # Find Blender
        blender_path = find_blender()
        if not blender_path:
            print("[SAM4D Export] Blender not found!")
            print("[SAM4D Export] Install Blender or ensure ComfyUI-SAM3DBody is installed")
            return ("",)
        
        # Use ComfyUI output directory
        output_dir = COMFYUI_OUTPUT
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate incremental filename to avoid overwriting
        output_path = self._get_incremental_filepath(output_dir, filename, ".fbx")
        print(f"[SAM4D Export] Output file: {os.path.basename(output_path)}")
        
        # Build JSON for Blender script
        up_axis = "Y" if "Y-up" in coordinate_system else "Z"
        
        # Get camera intrinsics
        focal_length = None
        image_width = 1920
        image_height = 1080
        
        if camera_intrinsics:
            focal_length = camera_intrinsics.get("focal_length")
            image_width = camera_intrinsics.get("width", 1920)
            image_height = camera_intrinsics.get("height", 1080)
            print(f"[SAM4D Export] Using camera intrinsics: focal={focal_length:.1f}px, size={image_width}x{image_height}")
        
        frames_data = []
        for i, vertices in enumerate(seq.vertices):
            # Apply flip_x if needed (mirror on X axis)
            if flip_x:
                if isinstance(vertices, np.ndarray):
                    vertices = vertices.copy()
                    vertices[:, 0] = -vertices[:, 0]
                else:
                    vertices = [[-v[0], v[1], v[2]] for v in vertices]
            
            frame_data = {
                "frame_index": i,
                "vertices": vertices.tolist() if isinstance(vertices, np.ndarray) else vertices,
                "image_size": [image_width, image_height],
            }
            
            # Add focal length per frame if available
            if camera_intrinsics and "per_frame_focal" in camera_intrinsics:
                per_frame = camera_intrinsics["per_frame_focal"]
                if i < len(per_frame):
                    frame_data["focal_length"] = per_frame[i]
                elif focal_length:
                    frame_data["focal_length"] = focal_length
            elif focal_length:
                frame_data["focal_length"] = focal_length
            
            # Handle params - can be list of dicts OR dict of lists
            if seq.params:
                if isinstance(seq.params, dict) and "joint_coords" in seq.params:
                    # New format: dict of lists {"joint_coords": [...], "joint_rotations": [...]}
                    if "joint_coords" in seq.params and i < len(seq.params.get("joint_coords", [])):
                        jc = seq.params["joint_coords"][i]
                        if jc is not None:
                            jc_arr = np.array(jc) if not isinstance(jc, np.ndarray) else jc.copy()
                            if flip_x:
                                jc_arr[:, 0] = -jc_arr[:, 0]  # Flip X for joint coords too
                            frame_data['joint_coords'] = jc_arr.tolist()
                    if "joint_rotations" in seq.params and i < len(seq.params.get("joint_rotations", [])):
                        jr = seq.params["joint_rotations"][i]
                        if jr is not None:
                            frame_data['joint_rotations'] = jr.tolist() if hasattr(jr, 'tolist') else jr
                    if "camera_t" in seq.params and i < len(seq.params.get("camera_t", [])):
                        pct = seq.params["camera_t"][i]
                        if pct is not None:
                            frame_data['pred_cam_t'] = pct.tolist() if hasattr(pct, 'tolist') else pct
                elif isinstance(seq.params, list) and i < len(seq.params):
                    # Old format: list of dicts [{frame0_params}, {frame1_params}]
                    params = seq.params[i]
                    if isinstance(params, dict):
                        if 'joint_coords' in params:
                            jc = params['joint_coords']
                            jc_arr = np.array(jc) if not isinstance(jc, np.ndarray) else jc.copy()
                            if flip_x:
                                jc_arr[:, 0] = -jc_arr[:, 0]  # Flip X for joint coords too
                            frame_data['joint_coords'] = jc_arr.tolist()
                        if 'joint_rotations' in params:
                            jr = params['joint_rotations']
                            frame_data['joint_rotations'] = jr.tolist() if hasattr(jr, 'tolist') else jr
                        if 'pred_cam_t' in params:
                            pct = params['pred_cam_t']
                            frame_data['pred_cam_t'] = pct.tolist() if hasattr(pct, 'tolist') else pct
            frames_data.append(frame_data)
        
        export_data = {
            "fps": fps if fps > 0 else seq.fps,
            "frame_count": seq.frame_count,
            "faces": seq.faces.tolist() if seq.faces is not None else None,
            "frames": frames_data,
            "world_translation_mode": "root",  # Character at origin
            "skeleton_mode": "positions",
            "sensor_width": sensor_width,
        }
        
        # Write temp JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(export_data, f)
            json_path = f.name
        
        try:
            cmd = [
                blender_path,
                "--background",
                "--python", BLENDER_SCRIPT,
                "--",
                json_path,
                output_path,
                up_axis,
                "1",  # include_mesh
                "1" if include_camera else "0",
            ]
            
            print(f"[SAM4D Export] Exporting {seq.frame_count} frames to FBX...")
            print(f"[SAM4D Export] Include camera: {include_camera}")
            print(f"[SAM4D Export] Output: {output_path}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=BLENDER_TIMEOUT,
            )
            
            if result.returncode != 0:
                error = result.stderr[:500] if result.stderr else "Unknown error"
                print(f"[SAM4D Export] Blender error: {error}")
                return ("",)
            
            if not os.path.exists(output_path):
                print("[SAM4D Export] FBX not created")
                return ("",)
            
            print(f"[SAM4D Export] FBX saved: {output_path}")
            return (output_path,)
            
        except subprocess.TimeoutExpired:
            print("[SAM4D Export] Blender timed out")
            return ("",)
        except Exception as e:
            print(f"[SAM4D Export] Error: {e}")
            return ("",)
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)


class SAM4DExportCharacterAlembic:
    """
    Export mesh sequence as Alembic file.
    
    Ideal for VFX pipelines (Houdini, Nuke).
    Falls back to OBJ sequence if Alembic not available.
    """
    
    COORD_SYSTEMS = list(COORD_SYSTEMS.keys())
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("SAM4D_MESH_SEQUENCE",),
                "filename": ("STRING", {
                    "default": "sam4d_character",
                }),
            },
            "optional": {
                "coordinate_system": (cls.COORD_SYSTEMS, {"default": "Y-up (Maya/Blender)"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("abc_filepath",)
    FUNCTION = "export"
    CATEGORY = "SAM4DBodyCapture/Export"
    OUTPUT_NODE = True
    
    def export(
        self,
        mesh_sequence: dict,
        filename: str = "sam4d_character",
        coordinate_system: str = "Y-up (Maya/Blender)",
    ):
        seq = SAM4DMeshSequence.from_dict(mesh_sequence)
        
        output_dir = COMFYUI_OUTPUT
        os.makedirs(output_dir, exist_ok=True)
        
        # Check for Alembic
        try:
            import alembic
            from alembic import Abc, AbcGeom
            has_alembic = True
        except ImportError:
            has_alembic = False
        
        if not has_alembic:
            print("[SAM4D Export] Alembic not available, exporting OBJ sequence")
            obj_dir = os.path.join(output_dir, f"{filename}_obj")
            export_obj_sequence(obj_dir, seq, "mesh", coordinate_system)
            return (obj_dir,)
        
        filepath = os.path.join(output_dir, f"{filename}.abc")
        
        print(f"[SAM4D Export] Exporting Alembic: {filepath}")
        
        export_alembic(filepath, seq, coordinate_system)
        
        print(f"[SAM4D Export] Alembic saved")
        
        return (filepath,)


class SAM4DFBXViewer:
    """
    Display animated FBX in ComfyUI UI.
    
    Shows skeletal animation playback with play/pause controls,
    timeline scrubber, and adjustable playback speed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fbx_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to animated FBX file"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_path",)
    FUNCTION = "view_animation"
    OUTPUT_NODE = True
    CATEGORY = "SAM4DBodyCapture/Export"

    def view_animation(self, fbx_path: str):
        """Display animated FBX playback in ComfyUI UI."""
        try:
            print(f"[SAM4D FBX Viewer] Displaying: {fbx_path}")

            return {
                "ui": {
                    "fbx_path": [fbx_path]
                },
                "result": (fbx_path,)
            }

        except Exception as e:
            print(f"[SAM4D FBX Viewer] Error: {e}")
            return {
                "ui": {
                    "fbx_path": [""]
                },
                "result": ("",)
            }


class SAM4DExportCameraFBX:
    """
    Export camera animation as FBX file.
    """
    
    COORD_SYSTEMS = list(COORD_SYSTEMS.keys())
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_positions": ("NUMPY",),  # [T, 3]
                "camera_rotations": ("NUMPY",),  # [T, 3] or [T, 4]
                "output_path": ("STRING", {"default": "outputs/sam4d_camera"}),
            },
            "optional": {
                "focal_lengths": ("NUMPY",),
                "fps": ("FLOAT", {"default": 30.0}),
                "coordinate_system": (cls.COORD_SYSTEMS, {"default": "Y-up (Maya/Blender)"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_filepath",)
    FUNCTION = "export"
    CATEGORY = "SAM4DBodyCapture/Export"
    OUTPUT_NODE = True
    
    def export(
        self,
        camera_positions: np.ndarray,
        camera_rotations: np.ndarray,
        output_path: str,
        focal_lengths: np.ndarray = None,
        fps: float = 30.0,
        coordinate_system: str = "Y-up (Maya/Blender)",
    ):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        filepath = output_path + ".fbx"
        
        positions = [camera_positions[i] for i in range(len(camera_positions))]
        rotations = [camera_rotations[i] for i in range(len(camera_rotations))]
        focals = list(focal_lengths) if focal_lengths is not None else None
        
        export_camera_fbx(filepath, positions, rotations, focals, fps, coordinate_system)
        
        return (filepath,)


class SAM4DExportCameraJSON:
    """
    Export camera data as JSON file.
    
    Universal format for:
    - Custom importers
    - Web viewers
    - Documentation
    """
    
    COORD_SYSTEMS = list(COORD_SYSTEMS.keys())
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_positions": ("NUMPY",),
                "camera_rotations": ("NUMPY",),
                "output_path": ("STRING", {"default": "outputs/sam4d_camera"}),
            },
            "optional": {
                "focal_lengths": ("NUMPY",),
                "sensor_width": ("FLOAT", {"default": 36.0}),
                "fps": ("FLOAT", {"default": 30.0}),
                "coordinate_system": (cls.COORD_SYSTEMS, {"default": "Y-up (Maya/Blender)"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_filepath",)
    FUNCTION = "export"
    CATEGORY = "SAM4DBodyCapture/Export"
    OUTPUT_NODE = True
    
    def export(
        self,
        camera_positions: np.ndarray,
        camera_rotations: np.ndarray,
        output_path: str,
        focal_lengths: np.ndarray = None,
        sensor_width: float = 36.0,
        fps: float = 30.0,
        coordinate_system: str = "Y-up (Maya/Blender)",
    ):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        filepath = output_path + ".json"
        
        positions = [camera_positions[i] for i in range(len(camera_positions))]
        rotations = [camera_rotations[i] for i in range(len(camera_rotations))]
        focals = list(focal_lengths) if focal_lengths is not None else None
        
        export_camera_json(filepath, positions, rotations, focals, sensor_width, fps, coordinate_system)
        
        return (filepath,)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "SAM4DExportCharacterFBX": SAM4DExportCharacterFBX,
    "SAM4DExportCharacterAlembic": SAM4DExportCharacterAlembic,
    "SAM4DExportCameraFBX": SAM4DExportCameraFBX,
    "SAM4DExportCameraJSON": SAM4DExportCameraJSON,
    "SAM4DFBXViewer": SAM4DFBXViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM4DExportCharacterFBX": "📦 Export Character FBX",
    "SAM4DExportCharacterAlembic": "📦 Export Character Alembic",
    "SAM4DExportCameraFBX": "🎥 Export Camera FBX",
    "SAM4DExportCameraJSON": "🎥 Export Camera JSON",
    "SAM4DFBXViewer": "🎥 FBX Animation Viewer",
}
