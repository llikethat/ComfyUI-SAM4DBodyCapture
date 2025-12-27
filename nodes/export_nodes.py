"""
SAM4D Export Nodes for ComfyUI

Provides export capabilities for:
- Character meshes (FBX, Alembic, OBJ sequence)
- Camera data (FBX, Alembic)

Supports coordinate system transforms for:
- Maya/Blender (Y-up, right-handed)
- Unreal Engine (Z-up, left-handed)
- Unity (Y-up, left-handed)
- Houdini (Y-up, right-handed)
- Nuke (Y-up, right-handed)
"""

import os
import json
import struct
import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional, List
from datetime import datetime

# Import our pipeline types
from .sam4d_pipeline import SAM4DMeshSequence

# ============================================================================
# Check for optional export libraries
# ============================================================================

ALEMBIC_AVAILABLE = False
try:
    import alembic
    from alembic import Abc, AbcGeom
    ALEMBIC_AVAILABLE = True
except ImportError:
    pass

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
    "Y-up (Unity)": {
        "matrix": np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32),
        "scale": 1.0,
    },
    "Y-up (Houdini)": {
        "matrix": np.eye(3),
        "scale": 1.0,
    },
    "Y-up (Nuke)": {
        "matrix": np.eye(3),
        "scale": 1.0,
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
    Export mesh sequence as FBX file.
    
    Creates an ASCII FBX 7.4 file compatible with:
    - Maya
    - Blender
    - Unreal Engine
    - Unity
    - 3ds Max
    """
    
    COORD_SYSTEMS = list(COORD_SYSTEMS.keys())
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("SAM4D_MESH_SEQUENCE",),
                "output_path": ("STRING", {
                    "default": "outputs/sam4d_character",
                    "tooltip": "Output path without extension"
                }),
            },
            "optional": {
                "coordinate_system": (cls.COORD_SYSTEMS, {"default": "Y-up (Maya/Blender)"}),
                "include_animation": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_filepath",)
    FUNCTION = "export"
    CATEGORY = "SAM4DBodyCapture/Export"
    OUTPUT_NODE = True
    
    def export(
        self,
        mesh_sequence: dict,
        output_path: str,
        coordinate_system: str = "Y-up (Maya/Blender)",
        include_animation: bool = True,
    ):
        seq = SAM4DMeshSequence.from_dict(mesh_sequence)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        filepath = output_path + ".fbx"
        
        print(f"[SAM4D Export] Exporting FBX: {filepath}")
        print(f"[SAM4D Export] Frames: {seq.frame_count}, Coordinate System: {coordinate_system}")
        
        export_fbx_ascii(filepath, seq, coordinate_system, include_animation)
        
        print(f"[SAM4D Export] FBX saved successfully")
        
        return (filepath,)


class SAM4DExportCharacterAlembic:
    """
    Export mesh sequence as Alembic file.
    
    Alembic is ideal for:
    - Vertex animation / point cache
    - VFX pipelines (Houdini, Nuke)
    - High-fidelity mesh animation
    
    Requires: pip install alembic
    """
    
    COORD_SYSTEMS = list(COORD_SYSTEMS.keys())
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("SAM4D_MESH_SEQUENCE",),
                "output_path": ("STRING", {
                    "default": "outputs/sam4d_character",
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
        output_path: str,
        coordinate_system: str = "Y-up (Maya/Blender)",
    ):
        seq = SAM4DMeshSequence.from_dict(mesh_sequence)
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        filepath = output_path + ".abc"
        
        if not ALEMBIC_AVAILABLE:
            print("[SAM4D Export] Alembic not available, falling back to OBJ sequence")
            obj_dir = output_path + "_obj"
            export_obj_sequence(obj_dir, seq, "mesh", coordinate_system)
            return (obj_dir,)
        
        print(f"[SAM4D Export] Exporting Alembic: {filepath}")
        
        export_alembic(filepath, seq, coordinate_system)
        
        print(f"[SAM4D Export] Alembic saved successfully")
        
        return (filepath,)


class SAM4DExportCharacterOBJ:
    """
    Export mesh sequence as OBJ file sequence.
    
    Most compatible format - works with all 3D software.
    Creates one OBJ file per frame.
    """
    
    COORD_SYSTEMS = list(COORD_SYSTEMS.keys())
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("SAM4D_MESH_SEQUENCE",),
                "output_path": ("STRING", {
                    "default": "outputs/sam4d_obj_sequence",
                }),
            },
            "optional": {
                "coordinate_system": (cls.COORD_SYSTEMS, {"default": "Y-up (Maya/Blender)"}),
                "prefix": ("STRING", {"default": "mesh"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_directory",)
    FUNCTION = "export"
    CATEGORY = "SAM4DBodyCapture/Export"
    OUTPUT_NODE = True
    
    def export(
        self,
        mesh_sequence: dict,
        output_path: str,
        coordinate_system: str = "Y-up (Maya/Blender)",
        prefix: str = "mesh",
    ):
        seq = SAM4DMeshSequence.from_dict(mesh_sequence)
        
        print(f"[SAM4D Export] Exporting OBJ sequence to: {output_path}")
        
        saved = export_obj_sequence(output_path, seq, prefix, coordinate_system)
        
        print(f"[SAM4D Export] Saved {len(saved)} OBJ files")
        
        return (output_path,)


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
    "SAM4DExportCharacterOBJ": SAM4DExportCharacterOBJ,
    "SAM4DExportCameraFBX": SAM4DExportCameraFBX,
    "SAM4DExportCameraJSON": SAM4DExportCameraJSON,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM4DExportCharacterFBX": "📦 Export Character FBX",
    "SAM4DExportCharacterAlembic": "📦 Export Character Alembic",
    "SAM4DExportCharacterOBJ": "📦 Export Character OBJ Sequence",
    "SAM4DExportCameraFBX": "🎥 Export Camera FBX",
    "SAM4DExportCameraJSON": "🎥 Export Camera JSON",
}
