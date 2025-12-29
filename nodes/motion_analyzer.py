# Copyright (c) 2025 SAM4DBodyCapture
# SPDX-License-Identifier: MIT
"""
Motion Analyzer Node for SAM4DBodyCapture v0.5.0

Analyzes subject motion from SAM3DBody mesh sequence outputs:
- Height estimation from mesh (with user override)
- Pelvis/joint tracking (2D + 3D positions)
- Foot contact detection
- Motion vector debug overlay

Part of the Motion Disambiguation Pipeline:
[Motion Analyzer] → [Camera Solver] → [Motion Decoder]
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Optional, Tuple, Any


# SMPL Joint Indices
class SMPLJoints:
    """SMPL skeleton joint indices."""
    PELVIS = 0
    LEFT_HIP = 1
    RIGHT_HIP = 2
    SPINE1 = 3
    LEFT_KNEE = 4
    RIGHT_KNEE = 5
    SPINE2 = 6
    LEFT_ANKLE = 7
    RIGHT_ANKLE = 8
    SPINE3 = 9
    LEFT_FOOT = 10
    RIGHT_FOOT = 11
    NECK = 12
    LEFT_COLLAR = 13
    RIGHT_COLLAR = 14
    HEAD = 15
    LEFT_SHOULDER = 16
    RIGHT_SHOULDER = 17
    LEFT_ELBOW = 18
    RIGHT_ELBOW = 19
    LEFT_WRIST = 20
    RIGHT_WRIST = 21
    
    # Joint groups for analysis
    LOWER_BODY = [PELVIS, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, 
                  LEFT_ANKLE, RIGHT_ANKLE, LEFT_FOOT, RIGHT_FOOT]
    UPPER_BODY = [SPINE1, SPINE2, SPINE3, NECK, HEAD,
                  LEFT_COLLAR, RIGHT_COLLAR, LEFT_SHOULDER, RIGHT_SHOULDER,
                  LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]


def estimate_height_from_mesh(
    vertices: np.ndarray,
    joint_coords: np.ndarray,
) -> Dict[str, float]:
    """
    Estimate subject height from mesh and joints.
    
    SMPL meshes are approximately metric (life-sized).
    We measure both mesh bounding box and joint chain for robustness.
    
    Args:
        vertices: [V, 3] mesh vertices
        joint_coords: [J, 3] joint positions
    
    Returns:
        dict with height measurements
    """
    # Method 1: Mesh bounding box height
    mesh_min_y = vertices[:, 1].min()
    mesh_max_y = vertices[:, 1].max()
    mesh_height = mesh_max_y - mesh_min_y
    
    # Method 2: Joint chain measurement (more robust to pose)
    # Leg length: pelvis → knee → ankle
    pelvis = joint_coords[SMPLJoints.PELVIS]
    
    left_knee = joint_coords[SMPLJoints.LEFT_KNEE]
    left_ankle = joint_coords[SMPLJoints.LEFT_ANKLE]
    right_knee = joint_coords[SMPLJoints.RIGHT_KNEE]
    right_ankle = joint_coords[SMPLJoints.RIGHT_ANKLE]
    
    left_upper_leg = np.linalg.norm(left_knee - pelvis)
    left_lower_leg = np.linalg.norm(left_ankle - left_knee)
    left_leg = left_upper_leg + left_lower_leg
    
    right_upper_leg = np.linalg.norm(right_knee - pelvis)
    right_lower_leg = np.linalg.norm(right_ankle - right_knee)
    right_leg = right_upper_leg + right_lower_leg
    
    avg_leg_length = (left_leg + right_leg) / 2
    
    # Torso + head: pelvis → spine → neck → head
    head = joint_coords[SMPLJoints.HEAD]
    torso_head_length = np.linalg.norm(head - pelvis)
    
    # Estimate full standing height
    # Joint chain gives us pelvis-to-head + pelvis-to-ankle
    # Full height ≈ leg_length + torso_head_length (with some overlap at pelvis)
    estimated_height = avg_leg_length + torso_head_length * 0.95  # Slight adjustment for overlap
    
    return {
        "mesh_height": float(mesh_height),
        "mesh_min_y": float(mesh_min_y),
        "mesh_max_y": float(mesh_max_y),
        "estimated_height": float(estimated_height),
        "leg_length": float(avg_leg_length),
        "torso_head_length": float(torso_head_length),
        "left_leg_length": float(left_leg),
        "right_leg_length": float(right_leg),
    }


def project_point_weak_perspective(
    point_3d: np.ndarray,
    cam_t: np.ndarray,
    focal_length: float,
    image_size: Tuple[int, int],
) -> np.ndarray:
    """
    Project 3D point to 2D using weak perspective projection.
    
    This matches SAM3DBody's camera model.
    
    Args:
        point_3d: [3] point in body-local space
        cam_t: [3] camera translation (tx, ty, tz)
        focal_length: focal length in pixels
        image_size: (width, height)
    
    Returns:
        [2] screen position (x, y)
    """
    tx, ty, tz = cam_t[0], cam_t[1], cam_t[2]
    
    # Weak perspective: x_screen = focal * (X + tx) / tz + cx
    cx, cy = image_size[0] / 2, image_size[1] / 2
    
    x_screen = focal_length * (point_3d[0] + tx) / tz + cx
    y_screen = focal_length * (point_3d[1] + ty) / tz + cy
    
    return np.array([x_screen, y_screen])


def detect_foot_contact(
    joint_coords: np.ndarray,
    vertices: np.ndarray,
    threshold_ratio: float = 0.05,
) -> str:
    """
    Detect if feet are in contact with ground.
    
    Heuristics:
    - Foot Y position close to minimum mesh Y (ground plane)
    - Uses ratio of leg length as threshold for robustness
    
    Args:
        joint_coords: [J, 3] joint positions
        vertices: [V, 3] mesh vertices
        threshold_ratio: Threshold as ratio of leg length (default 5%)
    
    Returns:
        "both", "left", "right", or "none"
    """
    # Ground plane estimate (lowest point of mesh)
    ground_y = vertices[:, 1].min()
    
    # Get ankle/foot positions
    left_ankle = joint_coords[SMPLJoints.LEFT_ANKLE]
    right_ankle = joint_coords[SMPLJoints.RIGHT_ANKLE]
    
    # Calculate leg length for adaptive threshold
    pelvis = joint_coords[SMPLJoints.PELVIS]
    left_knee = joint_coords[SMPLJoints.LEFT_KNEE]
    right_knee = joint_coords[SMPLJoints.RIGHT_KNEE]
    
    left_leg = np.linalg.norm(left_knee - pelvis) + np.linalg.norm(left_ankle - left_knee)
    right_leg = np.linalg.norm(right_knee - pelvis) + np.linalg.norm(right_ankle - right_knee)
    avg_leg = (left_leg + right_leg) / 2
    
    # Adaptive threshold based on leg length
    threshold = avg_leg * threshold_ratio
    
    # Check contact
    left_contact = abs(left_ankle[1] - ground_y) < threshold
    right_contact = abs(right_ankle[1] - ground_y) < threshold
    
    if left_contact and right_contact:
        return "both"
    elif left_contact:
        return "left"
    elif right_contact:
        return "right"
    else:
        return "none"


def create_motion_debug_overlay(
    images: np.ndarray,
    subject_motion: Dict,
    scale_info: Dict,
    arrow_scale: float = 10.0,
    show_skeleton: bool = True,
) -> np.ndarray:
    """
    Create debug visualization with motion vectors overlaid on video.
    
    Shows:
    - Green dot: Pelvis position
    - Yellow arrow: Velocity vector (scaled)
    - Foot contact state (color-coded text)
    - Depth estimate
    - Skeleton lines (optional)
    
    Args:
        images: [N, H, W, 3] video frames (0-255 uint8 or 0-1 float)
        subject_motion: Motion analysis results
        scale_info: Scale/height info
        arrow_scale: Multiplier for velocity arrows
        show_skeleton: Draw skeleton connections
    
    Returns:
        [N, H, W, 3] annotated frames
    """
    # Convert to uint8 if needed
    if images.dtype == np.float32 or images.dtype == np.float64:
        if images.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
        else:
            images = images.astype(np.uint8)
    
    output = images.copy()
    num_frames = len(images)
    
    # Colors (BGR for OpenCV)
    COLOR_PELVIS = (0, 255, 0)       # Green
    COLOR_VELOCITY = (0, 255, 255)   # Yellow
    COLOR_SKELETON = (255, 128, 0)   # Orange
    COLOR_GROUNDED = (0, 255, 0)     # Green
    COLOR_AIRBORNE = (0, 0, 255)     # Red
    COLOR_PARTIAL = (0, 255, 255)    # Yellow
    COLOR_TEXT = (255, 255, 255)     # White
    
    # Skeleton connections for visualization
    SKELETON_CONNECTIONS = [
        # Spine
        (SMPLJoints.PELVIS, SMPLJoints.SPINE1),
        (SMPLJoints.SPINE1, SMPLJoints.SPINE2),
        (SMPLJoints.SPINE2, SMPLJoints.SPINE3),
        (SMPLJoints.SPINE3, SMPLJoints.NECK),
        (SMPLJoints.NECK, SMPLJoints.HEAD),
        # Left leg
        (SMPLJoints.PELVIS, SMPLJoints.LEFT_HIP),
        (SMPLJoints.LEFT_HIP, SMPLJoints.LEFT_KNEE),
        (SMPLJoints.LEFT_KNEE, SMPLJoints.LEFT_ANKLE),
        (SMPLJoints.LEFT_ANKLE, SMPLJoints.LEFT_FOOT),
        # Right leg
        (SMPLJoints.PELVIS, SMPLJoints.RIGHT_HIP),
        (SMPLJoints.RIGHT_HIP, SMPLJoints.RIGHT_KNEE),
        (SMPLJoints.RIGHT_KNEE, SMPLJoints.RIGHT_ANKLE),
        (SMPLJoints.RIGHT_ANKLE, SMPLJoints.RIGHT_FOOT),
        # Left arm
        (SMPLJoints.SPINE3, SMPLJoints.LEFT_COLLAR),
        (SMPLJoints.LEFT_COLLAR, SMPLJoints.LEFT_SHOULDER),
        (SMPLJoints.LEFT_SHOULDER, SMPLJoints.LEFT_ELBOW),
        (SMPLJoints.LEFT_ELBOW, SMPLJoints.LEFT_WRIST),
        # Right arm
        (SMPLJoints.SPINE3, SMPLJoints.RIGHT_COLLAR),
        (SMPLJoints.RIGHT_COLLAR, SMPLJoints.RIGHT_SHOULDER),
        (SMPLJoints.RIGHT_SHOULDER, SMPLJoints.RIGHT_ELBOW),
        (SMPLJoints.RIGHT_ELBOW, SMPLJoints.RIGHT_WRIST),
    ]
    
    for i in range(num_frames):
        frame = output[i]
        
        # Get 2D joint positions for this frame
        joints_2d = subject_motion.get("joints_2d")
        if joints_2d is not None and i < len(joints_2d) and joints_2d[i] is not None:
            joints_2d_frame = joints_2d[i]
            
            # Draw skeleton
            if show_skeleton:
                for j1, j2 in SKELETON_CONNECTIONS:
                    if j1 < len(joints_2d_frame) and j2 < len(joints_2d_frame):
                        pt1 = joints_2d_frame[j1]
                        pt2 = joints_2d_frame[j2]
                        if pt1 is not None and pt2 is not None:
                            cv2.line(frame, 
                                    (int(pt1[0]), int(pt1[1])),
                                    (int(pt2[0]), int(pt2[1])),
                                    COLOR_SKELETON, 1, cv2.LINE_AA)
            
            # Draw joint dots
            for j, pt in enumerate(joints_2d_frame):
                if pt is not None:
                    color = COLOR_PELVIS if j == SMPLJoints.PELVIS else (128, 128, 255)
                    radius = 6 if j == SMPLJoints.PELVIS else 3
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), radius, color, -1)
        
        # Draw pelvis position (larger, green)
        pelvis_2d = subject_motion.get("pelvis_2d")
        if pelvis_2d is not None and i < len(pelvis_2d):
            px, py = pelvis_2d[i]
            cv2.circle(frame, (int(px), int(py)), 8, COLOR_PELVIS, -1)
            cv2.circle(frame, (int(px), int(py)), 10, (0, 0, 0), 2)  # Black outline
        
        # Draw velocity arrow
        velocity_2d = subject_motion.get("velocity_2d")
        if velocity_2d is not None and i > 0 and (i-1) < len(velocity_2d):
            vx, vy = velocity_2d[i-1]
            if pelvis_2d is not None and i < len(pelvis_2d):
                px, py = pelvis_2d[i]
                # Scale velocity for visibility
                end_x = int(px + vx * arrow_scale)
                end_y = int(py + vy * arrow_scale)
                cv2.arrowedLine(frame, (int(px), int(py)), (end_x, end_y),
                               COLOR_VELOCITY, 2, tipLength=0.3)
        
        # Draw info text
        y_offset = 30
        line_height = 30
        
        # Foot contact
        foot_contact = subject_motion.get("foot_contact", [])
        if i < len(foot_contact):
            foot = foot_contact[i]
            foot_colors = {
                "both": COLOR_GROUNDED,
                "left": COLOR_PARTIAL,
                "right": COLOR_PARTIAL,
                "none": COLOR_AIRBORNE,
            }
            foot_color = foot_colors.get(foot, COLOR_TEXT)
            cv2.putText(frame, f"Feet: {foot}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, foot_color, 2)
            y_offset += line_height
        
        # Depth estimate
        depth_estimate = subject_motion.get("depth_estimate", [])
        if i < len(depth_estimate):
            depth = depth_estimate[i]
            cv2.putText(frame, f"Depth: {depth:.2f}m", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
            y_offset += line_height
        
        # Apparent height
        apparent_height = subject_motion.get("apparent_height", [])
        if i < len(apparent_height):
            height_px = apparent_height[i]
            cv2.putText(frame, f"Height: {height_px:.0f}px", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
            y_offset += line_height
        
        # Scale info (only on first frame or if changed)
        if i == 0:
            actual_height = scale_info.get("actual_height_m", 1.70)
            cv2.putText(frame, f"Subject: {actual_height:.2f}m", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 255, 128), 2)
            y_offset += line_height
        
        # Frame number
        cv2.putText(frame, f"Frame: {i}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    return output


class SAM4DMotionAnalyzer:
    """
    Analyze subject motion from SAM3DBody mesh sequence.
    
    This node extracts:
    - Subject height (estimated from mesh, with user override option)
    - Per-frame pelvis/joint positions (2D screen + 3D world)
    - Per-frame velocity (2D and 3D)
    - Foot contact detection (both/left/right/none)
    - Apparent height in pixels (depth indicator)
    
    Outputs:
    - SUBJECT_MOTION: Per-frame motion data
    - SCALE_INFO: Height and scale factor
    - debug_overlay: Motion vectors on video (optional)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("SAM4D_MESH_SEQUENCE", {
                    "tooltip": "Mesh sequence from SAM3DBody Batch Process"
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Original video frames for debug overlay"
                }),
                "subject_height_m": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Subject height in meters. 0 = auto-estimate (~1.70m)"
                }),
                "reference_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Frame to use for height estimation (should be standing pose)"
                }),
                "default_height_m": ("FLOAT", {
                    "default": 1.70,
                    "min": 0.5,
                    "max": 2.5,
                    "step": 0.01,
                    "tooltip": "Default height assumption when auto-estimating"
                }),
                "foot_contact_threshold": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.01,
                    "max": 0.20,
                    "step": 0.01,
                    "tooltip": "Foot contact threshold as ratio of leg length"
                }),
                "show_debug": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Generate debug overlay with motion vectors"
                }),
                "show_skeleton": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Draw skeleton on debug overlay"
                }),
                "arrow_scale": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 50.0,
                    "tooltip": "Scale factor for velocity arrows in debug view"
                }),
            }
        }
    
    RETURN_TYPES = ("SUBJECT_MOTION", "SCALE_INFO", "IMAGE", "STRING")
    RETURN_NAMES = ("subject_motion", "scale_info", "debug_overlay", "debug_info")
    FUNCTION = "analyze"
    CATEGORY = "SAM4DBodyCapture/Motion"
    
    def analyze(
        self,
        mesh_sequence: Dict,
        images: torch.Tensor = None,
        subject_height_m: float = 0.0,
        reference_frame: int = 0,
        default_height_m: float = 1.70,
        foot_contact_threshold: float = 0.05,
        show_debug: bool = True,
        show_skeleton: bool = True,
        arrow_scale: float = 10.0,
    ) -> Tuple[Dict, Dict, torch.Tensor, str]:
        """
        Analyze subject motion from mesh sequence.
        """
        print("\n[Motion Analyzer] ========== SUBJECT MOTION ANALYSIS ==========")
        
        # Extract data from mesh sequence
        vertices_list = mesh_sequence.get("vertices", [])
        params = mesh_sequence.get("params", {})
        joint_coords_list = params.get("joint_coords", [])
        camera_t_list = params.get("camera_t", [])
        focal_length_list = params.get("focal_length", [])
        
        num_frames = len(vertices_list)
        if num_frames == 0:
            print("[Motion Analyzer] ERROR: No frames in mesh sequence!")
            return ({}, {}, torch.zeros(1, 64, 64, 3), "Error: No frames")
        
        print(f"[Motion Analyzer] Processing {num_frames} frames...")
        
        # Get image size from first camera_t or default
        # SAM3DBody typically works with the input image size
        image_size = (1920, 1080)  # Default
        if images is not None:
            _, H, W, _ = images.shape
            image_size = (W, H)
            print(f"[Motion Analyzer] Image size from input: {W}x{H}")
        
        # ===== HEIGHT ESTIMATION =====
        ref_frame = min(reference_frame, num_frames - 1)
        ref_vertices = vertices_list[ref_frame]
        ref_joints = joint_coords_list[ref_frame]
        
        # Convert to numpy if needed
        if isinstance(ref_vertices, torch.Tensor):
            ref_vertices = ref_vertices.cpu().numpy()
        if isinstance(ref_joints, torch.Tensor):
            ref_joints = ref_joints.cpu().numpy()
        
        # Estimate height from mesh
        height_est = estimate_height_from_mesh(ref_vertices, ref_joints)
        
        # Determine actual height (user override or estimate)
        if subject_height_m > 0:
            actual_height = subject_height_m
            height_source = "user_input"
            print(f"[Motion Analyzer] Using user-specified height: {actual_height:.2f}m")
        else:
            # Use default assumption and calculate scale
            actual_height = default_height_m
            height_source = "auto_estimate"
            print(f"[Motion Analyzer] Using default height assumption: {actual_height:.2f}m")
        
        # Calculate scale factor (mesh units to meters)
        estimated_mesh_height = height_est["estimated_height"]
        if estimated_mesh_height > 0:
            scale_factor = actual_height / estimated_mesh_height
        else:
            scale_factor = 1.0
        
        scale_info = {
            "mesh_height": height_est["mesh_height"],
            "estimated_height": estimated_mesh_height,
            "actual_height_m": actual_height,
            "scale_factor": scale_factor,
            "leg_length": height_est["leg_length"],
            "torso_head_length": height_est["torso_head_length"],
            "height_source": height_source,
            "reference_frame": ref_frame,
        }
        
        print(f"[Motion Analyzer] Mesh height: {height_est['mesh_height']:.3f} units")
        print(f"[Motion Analyzer] Estimated height: {estimated_mesh_height:.3f} units")
        print(f"[Motion Analyzer] Scale factor: {scale_factor:.3f} (mesh → meters)")
        print(f"[Motion Analyzer] Leg length: {height_est['leg_length']:.3f} units")
        
        # ===== PER-FRAME ANALYSIS =====
        subject_motion = {
            "pelvis_2d": [],          # [N, 2] screen position
            "pelvis_3d": [],          # [N, 3] world position (meters)
            "joints_2d": [],          # [N, J, 2] all joints screen position
            "joints_3d": [],          # [N, J, 3] all joints world position
            "velocity_2d": [],        # [N-1, 2] screen velocity
            "velocity_3d": [],        # [N-1, 3] world velocity
            "apparent_height": [],    # [N] height in pixels
            "depth_estimate": [],     # [N] depth in meters
            "foot_contact": [],       # [N] "both"/"left"/"right"/"none"
            "camera_t": [],           # [N, 3] camera translation
            "focal_length": [],       # [N] focal length
            "image_size": image_size,
            "num_frames": num_frames,
            "scale_factor": scale_factor,
        }
        
        for i in range(num_frames):
            # Get frame data
            vertices = vertices_list[i]
            joint_coords = joint_coords_list[i] if i < len(joint_coords_list) else None
            camera_t = camera_t_list[i] if i < len(camera_t_list) else np.array([0, 0, 5])
            focal = focal_length_list[i] if i < len(focal_length_list) else 1000.0
            
            # Convert to numpy
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.cpu().numpy()
            if isinstance(joint_coords, torch.Tensor):
                joint_coords = joint_coords.cpu().numpy()
            if isinstance(camera_t, torch.Tensor):
                camera_t = camera_t.cpu().numpy()
            if isinstance(focal, torch.Tensor):
                focal = focal.cpu().item()
            
            # Handle None values
            if joint_coords is None:
                print(f"[Motion Analyzer] Warning: No joint coords for frame {i}")
                joint_coords = np.zeros((22, 3))
            if camera_t is None:
                camera_t = np.array([0, 0, 5])
            
            # Flatten camera_t if needed
            if len(camera_t.shape) > 1:
                camera_t = camera_t.flatten()[:3]
            
            subject_motion["camera_t"].append(camera_t.copy())
            subject_motion["focal_length"].append(float(focal))
            
            # 1. Pelvis position
            pelvis_3d = joint_coords[SMPLJoints.PELVIS] * scale_factor
            subject_motion["pelvis_3d"].append(pelvis_3d.copy())
            
            pelvis_2d = project_point_weak_perspective(
                joint_coords[SMPLJoints.PELVIS], camera_t, focal, image_size
            )
            subject_motion["pelvis_2d"].append(pelvis_2d.copy())
            
            # 2. All joints (2D and 3D)
            joints_3d_frame = joint_coords * scale_factor
            subject_motion["joints_3d"].append(joints_3d_frame.copy())
            
            joints_2d_frame = []
            for j in range(len(joint_coords)):
                pt_2d = project_point_weak_perspective(
                    joint_coords[j], camera_t, focal, image_size
                )
                joints_2d_frame.append(pt_2d)
            subject_motion["joints_2d"].append(np.array(joints_2d_frame))
            
            # 3. Apparent height (pixels)
            head_2d = project_point_weak_perspective(
                joint_coords[SMPLJoints.HEAD], camera_t, focal, image_size
            )
            # Use lower of two ankles for feet
            left_ankle_2d = project_point_weak_perspective(
                joint_coords[SMPLJoints.LEFT_ANKLE], camera_t, focal, image_size
            )
            right_ankle_2d = project_point_weak_perspective(
                joint_coords[SMPLJoints.RIGHT_ANKLE], camera_t, focal, image_size
            )
            feet_y = max(left_ankle_2d[1], right_ankle_2d[1])
            apparent_height = abs(feet_y - head_2d[1])
            subject_motion["apparent_height"].append(apparent_height)
            
            # 4. Depth estimate (from camera_t.z, scaled)
            depth_m = camera_t[2] * scale_factor
            subject_motion["depth_estimate"].append(depth_m)
            
            # 5. Foot contact detection
            foot_contact = detect_foot_contact(
                joint_coords, vertices, foot_contact_threshold
            )
            subject_motion["foot_contact"].append(foot_contact)
        
        # ===== CALCULATE VELOCITIES =====
        pelvis_2d_arr = np.array(subject_motion["pelvis_2d"])
        pelvis_3d_arr = np.array(subject_motion["pelvis_3d"])
        
        if num_frames > 1:
            velocity_2d = np.diff(pelvis_2d_arr, axis=0)
            velocity_3d = np.diff(pelvis_3d_arr, axis=0)
        else:
            velocity_2d = np.zeros((0, 2))
            velocity_3d = np.zeros((0, 3))
        
        subject_motion["velocity_2d"] = velocity_2d.tolist()
        subject_motion["velocity_3d"] = velocity_3d.tolist()
        
        # ===== STATISTICS =====
        avg_velocity_2d = np.mean(np.abs(velocity_2d)) if len(velocity_2d) > 0 else 0
        max_velocity_2d = np.max(np.abs(velocity_2d)) if len(velocity_2d) > 0 else 0
        
        grounded_count = sum(1 for fc in subject_motion["foot_contact"] if fc in ["both", "left", "right"])
        airborne_count = sum(1 for fc in subject_motion["foot_contact"] if fc == "none")
        
        print(f"\n[Motion Analyzer] ----- MOTION STATISTICS -----")
        print(f"[Motion Analyzer] Frames: {num_frames}")
        print(f"[Motion Analyzer] Avg 2D velocity: {avg_velocity_2d:.2f} px/frame")
        print(f"[Motion Analyzer] Max 2D velocity: {max_velocity_2d:.2f} px/frame")
        print(f"[Motion Analyzer] Grounded frames: {grounded_count} ({100*grounded_count/num_frames:.1f}%)")
        print(f"[Motion Analyzer] Airborne frames: {airborne_count} ({100*airborne_count/num_frames:.1f}%)")
        print(f"[Motion Analyzer] Depth range: {min(subject_motion['depth_estimate']):.2f}m - {max(subject_motion['depth_estimate']):.2f}m")
        
        # ===== DEBUG INFO STRING =====
        debug_info = (
            f"=== Motion Analysis Results ===\n"
            f"Frames: {num_frames}\n"
            f"Subject height: {actual_height:.2f}m ({height_source})\n"
            f"Scale factor: {scale_factor:.3f}\n"
            f"Avg 2D velocity: {avg_velocity_2d:.2f} px/frame\n"
            f"Max 2D velocity: {max_velocity_2d:.2f} px/frame\n"
            f"Grounded: {grounded_count}/{num_frames} frames\n"
            f"Depth range: {min(subject_motion['depth_estimate']):.2f}m - {max(subject_motion['depth_estimate']):.2f}m\n"
        )
        
        # ===== DEBUG OVERLAY =====
        if show_debug and images is not None:
            print(f"[Motion Analyzer] Generating debug overlay...")
            
            # Convert images to numpy
            if isinstance(images, torch.Tensor):
                images_np = images.cpu().numpy()
            else:
                images_np = images
            
            # Create overlay
            overlay = create_motion_debug_overlay(
                images_np,
                subject_motion,
                scale_info,
                arrow_scale=arrow_scale,
                show_skeleton=show_skeleton,
            )
            
            # Convert back to tensor
            if overlay.dtype == np.uint8:
                overlay = overlay.astype(np.float32) / 255.0
            debug_overlay = torch.from_numpy(overlay).float()
        else:
            # Return empty tensor if no debug
            debug_overlay = torch.zeros(1, 64, 64, 3)
        
        print(f"[Motion Analyzer] =============================================\n")
        
        return (subject_motion, scale_info, debug_overlay, debug_info)


class SAM4DScaleInfoDisplay:
    """Display scale/height information from Motion Analyzer."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scale_info": ("SCALE_INFO",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "display"
    CATEGORY = "SAM4DBodyCapture/Motion"
    OUTPUT_NODE = True
    
    def display(self, scale_info: Dict) -> Tuple[str]:
        info = (
            "=== Scale Info ===\n"
            f"Actual height: {scale_info.get('actual_height_m', 'N/A'):.2f}m\n"
            f"Mesh height: {scale_info.get('mesh_height', 'N/A'):.3f} units\n"
            f"Estimated height: {scale_info.get('estimated_height', 'N/A'):.3f} units\n"
            f"Scale factor: {scale_info.get('scale_factor', 'N/A'):.3f}\n"
            f"Leg length: {scale_info.get('leg_length', 'N/A'):.3f} units\n"
            f"Torso+head: {scale_info.get('torso_head_length', 'N/A'):.3f} units\n"
            f"Source: {scale_info.get('height_source', 'N/A')}\n"
            f"Reference frame: {scale_info.get('reference_frame', 'N/A')}\n"
        )
        print(info)
        return (info,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SAM4DMotionAnalyzer": SAM4DMotionAnalyzer,
    "SAM4DScaleInfoDisplay": SAM4DScaleInfoDisplay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM4DMotionAnalyzer": "📊 Motion Analyzer",
    "SAM4DScaleInfoDisplay": "📏 Scale Info Display",
}
