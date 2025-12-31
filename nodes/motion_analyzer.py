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

Joint Index Reference (MHR 70-Joint / 127-Joint formats share same body indices):
- Index 0: body_world (global trajectory)
- Index 1-21: Core body joints
- Index 22+: Hand joints (70-joint) or Hand+Face (127-joint)
"""

# Version for logging
VERSION = "0.5.0-debug11"

import numpy as np
import torch
import cv2
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta

# IST timezone (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

def get_timestamp():
    """Get current timestamp in IST format."""
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


# ============================================================================
# Unified Body Joint Structure (works for both 70-joint and 127-joint formats)
# From SAM3DBody MHR model - same indices for both pred_keypoints and joint_coords
# ============================================================================
class BodyJoints:
    """Unified joint indices for SAM3DBody outputs.
    
    Both pred_keypoints_2d/3d (70 joints) and joint_coords (127 joints)
    share the same body joint indices (0-21).
    
    CRITICAL: Index 0 is body_world (global trajectory), NOT pelvis!
    
    debug10: Corrected indices based on actual SAM3DBody MHR structure.
    Previous versions used wrong indices from an incorrect reference.
    """
    # Global trajectory (index 0)
    BODY_WORLD = 0   # Global movement (X, Y, Z translation in world space)
    
    # Core body (1-21)
    PELVIS = 1       # Anatomical root for spine and legs
    SPINE_1 = 2      # Lower spine
    SPINE_2 = 3      # Upper spine / shoulder connection
    NECK = 4         # Base of neck
    HEAD = 5         # Top of head / skull
    
    # Left leg (6-8)
    L_HIP = 6        # Primary left hip rotation
    L_KNEE = 7       # Left knee flexion/extension
    L_ANKLE = 8      # Left ankle joint
    
    # Right leg (9-11)
    R_HIP = 9        # Primary right hip rotation
    R_KNEE = 10      # Right knee flexion/extension
    R_ANKLE = 11     # Right ankle joint
    
    # Feet (12-15)
    L_FOOT_HEEL = 12 # Left foot heel
    L_FOOT_TOE = 13  # Left foot toe
    R_FOOT_HEEL = 14 # Right foot heel
    R_FOOT_TOE = 15  # Right foot toe
    
    # Left arm (16-18)
    L_SHOULDER = 16  # Left shoulder joint
    L_ELBOW = 17     # Left elbow articulation
    L_WRIST = 18     # Left wrist joint
    
    # Right arm (19-21)
    R_SHOULDER = 19  # Right shoulder joint
    R_ELBOW = 20     # Right elbow articulation
    R_WRIST = 21     # Right wrist joint
    
    # Backward-compatible aliases (snake_case to UPPER_CASE with L/R prefix)
    LEFT_HIP = L_HIP
    LEFT_KNEE = L_KNEE
    LEFT_ANKLE = L_ANKLE
    RIGHT_HIP = R_HIP
    RIGHT_KNEE = R_KNEE
    RIGHT_ANKLE = R_ANKLE
    LEFT_SHOULDER = L_SHOULDER
    LEFT_ELBOW = L_ELBOW
    LEFT_WRIST = L_WRIST
    RIGHT_SHOULDER = R_SHOULDER
    RIGHT_ELBOW = R_ELBOW
    RIGHT_WRIST = R_WRIST
    LEFT_FOOT_HEEL = L_FOOT_HEEL
    LEFT_FOOT_TOE = L_FOOT_TOE
    RIGHT_FOOT_HEEL = R_FOOT_HEEL
    RIGHT_FOOT_TOE = R_FOOT_TOE
    
    # Hand joints (22+)
    L_HAND_START = 22
    L_HAND_END = 45
    R_HAND_START = 46
    R_HAND_END = 69
    
    NUM_BODY_JOINTS = 22  # Indices 0-21 (including body_world)
    NUM_70_JOINTS = 70    # Body + hands
    NUM_127_JOINTS = 127  # Body + hands + feet details + face
    
    # Joint names for labeling
    JOINT_NAMES = {
        0: "body_world", 1: "pelvis", 2: "spine1", 3: "spine2",
        4: "neck", 5: "head",
        6: "L_hip", 7: "L_knee", 8: "L_ankle",
        9: "R_hip", 10: "R_knee", 11: "R_ankle",
        12: "L_heel", 13: "L_toe", 14: "R_heel", 15: "R_toe",
        16: "L_shldr", 17: "L_elbow", 18: "L_wrist",
        19: "R_shldr", 20: "R_elbow", 21: "R_wrist",
    }
    
    # Skeleton connections for visualization
    CONNECTIONS = [
        # Spine chain
        (PELVIS, SPINE_1), (SPINE_1, SPINE_2), (SPINE_2, NECK), (NECK, HEAD),
        # Left leg
        (PELVIS, L_HIP), (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), 
        (L_ANKLE, L_FOOT_HEEL), (L_ANKLE, L_FOOT_TOE),
        # Right leg
        (PELVIS, R_HIP), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
        (R_ANKLE, R_FOOT_HEEL), (R_ANKLE, R_FOOT_TOE),
        # Left arm
        (SPINE_2, L_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
        # Right arm
        (SPINE_2, R_SHOULDER), (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
        # Hip connection
        (L_HIP, R_HIP),
        # Shoulder connection
        (L_SHOULDER, R_SHOULDER),
    ]


# Backward compatibility aliases
SAM3DJoints = BodyJoints
MHRJoints = BodyJoints
SMPLHJoints = BodyJoints


def to_numpy(data):
    """Convert tensor or list to numpy array."""
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    if isinstance(data, np.ndarray):
        return data.copy()
    return np.array(data)


def project_points_to_2d(
    points_3d: np.ndarray,
    focal_length: float,
    cam_t: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """
    Project 3D points to 2D using SAM3DBody's camera model.
    
    debug10: Fixed to match mesh_overlay.py projection exactly.
    The mesh overlay applies a 180° rotation around X axis after translation,
    which negates Y and Z. This is required for correct alignment.
    
    Args:
        points_3d: (N, 3) array of 3D points
        focal_length: focal length in pixels
        cam_t: camera translation [tx, ty, tz]
        image_width, image_height: image dimensions
        
    Returns:
        points_2d: (N, 2) array of 2D points
    """
    points_3d = np.array(points_3d).copy()
    cam_t = np.array(cam_t).flatten()
    
    # Camera center (principal point)
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    if len(cam_t) < 3:
        # Fallback if cam_t is incomplete
        return np.column_stack([
            np.full(len(points_3d), cx),
            np.full(len(points_3d), cy)
        ])
    
    tx, ty, tz = cam_t[0], cam_t[1], cam_t[2]
    
    # SAM3DBody camera model (same as mesh_overlay.py fallback):
    # 1. Apply camera translation
    X = points_3d[:, 0] + tx
    Y = points_3d[:, 1] + ty
    Z = points_3d[:, 2] + tz
    
    # 2. Apply 180° rotation around X axis (same as SAM3DBody renderer.py line 209)
    # This is critical for matching mesh overlay alignment!
    Y = -Y
    Z = -Z
    
    # 3. Avoid division by zero
    Z = np.where(np.abs(Z) < 0.1, 0.1, Z)
    
    # 4. Perspective projection
    x_2d = focal_length * X / Z + cx
    y_2d = focal_length * Y / Z + cy
    
    return np.stack([x_2d, y_2d], axis=1)


def estimate_height_from_keypoints(
    keypoints_3d: np.ndarray,
    skeleton_mode: str = "simple",
) -> Dict[str, float]:
    """
    Estimate subject height from 3D keypoints.
    
    Args:
        keypoints_3d: [J, 3] keypoint positions
        skeleton_mode: "simple" (18-joint) or "full" (127-joint)
    
    Returns:
        dict with height measurements
    """
    if skeleton_mode == "simple":
        # SAM3DBody 18-joint
        J = SAM3DJoints
        pelvis = keypoints_3d[J.PELVIS]
        head = keypoints_3d[J.HEAD]
        left_hip = keypoints_3d[J.LEFT_HIP]
        left_knee = keypoints_3d[J.LEFT_KNEE]
        left_ankle = keypoints_3d[J.LEFT_ANKLE]
        right_hip = keypoints_3d[J.RIGHT_HIP]
        right_knee = keypoints_3d[J.RIGHT_KNEE]
        right_ankle = keypoints_3d[J.RIGHT_ANKLE]
    else:
        # SMPL-H 127-joint (use first 22)
        J = SMPLHJoints
        pelvis = keypoints_3d[J.PELVIS]
        head = keypoints_3d[J.HEAD]
        left_hip = keypoints_3d[J.LEFT_HIP]
        left_knee = keypoints_3d[J.LEFT_KNEE]
        left_ankle = keypoints_3d[J.LEFT_ANKLE]
        right_hip = keypoints_3d[J.RIGHT_HIP]
        right_knee = keypoints_3d[J.RIGHT_KNEE]
        right_ankle = keypoints_3d[J.RIGHT_ANKLE]
    
    # Leg length: hip → knee → ankle
    left_upper_leg = np.linalg.norm(left_knee - left_hip)
    left_lower_leg = np.linalg.norm(left_ankle - left_knee)
    left_leg = left_upper_leg + left_lower_leg
    
    right_upper_leg = np.linalg.norm(right_knee - right_hip)
    right_lower_leg = np.linalg.norm(right_ankle - right_knee)
    right_leg = right_upper_leg + right_lower_leg
    
    avg_leg_length = (left_leg + right_leg) / 2
    
    # Torso + head: pelvis → head
    torso_head_length = np.linalg.norm(head - pelvis)
    
    # Estimate full standing height
    # Full height ≈ leg_length + torso_head_length (with overlap adjustment)
    estimated_height = avg_leg_length + torso_head_length * 0.95
    
    return {
        "estimated_height": float(estimated_height),
        "leg_length": float(avg_leg_length),
        "torso_head_length": float(torso_head_length),
        "left_leg_length": float(left_leg),
        "right_leg_length": float(right_leg),
    }


def estimate_height_from_mesh(vertices: np.ndarray) -> Dict[str, float]:
    """
    Estimate height from mesh bounding box.
    """
    mesh_min_y = vertices[:, 1].min()
    mesh_max_y = vertices[:, 1].max()
    mesh_height = mesh_max_y - mesh_min_y
    
    return {
        "mesh_height": float(mesh_height),
        "mesh_min_y": float(mesh_min_y),
        "mesh_max_y": float(mesh_max_y),
    }


def detect_foot_contact(
    keypoints_3d: np.ndarray,
    vertices: np.ndarray,
    skeleton_mode: str = "simple",
    threshold_ratio: float = 0.05,
) -> str:
    """
    Detect if feet are in contact with ground.
    
    Args:
        keypoints_3d: [J, 3] keypoint positions
        vertices: [V, 3] mesh vertices
        skeleton_mode: "simple" (18-joint) or "full" (127-joint)
        threshold_ratio: Threshold as ratio of leg length
    
    Returns:
        "both", "left", "right", or "none"
    """
    if skeleton_mode == "simple":
        J = SAM3DJoints
        left_ankle = keypoints_3d[J.LEFT_ANKLE]
        right_ankle = keypoints_3d[J.RIGHT_ANKLE]
        left_hip = keypoints_3d[J.LEFT_HIP]
        left_knee = keypoints_3d[J.LEFT_KNEE]
        right_hip = keypoints_3d[J.RIGHT_HIP]
        right_knee = keypoints_3d[J.RIGHT_KNEE]
    else:
        J = SMPLHJoints
        left_ankle = keypoints_3d[J.LEFT_ANKLE]
        right_ankle = keypoints_3d[J.RIGHT_ANKLE]
        left_hip = keypoints_3d[J.LEFT_HIP]
        left_knee = keypoints_3d[J.LEFT_KNEE]
        right_hip = keypoints_3d[J.RIGHT_HIP]
        right_knee = keypoints_3d[J.RIGHT_KNEE]
    
    # Ground plane estimate (lowest point of mesh)
    ground_y = vertices[:, 1].min()
    
    # Calculate leg length for adaptive threshold
    left_leg = np.linalg.norm(left_knee - left_hip) + np.linalg.norm(left_ankle - left_knee)
    right_leg = np.linalg.norm(right_knee - right_hip) + np.linalg.norm(right_ankle - right_knee)
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
    skeleton_mode: str = "simple",
    arrow_scale: float = 10.0,
    show_skeleton: bool = True,
) -> np.ndarray:
    """
    Create debug visualization with joint markers overlaid on video.
    
    debug11: Uses pred_keypoints_2d directly (already in pixel coordinates).
    The joints_2d data in subject_motion comes from pred_keypoints_2d which
    is already properly aligned with the video frame.
    
    Uses unified BodyJoints indices (same for all SAM3DBody outputs).
    Only individual joint dots are shown - no skeleton lines since
    joints have independent translational data.
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
    COLOR_JOINTS = (255, 128, 128)   # Light blue
    COLOR_HEAD = (0, 255, 255)       # Yellow - for head
    COLOR_HANDS = (255, 0, 255)      # Magenta - for wrists
    COLOR_FEET = (255, 128, 0)       # Orange - for ankles
    COLOR_GROUNDED = (0, 255, 0)     # Green
    COLOR_AIRBORNE = (0, 0, 255)     # Red
    COLOR_PARTIAL = (0, 255, 255)    # Yellow
    COLOR_TEXT = (255, 255, 255)     # White
    COLOR_LABEL = (255, 255, 255)    # White for joint labels
    
    # debug10: Use unified BodyJoints indices
    pelvis_idx = BodyJoints.PELVIS          # 1
    head_idx = BodyJoints.HEAD              # 5
    left_wrist_idx = BodyJoints.L_WRIST     # 18
    right_wrist_idx = BodyJoints.R_WRIST    # 21
    left_ankle_idx = BodyJoints.L_ANKLE     # 8
    right_ankle_idx = BodyJoints.R_ANKLE    # 11
    
    # Special joint indices for coloring
    special_joints = {
        pelvis_idx: (COLOR_PELVIS, 8),      # Green, large
        head_idx: (COLOR_HEAD, 6),          # Yellow, medium
        left_wrist_idx: (COLOR_HANDS, 5),   # Magenta
        right_wrist_idx: (COLOR_HANDS, 5),  # Magenta
        left_ankle_idx: (COLOR_FEET, 5),    # Orange
        right_ankle_idx: (COLOR_FEET, 5),   # Orange
    }
    
    for i in range(num_frames):
        frame = output[i]
        
        # Get 2D joint positions for this frame
        joints_2d = subject_motion.get("joints_2d")
        if joints_2d is not None and i < len(joints_2d) and joints_2d[i] is not None:
            joints_2d_frame = np.array(joints_2d[i])
            
            # Draw joint dots with labels (unified format - first 22 are body joints)
            if show_skeleton:
                num_body_joints = min(BodyJoints.NUM_BODY_JOINTS, len(joints_2d_frame))
                for j in range(num_body_joints):
                    pt = joints_2d_frame[j]
                    if pt is not None and len(pt) >= 2:
                        x, y = int(pt[0]), int(pt[1])
                        # Skip invalid coordinates
                        if x < 0 or y < 0 or x > 10000 or y > 10000:
                            continue
                        # Use special color/size for key joints
                        if j in special_joints:
                            color, radius = special_joints[j]
                        else:
                            color = COLOR_JOINTS
                            radius = 3
                        cv2.circle(frame, (x, y), radius, color, -1)
                        # Add black outline for visibility
                        cv2.circle(frame, (x, y), radius, (0, 0, 0), 1)
                        
                        # Draw joint label (index and short name)
                        if j in BodyJoints.JOINT_NAMES:
                            label = f"{j}"  # Just index for clarity
                            cv2.putText(frame, label, (x + 5, y - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_LABEL, 1)
        
        # Draw pelvis position with larger black outline
        pelvis_2d = subject_motion.get("pelvis_2d")
        if pelvis_2d is not None and i < len(pelvis_2d):
            px, py = pelvis_2d[i]
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
        line_height = 25
        
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
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, foot_color, 2)
            y_offset += line_height
        
        # Depth estimate
        depth_estimate = subject_motion.get("depth_estimate", [])
        if i < len(depth_estimate):
            depth = depth_estimate[i]
            cv2.putText(frame, f"Depth: {depth:.2f}m", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
            y_offset += line_height
        
        # Apparent height
        apparent_height = subject_motion.get("apparent_height", [])
        if i < len(apparent_height):
            height_px = apparent_height[i]
            cv2.putText(frame, f"Height: {height_px:.0f}px", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
            y_offset += line_height
        
        # Frame number
        cv2.putText(frame, f"Frame: {i}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    return output


class SAM4DMotionAnalyzer:
    """
    Analyze subject motion from SAM3DBody mesh sequence.
    
    This node extracts:
    - Subject height (estimated from keypoints, with user override option)
    - Per-frame pelvis/joint positions (2D screen + 3D world)
    - Per-frame velocity (2D and 3D)
    - Foot contact detection (both/left/right/none)
    - Apparent height in pixels (depth indicator)
    
    debug11 Changes (CRITICAL FIX):
    - Use pred_keypoints_2d DIRECTLY (already in pixel coords) for 2D overlay
    - Do NOT project joint_coords - it's in LOCAL body space, not world space!
    - Fallback: project pred_keypoints_3d (18-joint world space) if 2D unavailable
    - Uses unified BodyJoints indices (same indices work for all data sources)
    
    Data Sources:
    - pred_keypoints_2d: 70 joints, already in pixel coordinates (USE THIS FOR 2D!)
    - pred_keypoints_3d: 18 joints, world space coordinates
    - joint_coords: 127 joints, LOCAL body space (NOT for 2D projection!)
    
    Skeleton Modes:
    - "Simple Skeleton" (default): Uses 18-joint keypoints
    - "Full Skeleton": Uses 127-joint SMPL-H skeleton
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
                "skeleton_mode": (["Simple Skeleton", "Full Skeleton"], {
                    "default": "Simple Skeleton",
                    "tooltip": "Simple: 18-joint keypoints, Full: 127-joint SMPL-H"
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
                    "default": 0.10,
                    "min": 0.01,
                    "max": 0.30,
                    "step": 0.01,
                    "tooltip": "Foot contact threshold as ratio of leg length"
                }),
                "show_debug": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Generate debug overlay with motion vectors"
                }),
                "show_skeleton": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Draw joint markers on debug overlay (no lines, just dots)"
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
        skeleton_mode: str = "Simple Skeleton",
        subject_height_m: float = 0.0,
        reference_frame: int = 0,
        default_height_m: float = 1.70,
        foot_contact_threshold: float = 0.10,
        show_debug: bool = True,
        show_skeleton: bool = True,
        arrow_scale: float = 10.0,
    ) -> Tuple[Dict, Dict, torch.Tensor, str]:
        """
        Analyze subject motion from mesh sequence.
        """
        print(f"\n[Motion Analyzer v{VERSION}] ========== SUBJECT MOTION ANALYSIS ==========")
        
        # Determine skeleton mode
        use_simple = skeleton_mode == "Simple Skeleton"
        mode_str = "simple" if use_simple else "full"
        print(f"[{get_timestamp()}] [Motion Analyzer] Skeleton mode: {skeleton_mode}")
        
        # Extract data from mesh sequence
        vertices_list = mesh_sequence.get("vertices", [])
        params = mesh_sequence.get("params", {})
        
        # Get keypoint data
        keypoints_2d_list = params.get("keypoints_2d", [])
        keypoints_3d_list = params.get("keypoints_3d", [])
        joint_coords_list = params.get("joint_coords", [])  # 127-joint fallback
        camera_t_list = params.get("camera_t", [])
        focal_length_list = params.get("focal_length", [])
        
        num_frames = len(vertices_list)
        if num_frames == 0:
            print("[Motion Analyzer] ERROR: No frames in mesh sequence!")
            return ({}, {}, torch.zeros(1, 64, 64, 3), "Error: No frames")
        
        print(f"[{get_timestamp()}] [Motion Analyzer] Processing {num_frames} frames...")
        
        # Check what keypoint data is available
        has_kp_2d = len(keypoints_2d_list) > 0 and keypoints_2d_list[0] is not None
        has_kp_3d = len(keypoints_3d_list) > 0 and keypoints_3d_list[0] is not None
        has_joint_coords = len(joint_coords_list) > 0 and joint_coords_list[0] is not None
        
        print(f"[{get_timestamp()}] [Motion Analyzer] Data available: keypoints_2d={has_kp_2d}, keypoints_3d={has_kp_3d}, joint_coords={has_joint_coords}")
        
        # Decide which 3D keypoints to use
        if use_simple and has_kp_3d:
            kp_source = "keypoints_3d"
            print(f"[{get_timestamp()}] [Motion Analyzer] Using 18-joint keypoints_3d for analysis")
        elif has_joint_coords:
            kp_source = "joint_coords"
            print(f"[{get_timestamp()}] [Motion Analyzer] Using 127-joint joint_coords for analysis")
        elif has_kp_3d:
            kp_source = "keypoints_3d"
            print(f"[{get_timestamp()}] [Motion Analyzer] Fallback to 18-joint keypoints_3d")
        else:
            print("[Motion Analyzer] ERROR: No 3D keypoint data available!")
            return ({}, {}, torch.zeros(1, 64, 64, 3), "Error: No keypoint data")
        
        # Get image size
        image_size = (1920, 1080)  # Default
        if images is not None:
            _, H, W, _ = images.shape
            image_size = (W, H)
            print(f"[{get_timestamp()}] [Motion Analyzer] Image size: {W}x{H}")
        
        # ===== HEIGHT ESTIMATION =====
        ref_frame = min(reference_frame, num_frames - 1)
        ref_vertices = to_numpy(vertices_list[ref_frame])
        
        # Get reference keypoints for height estimation
        if kp_source == "keypoints_3d":
            ref_keypoints = to_numpy(keypoints_3d_list[ref_frame])
        else:
            ref_keypoints = to_numpy(joint_coords_list[ref_frame])
        
        # Handle shape
        if ref_keypoints is not None and ref_keypoints.ndim == 3:
            ref_keypoints = ref_keypoints.squeeze(0)
        
        # Estimate height from mesh and keypoints
        mesh_height_info = estimate_height_from_mesh(ref_vertices)
        kp_height_info = estimate_height_from_keypoints(ref_keypoints, mode_str if kp_source == "keypoints_3d" else "full")
        
        # Determine actual height
        if subject_height_m > 0:
            actual_height = subject_height_m
            height_source = "user_input"
            print(f"[{get_timestamp()}] [Motion Analyzer] Using user-specified height: {actual_height:.2f}m")
        else:
            actual_height = default_height_m
            height_source = "auto_estimate"
            print(f"[{get_timestamp()}] [Motion Analyzer] Using default height: {actual_height:.2f}m")
        
        # Calculate scale factor
        estimated_height = kp_height_info["estimated_height"]
        if estimated_height > 0:
            scale_factor = actual_height / estimated_height
        else:
            scale_factor = 1.0
        
        scale_info = {
            "mesh_height": mesh_height_info["mesh_height"],
            "estimated_height": estimated_height,
            "actual_height_m": actual_height,
            "scale_factor": scale_factor,
            "leg_length": kp_height_info["leg_length"],
            "torso_head_length": kp_height_info["torso_head_length"],
            "height_source": height_source,
            "reference_frame": ref_frame,
            "skeleton_mode": skeleton_mode,
            "keypoint_source": kp_source,
        }
        
        print(f"[{get_timestamp()}] [Motion Analyzer] Mesh height: {mesh_height_info['mesh_height']:.3f} units")
        print(f"[{get_timestamp()}] [Motion Analyzer] Estimated height (from joints): {estimated_height:.3f} units")
        print(f"[{get_timestamp()}] [Motion Analyzer] Scale factor: {scale_factor:.3f}")
        print(f"[{get_timestamp()}] [Motion Analyzer] Leg length: {kp_height_info['leg_length']:.3f} units")
        print(f"[{get_timestamp()}] [Motion Analyzer] Torso+head: {kp_height_info['torso_head_length']:.3f} units")
        
        # ===== PER-FRAME ANALYSIS =====
        # ===== JOINT INDICES =====
        # debug10: Use unified BodyJoints indices for all data sources
        # All SAM3DBody outputs (joint_coords, pred_keypoints_3d) use same indices:
        # - Index 0 = body_world (global trajectory)
        # - Index 1 = pelvis (anatomical root)
        # - Indices 5, 8, 11 = head, L_ankle, R_ankle
        
        # Use unified indices for both 2D and 3D
        pelvis_idx = BodyJoints.PELVIS           # 1
        head_idx = BodyJoints.HEAD               # 5
        left_ankle_idx = BodyJoints.L_ANKLE      # 8
        right_ankle_idx = BodyJoints.R_ANKLE     # 11
        
        print(f"[{get_timestamp()}] [Motion Analyzer] debug10: Using unified BodyJoints indices")
        print(f"[{get_timestamp()}] [Motion Analyzer] Indices: pelvis={pelvis_idx}, head={head_idx}, L_ankle={left_ankle_idx}, R_ankle={right_ankle_idx}")
        
        # Track body_world (global trajectory) if using joint_coords
        track_body_world = (kp_source == "joint_coords")
        if track_body_world:
            print(f"[{get_timestamp()}] [Motion Analyzer] Tracking body_world (idx 0) for global trajectory")
        
        subject_motion = {
            "pelvis_2d": [],
            "pelvis_3d": [],
            "body_world_3d": [],  # Global trajectory from joint_coords[0]
            "joints_2d": [],
            "joints_3d": [],
            "velocity_2d": [],
            "velocity_3d": [],
            "apparent_height": [],
            "depth_estimate": [],
            "foot_contact": [],
            "camera_t": [],
            "focal_length": [],
            "image_size": image_size,
            "num_frames": num_frames,
            "scale_factor": scale_factor,
            "skeleton_mode": skeleton_mode,
            "keypoint_source": kp_source,
        }
        
        for i in range(num_frames):
            # Get frame data
            vertices = to_numpy(vertices_list[i])
            camera_t = to_numpy(camera_t_list[i]) if i < len(camera_t_list) else np.array([0, 0, 5])
            focal = focal_length_list[i] if i < len(focal_length_list) else 1000.0
            
            if isinstance(focal, torch.Tensor):
                focal = focal.cpu().item()
            if camera_t is None:
                camera_t = np.array([0, 0, 5])
            if len(camera_t.shape) > 1:
                camera_t = camera_t.flatten()[:3]
            
            subject_motion["camera_t"].append(camera_t.copy())
            subject_motion["focal_length"].append(float(focal))
            
            # Get 3D keypoints for analysis (height estimation, foot contact, etc.)
            if kp_source == "keypoints_3d":
                keypoints_3d = to_numpy(keypoints_3d_list[i]) if i < len(keypoints_3d_list) else None
            else:
                keypoints_3d = to_numpy(joint_coords_list[i]) if i < len(joint_coords_list) else None
            
            if keypoints_3d is None:
                print(f"[{get_timestamp()}] [Motion Analyzer] Warning: No keypoints for frame {i}")
                keypoints_3d = np.zeros((18 if kp_source == "keypoints_3d" else 127, 3))
            
            # Handle shape
            if keypoints_3d.ndim == 3:
                keypoints_3d = keypoints_3d.squeeze(0)
            
            # ===== GET 2D JOINTS FOR VISUALIZATION =====
            # debug10 fix: joint_coords is in LOCAL body space, NOT world space!
            # For 2D overlay, we must use either:
            # 1. pred_keypoints_2d directly (already in pixel coords) - BEST
            # 2. Project pred_keypoints_3d (18-joint world coords) - FALLBACK
            # Do NOT project joint_coords - it's local/relative coordinates!
            
            if has_kp_2d and i < len(keypoints_2d_list) and keypoints_2d_list[i] is not None:
                # BEST: Use pred_keypoints_2d directly - already in pixel coordinates
                keypoints_2d = to_numpy(keypoints_2d_list[i])
                if keypoints_2d.ndim == 3:
                    keypoints_2d = keypoints_2d.squeeze(0)
                # Take only x,y (might have confidence as 3rd column)
                if keypoints_2d.shape[1] >= 2:
                    joints_2d = keypoints_2d[:, :2]
                else:
                    joints_2d = keypoints_2d
                joints_2d_source = "pred_keypoints_2d (direct pixel coords)"
                
                if i == 0:
                    print(f"[{get_timestamp()}] [Motion Analyzer] debug10: Using pred_keypoints_2d directly (already in pixel coords)")
                    print(f"[{get_timestamp()}] [Motion Analyzer] pred_keypoints_2d shape: {joints_2d.shape}")
                    
            elif has_kp_3d and i < len(keypoints_3d_list) and keypoints_3d_list[i] is not None:
                # FALLBACK: Project pred_keypoints_3d (18-joint world space) to 2D
                kp3d_for_projection = to_numpy(keypoints_3d_list[i])
                if kp3d_for_projection.ndim == 3:
                    kp3d_for_projection = kp3d_for_projection.squeeze(0)
                    
                joints_2d = project_points_to_2d(
                    kp3d_for_projection, focal, camera_t, image_size[0], image_size[1]
                )
                joints_2d_source = f"projected from keypoints_3d (18-joint world coords)"
                
                if i == 0:
                    print(f"[{get_timestamp()}] [Motion Analyzer] debug10: Projecting keypoints_3d (world space) → 2D")
                    print(f"[{get_timestamp()}] [Motion Analyzer] Projection: focal={focal:.1f}px, cam_t=[{camera_t[0]:.3f}, {camera_t[1]:.3f}, {camera_t[2]:.3f}]")
            else:
                # LAST RESORT: Use center of image as fallback
                joints_2d = np.zeros((22, 2))
                joints_2d[:, 0] = image_size[0] / 2
                joints_2d[:, 1] = image_size[1] / 2
                joints_2d_source = "fallback (no 2D data available)"
                
                if i == 0:
                    print(f"[{get_timestamp()}] [Motion Analyzer] WARNING: No 2D keypoint data available!")
            
            subject_motion["joints_2d"].append(joints_2d)
            subject_motion["joints_3d"].append(keypoints_3d * scale_factor)
            
            # Detailed joint position logging for Frame 0
            if i == 0:
                print(f"[{get_timestamp()}] [Motion Analyzer] ===== JOINT POSITIONS (Frame 0) =====")
                print(f"[{get_timestamp()}] [Motion Analyzer] joints_2d shape: {joints_2d.shape}")
                print(f"[{get_timestamp()}] [Motion Analyzer] joints_2d source: {joints_2d_source}")
                print(f"[{get_timestamp()}] [Motion Analyzer] Image size: {image_size[0]}x{image_size[1]}")
                print(f"[{get_timestamp()}] [Motion Analyzer] --- Body Joints (unified indices 0-21) ---")
                for j_idx in range(min(22, len(joints_2d))):
                    j_name = BodyJoints.JOINT_NAMES.get(j_idx, f"joint{j_idx}")
                    j_x, j_y = joints_2d[j_idx][0], joints_2d[j_idx][1]
                    print(f"[{get_timestamp()}] [Motion Analyzer]   [{j_idx:2d}] {j_name:10s}: x={j_x:7.1f}, y={j_y:7.1f}")
                print(f"[{get_timestamp()}] [Motion Analyzer] ========================================")
            
            # Pelvis position - use unified indices for both 2D and 3D
            pelvis_3d = keypoints_3d[pelvis_idx] * scale_factor
            pelvis_2d = joints_2d[pelvis_idx] if pelvis_idx < len(joints_2d) else joints_2d[0]
            subject_motion["pelvis_3d"].append(pelvis_3d.copy())
            subject_motion["pelvis_2d"].append(pelvis_2d.copy())
            
            # Body world (global trajectory) - index 0
            if track_body_world and len(keypoints_3d) >= 22:
                body_world_3d = keypoints_3d[BodyJoints.BODY_WORLD] * scale_factor
                subject_motion["body_world_3d"].append(body_world_3d.copy())
                # Log body_world position for first frame
                if i == 0:
                    print(f"[{get_timestamp()}] [Motion Analyzer] body_world[0] (global trajectory): X={body_world_3d[0]:.3f}, Y={body_world_3d[1]:.3f}, Z={body_world_3d[2]:.3f}")
            else:
                subject_motion["body_world_3d"].append(None)
            
            # Apparent height (pixels) - use unified indices
            head_2d = joints_2d[head_idx] if head_idx < len(joints_2d) else joints_2d[0]
            left_ankle_2d = joints_2d[left_ankle_idx] if left_ankle_idx < len(joints_2d) else joints_2d[0]
            right_ankle_2d = joints_2d[right_ankle_idx] if right_ankle_idx < len(joints_2d) else joints_2d[0]
            feet_y = max(left_ankle_2d[1], right_ankle_2d[1])
            apparent_height = abs(feet_y - head_2d[1])
            subject_motion["apparent_height"].append(apparent_height)
            
            # Depth estimate
            depth_m = camera_t[2] * scale_factor
            subject_motion["depth_estimate"].append(depth_m)
            
            # Foot contact detection
            skeleton_mode_str = "simple" if kp_source == "keypoints_3d" else "full"
            foot_contact = detect_foot_contact(
                keypoints_3d, vertices, skeleton_mode_str, foot_contact_threshold
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
        print(f"[{get_timestamp()}] [Motion Analyzer] Frames: {num_frames}")
        print(f"[{get_timestamp()}] [Motion Analyzer] Avg 2D velocity: {avg_velocity_2d:.2f} px/frame")
        print(f"[{get_timestamp()}] [Motion Analyzer] Max 2D velocity: {max_velocity_2d:.2f} px/frame")
        print(f"[{get_timestamp()}] [Motion Analyzer] Grounded frames: {grounded_count} ({100*grounded_count/num_frames:.1f}%)")
        print(f"[{get_timestamp()}] [Motion Analyzer] Airborne frames: {airborne_count} ({100*airborne_count/num_frames:.1f}%)")
        print(f"[{get_timestamp()}] [Motion Analyzer] Depth range: {min(subject_motion['depth_estimate']):.2f}m - {max(subject_motion['depth_estimate']):.2f}m")
        
        # ===== DEBUG INFO STRING =====
        debug_info = (
            f"=== Motion Analysis Results ===\n"
            f"Frames: {num_frames}\n"
            f"Skeleton: {skeleton_mode} ({kp_source})\n"
            f"Subject height: {actual_height:.2f}m ({height_source})\n"
            f"Scale factor: {scale_factor:.3f}\n"
            f"Avg 2D velocity: {avg_velocity_2d:.2f} px/frame\n"
            f"Max 2D velocity: {max_velocity_2d:.2f} px/frame\n"
            f"Grounded: {grounded_count}/{num_frames} frames\n"
            f"Depth range: {min(subject_motion['depth_estimate']):.2f}m - {max(subject_motion['depth_estimate']):.2f}m\n"
        )
        
        # ===== DEBUG OVERLAY =====
        if show_debug and images is not None:
            print(f"[{get_timestamp()}] [Motion Analyzer] Generating debug overlay...")
            
            images_np = images.cpu().numpy() if isinstance(images, torch.Tensor) else images
            
            overlay = create_motion_debug_overlay(
                images_np,
                subject_motion,
                scale_info,
                skeleton_mode=skeleton_mode_str,
                arrow_scale=arrow_scale,
                show_skeleton=show_skeleton,
            )
            
            if overlay.dtype == np.uint8:
                overlay = overlay.astype(np.float32) / 255.0
            debug_overlay = torch.from_numpy(overlay).float()
        else:
            debug_overlay = torch.zeros(1, 64, 64, 3)
        
        print(f"[{get_timestamp()}] [Motion Analyzer] =============================================\n")
        
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
            f"Skeleton: {scale_info.get('skeleton_mode', 'N/A')}\n"
            f"Keypoint source: {scale_info.get('keypoint_source', 'N/A')}\n"
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
