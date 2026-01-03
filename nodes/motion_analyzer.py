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
VERSION = "0.5.0-debug19"

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
# Joint Index Definitions - MHR (70-joint) and SMPLH (127-joint) have DIFFERENT layouts!
# debug12: Separate classes for each format with correct indices
# ============================================================================

class MHRJoints:
    """MHR 70-joint format indices (used by pred_keypoints_2d).
    
    This is a detection-based format where joints are ordered differently
    from the SMPLH rigging format.
    """
    # Key body joints
    HEAD = 0              # Head/nose
    NECK = 1              # Neck
    R_SHOULDER = 2        # Right shoulder
    R_ELBOW = 3           # Right elbow
    R_WRIST = 4           # Right wrist
    L_SHOULDER = 5        # Left shoulder
    L_ELBOW = 6           # Left elbow
    L_WRIST = 7           # Left wrist
    R_HIP = 8             # Right hip
    R_KNEE = 9            # Right knee
    R_ANKLE = 10          # Right ankle
    L_HIP = 11            # Left hip
    L_KNEE = 12           # Left knee
    L_ANKLE = 13          # Left ankle
    R_EYE = 14            # Right eye
    L_EYE = 15            # Left eye
    R_EAR = 16            # Right ear
    L_EAR = 17            # Left ear
    
    # Pelvis approximation (midpoint of hips, use R_HIP or L_HIP as proxy)
    PELVIS = 8            # Use R_HIP as pelvis proxy
    
    # Aliases
    LEFT_SHOULDER = L_SHOULDER
    LEFT_ELBOW = L_ELBOW
    LEFT_WRIST = L_WRIST
    LEFT_HIP = L_HIP
    LEFT_KNEE = L_KNEE
    LEFT_ANKLE = L_ANKLE
    RIGHT_SHOULDER = R_SHOULDER
    RIGHT_ELBOW = R_ELBOW
    RIGHT_WRIST = R_WRIST
    RIGHT_HIP = R_HIP
    RIGHT_KNEE = R_KNEE
    RIGHT_ANKLE = R_ANKLE
    
    NUM_JOINTS = 70
    NUM_BODY_JOINTS = 18  # First 18 are body joints
    
    # Joint names for labeling
    JOINT_NAMES = {
        0: "head", 1: "neck", 2: "R_shldr", 3: "R_elbow", 4: "R_wrist",
        5: "L_shldr", 6: "L_elbow", 7: "L_wrist",
        8: "R_hip", 9: "R_knee", 10: "R_ankle",
        11: "L_hip", 12: "L_knee", 13: "L_ankle",
        14: "R_eye", 15: "L_eye", 16: "R_ear", 17: "L_ear",
    }


class SMPLHJoints:
    """SMPLH 127-joint format indices (used by joint_coords).
    
    This is a rigging-based format following SMPL-H body model structure.
    Index 0 is body_world (global trajectory), anatomical joints start at 1.
    """
    # Global trajectory
    BODY_WORLD = 0        # Global movement (X, Y, Z translation)
    
    # Core body (1-21)
    PELVIS = 1            # Anatomical root
    L_HIP = 2             # Left hip
    R_HIP = 3             # Right hip
    SPINE_1 = 4           # Lower spine
    L_KNEE = 5            # Left knee
    R_KNEE = 6            # Right knee
    SPINE_2 = 7           # Mid spine
    L_ANKLE = 8           # Left ankle
    R_ANKLE = 9           # Right ankle
    SPINE_3 = 10          # Upper spine
    L_FOOT = 11           # Left foot
    R_FOOT = 12           # Right foot
    NECK = 13             # Neck
    L_COLLAR = 14         # Left collar
    R_COLLAR = 15         # Right collar
    HEAD = 16             # Head
    L_SHOULDER = 17       # Left shoulder
    R_SHOULDER = 18       # Right shoulder
    L_ELBOW = 19          # Left elbow
    R_ELBOW = 20          # Right elbow
    L_WRIST = 21          # Left wrist
    R_WRIST = 22          # Right wrist
    
    # Aliases
    LEFT_HIP = L_HIP
    LEFT_KNEE = L_KNEE
    LEFT_ANKLE = L_ANKLE
    LEFT_SHOULDER = L_SHOULDER
    LEFT_ELBOW = L_ELBOW
    LEFT_WRIST = L_WRIST
    RIGHT_HIP = R_HIP
    RIGHT_KNEE = R_KNEE
    RIGHT_ANKLE = R_ANKLE
    RIGHT_SHOULDER = R_SHOULDER
    RIGHT_ELBOW = R_ELBOW
    RIGHT_WRIST = R_WRIST
    
    NUM_JOINTS = 127
    NUM_BODY_JOINTS = 24  # First 24 are body joints (0-23)
    
    # Joint names for labeling
    JOINT_NAMES = {
        0: "body_world", 1: "pelvis", 2: "L_hip", 3: "R_hip",
        4: "spine1", 5: "L_knee", 6: "R_knee", 7: "spine2",
        8: "L_ankle", 9: "R_ankle", 10: "spine3", 11: "L_foot", 12: "R_foot",
        13: "neck", 14: "L_collar", 15: "R_collar", 16: "head",
        17: "L_shldr", 18: "R_shldr", 19: "L_elbow", 20: "R_elbow",
        21: "L_wrist", 22: "R_wrist",
    }


# Backward compatibility - BodyJoints defaults to SMPLH
class BodyJoints(SMPLHJoints):
    """Alias for SMPLHJoints (default joint format)."""
    pass


# SAM3DJoints alias for backward compatibility
SAM3DJoints = SMPLHJoints


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


def get_global_trajectory_point(
    mesh_sequence: dict,
    frame_index: int
) -> Optional[np.ndarray]:
    """
    Returns the Body World (Global Trajectory) coordinate for a given frame.
    
    This function extracts the global root position (index 0 in joint_coords)
    which represents the character's world-space position separate from
    local body pose.
    
    Args:
        mesh_sequence: SAM4D mesh sequence dict with params containing joint_coords
        frame_index: Frame index to extract trajectory from
        
    Returns:
        np.ndarray of shape (3,) with [X, Y, Z] world coordinates, or None if unavailable
        
    Note:
        - Must use 'joint_coords' (127 joints), not 'keypoints_3d'
        - Index 0 = body_world (global trajectory)
        - Index 1 = pelvis (anatomical root)
    """
    params = mesh_sequence.get("params", {})
    
    # Must use joint_coords (127-joint SMPLH format)
    joint_coords = params.get("joint_coords")
    
    if joint_coords is None:
        return None
    
    # Check bounds
    if frame_index < 0 or frame_index >= len(joint_coords):
        return None
    
    # Extract frame data
    current_frame_joints = joint_coords[frame_index]
    
    # Handle both tensor and numpy formats
    if hasattr(current_frame_joints, "cpu"):
        current_frame_joints = current_frame_joints.cpu().numpy()
    elif not isinstance(current_frame_joints, np.ndarray):
        current_frame_joints = np.array(current_frame_joints)
    
    # Return index 0 (body_world / global trajectory)
    # This is distinct from pelvis (index 1)
    if len(current_frame_joints) > 0:
        return current_frame_joints[0].copy()
    
    return None


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
    
    debug12: Uses DYNAMIC index mapping based on keypoint source.
    - MHR format (pred_keypoints_2d): HEAD=0, PELVIS=8, L_ANKLE=13, R_ANKLE=10
    - SMPLH format (joint_coords): HEAD=16, PELVIS=1, L_ANKLE=8, R_ANKLE=9
    
    Only individual joint dots are shown - no skeleton lines.
    """
    # Convert to uint8 if needed
    if images.dtype == np.float32 or images.dtype == np.float64:
        if images.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
        else:
            images = images.astype(np.uint8)
    
    output = images.copy()
    num_frames = len(images)
    
    # Get keypoint source to determine which index mapping to use
    kp_source = subject_motion.get("keypoint_source", "keypoints_3d")
    joints_2d_source = subject_motion.get("joints_2d_source", "unknown")
    
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
    
    # debug19: DYNAMIC INDEX MAPPING based on source
    # pred_keypoints_2d = MHR format (confirmed correct!)
    # joint_coords projected = SMPLH format (fallback)
    use_mhr_indices = "pred_keypoints_2d" in joints_2d_source or "MHR" in joints_2d_source
    
    if use_mhr_indices:
        # MHR 70-joint format (pred_keypoints_2d - confirmed correct!)
        idx_map = {
            "HEAD": MHRJoints.HEAD,              # 0
            "PELVIS": MHRJoints.PELVIS,          # 8 (R_HIP as proxy)
            "L_WRIST": MHRJoints.L_WRIST,        # 7
            "R_WRIST": MHRJoints.R_WRIST,        # 4
            "L_ANKLE": MHRJoints.L_ANKLE,        # 13
            "R_ANKLE": MHRJoints.R_ANKLE,        # 10
        }
        max_draw_index = MHRJoints.NUM_BODY_JOINTS  # 18
        joint_names = MHRJoints.JOINT_NAMES
        format_name = "MHR"
    else:
        # SMPLH 127-joint format (joint_coords projected - fallback)
        idx_map = {
            "HEAD": SMPLHJoints.HEAD,            # 16
            "PELVIS": SMPLHJoints.PELVIS,        # 1
            "L_WRIST": SMPLHJoints.L_WRIST,      # 21
            "R_WRIST": SMPLHJoints.R_WRIST,      # 22
            "L_ANKLE": SMPLHJoints.L_ANKLE,      # 8
            "R_ANKLE": SMPLHJoints.R_ANKLE,      # 9
        }
        max_draw_index = SMPLHJoints.NUM_BODY_JOINTS  # 24
        joint_names = SMPLHJoints.JOINT_NAMES
        format_name = "SMPLH"
    
    # Log which format we're using (once)
    print(f"[Motion Analyzer] debug19: Using {format_name} index mapping for overlay")
    print(f"[Motion Analyzer] debug19: HEAD={idx_map['HEAD']}, PELVIS={idx_map['PELVIS']}, L_ANKLE={idx_map['L_ANKLE']}, R_ANKLE={idx_map['R_ANKLE']}")
    
    # Special joint indices for coloring (using dynamic map)
    special_joints = {
        idx_map["PELVIS"]: (COLOR_PELVIS, 8),      # Green, large
        idx_map["HEAD"]: (COLOR_HEAD, 6),          # Yellow, medium
        idx_map["L_WRIST"]: (COLOR_HANDS, 5),      # Magenta
        idx_map["R_WRIST"]: (COLOR_HANDS, 5),      # Magenta
        idx_map["L_ANKLE"]: (COLOR_FEET, 5),       # Orange
        idx_map["R_ANKLE"]: (COLOR_FEET, 5),       # Orange
    }
    
    for i in range(num_frames):
        frame = output[i]
        
        # Get 2D joint positions for this frame
        joints_2d = subject_motion.get("joints_2d")
        if joints_2d is not None and i < len(joints_2d) and joints_2d[i] is not None:
            joints_2d_frame = np.array(joints_2d[i])
            
            # Draw joint dots with labels
            if show_skeleton:
                num_body_joints = min(max_draw_index, len(joints_2d_frame))
                for j in range(num_body_joints):
                    pt = joints_2d_frame[j]
                    if pt is not None and len(pt) >= 2:
                        x, y = int(pt[0]), int(pt[1])
                        # Skip invalid coordinates
                        if x < 0 or y < 0 or x >= frame.shape[1] or y >= frame.shape[0]:
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
                        
                        # Draw joint label (index)
                        if j in joint_names:
                            label = f"{j}"
                            cv2.putText(frame, label, (x + 5, y - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_LABEL, 1)
        
        # Draw pelvis tracking using dynamic index
        pelvis_idx = idx_map["PELVIS"]
        if joints_2d is not None and i < len(joints_2d) and joints_2d[i] is not None:
            joints_2d_frame = np.array(joints_2d[i])
            if pelvis_idx < len(joints_2d_frame):
                pelvis_pt = joints_2d_frame[pelvis_idx]
                px, py = int(pelvis_pt[0]), int(pelvis_pt[1])
                if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                    cv2.circle(frame, (px, py), 10, (0, 0, 0), 2)  # Black ring
        
        # Draw velocity arrow
        velocity_2d = subject_motion.get("velocity_2d")
        pelvis_2d = subject_motion.get("pelvis_2d")
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
    
    debug12 Changes:
    - DYNAMIC index mapping based on 2D source format
    - MHR format (pred_keypoints_2d): HEAD=0, PELVIS=8, L_ANKLE=13, R_ANKLE=10
    - SMPLH format (projected): HEAD=16, PELVIS=1, L_ANKLE=8, R_ANKLE=9
    - Uses pred_keypoints_2d directly when available (MHR format, pixel coords)
    - Fallback: project pred_keypoints_3d (SMPLH format) if 2D unavailable
    
    Data Sources:
    - pred_keypoints_2d: 70 joints, MHR format, already in pixel coordinates
    - pred_keypoints_3d: 18 joints, SMPLH format, world space coordinates
    - joint_coords: 127 joints, SMPLH format, LOCAL body space (NOT for 2D!)
    
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
    
    RETURN_TYPES = ("SUBJECT_MOTION", "SCALE_INFO", "IMAGE", "STRING", "MOTION_ANALYSIS")
    RETURN_NAMES = ("subject_motion", "scale_info", "debug_overlay", "debug_info", "motion_analysis")
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
        sequence_fps = mesh_sequence.get("fps", 30.0)  # Extract FPS for metadata
        
        # Get keypoint data
        keypoints_2d_list = params.get("keypoints_2d", [])
        keypoints_3d_list = params.get("keypoints_3d", [])
        joint_coords_list = params.get("joint_coords", [])  # 127-joint fallback
        camera_t_list = params.get("camera_t", [])
        focal_length_list = params.get("focal_length", [])
        
        num_frames = len(vertices_list)
        if num_frames == 0:
            print("[Motion Analyzer] ERROR: No frames in mesh sequence!")
            return ({}, {}, torch.zeros(1, 64, 64, 3), "Error: No frames", {})
        
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
            return ({}, {}, torch.zeros(1, 64, 64, 3), "Error: No keypoint data", {})
        
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
        # debug19: Use SMPLH indices since we PROJECT joint_coords to 2D
        # This matches the mesh renderer which uses joint_coords
        # Priority: joint_coords (SMPLH) > keypoints_3d (MHR fallback)
        
        # Determine which format will be used for 2D joints
        # debug19: CONFIRMED pred_keypoints_2d (MHR) is correct!
        # Use MHR format when pred_keypoints_2d is available
        use_mhr_for_2d = has_kp_2d  # pred_keypoints_2d = MHR format (confirmed correct)
        
        if use_mhr_for_2d:
            # MHR format indices (for pred_keypoints_2d - confirmed correct!)
            pelvis_idx_2d = MHRJoints.PELVIS       # 8 (R_HIP as proxy)
            head_idx_2d = MHRJoints.HEAD           # 0
            left_ankle_idx_2d = MHRJoints.L_ANKLE  # 13
            right_ankle_idx_2d = MHRJoints.R_ANKLE # 10
            joint_names_2d = MHRJoints.JOINT_NAMES
            format_2d = "MHR"
        else:
            # SMPLH format indices (fallback for projected joint_coords)
            pelvis_idx_2d = SMPLHJoints.PELVIS       # 1
            head_idx_2d = SMPLHJoints.HEAD           # 16
            left_ankle_idx_2d = SMPLHJoints.L_ANKLE  # 8
            right_ankle_idx_2d = SMPLHJoints.R_ANKLE # 9
            joint_names_2d = SMPLHJoints.JOINT_NAMES
            format_2d = "SMPLH"
        
        # 3D indices always use SMPLH format (for joint_coords or keypoints_3d)
        pelvis_idx_3d = SMPLHJoints.PELVIS
        
        print(f"[{get_timestamp()}] [Motion Analyzer] debug19: 2D format = {format_2d}")
        print(f"[{get_timestamp()}] [Motion Analyzer] 2D indices: pelvis={pelvis_idx_2d}, head={head_idx_2d}, L_ankle={left_ankle_idx_2d}, R_ankle={right_ankle_idx_2d}")
        
        # Track body_world (global trajectory) if using joint_coords
        track_body_world = (kp_source == "joint_coords")
        if track_body_world:
            print(f"[{get_timestamp()}] [Motion Analyzer] Tracking body_world (idx 0) for global trajectory")
        
        # Initialize subject_motion with joints_2d_source tracking
        joints_2d_source_str = ""  # Will be set in first frame
        
        subject_motion = {
            "pelvis_2d": [],
            "pelvis_3d": [],
            "body_world_3d": [],  # Global trajectory from joint_coords[0] (usually zeros)
            "camera_t_trajectory": [],  # Camera translation per frame (actual world trajectory)
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
            "joints_2d_source": "",  # Will be updated after first frame
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
            # debug19: CONFIRMED pred_keypoints_2d is CORRECT!
            # Joint Debug Overlay showed:
            # - BLUE (pred_keypoints_2d) aligns with athlete
            # - RED (joint_coords projected) is wrong
            # 
            # Use pred_keypoints_2d directly with MHR format (70 joints)
            
            if has_kp_2d and i < len(keypoints_2d_list) and keypoints_2d_list[i] is not None:
                # BEST: Use pred_keypoints_2d directly - confirmed correct!
                kp2d = to_numpy(keypoints_2d_list[i])
                if kp2d.ndim == 3:
                    kp2d = kp2d.squeeze(0)
                if kp2d.shape[1] >= 2:
                    joints_2d = kp2d[:, :2].copy()
                else:
                    joints_2d = kp2d.copy()
                
                joints_2d_source = "pred_keypoints_2d (MHR 70-joint) - DIRECT"
                format_2d = "MHR"  # Use MHR indices since pred_keypoints_2d is MHR format
                
                if i == 0:
                    print(f"[{get_timestamp()}] [Motion Analyzer] debug19: Using pred_keypoints_2d DIRECTLY (confirmed correct)")
                    print(f"[{get_timestamp()}] [Motion Analyzer] pred_keypoints_2d shape: {joints_2d.shape}")
                    print(f"[{get_timestamp()}] [Motion Analyzer] Head (MHR idx 0): ({joints_2d[0,0]:.1f}, {joints_2d[0,1]:.1f})")
                    if len(joints_2d) > 9:
                        print(f"[{get_timestamp()}] [Motion Analyzer] Pelvis area (MHR idx 9): ({joints_2d[9,0]:.1f}, {joints_2d[9,1]:.1f})")
                    subject_motion["joints_2d_source"] = joints_2d_source
            
            elif has_joint_coords and i < len(joint_coords_list) and joint_coords_list[i] is not None:
                # FALLBACK: Project joint_coords if pred_keypoints_2d not available
                jc = to_numpy(joint_coords_list[i])
                if jc.ndim == 3:
                    jc = jc.squeeze(0)
                
                joints_2d = project_points_to_2d(
                    jc, focal, camera_t, image_size[0], image_size[1]
                )
                joints_2d_source = "projected joint_coords (SMPLH 127-joint) - FALLBACK"
                format_2d = "SMPLH"
                
                if i == 0:
                    print(f"[{get_timestamp()}] [Motion Analyzer] debug19: FALLBACK to joint_coords projection")
                    subject_motion["joints_2d_source"] = joints_2d_source
            else:
                # LAST RESORT: Use center of image as fallback
                joints_2d = np.zeros((22, 2))
                joints_2d[:, 0] = image_size[0] / 2
                joints_2d[:, 1] = image_size[1] / 2
                joints_2d_source = "fallback (no data available)"
                format_2d = "MHR"
                
                if i == 0:
                    print(f"[{get_timestamp()}] [Motion Analyzer] WARNING: No 2D keypoint data!")
                    subject_motion["joints_2d_source"] = joints_2d_source
            
            subject_motion["joints_2d"].append(joints_2d)
            subject_motion["joints_3d"].append(keypoints_3d * scale_factor)
            
            # Detailed joint position logging for Frame 0
            if i == 0:
                print(f"[{get_timestamp()}] [Motion Analyzer] ===== JOINT POSITIONS (Frame 0) =====")
                print(f"[{get_timestamp()}] [Motion Analyzer] joints_2d shape: {joints_2d.shape}")
                print(f"[{get_timestamp()}] [Motion Analyzer] joints_2d source: {joints_2d_source}")
                print(f"[{get_timestamp()}] [Motion Analyzer] 2D format: {format_2d}")
                print(f"[{get_timestamp()}] [Motion Analyzer] Image size: {image_size[0]}x{image_size[1]}")
                print(f"[{get_timestamp()}] [Motion Analyzer] --- Body Joints ({format_2d} indices) ---")
                num_to_print = min(18 if format_2d == "MHR" else 24, len(joints_2d))
                for j_idx in range(num_to_print):
                    j_name = joint_names_2d.get(j_idx, f"joint{j_idx}")
                    j_x, j_y = joints_2d[j_idx][0], joints_2d[j_idx][1]
                    print(f"[{get_timestamp()}] [Motion Analyzer]   [{j_idx:2d}] {j_name:10s}: x={j_x:7.1f}, y={j_y:7.1f}")
                print(f"[{get_timestamp()}] [Motion Analyzer] ========================================")
                
                # ===== PROJECTION COMPARISON (Frame 0) =====
                # Compare ground truth (pred_keypoints_2d) vs different projection formulas
                # This helps identify the correct projection for pred_keypoints_3d
                if has_kp_2d and has_kp_3d:
                    gt_2d = to_numpy(keypoints_2d_list[0])
                    kp_3d = to_numpy(keypoints_3d_list[0])
                    if gt_2d.ndim == 3:
                        gt_2d = gt_2d.squeeze(0)
                    if kp_3d.ndim == 3:
                        kp_3d = kp_3d.squeeze(0)
                    if gt_2d.shape[1] >= 2:
                        gt_2d = gt_2d[:, :2]
                    
                    cx = image_size[0] / 2.0
                    cy = image_size[1] / 2.0
                    tx, ty, tz = camera_t[0], camera_t[1], camera_t[2]
                    
                    # Test multiple projection formulas
                    def project_formula_A(pts):
                        """Current formula: translate + 180° X rotation"""
                        X = pts[:, 0] + tx
                        Y = -(pts[:, 1] + ty)  # Negate Y
                        Z = -(pts[:, 2] + tz)  # Negate Z
                        Z = np.where(np.abs(Z) < 0.1, 0.1, Z)
                        return np.stack([focal * X / Z + cx, focal * Y / Z + cy], axis=1)
                    
                    def project_formula_B(pts):
                        """Simple perspective: translate only, no rotation"""
                        X = pts[:, 0] + tx
                        Y = pts[:, 1] + ty
                        Z = pts[:, 2] + tz
                        Z = np.where(np.abs(Z) < 0.1, 0.1, Z)
                        return np.stack([focal * X / Z + cx, focal * Y / Z + cy], axis=1)
                    
                    def project_formula_C(pts):
                        """Weak perspective (orthographic-like)"""
                        s = focal / tz  # Scale factor based on depth
                        X = pts[:, 0] * s + cx + tx * s
                        Y = pts[:, 1] * s + cy + ty * s
                        return np.stack([X, Y], axis=1)
                    
                    def project_formula_D(pts):
                        """MHR-style: weak perspective with pred_cam_t as [s, tx, ty]"""
                        # In MHR, pred_cam_t might be [scale, tx_2d, ty_2d]
                        # where tx_2d, ty_2d are already in pixel space
                        s = tz  # Assume tz is actually scale
                        X = pts[:, 0] * s * focal + cx + tx * focal
                        Y = pts[:, 1] * s * focal + cy + ty * focal
                        return np.stack([X, Y], axis=1)
                    
                    print(f"[DEBUG] ========== PROJECTION COMPARISON (Frame 0) ==========")
                    print(f"[DEBUG] Image size: {image_size[0]}x{image_size[1]}, Focal: {focal:.1f}px")
                    print(f"[DEBUG] pred_cam_t: tx={tx:.4f}, ty={ty:.4f}, tz={tz:.4f}")
                    print(f"[DEBUG] Ground truth: pred_keypoints_2d ({gt_2d.shape[0]} joints)")
                    print(f"[DEBUG] 3D source: pred_keypoints_3d ({kp_3d.shape[0]} joints)")
                    
                    # Test each formula
                    formulas = [
                        ("A: translate+180°rot", project_formula_A),
                        ("B: translate only", project_formula_B),
                        ("C: weak perspective", project_formula_C),
                        ("D: MHR-style", project_formula_D),
                    ]
                    
                    best_formula = None
                    best_error = float('inf')
                    
                    print(f"[DEBUG]")
                    print(f"[DEBUG] Testing projection formulas:")
                    
                    for name, func in formulas:
                        try:
                            proj = func(kp_3d)
                            errors = np.sqrt(np.sum((proj[:18] - gt_2d[:18])**2, axis=1))
                            avg_error = np.mean(errors)
                            print(f"[DEBUG]   {name}: avg error = {avg_error:.1f}px")
                            if avg_error < best_error:
                                best_error = avg_error
                                best_formula = name
                        except Exception as e:
                            print(f"[DEBUG]   {name}: ERROR - {e}")
                    
                    print(f"[DEBUG]")
                    print(f"[DEBUG] Best formula: {best_formula} (error: {best_error:.1f}px)")
                    
                    # Show detailed comparison for best formula (or current if all fail)
                    projected_2d = project_points_to_2d(
                        kp_3d, focal, camera_t, image_size[0], image_size[1]
                    )
                    
                    print(f"[DEBUG]")
                    print(f"[DEBUG] Joint | Ground Truth (x,y) | Current Proj (x,y) | Diff (dx, dy)")
                    print(f"[DEBUG] ------|-------------------|-------------------|---------------")
                    
                    total_dx, total_dy = 0.0, 0.0
                    num_compared = min(len(gt_2d), len(projected_2d), 10)  # First 10 joints
                    
                    for j in range(num_compared):
                        gt_x, gt_y = gt_2d[j][0], gt_2d[j][1]
                        proj_x, proj_y = projected_2d[j][0], projected_2d[j][1]
                        dx = proj_x - gt_x
                        dy = proj_y - gt_y
                        total_dx += dx
                        total_dy += dy
                        print(f"[DEBUG]   {j:3d}  | ({gt_x:7.1f},{gt_y:7.1f}) | ({proj_x:7.1f},{proj_y:7.1f}) | ({dx:+6.1f}, {dy:+6.1f})")
                    
                    print(f"[DEBUG] ------|-------------------|-------------------|---------------")
                    avg_dx = total_dx / num_compared if num_compared > 0 else 0
                    avg_dy = total_dy / num_compared if num_compared > 0 else 0
                    print(f"[DEBUG] AVERAGE OFFSET: dx={avg_dx:+.1f}px, dy={avg_dy:+.1f}px")
                    print(f"[DEBUG] ==========================================================")
            
            # Pelvis position - use 2D indices for 2D, 3D indices for 3D
            pelvis_3d = keypoints_3d[pelvis_idx_3d] * scale_factor if pelvis_idx_3d < len(keypoints_3d) else np.zeros(3)
            pelvis_2d = joints_2d[pelvis_idx_2d] if pelvis_idx_2d < len(joints_2d) else joints_2d[0]
            subject_motion["pelvis_3d"].append(pelvis_3d.copy())
            subject_motion["pelvis_2d"].append(pelvis_2d.copy())
            
            # Body world (global trajectory)
            # NOTE: In SMPLH, joint_coords[0] (body_world) is typically zeros because
            # the body is at origin. The actual world trajectory comes from pred_cam_t
            # (camera translation), which represents camera position relative to body.
            # Body world position = -pred_cam_t (inverse of camera offset)
            if track_body_world:
                # Track both body_world (usually zeros) and camera_t (actual trajectory)
                body_world_3d = get_global_trajectory_point(mesh_sequence, i)
                if body_world_3d is not None:
                    body_world_3d = body_world_3d * scale_factor
                else:
                    body_world_3d = np.zeros(3)
                
                subject_motion["body_world_3d"].append(body_world_3d.copy())
                
                # Also track camera_t as the effective world trajectory
                # The body's world position ≈ -camera_t (body at origin, camera moves)
                if i < len(camera_t_list) and camera_t_list[i] is not None:
                    cam_t_frame = to_numpy(camera_t_list[i]).flatten()
                    subject_motion["camera_t_trajectory"].append(cam_t_frame.copy())
                    
                    if i == 0:
                        # Log both values for debugging
                        print(f"[{get_timestamp()}] [Motion Analyzer] body_world[0] (joint_coords[0]): X={body_world_3d[0]:.3f}, Y={body_world_3d[1]:.3f}, Z={body_world_3d[2]:.3f}")
                        if np.allclose(body_world_3d, 0):
                            print(f"[{get_timestamp()}] [Motion Analyzer] NOTE: body_world=0 is normal (body at origin)")
                        print(f"[{get_timestamp()}] [Motion Analyzer] pred_cam_t[0] (camera trajectory): X={cam_t_frame[0]:.3f}, Y={cam_t_frame[1]:.3f}, Z={cam_t_frame[2]:.3f}")
                else:
                    subject_motion["camera_t_trajectory"].append(None)
                    if i == 0:
                        print(f"[{get_timestamp()}] [Motion Analyzer] body_world[0]: X={body_world_3d[0]:.3f}, Y={body_world_3d[1]:.3f}, Z={body_world_3d[2]:.3f}")
            else:
                subject_motion["body_world_3d"].append(None)
                subject_motion["camera_t_trajectory"].append(None)
            
            # Apparent height (pixels) - use 2D indices
            head_2d = joints_2d[head_idx_2d] if head_idx_2d < len(joints_2d) else joints_2d[0]
            left_ankle_2d = joints_2d[left_ankle_idx_2d] if left_ankle_idx_2d < len(joints_2d) else joints_2d[0]
            right_ankle_2d = joints_2d[right_ankle_idx_2d] if right_ankle_idx_2d < len(joints_2d) else joints_2d[0]
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
        
        # ===== BUILD MOTION_ANALYSIS FOR FBX EXPORT =====
        # This dict provides FBX-compatible metadata with correct data types
        # format_2d is defined earlier in the function ("MHR" or "SMPLH")
        motion_analysis = {
            # Motion stats
            "skeleton_mode": scale_info.get("skeleton_mode", "unknown"),
            "keypoint_source": scale_info.get("keypoint_source", "unknown"),
            "joint_indices_format": format_2d,
            "num_frames": int(num_frames),
            "fps": float(sequence_fps),  # From mesh_sequence
            "avg_velocity_2d": float(np.mean([np.linalg.norm(v) for v in subject_motion.get("velocity_2d", [[0,0]])])) if subject_motion.get("velocity_2d") else 0.0,
            "max_velocity_2d": float(np.max([np.linalg.norm(v) for v in subject_motion.get("velocity_2d", [[0,0]])])) if subject_motion.get("velocity_2d") else 0.0,
            "grounded_frames": int(sum(1 for f in subject_motion.get("foot_contact", []) if f in ["both", "left", "right"])),
            "airborne_frames": int(sum(1 for f in subject_motion.get("foot_contact", []) if f == "none")),
            "depth_min": float(min(subject_motion.get("depth_estimate", [0]))) if subject_motion.get("depth_estimate") else 0.0,
            "depth_max": float(max(subject_motion.get("depth_estimate", [0]))) if subject_motion.get("depth_estimate") else 0.0,
            # Scale info
            "actual_height_m": float(scale_info.get("actual_height_m", 0)),
            "mesh_height_units": float(scale_info.get("mesh_height_units", 0)),
            "estimated_height_units": float(scale_info.get("estimated_height_units", 0)),
            "scale_factor": float(scale_info.get("scale_factor", 1)),
            "leg_length_units": float(scale_info.get("leg_length_units", 0)),
            "torso_head_units": float(scale_info.get("torso_head_units", 0)),
            "height_source": str(scale_info.get("source", "unknown")),
        }
        
        return (subject_motion, scale_info, debug_overlay, debug_info, motion_analysis)


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
