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

Joint Index Reference:
- SAM3DBody outputs 18-joint keypoints (pred_keypoints_2d/3d)
- Also outputs 127-joint full skeleton (pred_joint_coords)
- This module uses 18-joint by default, 127-joint for full skeleton mode
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Optional, Tuple, Any


# ============================================================================
# SAM3DBody 18-Joint Skeleton (Simple Mode)
# This is what pred_keypoints_2d and pred_keypoints_3d use
# ============================================================================
class SAM3DJoints:
    """SAM3DBody 18-joint skeleton indices (pred_keypoints_2d/3d)."""
    PELVIS = 0
    SPINE1 = 1
    SPINE2 = 2
    SPINE3 = 3
    NECK = 4
    HEAD = 5
    LEFT_SHOULDER = 6
    LEFT_ELBOW = 7
    LEFT_WRIST = 8
    RIGHT_SHOULDER = 9
    RIGHT_ELBOW = 10
    RIGHT_WRIST = 11
    LEFT_HIP = 12
    LEFT_KNEE = 13
    LEFT_ANKLE = 14
    RIGHT_HIP = 15
    RIGHT_KNEE = 16
    RIGHT_ANKLE = 17
    
    NUM_JOINTS = 18
    
    # Skeleton connections for visualization
    CONNECTIONS = [
        # Spine to head
        (PELVIS, SPINE1), (SPINE1, SPINE2), (SPINE2, SPINE3), 
        (SPINE3, NECK), (NECK, HEAD),
        # Left arm
        (SPINE3, LEFT_SHOULDER), (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
        # Right arm
        (SPINE3, RIGHT_SHOULDER), (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
        # Left leg
        (PELVIS, LEFT_HIP), (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
        # Right leg
        (PELVIS, RIGHT_HIP), (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    ]


# ============================================================================
# SMPL-H 127-Joint Skeleton (Full Mode) - For future MHR integration
# This is what pred_joint_coords uses
# ============================================================================
class SMPLHJoints:
    """SMPL-H 127-joint skeleton indices (pred_joint_coords)."""
    # Main body joints (first 22)
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
    # Joints 22-126 are hand joints
    
    NUM_BODY_JOINTS = 22
    NUM_TOTAL_JOINTS = 127
    
    # Skeleton connections for body visualization
    CONNECTIONS = [
        # Spine
        (PELVIS, SPINE1), (SPINE1, SPINE2), (SPINE2, SPINE3), (SPINE3, NECK), (NECK, HEAD),
        # Left leg
        (PELVIS, LEFT_HIP), (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE), (LEFT_ANKLE, LEFT_FOOT),
        # Right leg
        (PELVIS, RIGHT_HIP), (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE), (RIGHT_ANKLE, RIGHT_FOOT),
        # Left arm
        (SPINE3, LEFT_COLLAR), (LEFT_COLLAR, LEFT_SHOULDER), (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
        # Right arm
        (SPINE3, RIGHT_COLLAR), (RIGHT_COLLAR, RIGHT_SHOULDER), (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    ]


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
    
    SAM3DBody's coordinates are already image-aligned:
    - Positive Y = UP in image space (lower Y pixel value)
    - NO Y negation needed
    
    Args:
        points_3d: (N, 3) array of 3D points
        focal_length: focal length in pixels
        cam_t: camera translation [tx, ty, tz]
        image_width, image_height: image dimensions
        
    Returns:
        points_2d: (N, 2) array of 2D points
    """
    points_3d = np.array(points_3d)
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
    
    # SAM3DBody camera model:
    # Points in camera space = points_3d + cam_t
    # NO Y negation - coordinates are already image-aligned
    X = points_3d[:, 0] + tx
    Y = points_3d[:, 1] + ty
    Z = points_3d[:, 2] + tz
    
    # Avoid division by zero
    Z = np.where(np.abs(Z) < 1e-6, 1e-6, Z)
    
    # Perspective projection
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
    
    Draws body joints based on skeleton mode:
    - Simple Skeleton (keypoints_3d): 18 joints, indices 0-17
    - Full Skeleton (joint_coords): 22 body joints from 127-joint format
    
    Uses pred_keypoints_2d directly for accurate positioning.
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
    COLOR_PELVIS = (0, 255, 0)       # Green - pelvis
    COLOR_HEAD = (0, 255, 255)       # Yellow - head
    COLOR_HANDS = (255, 0, 255)      # Magenta - wrists
    COLOR_FEET = (255, 128, 0)       # Orange - ankles/feet
    COLOR_JOINTS = (255, 128, 128)   # Light blue - other joints
    COLOR_SKELETON = (0, 200, 200)   # Cyan - skeleton lines
    COLOR_VELOCITY = (0, 255, 255)   # Yellow
    COLOR_GROUNDED = (0, 255, 0)     # Green
    COLOR_AIRBORNE = (0, 0, 255)     # Red
    COLOR_PARTIAL = (0, 255, 255)    # Yellow
    COLOR_TEXT = (255, 255, 255)     # White
    
    # Determine joint indices and connections based on 2D keypoint format
    # Note: pred_keypoints_2d always uses 70-joint MHR format (body joints 0-17)
    # joint_coords uses 127-joint SMPLH format (body joints 0-21)
    joints_2d_format = subject_motion.get("joints_2d_format", "keypoints_2d")
    
    if joints_2d_format == "joint_coords":
        # 127-joint SMPLH format - use first 22 body joints
        BODY_JOINTS = list(range(22))
        SKELETON_CONNECTIONS = SMPLHJoints.CONNECTIONS
        SPECIAL_JOINTS = {
            SMPLHJoints.PELVIS: (COLOR_PELVIS, 8),
            SMPLHJoints.HEAD: (COLOR_HEAD, 6),
            SMPLHJoints.LEFT_WRIST: (COLOR_HANDS, 5),
            SMPLHJoints.RIGHT_WRIST: (COLOR_HANDS, 5),
            SMPLHJoints.LEFT_ANKLE: (COLOR_FEET, 5),
            SMPLHJoints.RIGHT_ANKLE: (COLOR_FEET, 5),
            SMPLHJoints.LEFT_FOOT: (COLOR_FEET, 4),
            SMPLHJoints.RIGHT_FOOT: (COLOR_FEET, 4),
        }
    else:
        # 70-joint MHR format (from pred_keypoints_2d) - body joints 0-17
        BODY_JOINTS = list(range(18))
        SKELETON_CONNECTIONS = SAM3DJoints.CONNECTIONS
        SPECIAL_JOINTS = {
            SAM3DJoints.PELVIS: (COLOR_PELVIS, 8),
            SAM3DJoints.HEAD: (COLOR_HEAD, 6),
            SAM3DJoints.LEFT_WRIST: (COLOR_HANDS, 5),
            SAM3DJoints.RIGHT_WRIST: (COLOR_HANDS, 5),
            SAM3DJoints.LEFT_ANKLE: (COLOR_FEET, 5),
            SAM3DJoints.RIGHT_ANKLE: (COLOR_FEET, 5),
        }
    
    for i in range(num_frames):
        frame = output[i]
        h, w = frame.shape[:2]
        
        # Get 2D joint positions for this frame
        joints_2d_list = subject_motion.get("joints_2d")
        if joints_2d_list is not None and i < len(joints_2d_list) and joints_2d_list[i] is not None:
            joints_2d_frame = np.array(joints_2d_list[i])
            
            if show_skeleton and len(joints_2d_frame) >= len(BODY_JOINTS):
                # Draw skeleton connections first (behind joints)
                for (j1, j2) in SKELETON_CONNECTIONS:
                    if j1 < len(joints_2d_frame) and j2 < len(joints_2d_frame):
                        pt1 = joints_2d_frame[j1]
                        pt2 = joints_2d_frame[j2]
                        x1, y1 = int(pt1[0]), int(pt1[1])
                        x2, y2 = int(pt2[0]), int(pt2[1])
                        if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                            cv2.line(frame, (x1, y1), (x2, y2), COLOR_SKELETON, 2)
                
                # Draw body joints only
                for j in BODY_JOINTS:
                    if j < len(joints_2d_frame):
                        pt = joints_2d_frame[j]
                        x, y = int(pt[0]), int(pt[1])
                        
                        # Skip joints outside image bounds
                        if x < 0 or y < 0 or x >= w or y >= h:
                            continue
                        
                        # Use special color/size for key joints
                        if j in SPECIAL_JOINTS:
                            color, radius = SPECIAL_JOINTS[j]
                        else:
                            color = COLOR_JOINTS
                            radius = 4
                        
                        cv2.circle(frame, (x, y), radius, color, -1)
                        cv2.circle(frame, (x, y), radius, (0, 0, 0), 1)  # Black outline
        
        # Draw velocity arrow from pelvis
        velocity_2d = subject_motion.get("velocity_2d")
        pelvis_2d_list = subject_motion.get("pelvis_2d")
        if velocity_2d is not None and i > 0 and (i-1) < len(velocity_2d):
            vx, vy = velocity_2d[i-1]
            if pelvis_2d_list is not None and i < len(pelvis_2d_list):
                px, py = pelvis_2d_list[i]
                # Scale velocity for visibility
                end_x = int(px + vx * arrow_scale)
                end_y = int(py + vy * arrow_scale)
                cv2.arrowedLine(frame, (int(px), int(py)), (end_x, end_y),
                               COLOR_VELOCITY, 2, tipLength=0.3)
        
        # Draw info text
        y_offset = 30
        line_height = 25
        
        # Skeleton info
        if joints_2d_format == "joint_coords":
            skeleton_name = "Full (SMPLH 22)"
        else:
            skeleton_name = "Simple (MHR 18)"
        cv2.putText(frame, f"Skeleton: {skeleton_name}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
        y_offset += line_height
        
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
    
    Uses pred_keypoints_2d directly when available for accurate 2D positions.
    Falls back to projection from pred_keypoints_3d if 2D not available.
    
    Skeleton Modes:
    - "Simple Skeleton" (default): Uses 18-joint keypoints
    - "Full Skeleton": Uses 127-joint SMPL-H skeleton (for future MHR integration)
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
        print("\n[Motion Analyzer] ========== SUBJECT MOTION ANALYSIS ==========")
        
        # Determine skeleton mode
        use_simple = skeleton_mode == "Simple Skeleton"
        mode_str = "simple" if use_simple else "full"
        print(f"[Motion Analyzer] Skeleton mode: {skeleton_mode}")
        
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
        
        print(f"[Motion Analyzer] Processing {num_frames} frames...")
        
        # Check what keypoint data is available
        has_kp_2d = len(keypoints_2d_list) > 0 and keypoints_2d_list[0] is not None
        has_kp_3d = len(keypoints_3d_list) > 0 and keypoints_3d_list[0] is not None
        has_joint_coords = len(joint_coords_list) > 0 and joint_coords_list[0] is not None
        
        print(f"[Motion Analyzer] Data available: keypoints_2d={has_kp_2d}, keypoints_3d={has_kp_3d}, joint_coords={has_joint_coords}")
        
        # Debug: Print more details about available data
        print(f"[Motion Analyzer] ===== KEYPOINT DATA DEBUG =====")
        print(f"[Motion Analyzer] keypoints_2d_list length: {len(keypoints_2d_list)}")
        if len(keypoints_2d_list) > 0 and keypoints_2d_list[0] is not None:
            kp2d_sample = to_numpy(keypoints_2d_list[0])
            print(f"[Motion Analyzer] keypoints_2d[0] shape: {kp2d_sample.shape if hasattr(kp2d_sample, 'shape') else 'no shape'}")
            if kp2d_sample is not None and len(kp2d_sample) > 0:
                print(f"[Motion Analyzer] keypoints_2d[0] first 3 points: {kp2d_sample[:3] if len(kp2d_sample) >= 3 else kp2d_sample}")
        else:
            print(f"[Motion Analyzer] keypoints_2d[0] is None or empty")
        
        print(f"[Motion Analyzer] keypoints_3d_list length: {len(keypoints_3d_list)}")
        if len(keypoints_3d_list) > 0 and keypoints_3d_list[0] is not None:
            kp3d_sample = to_numpy(keypoints_3d_list[0])
            print(f"[Motion Analyzer] keypoints_3d[0] shape: {kp3d_sample.shape if hasattr(kp3d_sample, 'shape') else 'no shape'}")
        
        print(f"[Motion Analyzer] joint_coords_list length: {len(joint_coords_list)}")
        if len(joint_coords_list) > 0 and joint_coords_list[0] is not None:
            jc_sample = to_numpy(joint_coords_list[0])
            print(f"[Motion Analyzer] joint_coords[0] shape: {jc_sample.shape if hasattr(jc_sample, 'shape') else 'no shape'}")
        print(f"[Motion Analyzer] =================================")
        
        # Decide which 3D keypoints to use
        if use_simple and has_kp_3d:
            kp_source = "keypoints_3d"
            print(f"[Motion Analyzer] Using 18-joint keypoints_3d for analysis")
        elif has_joint_coords:
            kp_source = "joint_coords"
            print(f"[Motion Analyzer] Using 127-joint joint_coords for analysis")
        elif has_kp_3d:
            kp_source = "keypoints_3d"
            print(f"[Motion Analyzer] Fallback to 18-joint keypoints_3d")
        else:
            print("[Motion Analyzer] ERROR: No 3D keypoint data available!")
            return ({}, {}, torch.zeros(1, 64, 64, 3), "Error: No keypoint data")
        
        # Get image size
        image_size = (1920, 1080)  # Default
        if images is not None:
            _, H, W, _ = images.shape
            image_size = (W, H)
            print(f"[Motion Analyzer] Image size: {W}x{H}")
        
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
            print(f"[Motion Analyzer] Using user-specified height: {actual_height:.2f}m")
        else:
            actual_height = default_height_m
            height_source = "auto_estimate"
            print(f"[Motion Analyzer] Using default height: {actual_height:.2f}m")
        
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
        
        print(f"[Motion Analyzer] Mesh height: {mesh_height_info['mesh_height']:.3f} units")
        print(f"[Motion Analyzer] Estimated height (from joints): {estimated_height:.3f} units")
        print(f"[Motion Analyzer] Scale factor: {scale_factor:.3f}")
        print(f"[Motion Analyzer] Leg length: {kp_height_info['leg_length']:.3f} units")
        print(f"[Motion Analyzer] Torso+head: {kp_height_info['torso_head_length']:.3f} units")
        
        # ===== PER-FRAME ANALYSIS =====
        # Get joint indices based on mode
        if kp_source == "keypoints_3d":
            pelvis_idx = SAM3DJoints.PELVIS
            head_idx = SAM3DJoints.HEAD
            left_ankle_idx = SAM3DJoints.LEFT_ANKLE
            right_ankle_idx = SAM3DJoints.RIGHT_ANKLE
        else:
            pelvis_idx = SMPLHJoints.PELVIS
            head_idx = SMPLHJoints.HEAD
            left_ankle_idx = SMPLHJoints.LEFT_ANKLE
            right_ankle_idx = SMPLHJoints.RIGHT_ANKLE
        
        subject_motion = {
            "pelvis_2d": [],
            "pelvis_3d": [],
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
        
        # Initialize ground level tracking for foot contact detection
        ground_level = None  # Will be set on first frame
        
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
            
            # Get 3D keypoints
            if kp_source == "keypoints_3d":
                keypoints_3d = to_numpy(keypoints_3d_list[i]) if i < len(keypoints_3d_list) else None
            else:
                keypoints_3d = to_numpy(joint_coords_list[i]) if i < len(joint_coords_list) else None
            
            if keypoints_3d is None:
                print(f"[Motion Analyzer] Warning: No keypoints for frame {i}")
                keypoints_3d = np.zeros((18 if kp_source == "keypoints_3d" else 127, 3))
            
            # Handle shape
            if keypoints_3d.ndim == 3:
                keypoints_3d = keypoints_3d.squeeze(0)
            
            # Get 2D keypoints (use directly if available, otherwise project)
            # NOTE: pred_keypoints_2d always uses 70-joint MHR format (body joints 0-17)
            # This is different from joint_coords which uses 127-joint SMPLH format
            if has_kp_2d and i < len(keypoints_2d_list) and keypoints_2d_list[i] is not None:
                keypoints_2d = to_numpy(keypoints_2d_list[i])
                if keypoints_2d.ndim == 3:
                    keypoints_2d = keypoints_2d.squeeze(0)
                # Take only x,y (might have confidence as 3rd column)
                if keypoints_2d.shape[1] >= 2:
                    joints_2d = keypoints_2d[:, :2]
                else:
                    joints_2d = keypoints_2d
                # pred_keypoints_2d uses 70-joint format with body joints 0-17
                joints_2d_format = "keypoints_2d"  # Always 70-joint MHR format
                if i == 0:
                    print(f"[Motion Analyzer] Frame 0: Using pred_keypoints_2d DIRECTLY (shape={joints_2d.shape})")
                    print(f"[Motion Analyzer] Frame 0: First 3 2D joints: {joints_2d[:3]}")
                    
                    # Comprehensive joint position analysis
                    print(f"\n[Motion Analyzer] ===== JOINT POSITION ANALYSIS (Frame 0) =====")
                    print(f"[Motion Analyzer] Image size: {image_size[0]}x{image_size[1]}")
                    print(f"[Motion Analyzer] Image center: ({image_size[0]/2:.1f}, {image_size[1]/2:.1f})")
                    
                    # Find bounding box of all joints
                    valid_joints = joints_2d[~np.isnan(joints_2d).any(axis=1)]
                    if len(valid_joints) > 0:
                        min_x, min_y = valid_joints.min(axis=0)
                        max_x, max_y = valid_joints.max(axis=0)
                        center_x = (min_x + max_x) / 2
                        center_y = (min_y + max_y) / 2
                        print(f"[Motion Analyzer] Joint bounds: x=[{min_x:.1f}, {max_x:.1f}], y=[{min_y:.1f}, {max_y:.1f}]")
                        print(f"[Motion Analyzer] Joint center: ({center_x:.1f}, {center_y:.1f})")
                        print(f"[Motion Analyzer] Joint spread: {max_x - min_x:.1f}w x {max_y - min_y:.1f}h")
                    
                    # Sample some key joints to help identify the format
                    print(f"[Motion Analyzer] Sample joints:")
                    sample_indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
                    for idx in sample_indices:
                        if idx < len(joints_2d):
                            x, y = joints_2d[idx]
                            print(f"[Motion Analyzer]   Joint {idx:2d}: ({x:7.1f}, {y:7.1f})")
                    print(f"[Motion Analyzer] =============================================\n")
            else:
                # Project 3D to 2D - use the same format as keypoints_3d
                joints_2d = project_points_to_2d(
                    keypoints_3d, focal, camera_t, image_size[0], image_size[1]
                )
                joints_2d_format = kp_source  # Same format as 3D source
                if i == 0:
                    print(f"[Motion Analyzer] Frame 0: PROJECTING 3D→2D (focal={focal}, cam_t={camera_t})")
                    print(f"[Motion Analyzer] Frame 0: First 3 projected joints: {joints_2d[:3]}")
            
            # Store the 2D format on first frame
            if i == 0:
                subject_motion["joints_2d_format"] = joints_2d_format
            
            subject_motion["joints_2d"].append(joints_2d)
            subject_motion["joints_3d"].append(keypoints_3d * scale_factor)
            
            # Pelvis position - use pelvis_idx (0 for both formats)
            pelvis_2d = joints_2d[pelvis_idx] if len(joints_2d) > pelvis_idx else np.array([0, 0])
            pelvis_3d = keypoints_3d[pelvis_idx] * scale_factor if len(keypoints_3d) > pelvis_idx else np.zeros(3)
            
            subject_motion["pelvis_3d"].append(pelvis_3d.copy())
            subject_motion["pelvis_2d"].append(pelvis_2d.copy())
            
            # Apparent height (pixels) - use correct joint indices for skeleton mode
            min_joints_for_height = max(head_idx, left_ankle_idx, right_ankle_idx) + 1
            if len(joints_2d) >= min_joints_for_height:
                head_y = joints_2d[head_idx][1]  # Head
                left_ankle_y_2d = joints_2d[left_ankle_idx][1]  # Left ankle
                right_ankle_y_2d = joints_2d[right_ankle_idx][1]  # Right ankle
                feet_y = max(left_ankle_y_2d, right_ankle_y_2d)
                apparent_height = abs(feet_y - head_y)
            elif vertices is not None:
                # Fallback: use mesh bounding box
                mesh_2d = project_points_to_2d(vertices, focal, camera_t, image_size[0], image_size[1])
                mesh_min_y = mesh_2d[:, 1].min()
                mesh_max_y = mesh_2d[:, 1].max()
                apparent_height = mesh_max_y - mesh_min_y
            else:
                apparent_height = 0
            subject_motion["apparent_height"].append(apparent_height)
            
            # Depth estimate
            depth_m = camera_t[2] * scale_factor
            subject_motion["depth_estimate"].append(depth_m)
            
            # Foot contact detection using ankle joint Y positions
            # Compare against reference ground level (lowest seen across all frames)
            # Use the correct indices based on skeleton mode (set at start of function)
            min_joints_needed = max(left_ankle_idx, right_ankle_idx) + 1
            if len(keypoints_3d) >= min_joints_needed:
                left_ankle_y = keypoints_3d[left_ankle_idx][1]   # Left ankle Y
                right_ankle_y = keypoints_3d[right_ankle_idx][1]  # Right ankle Y
                
                # Initialize or update ground level estimate (lowest seen)
                current_lowest = min(left_ankle_y, right_ankle_y)
                if ground_level is None:
                    ground_level = current_lowest
                else:
                    ground_level = min(ground_level, current_lowest)
                
                # Check if ankles are near ground level
                # Note: In SAM3DBody coordinate system, Y increases upward
                # So "grounded" means Y is close to the minimum value
                ground_threshold = 0.05  # 5cm threshold (unscaled)
                left_grounded = abs(left_ankle_y - ground_level) < ground_threshold
                right_grounded = abs(right_ankle_y - ground_level) < ground_threshold
                
                if left_grounded and right_grounded:
                    foot_contact = "both"
                elif left_grounded:
                    foot_contact = "left"
                elif right_grounded:
                    foot_contact = "right"
                else:
                    foot_contact = "none"
            else:
                foot_contact = "none"
            
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
            print(f"[Motion Analyzer] Generating debug overlay...")
            
            images_np = images.cpu().numpy() if isinstance(images, torch.Tensor) else images
            
            # Convert skeleton_mode to short form for overlay function
            skeleton_mode_str = "simple" if skeleton_mode == "Simple Skeleton" else "full"
            
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
