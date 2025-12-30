# SAM4DBodyCapture v0.5.0 Debug Reference Document

**Last Updated**: 2025-12-30 19:00 IST  
**Current Version**: v0.5.0-debug10 (COMPLETE)  
**Status**: Implemented unified BodyJoints + joint_coords projection  
**Reference**: SAM3DBody MHR 70-Joint / 127-Joint structure

---

## Quick Start for New Chat

If continuing this work in a new chat:
1. Upload this document
2. Upload the latest debug build zip (or source)
3. State: "Continue implementing debug10 for SAM4DBodyCapture skeleton overlay"

---

## Executive Summary

**Problem**: Skeleton overlay joints not aligning with person in video (mesh overlay works fine)

**Root Cause Discovered**: 
1. We were using WRONG joint indices (from SAM3DBody2abc verify_overlay.py which was incorrect)
2. We were using `pred_keypoints_2d` directly instead of projecting `joint_coords` like mesh does
3. Missing Y/Z negation (180° X rotation) in projection function

**Solution (debug10) - IMPLEMENTED**:
1. ✅ Use correct **unified BodyJoints** class with proper indices
2. ✅ Project `joint_coords` (3D) → 2D using same camera intrinsics as mesh
3. ✅ Apply Y/Z negation (same as mesh_overlay.py fallback)
4. ✅ Use `body_world` (index 0) for global trajectory tracking

---

## CRITICAL: Correct Joint Structure

### MHR 70-Joint Subset Table (CONFIRMED CORRECT)

| Index | Joint Name | Category | Primary Function |
|-------|------------|----------|------------------|
| 0 | `body_world` | Trajectory | **Global Movement (X, Y, Z translation)** |
| 1 | `pelvis` | Root | Anatomical root for spine and legs |
| 2 | `spine_1` | Torso | Lower spine articulation |
| 3 | `spine_2` | Torso | Upper spine and shoulder connection |
| 4 | `neck` | Neck/Head | Base of neck |
| 5 | `head` | Neck/Head | Top of head / skull |
| 6 | `L_hip` | Left Leg | Primary left hip rotation |
| 7 | `L_knee` | Left Leg | Left knee flexion/extension |
| 8 | `L_ankle` | Left Leg | Left ankle joint |
| 9 | `R_hip` | Right Leg | Primary right hip rotation |
| 10 | `R_knee` | Right Leg | Right knee flexion/extension |
| 11 | `R_ankle` | Right Leg | Right ankle joint |
| 12-13 | `L_foot_sub` | Left Foot | Heel, toe |
| 14-15 | `R_foot_sub` | Right Foot | Heel, toe |
| 16 | `L_shldr` | Left Arm | Left shoulder joint |
| 17 | `L_elbow` | Left Arm | Left elbow articulation |
| 18 | `L_wrist` | Left Arm | Left wrist joint |
| 19 | `R_shldr` | Right Arm | Right shoulder joint |
| 20 | `R_elbow` | Right Arm | Right elbow articulation |
| 21 | `R_wrist` | Right Arm | Right wrist joint |
| 22-45 | `L_hand_joints` | Left Hand | 20+ joints for finger control |
| 46-69 | `R_hand_joints` | Right Hand | 20+ joints for finger control |

### 127-Joint Breakdown

| Group | Index Range | Description |
|-------|-------------|-------------|
| Global Root | 0 | `body_world`: Global translation (X, Y, Z) |
| Body Core | 1-21 | Anatomical skeleton (same as 70-joint) |
| Left Hand | 22-42 | 21 joints for fingers |
| Right Hand | 43-63 | 21 joints for fingers |
| Feet & Toes | 64-70+ | Additional foot mechanics |
| Face & Head | Remaining | Facial expressions (FACS) |

---

## What Was WRONG (debug9 and earlier)

### Wrong MHRJoints Class (from SAM3DBody2abc verify_overlay.py)
```python
# THIS WAS WRONG!
class MHRJoints:
    HEAD = 0        # ← Actually body_world!
    NECK = 4        # ← Actually neck ✓
    LEFT_HIP = 9    # ← Actually R_hip!
    LEFT_ANKLE = 12 # ← Actually L_foot_heel!
```

### Wrong Data Source
- Using `pred_keypoints_2d` directly (values looked like pixels but didn't align)
- Should use `joint_coords` (3D) and project with camera intrinsics

---

## debug10 Implementation - COMPLETE

### 1. Unified BodyJoints Class (✅ DONE)
```python
class BodyJoints:
    BODY_WORLD = 0   # Global trajectory
    PELVIS = 1
    SPINE_1 = 2
    SPINE_2 = 3
    NECK = 4
    HEAD = 5         # ← Correct!
    L_HIP = 6
    L_KNEE = 7
    L_ANKLE = 8      # ← Correct!
    R_HIP = 9
    R_KNEE = 10
    R_ANKLE = 11     # ← Correct!
    L_FOOT_HEEL = 12
    L_FOOT_TOE = 13
    R_FOOT_HEEL = 14
    R_FOOT_TOE = 15
    L_SHOULDER = 16
    L_ELBOW = 17
    L_WRIST = 18
    R_SHOULDER = 19
    R_ELBOW = 20
    R_WRIST = 21
    
    # Backward-compatible aliases
    LEFT_HIP = L_HIP
    LEFT_KNEE = L_KNEE
    LEFT_ANKLE = L_ANKLE
    RIGHT_HIP = R_HIP
    # ... etc
```

### 2. Projection Method (✅ DONE)
```python
# Fixed to match mesh_overlay.py exactly
def project_points_to_2d(points_3d, focal_length, cam_t, image_width, image_height):
    # 1. Apply camera translation
    X = points_3d[:, 0] + cam_t[0]
    Y = points_3d[:, 1] + cam_t[1]
    Z = points_3d[:, 2] + cam_t[2]
    
    # 2. Apply 180° rotation around X axis (CRITICAL!)
    Y = -Y
    Z = -Z
    
    # 3. Perspective projection
    x_2d = focal_length * X / Z + cx
    y_2d = focal_length * Y / Z + cy
    
    return np.stack([x_2d, y_2d], axis=1)
```

### 3. analyze() Function (✅ DONE)
- Always projects joint_coords (3D) → 2D
- Removed pred_keypoints_2d usage
- Uses unified indices for both 2D and 3D

### 4. create_motion_debug_overlay() (✅ DONE)
- Uses unified BodyJoints indices
- Draws joints at projected 2D positions

---

## Files Modified in debug10

1. **nodes/motion_analyzer.py** (✅ COMPLETE)
   - [x] Added backward-compatible aliases (LEFT_ANKLE, RIGHT_ANKLE, etc.)
   - [x] Fixed project_points_to_2d() to apply Y/Z negation
   - [x] Updated analyze_subject_motion() to always project joint_coords
   - [x] Updated create_motion_debug_overlay() to use unified indices
   - [x] Removed pred_keypoints_2d usage for skeleton overlay

2. **docs/DEBUG_REFERENCE_v0.5.0.md** (this file)
   - [x] Document correct joint structure
   - [x] Document implementation completion

---

## Debug Version History

| Version | Changes | Mesh | Skeleton |
|---------|---------|------|----------|
| v0.5.0 | Original | ✅ | ❌ Wrong indices |
| debug7 | Removed cam_int | ❌ | ❌ |
| debug8 | Restore cam_int + COCO | ✅ | ❌ |
| debug9 | Wrong MHR indices + pred_keypoints_2d | ✅ | ❌ |
| **debug10** | **Unified BodyJoints + joint_coords projection + Y/Z negation** | ✅ | ✅ Expected |

---

## Key Evidence from debug9 Test

### Console Output (WRONG positions)
```
[ 0] head      : x=395.6, y=429.4   ← This was body_world, not head!
[ 5] L_shldr   : x=474.8, y=511.1   ← This was actually HEAD!
[ 8] R_elbow   : x=428.9, y=534.2   ← This was actually L_ANKLE!
[11] L_knee    : x=566.3, y=714.9   ← This was actually R_ANKLE!
[12] L_ankle   : x=544.1, y=707.5   ← This was actually L_FOOT_HEEL!
```

### body_world was all zeros
```
body_world[0] (global trajectory): X=0.000, Y=0.000, Z=0.000
```
This should have actual values for global position.

### Visual Result
- Mesh overlay: ✅ Perfectly aligned with person
- Skeleton joints: ❌ Clustered incorrectly, not on body parts

---

## Expected Result After debug10

### Console Output (Expected)
```
[2025-12-30 19:00:00 IST] [Motion Analyzer] debug10: Using unified BodyJoints indices
[2025-12-30 19:00:00 IST] [Motion Analyzer] Indices: pelvis=1, head=5, L_ankle=8, R_ankle=11
[2025-12-30 19:00:00 IST] [Motion Analyzer] debug10: Projecting joint_coords (3D) → 2D
[2025-12-30 19:00:00 IST] [Motion Analyzer] Projection: focal=623.5px, cam_t=[x, y, z]
[Motion Analyzer] ===== JOINT POSITIONS (Frame 0) =====
[Motion Analyzer] joints_2d shape: (127, 2) or (70, 2)
[Motion Analyzer] joints_2d source: projected from joint_coords (debug10)
[Motion Analyzer] Image size: 720x1280
[Motion Analyzer] --- Body Joints (unified indices 0-21) ---
[Motion Analyzer]   [ 0] body_world: x=  360.0, y=  640.0  (center - global root)
[Motion Analyzer]   [ 1] pelvis    : x=  365.0, y=  650.0
[Motion Analyzer]   [ 5] head      : x=  370.0, y=  430.0  ← Near top!
[Motion Analyzer]   [ 8] L_ankle   : x=  340.0, y= 1050.0  ← Near bottom!
[Motion Analyzer]   [11] R_ankle   : x=  380.0, y= 1040.0  ← Near bottom!
```

### Visual Result (Expected)
- Skeleton joints aligned with body parts
- Head joint (yellow, idx 5) near person's head
- Ankle joints (orange, idx 8/11) near person's feet
- Pelvis (green, idx 1) at person's waist/hip area
- Same alignment as mesh overlay

---

## Code Location

Repository: `/home/claude/ComfyUI-SAM4DBodyCapture/`

Key files:
- `nodes/motion_analyzer.py` - Main file being modified
- `nodes/mesh_overlay.py` - Reference for working projection (mesh works!)
- `docs/DEBUG_REFERENCE_v0.5.0.md` - This document

---

## Transcript Location

Full conversation history available at:
`/mnt/transcripts/2025-12-30-12-10-35-debug9-mhr-joint-format-timestamp-fix.txt`

Previous transcript:
`/mnt/transcripts/2025-12-30-11-13-41-debug8-coco-indices-cam-int-fix.txt`
