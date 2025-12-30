# SAM4DBodyCapture v0.5.0 Debug Reference Document

**Last Updated**: 2025-12-30 15:30 IST  
**Current Version**: v0.5.0-debug7  
**Status**: Testing - Major fix for joint indices and 2D/3D separation  
**Working Reference**: SAM3DBody2abc implementation

---

## Executive Summary

The Motion Analyzer node in SAM4DBodyCapture v0.5.0 had skeleton and mesh overlay alignment issues caused by:
1. Wrong joint indices (using MHR format instead of COCO format for pred_keypoints_2d)
2. Incorrectly projecting joint_coords to 2D instead of using pred_keypoints_2d directly

**debug7 fixes these by:**
- Using COCO format indices for pred_keypoints_2d (the correct format)
- ALWAYS using pred_keypoints_2d for 2D visualization
- Using SMPLH or COCO indices appropriately for 3D analysis only

---

## Key Fix in debug7

### The Problem
SAM3DBody outputs `pred_keypoints_2d` in **COCO format** (17 joints), NOT MHR format (18 joints).

**COCO joint indices** (correct):
```
NOSE = 0, LEFT_EYE = 1, RIGHT_EYE = 2, LEFT_EAR = 3, RIGHT_EAR = 4
LEFT_SHOULDER = 5, RIGHT_SHOULDER = 6, LEFT_ELBOW = 7, RIGHT_ELBOW = 8
LEFT_WRIST = 9, RIGHT_WRIST = 10, LEFT_HIP = 11, RIGHT_HIP = 12
LEFT_KNEE = 13, RIGHT_KNEE = 14, LEFT_ANKLE = 15, RIGHT_ANKLE = 16
```

**What we had wrong** (MHR indices):
```
PELVIS = 0, LEFT_ANKLE = 14, RIGHT_ANKLE = 17  ← WRONG!
```

### The Solution (debug7)

1. **Created COCOJoints class** with correct COCO format indices
2. **Separated 2D and 3D indices**:
   - 2D indices: Always COCO (for pred_keypoints_2d visualization)
   - 3D indices: COCO or SMPLH based on skeleton mode (for analysis)
3. **Always use pred_keypoints_2d for 2D** - never project joint_coords

```python
# 2D indices - ALWAYS COCO format (pred_keypoints_2d)
pelvis_idx_2d = COCOJoints.PELVIS      # 11 (left hip as proxy)
head_idx_2d = COCOJoints.HEAD          # 0 (nose)
left_ankle_idx_2d = COCOJoints.LEFT_ANKLE   # 15
right_ankle_idx_2d = COCOJoints.RIGHT_ANKLE  # 16

# 3D indices - depends on kp_source
if kp_source == "keypoints_3d":
    # COCO format for pred_keypoints_3d
    left_ankle_idx_3d = COCOJoints.LEFT_ANKLE  # 15
else:
    # SMPLH format for joint_coords
    left_ankle_idx_3d = SMPLHJoints.LEFT_ANKLE  # 7
```

---

## Current Issues (debug6)

### Issue 1: Skeleton Projection Misaligned
- **Symptom**: Skeleton dots appear at TOP of frame (near YouTube icons) instead of on the dancer's body
- **Log evidence**:
  ```
  pred_keypoints_2d first 3 points: [[406.5, 446.96]...]   ← Y ~447 (correct position)
  projected joints first 3:         [[376.9, 182.22]...]   ← Y ~182 (wrong, at top!)
  ```
- **What we tried**:
  - debug5: No Y negation → skeleton at BOTTOM (Y ~1097)
  - debug6: Y negation for SMPLH → skeleton at TOP (Y ~182)
  - Neither matches the correct Y ~447 from pred_keypoints_2d

### Issue 2: Mesh Overlay Misaligned
- **Symptom**: Blue mesh overlay appears on torso but not correctly positioned
- **The mesh overlay uses pyrender which has its own coordinate handling
- **SAM3DBody2abc has a centroid-based offset correction that we're missing

### Issue 3: Foot Contact Detection
- **Current**: 30% grounded, 70% airborne
- **This may be incorrect due to the coordinate system issues
- **SAM3DBody2abc gets this right

---

## Joint Format Reference

### SAM3DBody outputs TWO different joint formats:

#### 1. pred_keypoints_2d / pred_keypoints_3d (70 joints - MHR format)
```
Body joints 0-17:
  0=PELVIS, 1=SPINE1, 2=SPINE2, 3=SPINE3, 4=NECK, 5=HEAD
  6=L_SHOULDER, 7=L_ELBOW, 8=L_WRIST
  9=R_SHOULDER, 10=R_ELBOW, 11=R_WRIST
  12=L_HIP, 13=L_KNEE, 14=L_ANKLE
  15=R_HIP, 16=R_KNEE, 17=R_ANKLE

Face joints 18-54 (37 points)
Hand joints 55-69 (15 points)
```

**Key indices for MHR**:
- LEFT_ANKLE = 14
- RIGHT_ANKLE = 17
- HEAD = 5
- PELVIS = 0

#### 2. pred_joint_coords (127 joints - SMPLH format)
```
Body joints 0-21:
  0=PELVIS, 1=L_HIP, 2=R_HIP, 3=SPINE1
  4=L_KNEE, 5=R_KNEE, 6=SPINE2
  7=L_ANKLE, 8=R_ANKLE, 9=SPINE3
  10=L_FOOT, 11=R_FOOT, 12=NECK
  13=L_COLLAR, 14=R_COLLAR, 15=HEAD
  16=L_SHOULDER, 17=R_SHOULDER
  18=L_ELBOW, 19=R_ELBOW
  20=L_WRIST, 21=R_WRIST

Hand joints 22-126 (105 points)
```

**Key indices for SMPLH**:
- LEFT_ANKLE = 7
- RIGHT_ANKLE = 8
- HEAD = 15
- PELVIS = 0

---

## What SAM3DBody2abc Does Correctly

### File: `/mnt/user-data/uploads/verify_overlay.py`

### KEY INSIGHT: Uses pred_keypoints_2d Directly! (lines 295-307)

**SAM3DBody2abc NEVER projects joint_coords to 2D for skeleton overlay!**

```python
if show_joints or show_skeleton:
    # Prefer 2D keypoints if available (already in image coordinates)
    if keypoints_2d is not None:
        # pred_keypoints_2d might be (N, 2) or (N, 3) with confidence
        if keypoints_2d.ndim == 2:
            joints_2d = keypoints_2d[:, :2] if keypoints_2d.shape[1] >= 2 else keypoints_2d
        else:
            joints_2d = keypoints_2d
        info_parts.append(f"Keypoints2D: {len(joints_2d)}")
    elif joint_coords is not None and focal_length is not None and cam_t is not None:
        # Fall back to projecting 3D joints (ONLY if no 2D available)
        joints_2d = project_points_to_2d(joint_coords, focal_length, cam_t, w, h)
```

**The 2D keypoints from SAM3DBody are already in correct image pixel coordinates!**

### 1. Projection Function (lines 33-77)
```python
def project_points_to_2d(points_3d, focal_length, cam_t, image_width, image_height):
    # NO Y negation - coordinates are already image-aligned
    X = points_3d[:, 0] + tx
    Y = points_3d[:, 1] + ty  # NO negation!
    Z = points_3d[:, 2] + tz
    
    x_2d = focal_length * X / Z + cx
    y_2d = focal_length * Y / Z + cy
```

### 2. Centroid-Based Offset Alignment (lines 664-701)
**THIS IS THE KEY MISSING PIECE FOR MESH OVERLAY**

SAM3DBody2abc computes an offset between the projected mesh centroid and the 2D keypoint centroid, then applies this offset to correct projection errors:

```python
# Get 2D keypoints (ground truth position)
keypoints_2d = output.get("pred_keypoints_2d")

# Project mesh vertices to 2D
mesh_2d = project_points_to_2d(vertices, focal, cam_t, width, height)

# Compute centroids
kp_centroid = keypoints_2d[:18].mean(axis=0)  # Only body joints
mesh_centroid = mesh_2d.mean(axis=0)

# Compute offset
offset = kp_centroid - mesh_centroid

# Apply offset to mesh projection
mesh_2d_corrected = mesh_2d + offset
```

---

## Debug Version History

| Version | Changes | Result |
|---------|---------|--------|
| debug3 | BFloat16 fix with autocast wrapper | ✅ No CUDA errors |
| debug4 | Joint index fix for SMPLH format | Skeleton scattered (wrong format) |
| debug5 | Project joint_coords to 2D, no Y negation | Skeleton at BOTTOM (Y ~1097) |
| debug6 | Y negation for SMPLH projection | Skeleton at TOP (Y ~182) |

---

## Proposed Fix Strategy

### Option A: Use pred_keypoints_2d for 2D (Recommended)
For skeleton overlay, **always use pred_keypoints_2d directly** since SAM3DBody already provides accurate 2D positions. Only use joint_coords for 3D analysis.

```python
# For 2D skeleton overlay - use pred_keypoints_2d (already correct)
if has_kp_2d:
    joints_2d = pred_keypoints_2d[:18]  # Body joints only
    joints_2d_format = "keypoints_2d"

# For 3D analysis - use joint_coords if Full Skeleton mode
if skeleton_mode == "full" and has_joint_coords:
    joints_3d = joint_coords
    # Use SMPLH indices (7, 8 for ankles, 15 for head)
else:
    joints_3d = keypoints_3d
    # Use MHR indices (14, 17 for ankles, 5 for head)
```

### Option B: Implement Centroid Offset Correction
If projecting joint_coords to 2D is needed, implement the centroid offset correction from SAM3DBody2abc.

### Option C: Match SAM3DBody2abc Exactly
Copy the exact projection and rendering logic from SAM3DBody2abc's verify_overlay.py.

---

## Key Files

### In SAM4DBodyCapture:
- `/nodes/motion_analyzer.py` - Motion analysis and skeleton overlay
- `/nodes/mesh_overlay.py` - Mesh rendering (uses pyrender)
- `/nodes/sam3dbody_integration.py` - SAM3DBody processing

### Reference (working):
- `/mnt/user-data/uploads/verify_overlay.py` - SAM3DBody2abc's working implementation

---

## Coordinate System Notes

### SAM3DBody Output Coordinates:
- **pred_keypoints_2d**: Already in image pixel coordinates (x, y)
- **pred_keypoints_3d**: Camera-relative 3D (x, y, z) - MHR format
- **pred_joint_coords**: Camera-relative 3D (x, y, z) - SMPLH format
- **camera_t**: Camera translation [tx, ty, tz]

### Image Coordinates:
- Origin: Top-left corner
- X: Left to right (0 to width)
- Y: Top to bottom (0 to height)

### 3D Camera Coordinates (SAM3DBody):
- Origin: Camera optical center
- X: Right
- Y: Down (in image space, so positive Y = lower in image)
- Z: Forward (depth into scene)

**The mystery**: Why does pred_keypoints_2d have correct Y values (~447 for head) but projecting joint_coords gives wrong values (~1097 or ~182)?

Possible reasons:
1. joint_coords uses a different coordinate convention than keypoints_3d
2. The camera_t values are calibrated for keypoints_3d, not joint_coords
3. There's a scale difference between the two formats

---

## Test Data Reference

### Video: Dance studio with 3 dancers
- Resolution: 720x1280 (portrait)
- Frames: 50
- Subject: Center dancer (woman in black)
- Height: 1.59m (user specified)

### Expected skeleton position:
- Head Y: ~447 pixels (upper portion of frame)
- Feet Y: ~800-900 pixels (lower portion)
- Center X: ~360-400 pixels (near center)

### Actual results (debug6):
- Projected joints Y: ~182 (way too high, at top of frame)

---

## Next Steps for New Chat

1. **Read this document first** to understand the full context

2. **Examine SAM3DBody2abc's verify_overlay.py** at `/mnt/user-data/uploads/verify_overlay.py`
   - Lines 33-77: Projection function
   - Lines 664-701: Centroid offset correction
   - Lines 80-140: Joint indices and connections

3. **Key decision**: Use pred_keypoints_2d directly for 2D skeleton (Option A) OR implement centroid offset (Option B)

4. **Test with the dance video** - same video has been used for all debug versions

5. **Compare log output**:
   - pred_keypoints_2d Y values should be ~400-800 range
   - Projected joint_coords should match after fix

---

## Files to Upload to New Chat

1. This document: `DEBUG_REFERENCE_v0.5.0.md`
2. Current codebase: `ComfyUI-SAM4DBodyCapture-v0.5.0-debug6.zip`
3. Reference implementation: `verify_overlay.py` from SAM3DBody2abc
4. Test screenshots showing the alignment issues

---

## Contact/Context

- Developer working on SAM4DBodyCapture ComfyUI node
- Uses SAM3DBody for mesh reconstruction
- Goal: Accurate skeleton overlay and foot contact detection for motion analysis
- Timezone: IST (UTC+5:30)
- Prefers timestamps in IST format in logs
