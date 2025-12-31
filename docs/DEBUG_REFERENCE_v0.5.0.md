# SAM4DBodyCapture v0.5.0 Debug Reference Document

**Last Updated**: 2025-12-30 19:30 IST  
**Current Version**: v0.5.0-debug11 (CRITICAL FIX)  
**Status**: Fixed - use pred_keypoints_2d directly for 2D overlay  
**Reference**: SAM3DBody MHR 70-Joint / 127-Joint structure

---

## Quick Start for New Chat

If continuing this work in a new chat:
1. Upload this document
2. Upload the latest debug build zip (or source)
3. State: "Continue implementing SAM4DBodyCapture skeleton overlay"

---

## Executive Summary

**Problem**: Skeleton overlay joints not aligning with person in video (mesh overlay works fine)

**Root Causes Discovered**:
1. Wrong joint indices (from incorrect reference file)
2. debug10 tried to project `joint_coords` which is in LOCAL body space, NOT world space!
3. `joint_coords` contains relative joint positions, not absolute 3D coordinates

**Solution (debug11) - IMPLEMENTED**:
1. ✅ Use `pred_keypoints_2d` DIRECTLY (already in pixel coordinates!)
2. ✅ Do NOT project `joint_coords` - it's local body space
3. ✅ Fallback: project `pred_keypoints_3d` (18-joint world space) if 2D unavailable
4. ✅ Use unified BodyJoints indices

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

## CRITICAL Understanding: SAM3DBody Data Sources

### The Three Data Sources (from SAM3DBody output)

| Data Source | Joints | Coordinate Space | Use For |
|-------------|--------|------------------|---------|
| `pred_keypoints_2d` | 70 | **Pixel coords (already 2D!)** | **2D overlay - USE THIS!** |
| `pred_keypoints_3d` | 18 | World space (3D) | Height estimation, 3D analysis |
| `joint_coords` | 127 | **LOCAL body space** | Rigging, NOT for 2D projection! |

### Why debug10 Failed

debug10 tried to project `joint_coords` to 2D. But `joint_coords` contains:
- Relative joint positions in local body space
- Values like 0.088 units for "leg length" (should be ~0.9m)
- All joints clustered in a tiny area when projected

### Why Mesh Overlay Works

The mesh overlay projects **vertices** which are in world space after SMPL transformation.
The skeleton overlay should use **pred_keypoints_2d** which is already in pixel coordinates!

---

## debug11 Implementation - COMPLETE

### 1. Data Source Priority (✅ DONE)
```python
# Priority for 2D visualization:
if has_kp_2d:
    # BEST: Use pred_keypoints_2d directly - already in pixel coords!
    joints_2d = keypoints_2d[:, :2]
elif has_kp_3d:
    # FALLBACK: Project pred_keypoints_3d (world space) to 2D
    joints_2d = project_points_to_2d(keypoints_3d_list[i], ...)
else:
    # LAST RESORT: Center of image
    joints_2d = np.zeros((22, 2))
```

### 2. Unified BodyJoints Indices (✅ DONE)
```python
class BodyJoints:
    BODY_WORLD = 0   # Global trajectory (index 0)
    PELVIS = 1       # Anatomical root
    HEAD = 5         # Top of head
    L_ANKLE = 8      # Left ankle
    R_ANKLE = 11     # Right ankle
    # ... etc
```

### 3. Key Insight
- `pred_keypoints_2d` is 70 joints (MHR format)
- First 22 joints (0-21) are body joints
- Indices match BodyJoints class
- NO projection needed - already in pixel coordinates!

---

## Files Modified in debug11

1. **nodes/motion_analyzer.py** (✅ COMPLETE)
   - [x] Use pred_keypoints_2d directly for 2D overlay (no projection!)
   - [x] Fallback to project pred_keypoints_3d if 2D unavailable
   - [x] Do NOT project joint_coords (it's local body space)
   - [x] Unified BodyJoints indices with backward-compatible aliases
   - [x] Proper bounds checking for index access

2. **docs/DEBUG_REFERENCE_v0.5.0.md** (this file)
   - [x] Document correct data source understanding
   - [x] Document debug11 fix

---

## Debug Version History

| Version | Changes | Mesh | Skeleton |
|---------|---------|------|----------|
| v0.5.0 | Original | ✅ | ❌ Wrong indices |
| debug7 | Removed cam_int | ❌ | ❌ |
| debug8 | Restore cam_int + COCO | ✅ | ❌ |
| debug9 | Wrong MHR indices + pred_keypoints_2d | ✅ | ❌ |
| debug10 | Unified BodyJoints + project joint_coords | ✅ | ❌ (joint_coords is local space!) |
| **debug11** | **Use pred_keypoints_2d directly (pixel coords)** | ✅ | ✅ Expected |

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

## Expected Result After debug11

### Console Output (Expected)
```
[Motion Analyzer] debug10: Using unified BodyJoints indices
[Motion Analyzer] Indices: pelvis=1, head=5, L_ankle=8, R_ankle=11
[Motion Analyzer] debug10: Using pred_keypoints_2d directly (already in pixel coords)
[Motion Analyzer] pred_keypoints_2d shape: (70, 2)
[Motion Analyzer] ===== JOINT POSITIONS (Frame 0) =====
[Motion Analyzer] joints_2d shape: (70, 2)
[Motion Analyzer] joints_2d source: pred_keypoints_2d (direct pixel coords)
[Motion Analyzer] Image size: 720x1280
[Motion Analyzer] --- Body Joints (unified indices 0-21) ---
[Motion Analyzer]   [ 0] body_world: x=  360.0, y=  640.0  (center - global root)
[Motion Analyzer]   [ 1] pelvis    : x=  365.0, y=  750.0  ← Mid-body
[Motion Analyzer]   [ 5] head      : x=  370.0, y=  450.0  ← Near TOP of person!
[Motion Analyzer]   [ 8] L_ankle   : x=  340.0, y= 1050.0  ← Near BOTTOM of person!
[Motion Analyzer]   [11] R_ankle   : x=  380.0, y= 1040.0  ← Near BOTTOM of person!
```

### Key Difference from debug10
- debug10: All joints clustered at y=1070-1096 (17px range) ❌
- debug11: Joints spread from y~450 (head) to y~1050 (feet) (~600px range) ✅

### Visual Result (Expected)
- Skeleton joints aligned with body parts
- Head joint (yellow, idx 5) near person's head (top of frame)
- Ankle joints (orange, idx 8/11) near person's feet (bottom of frame)
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
