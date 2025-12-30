# SAM4DBodyCapture v0.5.0 Debug Reference Document

**Last Updated**: 2025-12-30 18:00 IST  
**Current Version**: v0.5.0-debug9 (ready for test)  
**Status**: MHR format + SMPLH body_world fix + Joint labels + Timestamps  
**Reference**: SAM3DBody2abc v4.4.8 verify_overlay.py

---

## Quick Start for New Chat

If continuing this work in a new chat:
1. Upload this document
2. Upload the latest debug build zip
3. State: "Continue debugging SAM4DBodyCapture skeleton overlay alignment"

---

## Executive Summary

**debug9 changes:**
1. Uses **MHR format** for pred_keypoints_2d (NOT COCO) - based on visual inspection
2. Fixed **SMPLHJoints** indices - body_world=0, pelvis=1 (was pelvis=0)
3. Adds **body_world tracking** for global trajectory in world coordinates
4. Adds **joint index labels** on overlay for debugging
5. Adds **detailed joint position logging** for Frame 0
6. Adds **IST timestamps** to ALL node outputs

---

## CRITICAL: Two Different Joint Formats

### 1. pred_keypoints_2d/3d (70 joints) - MHR Format
Used for 2D overlay visualization:
```python
MHRJoints:
    HEAD = 0, NECK = 4
    L_SHOULDER = 5, R_SHOULDER = 6
    L_ELBOW = 7, R_ELBOW = 8
    L_HIP = 9, R_HIP = 10
    L_KNEE = 11, L_ANKLE = 12, L_HEEL = 13, L_TOE = 14
    R_KNEE = 15, R_ANKLE = 16, R_HEEL = 17, R_TOE = 18, 19
    L_WRIST = 20, R_WRIST = 21
```

### 2. pred_joint_coords (127 joints) - SMPLH Format
Used for 3D analysis and global trajectory:
```python
SMPLHJoints:
    BODY_WORLD = 0  # GLOBAL TRAJECTORY (X,Y,Z in world coordinates)
    PELVIS = 1      # Hip movement relative to body_world
    L_HIP = 2, R_HIP = 3
    SPINE1 = 4
    L_KNEE = 5, R_KNEE = 6
    SPINE2 = 7
    L_ANKLE = 8, R_ANKLE = 9
    SPINE3 = 10
    L_FOOT = 11, R_FOOT = 12
    NECK = 13
    L_COLLAR = 14, R_COLLAR = 15
    HEAD = 16
    L_SHOULDER = 17, R_SHOULDER = 18
    L_ELBOW = 19, R_ELBOW = 20
    L_WRIST = 21, R_WRIST = 22
```

**IMPORTANT**: `body_world[0]` gives the character's movement in X,Y,Z world coordinates!

---

## Debug Version History

| Version | Changes | Mesh | Skeleton |
|---------|---------|------|----------|
| v0.5.0 | Original | ✅ Correct | ❌ Wrong indices |
| debug7 | Removed cam_int + tried COCO | ❌ Broken | ❌ Broken |
| debug8 | Restore cam_int + COCO indices | ✅ Fixed | ❌ Still wrong |
| **debug9** | **MHR + SMPLH body_world fix** | ✅ | ⏳ Test |

---

## debug9 Console Output

Expected output for Frame 0:
```
[2025-12-30 18:00:00 IST] [Motion Analyzer] Skeleton mode: Full Skeleton
[2025-12-30 18:00:00 IST] [Motion Analyzer] Tracking body_world (idx 0) for global trajectory
[2025-12-30 18:00:00 IST] [Motion Analyzer] 2D indices (MHR): pelvis=9, head=0, L_ankle=12, R_ankle=16
[2025-12-30 18:00:00 IST] [Motion Analyzer] 3D indices (joint_coords): pelvis=1, head=16, L_ankle=8, R_ankle=9
[2025-12-30 18:00:00 IST] [Motion Analyzer] ===== JOINT POSITIONS (Frame 0) =====
[Motion Analyzer] joints_2d shape: (70, 2), source: pred_keypoints_2d
[Motion Analyzer]   [ 0] head      : x=  360.0, y=  430.0
[Motion Analyzer]   [ 9] L_hip     : x=  340.0, y=  700.0
[Motion Analyzer]   [12] L_ankle   : x=  350.0, y= 1000.0
...
[2025-12-30 18:00:00 IST] [Motion Analyzer] body_world[0] (global trajectory): X=0.123, Y=0.456, Z=3.000
```

---

## Expected Results

1. **Timestamps visible** from the very first node execution
2. **Joint numbers** on overlay match body parts (0=head, 9=L_hip, 12=L_ankle, etc.)
3. **Console log** shows correct X,Y positions for each joint
4. **body_world** position logged for global trajectory tracking
5. **3D indices** now correct: pelvis=1, head=16, L_ankle=8, R_ankle=9

---

## Key Code Snippets

### Getting Global Trajectory (body_world)
```python
# From pred_joint_coords (127 joints)
joints = output['pred_joint_coords']  # Shape: (127, 3)
body_world = joints[0]  # Global trajectory
world_x, world_y, world_z = body_world
pelvis = joints[1]  # Hip relative to body_world
```

### 2D Visualization (MHR Format)
```python
# From pred_keypoints_2d (70 joints)
joints_2d = output['pred_keypoints_2d']  # Shape: (70, 2)
head = joints_2d[0]      # Head position in pixels
left_hip = joints_2d[9]  # Left hip in pixels
left_ankle = joints_2d[12]  # Left ankle in pixels
```
