# SAM4DBodyCapture v0.5.0 Debug Reference Document

**Last Updated**: 2025-12-30 16:20 IST  
**Current Version**: v0.5.0-debug8 (awaiting test)  
**Status**: Fix for BOTH mesh overlay AND skeleton alignment  
**Working Reference**: v0.5.0 (mesh overlay correct, skeleton wrong)

---

## Quick Start for New Chat

If continuing this work in a new chat:
1. Upload this document
2. Upload the latest debug build zip
3. State: "Continue debugging SAM4DBodyCapture skeleton overlay alignment"

---

## Executive Summary

**TWO BUGS FOUND:**

1. **Mesh overlay wrong (debug7)** - Caused by removing `cam_int` parameter
2. **Skeleton indices wrong (ALL versions)** - Using MHR indices on COCO format data

**debug8 fixes BOTH:**
- Restores `cam_int` parameter (from working v0.5.0)
- Updates SAM3DJoints to use COCO format indices
- Separates 2D indices (always COCO) from 3D indices (COCO or SMPLH)

---

## Debug Version History

| Version | Changes | Mesh | Skeleton |
|---------|---------|------|----------|
| v0.5.0 | Original | ✅ Correct | ❌ Wrong indices |
| debug7 | Removed cam_int + COCO indices | ❌ Broken | ❌ Broken |
| **debug8** | **Restore cam_int + Fix COCO indices** | ⏳ Test | ⏳ Test |

---

## Root Cause Analysis

### Bug 1: Mesh Overlay (debug7)
**Missing `cam_int` in process_one_image()**
- Without camera intrinsics, SAM3DBody uses defaults
- Causes wrong `pred_cam_t` → wrong mesh projection
- Evidence: Depth doubled (2.88m → 6.15m)

### Bug 2: Skeleton Indices (ALL versions)
**pred_keypoints_2d is COCO format, not MHR!**

| Joint | Working Build (WRONG) | COCO Format (CORRECT) |
|-------|----------------------|----------------------|
| Index 0 | PELVIS | NOSE |
| Index 5 | HEAD | LEFT_SHOULDER |
| Index 14 | LEFT_ANKLE | RIGHT_KNEE |
| Index 15 | RIGHT_HIP | LEFT_ANKLE |
| Index 16 | RIGHT_KNEE | RIGHT_ANKLE |
| Index 17 | RIGHT_ANKLE | (doesn't exist) |

When drawing skeleton, we accessed `joints_2d[0]` expecting PELVIS but got NOSE!

---

## debug8 Fixes

### 1. Restored cam_int (from working v0.5.0)
```python
outputs = estimator.process_one_image(
    tmp_path,
    bboxes=bbox,
    masks=mask_np,
    cam_int=cam_int_tensor,  # ← RESTORED!
    ...
)
```

### 2. Fixed SAM3DJoints to COCO format
```python
class SAM3DJoints:
    NOSE = 0
    LEFT_EYE = 1
    ...
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    HEAD = NOSE          # 0
    PELVIS = LEFT_HIP    # 11
```

### 3. Separated 2D and 3D indices
```python
# 2D indices - ALWAYS COCO (for pred_keypoints_2d)
pelvis_idx_2d = SAM3DJoints.PELVIS      # 11
left_ankle_idx_2d = SAM3DJoints.LEFT_ANKLE   # 15

# 3D indices - depends on kp_source
if kp_source == "keypoints_3d":
    pelvis_idx_3d = SAM3DJoints.PELVIS  # COCO
else:
    pelvis_idx_3d = SMPLHJoints.PELVIS  # SMPLH
```

### 4. Overlay always uses COCO indices
The overlay drawing function now ALWAYS uses SAM3DJoints (COCO) indices regardless of skeleton_mode.

---

## Expected Results (debug8)

1. **Mesh overlay**: Cyan mesh aligned with person (like v0.5.0)
2. **Skeleton dots**: ON the person's body joints (not scattered)
3. **Depth range**: ~2.8-3.2m (not doubled like debug7)
4. **Logs**: Show `v0.5.0-debug8`

---

## Key Files Changed

| File | Changes |
|------|---------|
| sam3dbody_integration.py | Restored cam_int, added autocast wrapper |
| motion_analyzer.py | Fixed SAM3DJoints to COCO, separated 2D/3D indices |
