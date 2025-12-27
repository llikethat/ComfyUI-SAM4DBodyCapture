# Changelog

All notable changes to ComfyUI-SAM4DBodyCapture will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v0.3.0
- FBX export (skinned mesh with animation)
- Alembic export (point cache)
- Coordinate system presets

### Planned for v0.4.0
- Integration with SAM3DBody2abc v4.0.0 camera solver
- Hybrid workflow combining SAM4D with external camera solves

---

## [0.2.0] - 2025-12-27

### Added

#### SAM4D Pipeline Nodes
- **🎬 Load SAM4D Pipeline** - Central loader for all models
  - Depth Anything V2 (Large/Base/Small)
  - Diffusion-VAS amodal segmentation (optional)
  - Diffusion-VAS content completion (optional)
  - Configurable resolution (512x1024, 256x512, 384x768)

- **🔍 Detect Occlusions** - IoU-based occlusion detection
  - Compares modal vs predicted amodal masks
  - Per-object occlusion tracking
  - Configurable IoU threshold
  - Returns occlusion ranges for targeted completion

- **🎭 Complete Occluded Regions** - Selective amodal completion
  - Only processes frames marked as occluded
  - Optional RGB content completion
  - Margin expansion for smooth transitions

- **🗑️ Unload SAM4D Pipeline** - Memory management

#### Temporal Fusion & Mesh Nodes
- **🔄 Temporal Fusion** - Smooth mesh sequences
  - Gaussian filter smoothing
  - Bidirectional EMA smoothing
  - Separate vertex/parameter smoothing
  - Configurable strength

- **📦 Export Mesh Sequence** - Export to various formats
  - OBJ sequence (per-frame files)
  - NPZ (compressed numpy archive)
  - Coordinate system transforms (Maya/Blender/Unreal/Unity)

- **✨ Create Mesh Sequence** - Build sequences from frames
  - Collect per-frame mesh outputs
  - Support for multi-person tracking

- **👁️ Visualize Mesh Sequence** - Preview rendering
  - Simple orthographic projection
  - Point cloud visualization

#### Data Types
- `SAM4D_PIPELINE` - Pipeline configuration container
- `SAM4D_OCCLUSION_INFO` - Per-object occlusion tracking
- `SAM4D_MESH_SEQUENCE` - Temporal mesh container

### Technical Details
- Occlusion detection uses IoU < 0.7 threshold by default
- Largest connected component filtering for mask cleanup
- Bidirectional EMA for symmetric temporal smoothing
- Support for multi-person scenes with object ID tracking

---

## [0.1.1] - 2025-12-26

### Added
- **Complete Diffusion-VAS Pipeline Integration**
  - Correct pipeline signature: `pipeline(modal_input, condition, height, width, num_frames, ...)`
  - Proper preprocessing matching demo.py (resize, normalize to [-1,1], 3-channel expansion)
  - Proper postprocessing (threshold at 600, union with modal, resize back)
- **DiffusionVASWrapper** class with full lifecycle management
- **DepthEstimator** class supporting:
  - HuggingFace transformers Depth Anything V2 (Large/Base/Small)
  - Gradient-based fallback when model unavailable
- **Resolution selection** in loader node (512x1024, 256x512, 384x768)
- **Memory management**: `🎭 Unload VAS Models` node
- Configurable inference parameters (num_frames, guidance_scale, seed)

### Changed
- Pipeline now uses `trust_remote_code=True` for custom DiffusionVASPipeline
- Amodal segmentation takes depth as condition (not images)
- Content completion takes amodal masks as condition
- Proper tensor format: [1, B, 3, H, W] for pipeline input

### Technical Details
From demo.py analysis:
- Amodal: `pipeline_mask(modal_masks, depth_maps, ...)` → amodal masks
- Completion: `pipeline_rgb(modal_rgb, amodal_masks, ...)` → completed RGB
- Default parameters: fps=8, motion_bucket_id=127, noise_aug_strength=0.02

### Notes
- Requires diffusion-vas models from HuggingFace (auto-download)
- Model loading may fail if HF format incompatible - graceful fallback provided
- Depth estimation works independently and always available

---

## [0.1.0] - 2025-12-26

### Added
- Initial project structure
- **Diffusion-VAS Nodes** (skeleton implementation):
  - `🎭 Load Diffusion-VAS Models` - Model loading with auto-download
  - `🎭 Amodal Segmentation` - Generate complete masks from occluded views
  - `🎭 Content Completion` - Inpaint occluded regions
- Depth Anything V2 integration for pseudo-depth estimation
- Comprehensive documentation (README, LICENSE, requirements)
- MIT license with third-party acknowledgments

### Notes
- This is a skeleton release - actual Diffusion-VAS pipeline integration pending
- Nodes load and validate inputs but return placeholder outputs
- Full implementation requires vendor code from diffusion-vas repository

---

## Version Format

- **Major** (X.0.0): Breaking changes, stable release milestones
- **Minor** (0.X.0): New features, backwards compatible
- **Patch** (0.0.X): Bug fixes, small improvements

## GitHub Workflow

1. Development happens in feature branches
2. Versions are tagged (e.g., `v0.1.0`)
3. Tags can be downloaded as releases
4. Stable versions merged to `main` branch
