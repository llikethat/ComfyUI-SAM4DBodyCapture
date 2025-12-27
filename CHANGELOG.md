# Changelog

All notable changes to ComfyUI-SAM4DBodyCapture will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v0.4.0
- Integration with SAM3DBody2abc v4.0.0 camera solver
- Hybrid workflow combining SAM4D with external camera solves

---

## [0.3.6] - 2025-12-27

### Added
- **`hf_token` input** on VAS Loader node - Enter HuggingFace API token directly
- **`--token` CLI argument** for checkpoint_manager.py
- Token status shown in checkpoint status output

### How to Get Token
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is sufficient)
3. Paste in the `hf_token` field in VAS Loader node

### Usage Options
```bash
# Option 1: In ComfyUI node
# Paste token in 'hf_token' field

# Option 2: CLI with token
python checkpoint_manager.py --download all --token hf_xxx

# Option 3: Environment variable
export HF_TOKEN=hf_xxx
python checkpoint_manager.py --download all
```

---

## [0.3.5] - 2025-12-27

### Added
- **Checkpoint Manager** (`checkpoint_manager.py`) - Automatic model download and caching
- **`checkpoints/` folder** - Models downloaded here automatically on first use
- **`auto_download` option** - Toggle automatic model downloads in VAS Loader node
- **CLI for model management**: `python checkpoint_manager.py --download all`

### Changed
- Models now cached locally instead of re-downloading from HuggingFace each time
- Added `huggingface_hub` to requirements.txt

### Model Sizes
| Model | Size |
|-------|------|
| Depth-Anything-V2 Large | ~1.4GB |
| Diffusion-VAS Amodal | ~5GB |
| Diffusion-VAS Completion | ~5GB |

---

## [0.3.4] - 2025-12-27

### Fixed
- **Import errors** - Fixed `DiffusionVASUnloader` class name mismatch
- **Backward compatibility** - Added `DepthEstimator` class and function aliases for sam4d_pipeline.py
- All 17 nodes now load correctly

---

## [0.3.3] - 2025-12-27

### Added
- **External depth input** - Both `SAM4DOcclusionDetector` and `DiffusionVASAmodalSegmentation` now accept optional `depth_maps` input
- Connect any ComfyUI depth node: **Depth-Anything-V2**, **DepthCrafter**, **ZoeDepth**, etc.

### Changed
- Depth is now optional - uses gradient fallback if not provided
- More flexible pipeline: use your preferred depth estimation method

### Usage
```
[Depth-Anything-V2] → depth_maps → [SAM4D Occlusion Detector]
```

---

## [0.3.2] - 2025-12-27

### Added
- **🎥 FBX Animation Viewer** node - Interactive 3D viewer with play/pause controls
- **Bundled Diffusion-VAS** code in `lib/diffusion_vas/` for local pipeline loading
- **Blender integration** - Uses Blender from SAM3DBody for animated FBX export

### Fixed
- **FBX Export** now uses Blender via SAM3DBody for proper animated export
- **Output paths** now use ComfyUI's output directory (`folder_paths.get_output_directory()`)
- **Depth estimation** gracefully falls back to gradient when models unavailable

### Changed
- Simplified VAS nodes (removed depth model selection - uses gradient fallback)
- FBX export now takes `filename` parameter instead of full path

### Technical Details
- Blender path search: SAM3DBody bundled → workspace → system
- VAS pipeline requires local code (HuggingFace models need custom UNet)
- Web extension for FBX viewer with Three.js

---

## [0.3.1] - 2025-12-27

### Fixed
- **Create Mesh Sequence** now uses correct `SAM3D_OUTPUT` type (was MESH_DATA)
- Properly extracts `joint_coords`, `joint_rotations`, `camera` from SAM3DBody output

### Changed
- Simplified coordinate systems to only Y-up (Maya/Blender) and Z-up (Unreal)
- Removed OBJ sequence export (FBX and Alembic are better supported)

### Node Count: 16 Total
- Export nodes: 4 (removed OBJ sequence)
- Pipeline nodes: 4
- Mesh nodes: 4
- Diffusion-VAS nodes: 4

---

## [0.3.0] - 2025-12-27

### Added

#### Export Nodes
- **📦 Export Character FBX** - ASCII FBX 7.4 export
  - Compatible with Maya, Blender, Unreal, Unity, 3ds Max
  - Coordinate system presets for all major DCCs
  - Base mesh with topology

- **📦 Export Character Alembic** - Point cache export
  - Full vertex animation support
  - Ideal for VFX pipelines (Houdini, Nuke)
  - Requires: `pip install alembic` (fallback to OBJ if unavailable)

- **📦 Export Character OBJ Sequence** - Per-frame OBJ files
  - Most compatible format
  - Works with all 3D software

- **🎥 Export Camera FBX** - Camera animation export
  - Position, rotation, focal length

- **🎥 Export Camera JSON** - Universal camera format
  - For custom importers and web viewers

#### Coordinate System Support
- Y-up (Maya/Blender) - default
- Z-up (Unreal) - with 100x scale for cm
- Y-up (Unity) - left-handed
- Y-up (Houdini)
- Y-up (Nuke)

### Changed

#### Create Mesh Sequence Node
- Now accepts `mesh_data` input directly from SAM3DBody nodes
- Auto-extracts vertices, faces, and SMPL parameters
- Supports dict, tuple, or object formats
- No required inputs - all optional for flexibility

### Technical Details
- FBX export uses ASCII format (no binary SDK required)
- Alembic export via PyAlembic (optional dependency)
- Coordinate transforms apply rotation and scale
- SMPL params preserved through pipeline

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
