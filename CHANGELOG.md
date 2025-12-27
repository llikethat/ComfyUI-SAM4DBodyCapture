# Changelog

All notable changes to ComfyUI-SAM4DBodyCapture will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v0.5.0
- Body joint tracking using SAM-Body4D exact method
- Improved mesh quality

---

## [0.4.1] - 2025-12-28

### Added - SAM3DBody Integration

#### BFloat16 Fix
The original SAM3DBody uses `bfloat16` which doesn't work with sparse CUDA operations:
```
RuntimeError: "addmm_sparse_cuda" not implemented for 'BFloat16'
```

**Solution:** New integrated loader forces `float16` instead of `bfloat16`:
```yaml
# Original config (broken):
TRAIN:
  FP16_TYPE: bfloat16  # ← Sparse ops fail!

# Our fix:
TRAIN:
  FP16_TYPE: float16   # ← Works!
```

#### New Nodes

- **🧍 Load SAM3DBody (Fixed)** - Loads SAM3DBody with dtype fix
  - `force_fp16=True` by default (fixes sparse matrix error)
  - Auto-downloads from HuggingFace if token provided
  - Caches model for reuse
  
- **🎬 SAM3DBody Batch Process** - Process all video frames
  - Loops through each frame calling SAM3DBody
  - Collects mesh vertices, joints, rotations per frame
  - Accepts optional camera intrinsics from MoGe2
  - Returns `MESH_SEQUENCE` type
  
- **🔄 Temporal Mesh Smoothing** - Smooth mesh sequences
  - Vertex smoothing (reduces vertex jitter)
  - Joint smoothing (smoother joint positions)  
  - Rotation smoothing (smoother joint rotations)
  - Configurable window size and strength
  
- **ℹ️ Mesh Sequence Info** - Display sequence information

#### New Data Type

- **SAM4D_MESH_SEQUENCE** - Compatible with existing export nodes:
  ```python
  {
      "vertices": [np.array, ...],      # Per-frame vertices [V, 3]
      "faces": np.array,                # Shared topology [F, 3]
      "params": {
          "joint_coords": [...],
          "joint_rotations": [...],
          "camera_t": [...],
          "focal_length": [...],
          "body_pose": [...],
          ...
      },
      "frame_count": int,
      "fps": float,
      "person_ids": [int],
      "_type": "SAM4D_MESH_SEQUENCE",
  }
  ```

### Direct Export Connection

**No intermediate nodes needed!** SAM3DBody Batch Process outputs `SAM4D_MESH_SEQUENCE` which connects directly to Export nodes:

```
[SAM3DBody Batch Process] → SAM4D_MESH_SEQUENCE → [Export Character FBX]
                                    ↓
                          [Temporal Smoothing] (optional)
```

### Full SAM-Body4D Pipeline

The complete pipeline is now possible:

```
[Load Video Frames]
        ↓
[SAM3 Video Segmentation] → Get masks for person
        ↓
[Diffusion-VAS Amodal] → Complete occluded regions
        ↓
[MoGe2 Camera] → Get camera intrinsics
        ↓
[SAM3DBody Batch Process] → Mesh per frame (with BFloat16 fix!)
        ↓
[Temporal Mesh Smoothing] → Smooth animation
        ↓
[Export Character FBX] → Animated character
```

### Technical Details

**Why BFloat16 fails:**
- SAM3DBody's MHR head uses sparse matrix operations
- PyTorch's `addmm_sparse_cuda` only supports `float16` and `float32`
- The original config sets `FP16_TYPE: bfloat16`
- Our loader overrides this to `float16`

**Sparse operations in MHR:**
```python
# In mhr_head.py, the keypoint_mapping uses sparse matrices
model_keypoints_pred = self.keypoint_mapping @ model_vert_joints
```

---

## [0.4.0] - 2025-12-28

### Added - Camera & Visualization

#### MoGe2 Camera Intrinsics
Following SAM-Body4D approach, added MoGe2 integration for camera intrinsics:

- **MoGe2 Camera Intrinsics** node - Extracts FOV and focal length from images
  - Uses MoGe2 (Microsoft's monocular geometry estimation)
  - Formula: `focal_px = (image_width / 2) / tan(fov_x * pi / 180 / 2)`
  - Supports per-frame focal length for varying zoom
  - Outputs depth maps as bonus
  
- **Camera from FOV** node - Manual intrinsics from known FOV
  - For when you know your camera's FOV (e.g., iPhone ~70°, GoPro ~120°)
  
- **Camera Info** node - Display intrinsics information

#### Mesh Overlay Visualization
Preview 3D mesh alignment on original images:

- **Mesh Overlay Preview** node
  - Wireframe, solid, or skeleton render modes
  - Adjustable color, alpha, and line thickness
  - Uses camera intrinsics for proper projection
  
- **Depth Overlay Preview** node
  - Visualize depth maps with colormaps (viridis, plasma, jet, etc.)
  - Adjustable blend alpha

#### FBX Export Improvements
- Camera intrinsics now passed to FBX export
- Camera + character exported in **same FBX file** by default
- `include_camera` now defaults to `True`
- Added `sensor_width` parameter (default 36mm for 35mm equivalent)

### Changed
- Export node accepts `CAMERA_INTRINSICS` input for proper camera setup
- `world_translation_mode` defaults to `"root"` (character at origin)

### How to Use v0.4.0

**Basic Pipeline:**
```
[Load Video] → [SAM3 Segmentation] → [Diffusion-VAS Amodal] 
                                            ↓
                    [MoGe2 Camera] → [SAM3DBody Mesh] 
                                            ↓
                                    [Mesh Overlay Preview]
                                            ↓
                                    [Export Character FBX]
```

**Verify Alignment:**
1. Run MoGe2 Camera Intrinsics on your video
2. Connect camera_intrinsics to Mesh Overlay Preview
3. Check that mesh aligns with person in video
4. If good, export to FBX with same intrinsics

**Common FOV Values:**
| Camera | Horizontal FOV |
|--------|---------------|
| iPhone (standard) | ~70° |
| GoPro (wide) | ~120° |
| DSLR 50mm | ~40° |
| Webcam | ~60-90° |

---

## [0.3.16] - 2025-12-27

### Added
- **LOW VRAM MODE** - New option in VAS Loader for GPUs with <16GB VRAM
  - Uses `enable_sequential_cpu_offload()` instead of `enable_model_cpu_offload()`
  - Much slower but uses significantly less GPU memory
  - Enabled by default for safety
  
- **Attention Slicing** - Automatically enabled to reduce memory
- **VAE Slicing** - Automatically enabled to reduce memory
- **Dynamic decode_chunk_size** - Uses 2 in low_vram mode, 4 otherwise

### Changed
- Default resolution order changed to smallest first: 256x512, 384x768, 512x1024
- `low_vram` defaults to `True` for safety
- Pipeline clears CUDA cache before each chunk

### VRAM Requirements
The Diffusion-VAS model (based on SVD) is extremely memory-intensive:

| Mode | VRAM Needed | Notes |
|------|-------------|-------|
| `low_vram=True` | ~8-10GB | Sequential CPU offload, slower |
| `low_vram=False` | ~16-24GB | Model CPU offload, faster |

### If Still Getting OOM
1. Ensure `low_vram=True` in the loader
2. Use resolution `256x512` 
3. Use `chunk_size=4` or even `2`
4. Close other GPU applications
5. The model may simply not fit - consider using a cloud GPU

### Technical Details
Sequential CPU offload moves each layer to GPU only during forward pass,
then immediately moves it back to CPU. This allows running models larger
than VRAM but is significantly slower due to constant CPU↔GPU transfers.

---

## [0.3.15] - 2025-12-27

### Changed
- **Reduced default chunk_size from 25 to 12** - More conservative default for GPUs with limited VRAM
- Added VRAM logging before each chunk processing to help diagnose memory issues

### VRAM Guidelines
The Diffusion-VAS model (based on SVD) is very memory-hungry. Use these settings based on your GPU:

| VRAM | chunk_size | Notes |
|------|------------|-------|
| 8GB | 4-6 | May still struggle, use 256x512 resolution |
| 12GB | 8-10 | Conservative |
| 16GB | 12 (default) | Safe for most cases |
| 24GB | 16-20 | Comfortable |
| 32GB+ | 20-25 | Maximum efficiency |

### If You Still Get OOM
1. Reduce `chunk_size` to 8 or even 6
2. Use lower resolution (256x512 instead of 512x1024)
3. Close other GPU applications
4. Try `overlap=0` to reduce redundant processing

### Expected Log Output
```
[VAS] Chunking: 50 frames into chunks of 12 with 4 frame overlap
[VAS] Processing chunk 0: frames 0-11 (12 frames)
[VAS] VRAM before chunk: 5.50GB allocated, 6.00GB reserved
...
```

---

## [0.3.14] - 2025-12-27

### Added
- **Overlap Blending** - Chunks now overlap for smooth temporal transitions
- **`overlap` parameter** - Adjustable overlap frames (default 4, range 0-16) for all VAS/SAM4D nodes
- Linear crossfade blending in overlap regions prevents "seams" between chunks

### How Overlap Works
With `chunk_size=25` and `overlap=4` processing 50 frames:

```
Chunk 0: frames 0-24  (25 frames)
  └── Keep: 0-20, Fade out: 21-24
  
Chunk 1: frames 21-45 (25 frames) 
  └── Fade in: 21-24, Keep: 25-41, Fade out: 42-45
  
Chunk 2: frames 42-49 (8 frames)
  └── Fade in: 42-45, Keep: 46-49

Result: Smooth blended masks across all 50 frames
```

### Blending Strategy
- **Fade in**: First `overlap` frames of each chunk (except first chunk)
- **Fade out**: Last `overlap` frames of each chunk (except last chunk)
- **Linear weights**: Crossfade using position-based weights
- **Threshold**: Final masks thresholded at 0.5 back to binary

### Parameters
| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `overlap` | 4 | 0-16 | Frames to blend between chunks |

Set `overlap=0` to disable blending (hard cuts between chunks).

---

## [0.3.13] - 2025-12-27

### Added
- **Chunked Processing** - Long videos are now automatically processed in chunks to avoid OOM
- **`chunk_size` parameter** - Adjustable chunk size (default 25, range 4-64) for all VAS/SAM4D nodes
- Users with more VRAM can increase chunk_size, those with less can decrease it

### Fixed
- **OOM on long videos** - 50+ frame videos no longer crash with "Allocation on device" error
- Processing now splits videos into chunks, processes each, and concatenates results
- GPU cache is cleared between chunks to free memory

### Technical Details
The Diffusion-VAS pipeline (based on SVD) has memory constraints. Now we:
1. Check if frame count > max_chunk_size (default 25)
2. If so, split into chunks with overlap handling
3. Process each chunk separately
4. Concatenate results
5. Clear CUDA cache between chunks

```python
# Example: 50 frames with chunk_size=25
# Chunk 1: frames 0-24 (25 frames)
# Chunk 2: frames 25-49 (25 frames)
# Results concatenated → 50 amodal masks
```

### Affected Nodes
- DiffusionVASAmodalSegmentation (new `chunk_size` param)
- SAM4DOcclusionDetector (new `chunk_size` param)
- SAM4DAmodalCompletion (new `chunk_size` param)

---

## [0.3.12] - 2025-12-27

### Fixed
- **CRITICAL: num_frames Auto-Detection** - All nodes now default to `num_frames=0` which means "auto"
- The v0.3.11 fix was incomplete - nodes were still passing hardcoded `num_frames=25` to the wrapper
- Now properly uses SAM-Body4D approach: actual frame count is used by default

### Technical Details
The bug was that nodes had `num_frames=25` as default parameter value and passed it directly:

```python
# Before (broken):
def segment(self, ..., num_frames=25, ...):
    wrapper.run_amodal_segmentation(..., num_frames=num_frames)  # Always 25!

# After (fixed):
def segment(self, ..., num_frames=0, ...):
    actual_num_frames = B if num_frames == 0 else num_frames  # Auto-detect!
    wrapper.run_amodal_segmentation(..., num_frames=actual_num_frames)
```

### Changed
- `num_frames` parameter now accepts 0-256 range (was 1-64 or 4-64)
- Default `num_frames=0` means auto (use actual frame count)
- Added tooltips explaining the auto behavior

### Affected Nodes
- DiffusionVASAmodalSegmentation
- DiffusionVASContentCompletion  
- SAM4DOcclusionDetector
- SAM4DAmodalCompletion

---

## [0.3.11] - 2025-12-27

### Fixed
- **Dynamic Frame Count** - VAS now passes actual frame count like SAM-Body4D
- Removed unnecessary chunking/overlap complexity
- Videos of any length now work correctly

### Technical Details
The bug was passing `num_frames=25` (default) while providing 50 frames of data.
SAM-Body4D solution: pass `num_frames=actual_frame_count` dynamically.

```python
# Before (wrong):
num_frames=25  # hardcoded default

# After (SAM-Body4D approach):
num_frames=modal_pixels.shape[1]  # actual frame count
```

---

## [0.3.10] - 2025-12-27

### Fixed
- **VAS Module Registration** - Fixed `'models.diffusion_vas' is not a package` error
- Now properly registers submodules with `__path__` attribute
- Registers both `models.diffusion_vas.unet_diffusion_vas` and `models.diffusion_vas.pipeline_diffusion_vas`

---

## [0.3.9] - 2025-12-27

### Fixed
- **Critical: VAS Model Loading** - Fixed `No module named 'models.diffusion_vas'` error
- Added module alias registration so HuggingFace checkpoints can find our custom UNet class
- The downloaded VAS models now load correctly

### Technical Details
The HuggingFace VAS checkpoints reference `models.diffusion_vas.UNetDiffusionVAS` but our code is in `lib/diffusion_vas/`. The fix registers a module alias before loading.

---

## [0.3.8] - 2025-12-27

### Fixed
- **Pipeline Compatibility** - SAM4D nodes now accept BOTH `SAM4D_PIPELINE` AND `VAS_PIPELINE` types
- **Unified Pipeline Support** - You can use either loader with any downstream node
- **Fixed depth output** - Corrected variable reference in occlusion detector

### Pipeline Compatibility Matrix

| Loader | Outputs | Compatible With |
|--------|---------|-----------------|
| **Load SAM4D Pipeline** | `SAM4D_PIPELINE` | All SAM4D nodes, All VAS nodes |
| **Load Diffusion-VAS** | `VAS_PIPELINE` | All SAM4D nodes, All VAS nodes |

### When to Use Which Loader

| Loader | Best For |
|--------|----------|
| **Load SAM4D Pipeline** | Full body capture workflow with occlusion handling |
| **Load Diffusion-VAS** | Standalone amodal segmentation, simpler setup |

Both loaders are now fully interchangeable!

---

## [0.3.7] - 2025-12-27

### Fixed
- **SAM4DPipelineLoader** - Fixed missing method calls (`load_amodal_pipeline`, `run_amodal_segmentation`)
- **Compatibility layer** - Added aliases in DiffusionVASWrapper for SAM4D pipeline nodes
- **hf_token support** - Added to SAM4DPipelineLoader node

### Two Pipeline Options

**Option 1: SAM4D Pipeline (Recommended for full workflow)**
```
SAM4DPipelineLoader → SAM4DOcclusionDetector → SAM4DAmodalCompletion
     (SAM4D_PIPELINE)    ↓                          ↓
                    occlusion_info          completed masks/images
```

**Option 2: Standalone VAS (For amodal segmentation only)**
```
DiffusionVASLoader → DiffusionVASAmodalSegmentation
    (VAS_PIPELINE)         ↓
                      amodal_masks
```

### When to Use Which

| Use Case | Pipeline |
|----------|----------|
| Full body capture with occlusion handling | SAM4D Pipeline |
| Just amodal mask generation | Standalone VAS |
| Testing/debugging VAS models | Standalone VAS |

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
