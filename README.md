# ComfyUI-SAM4DBodyCapture

**Temporally Consistent 4D Human Mesh Recovery from Videos**

A ComfyUI package integrating SAM-Body4D and Diffusion-VAS for robust human body capture with occlusion handling, temporal smoothing, and mesh export.

[![Version](https://img.shields.io/badge/version-0.4.1-blue.svg)](https://github.com/llikethat/ComfyUI-SAM4DBodyCapture/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Features

### v0.4.1 - SAM3DBody Integration (BFloat16 Fix!)
- 🧍 **SAM3DBody Integration** - Full SAM-Body4D pipeline in ComfyUI
- 🔧 **BFloat16 Fix** - Solves sparse matrix CUDA error
- 🎬 **Batch Video Processing** - Process all frames through SAM3DBody
- 🔄 **Temporal Smoothing** - Smooth mesh animation jitter
- 📷 **MoGe2 → SAM3DBody** - Camera intrinsics properly passed

### v0.4.0 - Camera & Visualization
- 📷 **MoGe2 Camera** - Extract FOV/focal from images
- 👁️ **Mesh Overlay** - Preview 3D mesh on video
- 🎥 **Camera in FBX** - Character + camera same file

### v0.3.x - Export & Chunked Processing
- 📦 **FBX Export** - Character meshes for Maya, Blender, Unreal, Unity
- 📦 **Alembic Export** - Point cache for VFX pipelines
- 🔄 **Chunked Processing** - Handle long videos without OOM
- 🎛️ **Low VRAM Mode** - For GPUs with <16GB

### v0.2.0 - SAM4D Pipeline
- 🎬 **Complete Pipeline** - Occlusion detection and completion
- 🔍 **Smart Occlusion Detection** - IoU-based identification  
- 🎭 **Amodal Completion** - Recover complete masks using diffusion priors

## ⭐ Complete SAM-Body4D Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COMPLETE SAM-BODY4D PIPELINE                      │
└─────────────────────────────────────────────────────────────────────┘

[Load Video Frames]
        │
        ▼
┌───────────────────────┐
│ SAM3 Video Segmenter  │ ← Get identity-consistent masks
│ (external ComfyUI)    │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ Diffusion-VAS Amodal  │ ← Complete occluded body parts
│ (this package)        │
└───────────────────────┘
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
┌───────────────────────┐          ┌───────────────────────┐
│ MoGe2 Camera          │          │ SAM3DBody (Fixed)     │
│ Intrinsics            │ ──────── │ Batch Process         │
│ (this package)        │  cam_int │ (this package)        │
└───────────────────────┘          └───────────────────────┘
                                              │
                                              ▼
                                   ┌───────────────────────┐
                                   │ Temporal Smoothing    │
                                   │ (this package)        │
                                   └───────────────────────┘
                                              │
                                              ▼
                                   ┌───────────────────────┐
                                   │ Export Character FBX  │
                                   │ (this package)        │
                                   └───────────────────────┘
                                              │
                                              ▼
                                        📁 animated.fbx
```

## 📦 Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/llikethat/ComfyUI-SAM4DBodyCapture.git
cd ComfyUI-SAM4DBodyCapture
pip install -r requirements.txt
```

### Dependencies
- **ComfyUI-SAM3DBody** - Required for mesh generation
- **MoGe2** (optional) - For camera intrinsics: `pip install git+https://github.com/microsoft/MoGe.git`

### Model Downloads (Automatic)
Models download from HuggingFace on first use:

| Model | Size | Purpose |
|-------|------|---------|
| SAM3DBody | ~2GB | 3D mesh from images |
| Depth-Anything-V2-Large | ~700MB | Depth estimation |
| diffusion-vas-amodal-segmentation | ~2GB | Amodal mask prediction |
| diffusion-vas-content-completion | ~2GB | RGB inpainting (optional) |

## 🔧 Nodes (21 Total)

### SAM3DBody Integration (NEW in v0.4.1!)

| Node | Description |
|------|-------------|
| 🧍 **Load SAM3DBody (Fixed)** | Load SAM3DBody with BFloat16→Float16 fix |
| 🎬 **SAM3DBody Batch Process** | Process video frames → mesh sequence |
| 🔄 **Temporal Mesh Smoothing** | Smooth mesh vertices/joints over time |
| ℹ️ **Mesh Sequence Info** | Display mesh sequence information |

### Camera Nodes (v0.4.0)

| Node | Description |
|------|-------------|
| 📷 **MoGe2 Camera Intrinsics** | Extract FOV/focal from images |
| 📷 **Camera from FOV** | Manual intrinsics from known FOV |
| 📷 **Camera Info** | Display camera intrinsics |

### Visualization Nodes (v0.4.0)

| Node | Description |
|------|-------------|
| 👁️ **Mesh Overlay Preview** | Render 3D mesh on video |
| 👁️ **Depth Overlay Preview** | Visualize depth maps |

### SAM4D Pipeline Nodes

| Node | Description |
|------|-------------|
| 🎬 **Load SAM4D Pipeline** | Load all models (depth, amodal, completion) |
| 🔍 **Detect Occlusions** | Find frames with occluded body parts |
| 🎭 **Complete Occluded Regions** | Fill in missing mask/RGB regions |
| 🗑️ **Unload SAM4D Pipeline** | Free GPU memory |

### Temporal & Mesh Nodes

| Node | Description |
|------|-------------|
| 🔄 **Temporal Fusion** | Smooth vertex/parameter jitter |
| ✨ **Create Mesh Sequence** | Build sequence from SAM3DBody output |
| 👁️ **Visualize Mesh Sequence** | Preview mesh as point cloud |
| 📦 **Export Mesh Sequence** | NPZ compressed format |

### Export Nodes

| Node | Description |
|------|-------------|
| 📦 **Export Character FBX** | Animated FBX via Blender |
| 📦 **Export Character Alembic** | Point cache for VFX pipelines |
| 🎥 **Export Camera FBX** | Camera animation FBX |
| 🎥 **Export Camera JSON** | Universal camera format |

### Diffusion-VAS Nodes (Standalone)

| Node | Description |
|------|-------------|
| 🎭 **Load Diffusion-VAS Models** | Load VAS models only |
| 🎭 **Amodal Segmentation** | Generate complete masks |
| 🎭 **Content Completion** | Inpaint occluded RGB |
| 🎭 **Unload VAS Models** | Free VAS memory |

## 🔧 BFloat16 Fix (Why This Package Exists!)

The original **ComfyUI-SAM3DBody** crashes with this error:

```
RuntimeError: "addmm_sparse_cuda" not implemented for 'BFloat16'
```

**Root Cause:** SAM3DBody's config uses `bfloat16`, but PyTorch's sparse CUDA operations only support `float16`/`float32`.

**Our Fix:** The `Load SAM3DBody (Fixed)` node overrides the dtype:

```yaml
# Original SAM3DBody config (broken):
TRAIN:
  FP16_TYPE: bfloat16  # ← Causes sparse matrix error!

# Our fix:
TRAIN:
  FP16_TYPE: float16   # ← Works!
```

This is the ONLY change needed - everything else works identically.

## 📷 MoGe2 → SAM3DBody Connection

SAM-Body4D uses MoGe2 to estimate camera intrinsics, which improves 3D reconstruction accuracy:

```
┌─────────────────────┐
│ MoGe2 Camera        │
│ Intrinsics          │
└─────────────────────┘
         │
         │  CAMERA_INTRINSICS dict:
         │  {
         │    "focal_length": 1234.5,
         │    "fov_x": 65.0,
         │    "cx": 960.0,
         │    "cy": 540.0,
         │    "per_frame_focal": [...]
         │  }
         │
         ▼
┌─────────────────────┐
│ SAM3DBody Batch     │ ← Converts to 3x3 intrinsic matrix:
│ Process             │   [[fx, 0, cx],
└─────────────────────┘    [0, fy, cy],
                           [0,  0,  1]]
```

Without MoGe2, SAM3DBody uses a default FOV (~60°), which may not match your camera.

## 🔀 Two Loader Options (Fully Compatible)

Both loaders can be used with any downstream node!

### Option 1: SAM4D Pipeline Loader (Recommended)
For full body capture with occlusion handling and temporal processing.

```
┌─────────────────────┐
│ Load SAM4D Pipeline │──────────────────────────────────────┐
│ (SAM4D_PIPELINE or  │                                      │
│  VAS_PIPELINE)      │                                      │
└─────────────────────┘                                      │
         │                                                   │
         ▼                                                   ▼
┌─────────────────────┐     ┌─────────────────────────────────────┐
│  Detect Occlusions  │────▶│   Complete Occluded Regions         │
│     + depth_maps    │     │   (only processes occluded frames)  │
└─────────────────────┘     └─────────────────────────────────────┘
         │                               │
         ▼                               ▼
  occlusion_info              completed_masks, completed_images
```

### Option 2: Standalone VAS Loader
For amodal segmentation only (simpler setup).

```
┌─────────────────────────┐
│ Load Diffusion-VAS      │
│ (SAM4D_PIPELINE or      │
│  VAS_PIPELINE)          │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Amodal Segmentation    │
│     + depth_maps        │
└─────────────────────────┘
         │
         ▼
    amodal_masks
```

### Compatibility Matrix

| Loader | Can Connect To |
|--------|----------------|
| Load SAM4D Pipeline | ✅ All SAM4D nodes, ✅ All VAS nodes |
| Load Diffusion-VAS | ✅ All SAM4D nodes, ✅ All VAS nodes |

### When to Use Which

| Use Case | Loader |
|----------|--------|
| Full body capture workflow | **SAM4D Pipeline** |
| Just amodal mask generation | Either |
| Testing VAS models | **Diffusion-VAS** |
| Integration with SAM3DBody | **SAM4D Pipeline** |

## 🔄 Workflow

### Complete Pipeline
```
┌──────────────────────────────────────────────────────────────────┐
│                     SAM4D Pipeline Workflow                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   Video ─────► SAM3 ─────► Load SAM4D Pipeline                   │
│                  │                  │                             │
│                  ▼                  ▼                             │
│              Masks ─────► Detect Occlusions                      │
│                                │                                  │
│                    ┌───────────┴───────────┐                     │
│                    ▼                       ▼                      │
│              Occluded?              Not Occluded                 │
│                    │                       │                      │
│                    ▼                       │                      │
│         Complete Occluded ─────────────────┤                     │
│                    │                       │                      │
│                    └───────────┬───────────┘                     │
│                                ▼                                  │
│                          SAM3DBody                               │
│                                │                                  │
│                                ▼                                  │
│                       Temporal Fusion                            │
│                                │                                  │
│                                ▼                                  │
│                      Export Mesh Sequence                        │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Node Details

#### 🎬 Load SAM4D Pipeline
```
Inputs:
├── depth_model: Large | Base | Small
├── resolution: 512x1024 | 256x512 | 384x768
├── enable_amodal: bool (default: True)
├── enable_completion: bool (default: False)
├── device: cuda | cpu | auto
└── dtype: float16 | float32

Output:
└── SAM4D_PIPELINE
```

#### 🔍 Detect Occlusions
```
Inputs:
├── pipeline: SAM4D_PIPELINE
├── images: IMAGE (video frames)
├── masks: MASK (from SAM3)
├── iou_threshold: 0.7 (frames below this are occluded)
├── object_ids: "1" or "1,2,3" (multi-person)
└── num_frames: 25

Outputs:
├── occlusion_info: SAM4D_OCCLUSION_INFO
├── depth_maps: IMAGE
└── amodal_masks: MASK
```

#### 🔄 Temporal Fusion
```
Inputs:
├── mesh_sequence: SAM4D_MESH_SEQUENCE
├── method: gaussian | ema | none
├── smoothing_strength: 1.0 (higher = smoother)
├── smooth_vertices: bool
└── smooth_params: bool

Output:
└── smoothed_sequence: SAM4D_MESH_SEQUENCE
```

## 💻 Requirements

### Hardware
| Level | VRAM | Notes |
|-------|------|-------|
| Minimum | 12GB | Depth only, CPU fallback |
| Recommended | 16GB | Full pipeline |
| Optimal | 24GB | All features + high resolution |

### Software
- Python 3.10+
- CUDA 11.8+
- ComfyUI (latest)
- PyTorch 2.0+

## 📜 License

MIT License - see [LICENSE](LICENSE)

### Third-Party Components
| Component | License | Notes |
|-----------|---------|-------|
| SAM-Body4D | MIT | ✅ Commercial OK |
| Diffusion-VAS | MIT | ✅ Commercial OK |
| SAM 3D Body | SAM License | ⚠️ No military/nuclear |
| Depth Anything V2 | Apache 2.0 | ✅ Commercial OK |
| SVD (in VAS) | Stability AI | ⚠️ Check if revenue >$1M |

## 🗺️ Roadmap

- [x] v0.1.0 - Diffusion-VAS skeleton
- [x] v0.1.1 - Diffusion-VAS with depth
- [x] v0.2.0 - SAM4D pipeline integration
- [x] v0.3.0 - FBX/Alembic export
- [x] v0.3.1 - SAM3D_OUTPUT compatibility
- [x] v0.3.2 - Blender FBX export, FBX viewer
- [x] v0.3.3 - External depth input support
- [x] v0.3.4 - Import fixes
- [x] v0.3.5 - Checkpoint manager (auto model downloads)
- [x] v0.3.6 - HuggingFace token support
- [x] v0.3.7 - Pipeline compatibility fixes
- [x] v0.3.8 - Unified pipeline types (SAM4D/VAS interchangeable)
- [x] v0.3.9 - VAS model loading fix (module alias)
- [x] v0.3.10 - VAS module registration fix
- [x] v0.3.11 - Chunked processing for long videos
- [x] v0.3.12 - Fixed num_frames auto-detection (SAM-Body4D approach)
- [x] v0.3.13 - Chunked processing with adjustable chunk_size (OOM fix)
- [x] v0.3.14 - Overlap blending for smooth chunk transitions
- [x] v0.3.15 - Reduced default chunk_size to 12, added VRAM logging
- [x] v0.3.16 - Low VRAM mode with sequential CPU offload
- [x] v0.4.0 - MoGe2 camera intrinsics, mesh overlay visualization
- [ ] v0.5.0 - Body joint tracking using SAM-Body4D method
- [ ] v1.0.0 - First stable release  
- [ ] v1.0.0 - Stable release

## 🙏 Acknowledgments

- [SAM-Body4D](https://github.com/gaomingqi/sam-body4d) - Mingqi Gao et al.
- [Diffusion-VAS](https://github.com/Kaihua-Chen/diffusion-vas) - Kaihua Chen et al.
- [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) - Meta AI
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)

## 📚 Citation

```bibtex
@article{gao2025sambody4d,
    title={SAM-Body4D: Training-Free 4D Human Body Mesh Recovery from Videos},
    author={Gao, Mingqi and others},
    journal={arXiv:2512.08406},
    year={2025}
}

@inproceedings{chen2025diffvas,
    title={Using Diffusion Priors for Video Amodal Segmentation},
    author={Chen, Kaihua and others},
    booktitle={CVPR},
    year={2025}
}
```

## 📞 Support

- [Issues](https://github.com/llikethat/ComfyUI-SAM4DBodyCapture/issues)
- [Discussions](https://github.com/llikethat/ComfyUI-SAM4DBodyCapture/discussions)
