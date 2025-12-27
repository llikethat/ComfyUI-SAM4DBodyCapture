# ComfyUI-SAM4DBodyCapture

**Temporally Consistent 4D Human Mesh Recovery from Videos**

A ComfyUI package integrating SAM-Body4D and Diffusion-VAS for robust human body capture with camera and character export capabilities.

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/llikethat/ComfyUI-SAM4DBodyCapture/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Features

### Current (v0.1.0)
- 🎭 **Diffusion-VAS Integration** - Video Amodal Segmentation
  - Recover complete object masks even when heavily occluded
  - Inpaint occluded regions using diffusion priors
  - Depth-conditioned temporal modeling

### Planned
- 🏂 **SAM-Body4D** (v0.2.0) - Temporally consistent mesh recovery
- 📦 **Export Nodes** (v0.3.0) - FBX/Alembic for character and camera
- 🎥 **Camera Integration** (v0.4.0) - Integration with SAM3DBody2abc camera solver

## 📦 Installation

### From GitHub (Recommended)
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/llikethat/ComfyUI-SAM4DBodyCapture.git
cd ComfyUI-SAM4DBodyCapture
pip install -r requirements.txt
```

### Model Downloads
Models are automatically downloaded from HuggingFace on first use:

| Model | Size | Purpose |
|-------|------|---------|
| diffusion-vas-amodal-segmentation | ~2GB | Amodal mask generation |
| diffusion-vas-content-completion | ~2GB | Occluded region inpainting |
| Depth-Anything-V2-Large | ~700MB | Pseudo-depth estimation |

## 🔧 Nodes

### Diffusion-VAS Nodes

#### 🎭 Load Diffusion-VAS Models
Load all required models for amodal segmentation.

| Parameter | Description |
|-----------|-------------|
| depth_model | Depth Anything V2 variant (Large/Base/Small) |
| device | cuda, cpu, or auto |
| dtype | float16, bfloat16, or float32 |

#### 🎭 Amodal Segmentation
Generate complete object masks from partial/occluded views.

| Input | Type | Description |
|-------|------|-------------|
| vas_model | DIFFUSION_VAS_MODEL | From loader node |
| images | IMAGE | Video frames |
| modal_masks | MASK | Visible masks from SAM3 |

| Output | Type | Description |
|--------|------|-------------|
| amodal_masks | MASK | Complete object masks |
| depth_maps | IMAGE | Pseudo-depth visualization |

#### 🎭 Content Completion
Inpaint occluded regions of objects.

| Input | Type | Description |
|-------|------|-------------|
| vas_model | DIFFUSION_VAS_MODEL | From loader node |
| images | IMAGE | Video frames |
| modal_masks | MASK | Visible masks |
| amodal_masks | MASK | Complete masks |

| Output | Type | Description |
|--------|------|-------------|
| completed_content | IMAGE | Frames with inpainted regions |

## 🔄 Workflow

```
┌─────────────┐     ┌──────────────┐     ┌────────────────────┐
│ Load Video  │────▶│    SAM 3     │────▶│  Diffusion-VAS     │
│             │     │ (Segmentation)│     │ (Amodal + Complete)│
└─────────────┘     └──────────────┘     └─────────┬──────────┘
                                                   │
                           ┌───────────────────────┘
                           ▼
                    ┌──────────────┐     ┌────────────────────┐
                    │ SAM 3D Body  │────▶│  Export FBX/ABC    │
                    │ (Per-frame)  │     │ (Character+Camera) │
                    └──────────────┘     └────────────────────┘
```

## 💻 Requirements

### Hardware
- **Minimum**: NVIDIA GPU with 12GB VRAM
- **Recommended**: 16-24GB VRAM
- CPU mode available but slow

### Software
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- ComfyUI (latest)

## 📜 License

This project is licensed under MIT.

### Third-Party Licenses
| Component | License | Commercial Use |
|-----------|---------|----------------|
| SAM-Body4D | MIT | ✅ Allowed |
| Diffusion-VAS | MIT | ✅ Allowed |
| SAM 3D Body | SAM License | ⚠️ Restricted (no military/nuclear) |
| Stable Video Diffusion | Stability AI | ⚠️ Check for >$1M revenue |
| Depth Anything V2 | Apache 2.0 | ✅ Allowed |

## 🙏 Acknowledgments

This project builds upon:
- [SAM-Body4D](https://github.com/gaomingqi/sam-body4d) by Mingqi Gao et al.
- [Diffusion-VAS](https://github.com/Kaihua-Chen/diffusion-vas) by Kaihua Chen et al.
- [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) by Meta AI

## 📚 Citations

If you use this work, please cite:

```bibtex
@article{gao2025sambody4d,
    title={SAM-Body4D: Training-Free 4D Human Body Mesh Recovery from Videos},
    author={Gao, Mingqi and Miao, Yunqi and Han, Jungong},
    journal={arXiv preprint arXiv:2512.08406},
    year={2025}
}

@inproceedings{chen2025diffvas,
    title={Using Diffusion Priors for Video Amodal Segmentation},
    author={Chen, Kaihua and Ramanan, Deva and Khurana, Tarasha},
    booktitle={CVPR},
    year={2025}
}
```

## 🗺️ Roadmap

- [x] v0.1.0 - Diffusion-VAS nodes (skeleton)
- [ ] v0.1.1 - Complete Diffusion-VAS implementation
- [ ] v0.2.0 - SAM-Body4D integration
- [ ] v0.3.0 - FBX/Alembic export
- [ ] v0.4.0 - Camera solver integration
- [ ] v1.0.0 - First stable release

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## 📞 Support

- Issues: [GitHub Issues](https://github.com/llikethat/ComfyUI-SAM4DBodyCapture/issues)
- Discussions: [GitHub Discussions](https://github.com/llikethat/ComfyUI-SAM4DBodyCapture/discussions)
