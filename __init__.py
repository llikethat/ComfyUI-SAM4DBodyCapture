"""
ComfyUI-SAM4DBodyCapture

Temporally consistent 4D human mesh recovery from videos.
Integrates SAM-Body4D and Diffusion-VAS for ComfyUI.

GitHub: https://github.com/llikethat/ComfyUI-SAM4DBodyCapture

Version History:
- v0.1.0: Initial release - Diffusion-VAS skeleton
- v0.1.1: Diffusion-VAS with depth estimation
- v0.2.0: SAM-Body4D pipeline integration
- v0.3.0: Export nodes (FBX, Alembic, OBJ)
- v0.4.0: (Planned) Camera solver integration
- v1.0.0: (Planned) First stable release
"""

__version__ = "0.3.3"
__author__ = "llikethat"
__license__ = "MIT"

import os
import sys

# Add lib to path for vendored dependencies
_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
if _lib_path not in sys.path:
    sys.path.insert(0, _lib_path)

# Node imports with error handling
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ==================== Diffusion-VAS Nodes ====================
try:
    from .nodes import diffusion_vas
    
    NODE_CLASS_MAPPINGS.update({
        "SAM4D_DiffusionVASLoader": diffusion_vas.DiffusionVASLoader,
        "SAM4D_DiffusionVASAmodalSegmentation": diffusion_vas.DiffusionVASAmodalSegmentation,
        "SAM4D_DiffusionVASContentCompletion": diffusion_vas.DiffusionVASContentCompletion,
        "SAM4D_DiffusionVASUnload": diffusion_vas.DiffusionVASUnload,
    })
    
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "SAM4D_DiffusionVASLoader": "🎭 Load Diffusion-VAS Models",
        "SAM4D_DiffusionVASAmodalSegmentation": "🎭 Amodal Segmentation (VAS)",
        "SAM4D_DiffusionVASContentCompletion": "🎭 Content Completion (VAS)",
        "SAM4D_DiffusionVASUnload": "🎭 Unload VAS Models",
    })
    
    print(f"[SAM4DBodyCapture] Diffusion-VAS nodes loaded")
    
except ImportError as e:
    print(f"[SAM4DBodyCapture] Diffusion-VAS nodes not available: {e}")
except Exception as e:
    print(f"[SAM4DBodyCapture] Error loading Diffusion-VAS: {e}")

# ==================== SAM4D Pipeline Nodes (v0.2.0+) ====================
try:
    from .nodes import sam4d_pipeline
    
    NODE_CLASS_MAPPINGS.update({
        "SAM4D_PipelineLoader": sam4d_pipeline.SAM4DPipelineLoader,
        "SAM4D_OcclusionDetector": sam4d_pipeline.SAM4DOcclusionDetector,
        "SAM4D_AmodalCompletion": sam4d_pipeline.SAM4DAmodalCompletion,
        "SAM4D_PipelineUnload": sam4d_pipeline.SAM4DPipelineUnload,
    })
    
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "SAM4D_PipelineLoader": "🎬 Load SAM4D Pipeline",
        "SAM4D_OcclusionDetector": "🔍 Detect Occlusions",
        "SAM4D_AmodalCompletion": "🎭 Complete Occluded Regions",
        "SAM4D_PipelineUnload": "🗑️ Unload SAM4D Pipeline",
    })
    
    print(f"[SAM4DBodyCapture] SAM4D Pipeline nodes loaded")
    
except ImportError as e:
    print(f"[SAM4DBodyCapture] SAM4D Pipeline nodes not available: {e}")
except Exception as e:
    print(f"[SAM4DBodyCapture] Error loading SAM4D Pipeline: {e}")

# ==================== Temporal Fusion & Mesh Nodes (v0.2.0+) ====================
try:
    from .nodes import temporal_fusion
    
    NODE_CLASS_MAPPINGS.update({
        "SAM4D_TemporalFusion": temporal_fusion.SAM4DTemporalFusion,
        "SAM4D_ExportMeshSequence": temporal_fusion.SAM4DExportMeshSequence,
        "SAM4D_CreateMeshSequence": temporal_fusion.SAM4DCreateMeshSequence,
        "SAM4D_VisualizeMeshSequence": temporal_fusion.SAM4DVisualizeMeshSequence,
    })
    
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "SAM4D_TemporalFusion": "🔄 Temporal Fusion",
        "SAM4D_ExportMeshSequence": "📦 Export Mesh Sequence",
        "SAM4D_CreateMeshSequence": "✨ Create Mesh Sequence",
        "SAM4D_VisualizeMeshSequence": "👁️ Visualize Mesh Sequence",
    })
    
    print(f"[SAM4DBodyCapture] Temporal Fusion & Mesh nodes loaded")
    
except ImportError as e:
    print(f"[SAM4DBodyCapture] Temporal Fusion nodes not available: {e}")
except Exception as e:
    print(f"[SAM4DBodyCapture] Error loading Temporal Fusion: {e}")

# ==================== Export Nodes (v0.3.0+) ====================
try:
    from .nodes import export_nodes
    
    NODE_CLASS_MAPPINGS.update({
        "SAM4D_ExportCharacterFBX": export_nodes.SAM4DExportCharacterFBX,
        "SAM4D_ExportCharacterAlembic": export_nodes.SAM4DExportCharacterAlembic,
        "SAM4D_ExportCameraFBX": export_nodes.SAM4DExportCameraFBX,
        "SAM4D_ExportCameraJSON": export_nodes.SAM4DExportCameraJSON,
        "SAM4D_FBXViewer": export_nodes.SAM4DFBXViewer,
    })
    
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "SAM4D_ExportCharacterFBX": "📦 Export Character FBX",
        "SAM4D_ExportCharacterAlembic": "📦 Export Character Alembic",
        "SAM4D_ExportCameraFBX": "🎥 Export Camera FBX",
        "SAM4D_ExportCameraJSON": "🎥 Export Camera JSON",
        "SAM4D_FBXViewer": "🎥 FBX Animation Viewer",
    })
    
    print(f"[SAM4DBodyCapture] Export nodes loaded")
    
except ImportError as e:
    print(f"[SAM4DBodyCapture] Export nodes not available: {e}")
except Exception as e:
    print(f"[SAM4DBodyCapture] Error loading Export nodes: {e}")

# ==================== Final Setup ====================
print(f"[SAM4DBodyCapture] v{__version__} loaded {len(NODE_CLASS_MAPPINGS)} nodes")

# Web extension directory for FBX viewer
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# ==================== Summary ====================
print(f"[SAM4DBodyCapture] v{__version__} - Loaded {len(NODE_CLASS_MAPPINGS)} nodes")

if not NODE_CLASS_MAPPINGS:
    print(f"[SAM4DBodyCapture] Warning: No nodes loaded. Check dependencies.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
