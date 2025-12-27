"""
ComfyUI-SAM4DBodyCapture

Temporally consistent 4D human mesh recovery from videos.
Integrates SAM-Body4D and Diffusion-VAS for ComfyUI.

GitHub: https://github.com/llikethat/ComfyUI-SAM4DBodyCapture

Version History:
- v0.1.0: Initial release - Diffusion-VAS standalone node
- v0.2.0: (Planned) SAM-Body4D integration
- v0.3.0: (Planned) Export nodes (FBX/Alembic)
- v0.4.0: (Planned) Camera solver integration
- v1.0.0: (Planned) First stable release
"""

__version__ = "0.1.1"
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
    
    print(f"[SAM4DBodyCapture] v{__version__} - Diffusion-VAS nodes loaded")
    
except ImportError as e:
    print(f"[SAM4DBodyCapture] Diffusion-VAS nodes not available: {e}")
except Exception as e:
    print(f"[SAM4DBodyCapture] Error loading Diffusion-VAS: {e}")

# ==================== SAM-Body4D Nodes (v0.2.0+) ====================
try:
    from .nodes import sam_body4d
    
    NODE_CLASS_MAPPINGS.update({
        "SAM4D_Body4DProcessor": sam_body4d.Body4DProcessor,
        "SAM4D_TemporalFusion": sam_body4d.TemporalFusion,
    })
    
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "SAM4D_Body4DProcessor": "🏂 SAM-Body4D Processor",
        "SAM4D_TemporalFusion": "🏂 Temporal Mesh Fusion",
    })
    
    print(f"[SAM4DBodyCapture] SAM-Body4D nodes loaded")
    
except ImportError:
    # Expected in v0.1.0
    pass
except Exception as e:
    print(f"[SAM4DBodyCapture] Error loading SAM-Body4D: {e}")

# ==================== Export Nodes (v0.3.0+) ====================
try:
    from .nodes import export_nodes
    
    NODE_CLASS_MAPPINGS.update({
        "SAM4D_ExportCharacterFBX": export_nodes.ExportCharacterFBX,
        "SAM4D_ExportCharacterAlembic": export_nodes.ExportCharacterAlembic,
        "SAM4D_ExportCameraFBX": export_nodes.ExportCameraFBX,
        "SAM4D_ExportCameraAlembic": export_nodes.ExportCameraAlembic,
    })
    
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "SAM4D_ExportCharacterFBX": "📦 Export Character FBX",
        "SAM4D_ExportCharacterAlembic": "📦 Export Character Alembic",
        "SAM4D_ExportCameraFBX": "📦 Export Camera FBX",
        "SAM4D_ExportCameraAlembic": "📦 Export Camera Alembic",
    })
    
    print(f"[SAM4DBodyCapture] Export nodes loaded")
    
except ImportError:
    # Expected in v0.1.0 and v0.2.0
    pass
except Exception as e:
    print(f"[SAM4DBodyCapture] Error loading export nodes: {e}")

# ==================== Summary ====================
if NODE_CLASS_MAPPINGS:
    print(f"[SAM4DBodyCapture] Loaded {len(NODE_CLASS_MAPPINGS)} nodes")
else:
    print(f"[SAM4DBodyCapture] Warning: No nodes loaded. Check dependencies.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
