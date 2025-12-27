#!/usr/bin/env python3
"""
ComfyUI-SAM4DBodyCapture Installation Script

Usage:
    python install.py           # Install all dependencies
    python install.py --check   # Check if dependencies are installed
    python install.py --models  # Download models only
"""

import subprocess
import sys
import os
import argparse

def run_pip(args):
    """Run pip with given arguments."""
    return subprocess.run(
        [sys.executable, "-m", "pip"] + args,
        capture_output=True,
        text=True
    )

def check_package(package_name):
    """Check if a package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_requirements():
    """Install requirements from requirements.txt."""
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    
    if not os.path.exists(req_file):
        print("❌ requirements.txt not found!")
        return False
    
    print("📦 Installing dependencies...")
    result = run_pip(["install", "-r", req_file])
    
    if result.returncode != 0:
        print(f"❌ Installation failed: {result.stderr}")
        return False
    
    print("✅ Dependencies installed successfully!")
    return True

def check_dependencies():
    """Check all required dependencies."""
    required = {
        "torch": "torch",
        "torchvision": "torchvision",
        "numpy": "numpy",
        "PIL": "Pillow",
        "cv2": "opencv-python",
        "diffusers": "diffusers",
        "accelerate": "accelerate",
        "transformers": "transformers",
        "safetensors": "safetensors",
    }
    
    print("🔍 Checking dependencies...\n")
    
    all_ok = True
    for import_name, package_name in required.items():
        if check_package(import_name):
            print(f"  ✅ {package_name}")
        else:
            print(f"  ❌ {package_name} - NOT INSTALLED")
            all_ok = False
    
    # Check CUDA
    print("\n🔍 Checking GPU support...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  ✅ CUDA version: {torch.version.cuda}")
        else:
            print("  ⚠️ CUDA not available - will use CPU (slow)")
    except:
        print("  ❌ Could not check CUDA")
    
    print()
    if all_ok:
        print("✅ All dependencies satisfied!")
    else:
        print("❌ Some dependencies missing. Run: python install.py")
    
    return all_ok

def download_models():
    """Download required models from HuggingFace."""
    print("📥 Downloading models...")
    print("(This may take a while on first run)")
    
    try:
        from huggingface_hub import snapshot_download
        
        models = [
            ("kaihuac/diffusion-vas-amodal-segmentation", "Diffusion-VAS Amodal"),
            ("kaihuac/diffusion-vas-content-completion", "Diffusion-VAS Content"),
        ]
        
        for repo_id, name in models:
            print(f"\n📦 Downloading {name}...")
            try:
                snapshot_download(repo_id)
                print(f"  ✅ {name} downloaded")
            except Exception as e:
                print(f"  ❌ Failed to download {name}: {e}")
        
        print("\n✅ Model download complete!")
        print("Note: Models are cached in ~/.cache/huggingface/")
        
    except ImportError:
        print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Install ComfyUI-SAM4DBodyCapture")
    parser.add_argument("--check", action="store_true", help="Check dependencies only")
    parser.add_argument("--models", action="store_true", help="Download models only")
    args = parser.parse_args()
    
    print("=" * 50)
    print("ComfyUI-SAM4DBodyCapture Installer")
    print("=" * 50)
    print()
    
    if args.check:
        check_dependencies()
    elif args.models:
        download_models()
    else:
        # Full installation
        if install_requirements():
            print()
            check_dependencies()
            print()
            print("💡 To download models, run: python install.py --models")
            print("   (Models are also auto-downloaded on first use)")
    
    print()
    print("=" * 50)

if __name__ == "__main__":
    main()
