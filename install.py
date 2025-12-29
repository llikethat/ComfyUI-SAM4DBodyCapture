#!/usr/bin/env python3
"""
ComfyUI-SAM4DBodyCapture Installation Script

Usage:
    python install.py           # Install all dependencies
    python install.py --check   # Check if dependencies are installed
    python install.py --models  # Download models only
    python install.py --render  # Install rendering dependencies only (pyrender, etc.)
"""

import subprocess
import sys
import os
import argparse
import shutil

def run_pip(args):
    """Run pip with given arguments."""
    return subprocess.run(
        [sys.executable, "-m", "pip"] + args,
        capture_output=True,
        text=True
    )

def run_command(cmd, shell=False):
    """Run a shell command."""
    return subprocess.run(
        cmd,
        shell=shell,
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

def install_rendering_deps():
    """Install pyrender, trimesh, PyOpenGL and configure headless rendering."""
    print("\n📦 Installing rendering dependencies (pyrender, trimesh, PyOpenGL)...")
    
    # Install Python packages
    packages = ["pyrender>=0.1.45", "trimesh>=3.10.0", "PyOpenGL>=3.1.0"]
    
    for pkg in packages:
        print(f"  Installing {pkg}...")
        result = run_pip(["install", pkg])
        if result.returncode != 0:
            print(f"  ⚠️ Warning: Failed to install {pkg}")
        else:
            print(f"  ✅ {pkg}")
    
    # Try to install system dependencies for headless rendering
    print("\n📦 Setting up headless rendering...")
    
    # Check if we have sudo/root access
    has_sudo = shutil.which("sudo") is not None
    has_apt = shutil.which("apt-get") is not None
    
    if has_apt:
        print("  Attempting to install system libraries (may require sudo)...")
        
        apt_packages = ["libosmesa6-dev", "freeglut3-dev", "libgl1-mesa-glx", "libglib2.0-0"]
        
        for pkg in apt_packages:
            if has_sudo:
                cmd = ["sudo", "apt-get", "install", "-y", pkg]
            else:
                cmd = ["apt-get", "install", "-y", pkg]
            
            result = run_command(cmd)
            if result.returncode == 0:
                print(f"  ✅ {pkg}")
            else:
                print(f"  ⚠️ Could not install {pkg} (may already be installed or need manual install)")
    else:
        print("  ⚠️ apt-get not found - skipping system library installation")
        print("  For headless rendering, manually install: libosmesa6-dev freeglut3-dev")
    
    # Set up environment variable for headless rendering
    print("\n📦 Configuring PyOpenGL for headless rendering...")
    
    # Check if EGL is available
    egl_available = os.path.exists("/usr/lib/x86_64-linux-gnu/libEGL.so") or \
                    os.path.exists("/usr/lib/libEGL.so")
    
    osmesa_available = os.path.exists("/usr/lib/x86_64-linux-gnu/libOSMesa.so") or \
                       os.path.exists("/usr/lib/libOSMesa.so")
    
    if egl_available:
        print("  ✅ EGL available - recommended for headless rendering")
        print("  💡 Set environment: export PYOPENGL_PLATFORM=egl")
        
        # Try to set it in bashrc
        bashrc = os.path.expanduser("~/.bashrc")
        env_line = 'export PYOPENGL_PLATFORM=egl'
        
        try:
            # Check if already set
            if os.path.exists(bashrc):
                with open(bashrc, 'r') as f:
                    if env_line not in f.read():
                        with open(bashrc, 'a') as f2:
                            f2.write(f'\n# PyOpenGL headless rendering (added by SAM4DBodyCapture)\n{env_line}\n')
                        print(f"  ✅ Added to ~/.bashrc")
        except:
            pass
        
        # Set for current process
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        
    elif osmesa_available:
        print("  ✅ OSMesa available - using for headless rendering")
        print("  💡 Set environment: export PYOPENGL_PLATFORM=osmesa")
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    else:
        print("  ⚠️ Neither EGL nor OSMesa found")
        print("  💡 Fallback: export PYOPENGL_PLATFORM=egl")
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    
    # Verify pyrender works
    print("\n🔍 Verifying pyrender installation...")
    try:
        import pyrender
        import trimesh
        print(f"  ✅ pyrender {pyrender.__version__}")
        print(f"  ✅ trimesh {trimesh.__version__}")
        
        # Try to create a simple scene
        try:
            scene = pyrender.Scene()
            print("  ✅ pyrender Scene creation works")
        except Exception as e:
            print(f"  ⚠️ pyrender Scene test: {e}")
            print("  💡 May need to restart Python/ComfyUI after installation")
        
    except ImportError as e:
        print(f"  ❌ pyrender import failed: {e}")
        return False
    
    print("\n✅ Rendering dependencies installed!")
    print("\n💡 If mesh overlay doesn't work, try:")
    print("   1. Restart ComfyUI")
    print("   2. export PYOPENGL_PLATFORM=egl")
    print("   3. Or: export PYOPENGL_PLATFORM=osmesa")
    
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
    
    optional_render = {
        "pyrender": "pyrender",
        "trimesh": "trimesh",
        "OpenGL": "PyOpenGL",
    }
    
    print("🔍 Checking core dependencies...\n")
    
    all_ok = True
    for import_name, package_name in required.items():
        if check_package(import_name):
            print(f"  ✅ {package_name}")
        else:
            print(f"  ❌ {package_name} - NOT INSTALLED")
            all_ok = False
    
    print("\n🔍 Checking rendering dependencies (optional)...\n")
    
    render_ok = True
    for import_name, package_name in optional_render.items():
        if check_package(import_name):
            print(f"  ✅ {package_name}")
        else:
            print(f"  ⚠️ {package_name} - NOT INSTALLED (mesh overlay will use fallback)")
            render_ok = False
    
    # Check PYOPENGL_PLATFORM
    pyopengl_platform = os.environ.get("PYOPENGL_PLATFORM", "not set")
    print(f"\n  PYOPENGL_PLATFORM: {pyopengl_platform}")
    
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
        print("✅ All core dependencies satisfied!")
    else:
        print("❌ Some dependencies missing. Run: python install.py")
    
    if not render_ok:
        print("\n💡 For mesh overlay rendering, run: python install.py --render")
    
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
    parser.add_argument("--render", action="store_true", help="Install rendering dependencies (pyrender, etc.)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("ComfyUI-SAM4DBodyCapture Installer")
    print("=" * 60)
    print()
    
    if args.check:
        check_dependencies()
    elif args.models:
        download_models()
    elif args.render:
        install_rendering_deps()
    else:
        # Full installation (core dependencies only)
        if install_requirements():
            print()
            check_dependencies()
            print()
            print("💡 To download models, run: python install.py --models")
            print("   (Models are also auto-downloaded on first use)")
            print()
            print("💡 For mesh overlay rendering (optional), run: python install.py --render")
            print("   (Requires manual setup - see README for instructions)")
    
    print()
    print("=" * 60)

if __name__ == "__main__":
    main()
