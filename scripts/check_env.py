#!/usr/bin/env python3
"""
Environment and Hardware Check Script for Person Down Detection System
Validates Python version, virtual environment, CUDA/GPU support, installed packages,
and directory structure.
"""

import sys
import os
import platform
from pathlib import Path

REQUIRED_PACKAGES = [
    ("torch", "PyTorch"),
    ("torchvision", "TorchVision"),
    ("ultralytics", "Ultralytics YOLO"),
    ("cv2", "OpenCV"),
    ("numpy", "NumPy"),
    ("sklearn", "Scikit-Learn"),
    ("pandas", "Pandas"),
    ("scipy", "SciPy"),
    ("shapely", "Shapely"),
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("pydantic", "Pydantic"),
    ("yaml", "PyYAML"),
]

EXPECTED_DIRS = [
    "configs",
    "data/raw",
    "data/processed",
    "data/samples",
    "models/pretrained",
    "models/checkpoints",
    "logs",
    "scripts",
    "src/core",
    "src/dataset",
    "src/pipeline",
    "src/api",
    "tests",
]


def print_header(title: str):
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)


def check_python_environment():
    print_header("1. Python Environment")
    py_ver = sys.version.split()[0]
    in_venv = sys.prefix != sys.base_prefix
    print(f"Python Version    : {py_ver} ({platform.architecture()[0]})")
    print(f"Executable Path   : {sys.executable}")
    print(f"Platform          : {platform.platform()}")
    print(f"In Virtual Env    : {'[YES] ' + sys.prefix if in_venv else '[NO] Not running in a virtual environment'}")

    # Python version compatibility check (Target: 3.10 - 3.12)
    major, minor = sys.version_info.major, sys.version_info.minor
    if major == 3 and 10 <= minor <= 12:
        print("Python Status     : [OK] Compatible version for PyTorch/OpenCV")
        return True
    else:
        print(f"Python Status     : [WARNING] Python {major}.{minor} may have limited wheel support for PyTorch/OpenCV.")
        return False


def check_hardware_acceleration():
    print_header("2. Hardware Acceleration & PyTorch")
    try:
        import torch
        print(f"PyTorch Version   : {torch.__version__}")
        print(f"CUDA Available    : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version (PyTorch) : {torch.version.cuda}")
            print(f"cuDNN Version          : {torch.backends.cudnn.version()}")
            gpu_count = torch.cuda.device_count()
            print(f"GPU Count              : {gpu_count}")
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory / (1024 ** 3)
                print(f"  - Device [{i}]       : {props.name}")
                print(f"    Compute Capability : {props.major}.{props.minor}")
                print(f"    Total VRAM         : {total_mem:.2f} GB")
            print("Acceleration Status    : [OK] GPU acceleration is active and ready.")
            return True
        else:
            print("Acceleration Status    : [NOTE] Running in CPU-only mode (No CUDA acceleration detected).")
            return True
    except ImportError:
        print("Acceleration Status    : [FAIL] PyTorch is not installed.")
        return False


def check_packages():
    print_header("3. Package Dependency Verification")
    all_installed = True
    print(f"{'Package Name':<20} {'Display Name':<20} {'Status':<10} {'Version':<15}")
    print("-" * 65)

    for mod_name, disp_name in REQUIRED_PACKAGES:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"{mod_name:<20} {disp_name:<20} {'[OK]':<10} {ver:<15}")
        except ImportError:
            print(f"{mod_name:<20} {disp_name:<20} {'[MISSING]':<10} {'-':<15}")
            all_installed = False

    return all_installed


def check_directory_structure():
    print_header("4. Project Structure Verification")
    root_dir = Path(__file__).resolve().parent.parent
    all_exist = True
    for d in EXPECTED_DIRS:
        target = root_dir / d
        if target.exists() and target.is_dir():
            print(f"  [OK] {d}/")
        else:
            print(f"  [MISSING] {d}/")
            all_exist = False
    return all_exist


def main():
    print("\n" + "#" * 65)
    print(" PERSON DOWN DETECTION SYSTEM - PHASE 0 ENVIRONMENT CHECK")
    print("#" * 65)

    py_ok = check_python_environment()
    hw_ok = check_hardware_acceleration()
    pkg_ok = check_packages()
    dir_ok = check_directory_structure()

    print_header("Summary")
    print(f"Python Runtime     : {'PASSED' if py_ok else 'ATTENTION'}")
    print(f"Hardware / PyTorch : {'PASSED' if hw_ok else 'FAILED'}")
    print(f"Dependencies       : {'PASSED' if pkg_ok else 'FAILED'}")
    print(f"Project Structure  : {'PASSED' if dir_ok else 'FAILED'}")

    if py_ok and hw_ok and pkg_ok and dir_ok:
        print("\n>>> ALL PHASE 0 ENVIRONMENT CHECKS PASSED SUCCESSFULLY. <<<")
        sys.exit(0)
    else:
        print("\n>>> ONE OR MORE ENVIRONMENT CHECKS FAILED OR NEED ATTENTION. <<<")
        sys.exit(1)


if __name__ == "__main__":
    main()
