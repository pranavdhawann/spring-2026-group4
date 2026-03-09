import importlib
import sys

def check_package(pkg_name):
    try:
        importlib.import_module(pkg_name)
        return True, "Pass"
    except ImportError as e:
        return False, f"Fail ({e})"

def main():
    print("="*50)
    print("Chronos-2 Environment Check")
    print("="*50)

    # 1. Package Imports
    print("\n[1] Checking Packages:")
    packages = {
        "torch": "torch",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "peft": "peft",
        "datasets": "datasets",
        "gluonts": "gluonts",
        "pandas": "pandas",
        "numpy": "numpy",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
        "pyyaml": "yaml",
        "matplotlib": "matplotlib",
        "tqdm": "tqdm",
        "chronos-forecasting": "chronos"
    }

    all_passed = True
    for display_name, import_name in packages.items():
        passed, msg = check_package(import_name)
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {display_name:<20}: {msg}")
        if not passed:
            all_passed = False

    # 2. CUDA & GPU Check
    print("\n[2] Checking CUDA and GPU (g5.2xlarge expected):")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        status = "[PASS]" if cuda_available else "[FAIL]"
        print(f"  {status} CUDA Available       : {cuda_available}")

        if cuda_available:
            cuda_version = torch.version.cuda
            print(f"  [PASS] CUDA Version         : {cuda_version}")

            device_count = torch.cuda.device_count()
            print(f"  [PASS] GPU Count            : {device_count}")

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                name = props.name
                vram_gb = props.total_memory / (1024**3)
                print(f"  [PASS] GPU {i} Name           : {name}")
                print(f"  [PASS] GPU {i} VRAM           : {vram_gb:.2f} GB")
                
                if "A10G" in name or vram_gb >= 22.0:
                    print(f"  [PASS] GPU Suitability      : Looks suitable for LoRA (>=24GB ideal)")
                else:
                    print(f"  [WARN] GPU Suitability      : Might be constrained for large models")
        else:
            all_passed = False
            print("  [FAIL] CUDA not available, cannot check GPUs.")

    except ImportError:
            all_passed = False
            print("  [FAIL] torch not installed, cannot check CUDA/GPUs.")

    print("\n" + "="*50)
    if all_passed:
        print("🎉 SUMMARY: All checks PASSED. Environment is ready!")
    else:
        print("🤡 SUMMARY: Some checks FAILED. Please review the output above.")
    print("="*50)

if __name__ == "__main__":
    main()
