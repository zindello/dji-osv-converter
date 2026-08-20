# Phase 1 Setup Guide: Environment & Dependencies

## Environment Requirements

**Python Version:** 3.10 or higher (required by pyobjc and PyAV)
- Current system Python: 3.9 (too old)
- Recommended: Install via Homebrew or pyenv with proper OpenSSL support

## Step 1: Create Virtual Environment

```bash
# Option A: Using Homebrew Python (recommended)
brew install python@3.12
/usr/local/opt/python@3.12/bin/python3 -m venv venv

# Option B: Using pyenv (ensure OpenSSL is linked)
pyenv install 3.12.0
pyenv local 3.12.0
python -m venv venv
```

## Step 2: Activate Virtual Environment

```bash
source venv/bin/activate
```

## Step 3: Install Core Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install PyObjC (for macOS Objective-C interop)
pip install pyobjc

# Install protobuf (for metadata parsing)
pip install protobuf>=3.20.0

# Install PyAV (for HEVC video decoding)
pip install av>=10.0.0

# Verify installation
python -c "import objc; from google.protobuf import message; import av; print('✓ All dependencies installed')"
```

## Step 4: Update project.py setup.py

```python
install_requires=[
    "pyobjc>=11.0",
    "protobuf>=3.20.0",
    "PyAV>=10.0.0",
],
extras_require={
    "dev": [
        "pytest>=6.0",
        "pytest-cov>=2.0",
    ],
    "gpu": [
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
    ],
},
```

## Step 5: Verify No Conflicts

```bash
pip check
```

## Troubleshooting

### Issue: "unsupported hash type blake2b/blake2s"
**Cause:** PyEnv Python version has broken OpenSSL linkage  
**Solution:** Use Homebrew Python instead or rebuild pyenv with proper OpenSSL

### Issue: "PyAV not found" or version too old
**Cause:** PyAV requires Python 3.10+  
**Solution:** Verify Python version with `python --version`

### Issue: "pyobjc-framework-AppKit not found"
**Cause:** Framework-specific packages may not be available for all Python versions  
**Solution:** Install core pyobjc first, framework access handled automatically on macOS

## Next: Phase 2

Once dependencies are installed and verified, proceed with:
1. Creating `_objc_bridge.py` (PyObjC framework loading)
2. Rewriting `stitch_dualstitcher.py` for Objective-C compatibility
3. Testing framework initialization

## Manual Installation Test

```bash
source venv/bin/activate

# Test imports
python << 'EOF'
import objc
print(f"✓ PyObjC {objc.__version__}")

from google.protobuf import message
print(f"✓ Protobuf loaded")

import av
print(f"✓ PyAV {av.__version__}")

# Test framework discovery
from pathlib import Path
framework_path = Path("/Applications/DJI Studio.app/Contents/Frameworks/DualStitcher.framework/Versions/A/DualStitcher")
if framework_path.exists():
    print(f"✓ DualStitcher framework found")
else:
    print(f"✗ DualStitcher framework not found")
EOF
```

Expected output:
```
✓ PyObjC 11.x
✓ Protobuf loaded
✓ PyAV 10.x
✓ DualStitcher framework found
```

## Status

Phase 1 setup is **BLOCKED** on Python 3.10+ availability. Once Python environment is corrected:
- [ ] venv created with Python 3.10+
- [ ] All pip packages installed
- [ ] setup.py updated
- [ ] All imports verify successfully
- [ ] Framework discovery test passes

**Recommendation:** Use `brew install python@3.12` and rebuild venv to unblock Phase 1.
