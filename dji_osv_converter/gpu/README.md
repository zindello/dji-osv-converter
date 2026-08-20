# GPU Acceleration Layer

Modular GPU acceleration for video processing on multiple hardware targets.

## Architecture

```
GPUBackend (abstract interface)
├── MetalBackend (Apple Silicon - M1/M2/M3)
├── CUDABackend (NVIDIA - GeForce, Quadro, DataCenter)
├── OneAPIBackend (Intel - iGPU, Iris, Arc)
└── CPUBackend (fallback, always available)
```

## Key Design Principles

1. **Abstraction First**: Interface defines what must be accelerated, not how
2. **Detection Separate**: Auto-detection doesn't assume which backend caller wants
3. **Explicit Selection**: Caller chooses GPU by name (can use auto-detect recommendation)
4. **Graceful Fallback**: CPU backend works when GPU unavailable
5. **Modular Implementation**: Each backend is independent, can be developed/tested separately

## Operations (GPU-Accelerated)

1. **decode_hevc()** - Hardware HEVC decoding (50-100× faster than CPU)
2. **convert_color_space()** - RGB↔YUV conversion on GPU (5-10× faster)
3. **undistort_warp()** - Fisheye undistortion (20-50× faster) ⚠️ **CRITICAL PATH**
4. **blend_frames()** - Seam blending (5-20× faster)
5. **encode_hevc()** - Hardware HEVC encoding (50-100× faster)

## Usage

### Auto-Detect Best GPU

```python
from dji_osv_converter.gpu import factory

backend = factory.create_best_backend()
if backend is None:
    print("No GPU available, falling back to CPU")
    backend = factory.create_backend("cpu")

# Use backend
frame = backend.decode_hevc(hevc_data, width=3840, height=3840)
undistorted = backend.undistort_warp(frame, calibration)
```

### Explicit GPU Selection

```python
from dji_osv_converter.gpu import factory, detect

# Check what's available
available = detect.detect_available_gpus()
print(available)  # {"metal": True, "cuda": False, "cpu": True}

# Use Metal on M3 Mac
backend = factory.create_backend("metal")
if backend is None:
    print("Metal not available")
    backend = factory.create_backend("cpu")
```

### With Context Manager

```python
from dji_osv_converter.gpu import factory

with factory.create_best_backend() as backend:
    frame = backend.decode_hevc(hevc_data, 3840, 3840)
    # GPU resources automatically cleaned up on exit
```

## Implementation Status

| Backend | Status | Speedup | Priority |
|---------|--------|---------|----------|
| CPU | ✓ Partial (encode/decode via FFmpeg) | 1× baseline | Now |
| Metal | 🔴 Stubbed | 20-50× | High (M3 Mac) |
| CUDA | 🔴 Stubbed | 100-300× | Medium |
| oneAPI | 🔴 Stubbed | 50-200× | Low |

### Metal Implementation Plan

1. **VideoToolbox decode** (hardware HEVC decode)
   - Use `VTDecompressionSessionCreate()` for streaming decode
   - Output to `CVPixelBuffer` (Metal texture directly)

2. **Metal compute kernels** for:
   - Fisheye undistortion (critical path)
   - Color conversion (RGB↔YUV)
   - Seam blending

3. **VideoToolbox encode** (hardware HEVC encoder)
   - Use `VTCompressionSessionCreate()` for real-time encoding

### CUDA Implementation Plan

1. **NVIDIA NVDEC** (hardware decode)
2. **CUDA kernels** for processing
3. **NVIDIA NVENC** (hardware encode)

### Intel oneAPI Implementation Plan

1. **Intel Media SDK** or **Quick Sync** (hardware decode)
2. **Intel oneAPI/Level-Zero** for compute
3. **Intel Media SDK** encoder

## Performance Targets

| Operation | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| Decode HEVC (3840×3840) | 150-200ms | 10-20ms | 10-20× |
| Fisheye undistort | 5-7s | 100-300ms | 20-50× |
| Color convert | 500-1000ms | 50-100ms | 10-20× |
| Seam blend | 1-2s | 50-100ms | 20-40× |
| Encode HEVC | 2-5s | 100-200ms | 25-50× |
| **Total per frame** | **8-16s** | **300-800ms** | **15-30× speedup** |

**Real-time requirement**: 33ms per frame (30fps)

**CPU achieves**: 0.19 fps ❌
**GPU achieves**: 1.5-3 fps (Metal), 30+ fps (CUDA) ✓

## Testing

```bash
# Test GPU detection
python -c "from dji_osv_converter.gpu import detect; print(detect.detect_available_gpus())"

# Test backend instantiation
python -c "from dji_osv_converter.gpu import factory; b = factory.create_best_backend(); print(b)"

# Run tests (once implemented)
pytest dji_osv_converter/gpu/tests/
```

## Extending with New Backends

1. Inherit from `GPUBackend` in `new_backend.py`
2. Implement all abstract methods
3. Add to factory in `factory.py`
4. Add to detection in `detect.py` (if applicable)
5. Write tests

Example:

```python
# my_gpu_backend.py
from .backend import GPUBackend

class MyGPUBackend(GPUBackend):
    def __init__(self):
        super().__init__("mygpu")
    
    def initialize(self) -> bool:
        # Your GPU init logic
        return True
    
    def decode_hevc(self, hevc_data, width, height):
        # Your GPU decode logic
        pass
    
    # ... implement other methods
```

Then in `factory.py`:

```python
elif target == "mygpu":
    from .my_gpu_backend import MyGPUBackend
    backend = MyGPUBackend()
```

## References

- [Metal Framework](https://developer.apple.com/metal/)
- [VideoToolbox](https://developer.apple.com/documentation/videotoolbox)
- [NVIDIA NVDEC/NVENC](https://developer.nvidia.com/video-codec-sdk)
- [Intel Media SDK](https://github.com/Intel-Media-SDK/MediaSDK)
