import onnxruntime as _ort

# Load the CUDA/cuDNN DLLs shipped as pip packages (onnxruntime-gpu[cuda,cudnn]) before any
# session is created, and keep ORT's per-node warnings ("fallback mode", ScatterND) quiet.
if hasattr(_ort, "preload_dlls"):
    try:
        _ort.preload_dlls()
    except Exception:
        pass
_ort.set_default_logger_severity(3)
