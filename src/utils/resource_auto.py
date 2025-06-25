
import os
import psutil
        import pynvml
def get_system_memory_gb():
    """Return total system RAM in GB."""
    return psutil.virtual_memory().total / (1024 ** 3)

def get_available_memory_gb():
    """Return available system RAM in GB."""
    return psutil.virtual_memory().available / (1024 ** 3)

def get_gpu_memory_gb():
    """Return total and available GPU memory in GB (NVIDIA, Colab, etc)."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = info.total / (1024 ** 3)
        free = info.free / (1024 ** 3)
        return total, free
    except Exception:
        return None, None

def print_resource_summary():
    sys_total = get_system_memory_gb()
    sys_avail = get_available_memory_gb()
    gpu_total, gpu_free = get_gpu_memory_gb()
    print(f"[Resource] System RAM: {sys_avail:.1f} / {sys_total:.1f} GB available")
    if gpu_total:
        print(f"[Resource] GPU RAM: {gpu_free:.1f} / {gpu_total:.1f} GB available")
    else:
        print("[Resource] GPU not detected or pynvml not available.")

def get_optimal_resource_fraction(ram_fraction = 0.8, gpu_fraction = 0.8):
    """Return optimal RAM and GPU usage (GB) for current system."""
    sys_total = get_system_memory_gb()
    gpu_total, _ = get_gpu_memory_gb()
    ram_gb = int(sys_total * ram_fraction)
    gpu_gb = int(gpu_total * gpu_fraction) if gpu_total else None
    return ram_gb, gpu_gb