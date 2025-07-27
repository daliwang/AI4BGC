"""
GPU Monitoring Utilities
"""
import torch
import psutil
import logging
from typing import Dict, Any
import time
import pynvml

logger = logging.getLogger(__name__)

class GPUMonitor:
    """Monitor GPU memory and utilization."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_cuda = self.device.type == "cuda"
        
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get current GPU statistics."""
        if not self.is_cuda:
            return {"error": "No CUDA device available"}
            
        try:
            # Get GPU memory info
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
            memory_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # GB
            
            # Get GPU utilization (requires nvidia-ml-py)
            gpu_util = self._get_gpu_utilization()
            
            return {
                "memory_allocated_gb": round(memory_allocated, 2),
                "memory_reserved_gb": round(memory_reserved, 2),
                "memory_total_gb": round(memory_total, 2),
                "memory_usage_percent": round((memory_allocated / memory_total) * 100, 1),
                "gpu_utilization_percent": gpu_util,
                "device_name": torch.cuda.get_device_name(self.device)
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU stats: {e}")
            return {"error": str(e)}
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:

            pynvml.nvmlInit()
            # Check for None index
            if not hasattr(self.device, 'index') or self.device.index is None:
                logger.warning("GPU device index is None. Skipping GPU utilization monitoring.")
                return 0.0
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except ImportError:
            logger.warning("pynvml not available, cannot get GPU utilization")
            return 0.0
        except Exception as e:
            logger.warning(f"Failed to get GPU utilization: {e}")
            return 0.0
    
    def log_gpu_stats(self, prefix: str = ""):
        """Log current GPU statistics."""
        if not self.is_cuda:
            return
            
        stats = self.get_gpu_stats()
        if "error" in stats:
            logger.warning(f"{prefix}GPU stats error: {stats['error']}")
            return
            
        logger.info(
            f"{prefix}GPU: {stats['device_name']} | "
            f"Memory: {stats['memory_allocated_gb']}GB/{stats['memory_total_gb']}GB "
            f"({stats['memory_usage_percent']}%) | "
            f"Util: {stats['gpu_utilization_percent']}%"
        )
    
    def check_memory_threshold(self, threshold: float = 0.9) -> bool:
        """Check if GPU memory usage exceeds threshold."""
        if not self.is_cuda:
            return False
            
        stats = self.get_gpu_stats()
        if "error" in stats:
            return False
            
        return stats["memory_usage_percent"] / 100 > threshold
    
    def empty_cache(self):
        """Empty GPU cache."""
        if self.is_cuda:
            torch.cuda.empty_cache()
    
    
    def get_system_memory(self) -> Dict[str, float]:
        """Get system memory information."""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / 1024**3,
            "available_gb": memory.available / 1024**3,
            "used_gb": memory.used / 1024**3,
            "percent": memory.percent
        }

def log_memory_usage(prefix: str = ""):
    """Convenience function to log current memory usage."""
    monitor = GPUMonitor()
    monitor.log_gpu_stats(prefix)
    
    # Also log system memory
    sys_memory = monitor.get_system_memory()
    logger.info(
        f"{prefix}System Memory: {sys_memory['used_gb']:.1f}GB/"
        f"{sys_memory['total_gb']:.1f}GB ({sys_memory['percent']:.1f}%)"
    ) 