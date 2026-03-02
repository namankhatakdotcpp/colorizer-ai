import time
import threading
from collections import deque
import torch

class InternalMetrics:
    """
    Lightweight, in-memory metrics tracker for MVP/Hackathon dashboards.
    Safe for simple multi-threading use-cases.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.total_requests = 0
        self.total_latency_ms = 0.0
        self.recent_requests = deque()
        self.active_queue_size = 0
        
    def record_request(self, latency_ms: float):
        now = time.time()
        with self.lock:
            self.total_requests += 1
            self.total_latency_ms += latency_ms
            self.recent_requests.append(now)
            self._cleanup_old_requests(now)
            
    def _cleanup_old_requests(self, now: float):
        """Removes request timestamps older than 24 hours (86400 seconds)"""
        cutoff = now - 86400
        while self.recent_requests and self.recent_requests[0] < cutoff:
            self.recent_requests.popleft()

    def get_stats(self) -> dict:
        now = time.time()
        with self.lock:
            self._cleanup_old_requests(now)
            avg_latency = (self.total_latency_ms / self.total_requests) if self.total_requests > 0 else 0
            
            stats = {
                "total_requests": self.total_requests,
                "average_latency_ms": round(avg_latency, 2),
                "requests_last_24h": len(self.recent_requests),
                "active_inference_queue": self.active_queue_size
            }
            
        # GPU telemetry does not need our threading lock
        stats["gpu_memory_allocated_mb"] = 0.0
        if torch.cuda.is_available():
            stats["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
            
        return stats

metrics_tracker = InternalMetrics()
