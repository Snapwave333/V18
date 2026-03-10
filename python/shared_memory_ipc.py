import multiprocessing.shared_memory
import threading
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class SharedFrameMemory:
    """
    Manages a shared memory buffer for passing raw RGB frames
    between processes with zero serialization overhead.
    """
    def __init__(self, name: str, width: int, height: int, create: bool = False):
        self.name = name
        self.width = width
        self.height = height
        self.size = width * height * 3  # RGB bytes
        
        # Add metadata size: [frame_id (8 bytes), timestamp (8 bytes)]
        self.meta_size = 16 
        self.total_size = self.size + self.meta_size
        
        self.shm = None
        self._create = create
        self._lock = threading.Lock()
        
        try:
            if create:
                # Try to clean up existing if any
                try:
                    old_shm = multiprocessing.shared_memory.SharedMemory(name=self.name)
                    old_shm.unlink()
                except FileNotFoundError:
                    pass
                self.shm = multiprocessing.shared_memory.SharedMemory(
                    create=True, name=self.name, size=self.total_size
                )
            else:
                self.shm = multiprocessing.shared_memory.SharedMemory(name=self.name)
        except FileNotFoundError:
            # Expected if polling before the creator has initialized it
            raise
        except Exception as e:
            logger.error(f"Failed to initialize SharedMemory for {name}: {e}")
            raise

    def write_frame(self, frame_np: np.ndarray, frame_id: int):
        """Write a new frame and metadata to shared memory."""
        with self._lock:
            # Write meta
            meta_view = self.shm.buf[:self.meta_size]
            meta_arr = np.ndarray((2,), dtype=np.int64, buffer=meta_view)
            meta_arr[0] = int(frame_id)
            meta_arr[1] = int(time.time() * 1000) # ms timestamp
            
            # Write pixels
            frame_view = self.shm.buf[self.meta_size:]
            np.copyto(np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=frame_view), frame_np)

    def read_frame(self):
        """Read frame and metadata from shared memory. Returns (frame_np, frame_id, timestamp_ms)"""
        with self._lock:
            meta_view = self.shm.buf[:self.meta_size]
            meta_arr = np.ndarray((2,), dtype=np.int64, buffer=meta_view)
            frame_id = int(meta_arr[0])
            timestamp = int(meta_arr[1])
            
            if frame_id == 0 and timestamp == 0:
                return None, 0, 0
                
            frame_view = self.shm.buf[self.meta_size:]
            # Copy to avoid segmentation faults if shared memory is closed
            frame_np = np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=frame_view).copy()
            return frame_np, frame_id, timestamp

    def close(self):
        if self.shm:
            try:
                self.shm.close()
                if self._create:
                    self.shm.unlink()
            except Exception:
                pass
