import hashlib
import time
from typing import Dict, Tuple, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

class ImageCache:
    def __init__(self, max_size=100, expiration_time=3600, max_workers=4):
        self.cache: Dict[str, Tuple[float, Any]] = {}
        self.max_size = max_size
        self.expiration_time = expiration_time
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _generate_key(self, image_data: bytes) -> str:
        """Generate a unique key for the image data using SHA-256"""
        return hashlib.sha256(image_data).hexdigest()

    def get(self, image_data: bytes) -> Optional[Any]:
        """Retrieve cached result if it exists and hasn't expired"""
        if image_data is None:
            return None

        key = self._generate_key(image_data)
        if key in self.cache:
            timestamp, result = self.cache[key]
            if time.time() - timestamp < self.expiration_time:
                return result
            else:
                # Remove expired entry
                del self.cache[key]
        return None

    def set(self, image_data: bytes, result: Any) -> None:
        """Cache the result with current timestamp"""
        if image_data is None:
            return

        key = self._generate_key(image_data)
        self.cache[key] = (time.time(), result)

        # If cache exceeds max size, remove oldest entries
        if len(self.cache) > self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
            del self.cache[oldest_key]

    def process_batch(self, images: List[bytes], process_func) -> List[Any]:
        """
        Process multiple images concurrently
        """
        results = []
        
        def process_single(image: bytes) -> Any:
            # Check cache first
            result = self.get(image)
            if result is not None:
                return result
            
            # Process and cache if not found
            result = process_func(image)
            self.set(image, result)
            return result

        # Process images concurrently
        futures = [self.executor.submit(process_single, img) for img in images]
        results = [future.result() for future in futures]
        
        return results

    def __del__(self):
        """Cleanup executor on deletion"""
        self.executor.shutdown(wait=True)
