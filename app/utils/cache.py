import hashlib
import time
from typing import Dict, Tuple, Any, Optional

class ImageCache:
    def __init__(self, max_size=100, expiration_time=3600):  # 1 hour expiration by default
        self.cache: Dict[str, Tuple[float, Any]] = {}
        self.max_size = max_size
        self.expiration_time = expiration_time
    
    def _generate_key(self, image_data: bytes) -> str:
        """Generate a unique key for the image data using SHA-256"""
        return hashlib.sha256(image_data).hexdigest()
    
    def get(self, image_data: bytes) -> Optional[Any]:
        """Retrieve cached result if it exists and hasn't expired"""
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
        key = self._generate_key(image_data)
        self.cache[key] = (time.time(), result)
        
        # If cache exceeds max size, remove oldest entries
        if len(self.cache) > self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
            del self.cache[oldest_key]