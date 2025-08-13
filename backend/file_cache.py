from typing import OrderedDict, Optional, Dict
import time

# In-memory LRU cache for files
class InMemoryFileCache:
    def __init__(self, max_items: int = 50, max_bytes: int = 50 * 1024 * 1024):
        """
        max_items: max number of files to keep
        max_bytes: total bytes cap for all cached files
        """
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self._cache = OrderedDict()  # key -> (bytes, metadata)

    def set(self, key: str, data: bytes, metadata: Optional[Dict] = None):
        if key in self._cache:
            old_bytes = len(self._cache[key][0])
            self.current_bytes -= old_bytes
            del self._cache[key]
        self._cache[key] = (data, metadata or {"created_at": time.time(), "size": len(data)})
        self.current_bytes += len(data)
        self._cache.move_to_end(key, last=True)
        self._evict_if_needed()

    def get(self, key: str) -> Optional[bytes]:
        entry = self._cache.get(key)
        if not entry:
            return None
        
        # touch to mark as recently used
        self._cache.move_to_end(key, last=True)
        return entry[0]

    def get_metadata(self, key: str) -> Optional[Dict]:
        entry = self._cache.get(key)
        if not entry:
            return None
        return entry[1]

    def list_keys(self):
        # return most-recent-first
        return list(reversed(self._cache.keys()))

    def delete(self, key: str):
        entry = self._cache.pop(key, None)
        if entry:
            self.current_bytes -= len(entry[0])
            return True
        return False

    def clear(self):
        self._cache.clear()
        self.current_bytes = 0

    def _evict_if_needed(self):
        while (len(self._cache) > self.max_items) or (self.current_bytes > self.max_bytes):
            # pop least-recently-used (first item)
            k, (data, _) = self._cache.popitem(last=False)
            self.current_bytes -= len(data)