import heapq
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .events import Event


class EventHeap:
    """
    Priority queue for simulation events, ordered by event time.
    
    Automatically discards invalid events when retrieving next event.
    """
    
    def __init__(self):
        """Initialize empty event heap."""
        self._heap = []
    
    def add_event(self, event: 'Event'):
        """
        Add event to heap.
        
        Args:
            event: Event to add
        """
        heapq.heappush(self._heap, event)
    
    def get_next_event(self) -> Optional['Event']:
        """
        Get next valid event from heap.
        
        Automatically discards invalid events until a valid one is found.
        
        Returns:
            Next valid event, or None if heap is empty
        """
        while self._heap:
            event = heapq.heappop(self._heap)
            if event.valid:
                return event
            # Invalid event - discard and continue
            print(f"DISCARDED invalid event: {event}")
        
        return None
    
    def is_empty(self) -> bool:
        """
        Check if heap is empty.
        
        Returns:
            True if no events remain
        """
        return len(self._heap) == 0
    
    def size(self) -> int:
        """
        Get number of events in heap (including invalid ones).
        
        Returns:
            Total number of events
        """
        return len(self._heap)