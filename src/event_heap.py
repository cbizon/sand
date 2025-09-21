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
            import json
            discard_log = {
                "event_type": "EventDiscarded",
                "time": event.time,
                "discarded_event": str(event),
                "reason": "invalid"
            }
            
            # Add ball information based on event type
            if hasattr(event, 'ball1') and hasattr(event, 'ball2'):
                # BallBallCollision
                discard_log["ball1"] = event.ball1.index
                discard_log["ball2"] = event.ball2.index
                discard_log["event_subtype"] = "BallBallCollision"
            elif hasattr(event, 'ball') and hasattr(event, 'wall'):
                # BallWallCollision
                discard_log["ball"] = event.ball.index
                discard_log["wall"] = str(event.wall)
                discard_log["event_subtype"] = "BallWallCollision"
            elif hasattr(event, 'ball') and hasattr(event, 'new_cell'):
                # BallGridTransit
                discard_log["ball"] = event.ball.index
                discard_log["from_cell"] = list(event.ball.cell)
                discard_log["to_cell"] = list(event.new_cell)
                discard_log["event_subtype"] = "BallGridTransit"
            elif hasattr(event, 'ball'):
                # Other ball events
                discard_log["ball"] = event.ball.index
                
            print(json.dumps(discard_log))
        
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