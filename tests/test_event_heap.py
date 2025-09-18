import numpy as np
import pytest
from src.event_heap import EventHeap
from src.events import BallBallCollision, BallWallCollision, ExportEvent, EndEvent
from src.ball import Ball
from src.wall import Wall


class TestEventHeap:
    
    def test_empty_heap(self):
        heap = EventHeap()
        assert heap.is_empty()
        assert heap.size() == 0
        assert heap.get_next_event() is None
    
    def test_add_single_event(self):
        heap = EventHeap()
        
        # Create a simple export event
        event = ExportEvent(5.0)
        heap.add_event(event)
        
        assert not heap.is_empty()
        assert heap.size() == 1
        
        next_event = heap.get_next_event()
        assert next_event is event
        assert next_event.time == 5.0
        assert heap.is_empty()
    
    def test_event_ordering(self):
        heap = EventHeap()
        
        # Add events in non-chronological order
        event1 = ExportEvent(10.0)
        event2 = ExportEvent(5.0)
        event3 = ExportEvent(15.0)
        event4 = ExportEvent(7.5)
        
        heap.add_event(event1)
        heap.add_event(event2)
        heap.add_event(event3)
        heap.add_event(event4)
        
        assert heap.size() == 4
        
        # Should get events in chronological order
        assert heap.get_next_event().time == 5.0
        assert heap.get_next_event().time == 7.5
        assert heap.get_next_event().time == 10.0
        assert heap.get_next_event().time == 15.0
        assert heap.is_empty()
    
    def test_invalid_event_filtering(self):
        heap = EventHeap()
        
        # Create events and mark some as invalid
        event1 = ExportEvent(5.0)
        event2 = ExportEvent(10.0)
        event3 = ExportEvent(15.0)
        
        event2.valid = False  # Mark as invalid
        
        heap.add_event(event1)
        heap.add_event(event2)
        heap.add_event(event3)
        
        assert heap.size() == 3
        
        # Should skip invalid event
        assert heap.get_next_event() is event1
        assert heap.get_next_event() is event3  # event2 skipped
        assert heap.get_next_event() is None
    
    def test_all_invalid_events(self):
        heap = EventHeap()
        
        # Add events and mark all as invalid
        event1 = ExportEvent(5.0)
        event2 = ExportEvent(10.0)
        
        event1.valid = False
        event2.valid = False
        
        heap.add_event(event1)
        heap.add_event(event2)
        
        assert heap.size() == 2
        assert heap.get_next_event() is None  # Should return None after discarding all
        assert heap.is_empty()
    
    def test_mixed_event_types(self):
        heap = EventHeap()
        
        # Create different types of events
        ball1 = Ball(np.array([1.0, 1.0]), np.array([1.0, 0.0]), 0.1, 0, (1, 1))
        ball2 = Ball(np.array([2.0, 1.0]), np.array([-1.0, 0.0]), 0.1, 1, (2, 1))
        wall = Wall(0, 0.01, 1.0)
        
        collision_event = BallBallCollision(5.0, ball1, ball2)
        wall_event = BallWallCollision(8.0, ball1, wall)
        export_event = ExportEvent(10.0)
        end_event = EndEvent(20.0)
        
        heap.add_event(wall_event)
        heap.add_event(end_event)
        heap.add_event(collision_event)
        heap.add_event(export_event)
        
        # Should get events in time order regardless of type
        assert heap.get_next_event() is collision_event
        assert heap.get_next_event() is wall_event
        assert heap.get_next_event() is export_event
        assert heap.get_next_event() is end_event
    
    def test_same_time_events(self):
        heap = EventHeap()
        
        # Add events with same time
        event1 = ExportEvent(5.0)
        event2 = ExportEvent(5.0)
        event3 = ExportEvent(5.0)
        
        heap.add_event(event1)
        heap.add_event(event2)
        heap.add_event(event3)
        
        # All should be retrievable (order not guaranteed for same time)
        events = []
        while not heap.is_empty():
            events.append(heap.get_next_event())
        
        assert len(events) == 3
        assert all(event.time == 5.0 for event in events)
        assert set(events) == {event1, event2, event3}
    
    def test_heap_after_invalidation(self):
        heap = EventHeap()
        
        # Simulate the pattern used in simulation: 
        # add events, invalidate some, then get next
        ball1 = Ball(np.array([1.0, 1.0]), np.array([1.0, 0.0]), 0.1, 0, (1, 1))
        ball2 = Ball(np.array([2.0, 1.0]), np.array([-1.0, 0.0]), 0.1, 1, (2, 1))
        
        event1 = BallBallCollision(5.0, ball1, ball2)
        event2 = ExportEvent(10.0)
        event3 = ExportEvent(15.0)
        
        heap.add_event(event1)
        heap.add_event(event2)
        heap.add_event(event3)
        
        # Simulate collision processing - invalidate ball events
        ball1.invalidate_all_events()  # This should invalidate event1
        
        # Should skip invalidated event
        assert heap.get_next_event() is event2
        assert heap.get_next_event() is event3