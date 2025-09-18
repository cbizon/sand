import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch
from src.ball import Ball
from src.events import (
    Event, BallBallCollision, BallWallCollision, 
    BallGridTransit, ExportEvent, EndEvent
)


class MockEvent(Event):
    """Mock concrete Event class for testing abstract base class."""
    
    def get_participants(self):
        return []
    
    def process(self, **kwargs):
        pass


class TestEventBase:
    def test_event_initialization(self):
        """Test basic event initialization."""
        event = MockEvent(5.0)
        
        assert event.time == 5.0
        assert event.valid is True
    
    def test_event_ordering(self):
        """Test that events can be ordered by time for heap."""
        event1 = MockEvent(3.0)
        event2 = MockEvent(1.0)
        event3 = MockEvent(2.0)
        
        events = [event1, event2, event3]
        events.sort()
        
        assert events[0].time == 1.0
        assert events[1].time == 2.0
        assert events[2].time == 3.0
    
    def test_event_repr(self):
        """Test string representation."""
        event = MockEvent(2.5)
        repr_str = repr(event)
        
        assert "MockEvent" in repr_str
        assert "t=2.5" in repr_str
        assert "valid=True" in repr_str


class TestBallBallCollision:
    def test_initialization(self):
        """Test ball-ball collision initialization."""
        ball1 = Ball(np.array([1.0, 2.0]), np.array([0.5, -0.3]), 0.1, 0, (1, 2))
        ball2 = Ball(np.array([3.0, 4.0]), np.array([-0.2, 0.4]), 0.1, 1, (3, 4))
        
        collision = BallBallCollision(5.0, ball1, ball2)
        
        assert collision.time == 5.0
        assert collision.ball1 is ball1
        assert collision.ball2 is ball2
        assert collision.valid is True
        
        # Check that event was added to both balls
        assert collision in ball1.events
        assert collision in ball2.events
    
    def test_get_participants(self):
        """Test getting collision participants."""
        ball1 = Ball(np.array([1.0, 2.0]), np.array([0.5, -0.3]), 0.1, 0, (1, 2))
        ball2 = Ball(np.array([3.0, 4.0]), np.array([-0.2, 0.4]), 0.1, 1, (3, 4))
        
        collision = BallBallCollision(5.0, ball1, ball2)
        participants = collision.get_participants()
        
        assert len(participants) == 2
        assert ball1 in participants
        assert ball2 in participants
    
    def test_process(self):
        """Test ball-ball collision processing."""
        ball1 = Ball(np.array([1.0, 2.0]), np.array([0.5, -0.3]), 0.1, 0, (1, 2), time=1.0)
        ball2 = Ball(np.array([3.0, 4.0]), np.array([-0.2, 0.4]), 0.1, 1, (3, 4), time=1.0)
        
        collision = BallBallCollision(3.0, ball1, ball2)
        
        # Mock dependencies
        mock_physics = Mock()
        mock_event_generator = Mock()
        mock_event_heap = Mock()
        mock_event_generator.generate_events_for_ball.return_value = [Mock(), Mock()]
        
        # Mock the physics import
        with patch.dict('sys.modules', {'src.physics': mock_physics}):
            mock_physics.perform_ball_ball_collision = Mock()
            
            collision.process(
                ndim=2,
                gravity=False,
                ball_restitution=1.0,
                event_generator=mock_event_generator,
                event_heap=mock_event_heap
            )
        
        # Check that balls were updated to collision time
        assert ball1.time == 3.0
        assert ball2.time == 3.0
        
        # Check that collision was performed
        mock_physics.perform_ball_ball_collision.assert_called_once_with(ball1, ball2, 1.0)
        
        # Check that events were generated for both balls
        assert mock_event_generator.generate_events_for_ball.call_count == 2
        mock_event_generator.generate_events_for_ball.assert_any_call(ball1)
        mock_event_generator.generate_events_for_ball.assert_any_call(ball2)
        
        # Check that new events were added to heap
        assert mock_event_heap.add_event.call_count == 4  # 2 events per ball


class TestBallWallCollision:
    def test_initialization(self):
        """Test ball-wall collision initialization."""
        ball = Ball(np.array([1.0, 2.0]), np.array([0.5, -0.3]), 0.1, 0, (1, 2))
        wall = Mock()
        
        collision = BallWallCollision(3.0, ball, wall)
        
        assert collision.time == 3.0
        assert collision.ball is ball
        assert collision.wall is wall
        assert collision.valid is True
        
        # Check that event was added to ball
        assert collision in ball.events
    
    def test_get_participants(self):
        """Test getting collision participants."""
        ball = Ball(np.array([1.0, 2.0]), np.array([0.5, -0.3]), 0.1, 0, (1, 2))
        wall = Mock()
        
        collision = BallWallCollision(3.0, ball, wall)
        participants = collision.get_participants()
        
        assert len(participants) == 2
        assert ball in participants
        assert wall in participants
    
    def test_process(self):
        """Test ball-wall collision processing."""
        ball = Ball(np.array([1.0, 2.0]), np.array([0.5, -0.3]), 0.1, 0, (1, 2), time=1.0)
        wall = Mock()
        
        collision = BallWallCollision(3.0, ball, wall)
        
        # Mock dependencies
        mock_physics = Mock()
        mock_event_generator = Mock()
        mock_event_heap = Mock()
        mock_event_generator.generate_events_for_ball.return_value = [Mock(), Mock()]
        
        with patch.dict('sys.modules', {'src.physics': mock_physics}):
            mock_physics.perform_ball_wall_collision = Mock()
            
            collision.process(
                ndim=2,
                gravity=False,
                wall_restitution=1.0,
                event_generator=mock_event_generator,
                event_heap=mock_event_heap
            )
        
        # Check that ball was updated to collision time
        assert ball.time == 3.0
        
        # Check that collision was performed
        mock_physics.perform_ball_wall_collision.assert_called_once_with(ball, wall, 1.0)
        
        # Check that events were generated for ball
        mock_event_generator.generate_events_for_ball.assert_called_once_with(ball)
        
        # Check that new events were added to heap
        assert mock_event_heap.add_event.call_count == 2


class TestBallGridTransit:
    def test_initialization(self):
        """Test ball grid transit initialization."""
        ball = Ball(np.array([1.0, 2.0]), np.array([0.5, -0.3]), 0.1, 0, (1, 2))
        new_cell = (2, 2)
        
        transit = BallGridTransit(2.5, ball, new_cell)
        
        assert transit.time == 2.5
        assert transit.ball is ball
        assert transit.new_cell == new_cell
        assert transit.valid is True
        
        # Check that event was added to ball
        assert transit in ball.events
    
    def test_get_participants(self):
        """Test getting transit participants."""
        ball = Ball(np.array([1.0, 2.0]), np.array([0.5, -0.3]), 0.1, 0, (1, 2))
        
        transit = BallGridTransit(2.5, ball, (2, 2))
        participants = transit.get_participants()
        
        assert len(participants) == 1
        assert ball in participants
    
    def test_process(self):
        """Test ball grid transit processing."""
        ball = Ball(np.array([1.0, 2.0]), np.array([0.5, -0.3]), 0.1, 0, (1, 2))
        old_cell = ball.cell
        new_cell = (2, 2)
        
        transit = BallGridTransit(2.5, ball, new_cell)
        
        # Mock dependencies
        mock_grid = Mock()
        mock_event_generator = Mock()
        mock_event_heap = Mock()
        mock_event_generator.generate_ball_ball_events_for_new_cell.return_value = [Mock()]
        
        transit.process(
            grid=mock_grid,
            event_generator=mock_event_generator,
            event_heap=mock_event_heap
        )
        
        # Check that ball's cell was updated
        assert ball.cell == new_cell
        
        # Check that grid was updated
        mock_grid.move_ball.assert_called_once_with(ball.index, old_cell, new_cell)
        
        # Check that new ball-ball events were generated
        mock_event_generator.generate_ball_ball_events_for_new_cell.assert_called_once_with(ball, old_cell)
        
        # Check that new events were added to heap
        mock_event_heap.add_event.assert_called_once()


class TestExportEvent:
    def test_initialization(self):
        """Test export event initialization."""
        export = ExportEvent(10.0)
        
        assert export.time == 10.0
        assert export.valid is True
    
    def test_get_participants(self):
        """Test that export events have no participants."""
        export = ExportEvent(10.0)
        participants = export.get_participants()
        
        assert len(participants) == 0
    
    def test_process(self):
        """Test export event processing."""
        ball1 = Ball(np.array([1.0, 2.0]), np.array([0.5, -0.3]), 0.1, 0, (1, 2), time=5.0)
        ball2 = Ball(np.array([3.0, 4.0]), np.array([-0.2, 0.4]), 0.1, 1, (3, 4), time=5.0)
        balls = [ball1, ball2]
        
        export = ExportEvent(10.0)
        
        # Mock dependencies
        mock_output_manager = Mock()
        
        export.process(
            balls=balls,
            output_manager=mock_output_manager,
            ndim=2,
            gravity=False
        )
        
        # Check that output was written
        mock_output_manager.write_frame.assert_called_once()
        call_args = mock_output_manager.write_frame.call_args[0]
        
        # Check time
        assert call_args[0] == 10.0
        
        # Check positions (should be calculated for export time)
        positions = call_args[1]
        assert len(positions) == 2
        
        # Check velocities
        velocities = call_args[2]
        assert len(velocities) == 2
        np.testing.assert_array_equal(velocities[0], ball1.velocity)
        np.testing.assert_array_equal(velocities[1], ball2.velocity)


class TestEndEvent:
    def test_initialization(self):
        """Test end event initialization."""
        end = EndEvent(100.0)
        
        assert end.time == 100.0
        assert end.valid is True
    
    def test_get_participants(self):
        """Test that end events have no participants."""
        end = EndEvent(100.0)
        participants = end.get_participants()
        
        assert len(participants) == 0
    
    def test_process(self):
        """Test end event processing."""
        end = EndEvent(100.0)
        
        simulation_state = {'should_end': False}
        
        end.process(simulation_state=simulation_state)
        
        # Check that simulation was flagged to end
        assert simulation_state['should_end'] is True