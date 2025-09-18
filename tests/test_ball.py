import numpy as np
import pytest
from src.ball import Ball


class TestBall:
    def test_ball_initialization(self):
        """Test basic ball initialization."""
        position = np.array([1.0, 2.0])
        velocity = np.array([0.5, -0.3])
        radius = 0.1
        index = 0
        cell = (1, 2)
        
        ball = Ball(position, velocity, radius, index, cell)
        
        assert np.array_equal(ball.position, position)
        assert np.array_equal(ball.velocity, velocity)
        assert ball.radius == radius
        assert ball.index == index
        assert ball.cell == cell
        assert ball.time == 0.0
        assert len(ball.events) == 0
    
    def test_ball_initialization_3d(self):
        """Test ball initialization in 3D."""
        position = np.array([1.0, 2.0, 3.0])
        velocity = np.array([0.5, -0.3, 0.2])
        radius = 0.15
        index = 5
        cell = (1, 2, 3)
        time = 1.5
        
        ball = Ball(position, velocity, radius, index, cell, time)
        
        assert np.array_equal(ball.position, position)
        assert np.array_equal(ball.velocity, velocity)
        assert ball.cell == cell
        assert ball.time == time
    
    def test_position_independence(self):
        """Test that ball stores independent copies of position and velocity."""
        original_pos = np.array([1.0, 2.0])
        original_vel = np.array([0.5, -0.3])
        
        ball = Ball(original_pos, original_vel, 0.1, 0, (1, 2))
        
        # Modify original arrays
        original_pos[0] = 99.0
        original_vel[0] = 99.0
        
        # Ball should have unchanged values
        assert ball.position[0] == 1.0
        assert ball.velocity[0] == 0.5
    
    def test_get_position_at_time_no_gravity(self):
        """Test position calculation at future time without gravity."""
        position = np.array([1.0, 2.0])
        velocity = np.array([0.5, -0.3])
        ball = Ball(position, velocity, 0.1, 0, (1, 2), time=1.0)
        
        # Calculate position 2 seconds later
        new_pos = ball.get_position_at_time(3.0, ndim=2, gravity=False)
        expected_pos = np.array([1.0 + 0.5 * 2, 2.0 + (-0.3) * 2])
        
        np.testing.assert_array_almost_equal(new_pos, expected_pos)
    
    def test_get_position_at_time_with_gravity_2d(self):
        """Test position calculation with gravity in 2D."""
        position = np.array([1.0, 2.0])
        velocity = np.array([0.5, -0.3])
        ball = Ball(position, velocity, 0.1, 0, (1, 2), time=1.0)
        
        # Calculate position 2 seconds later with gravity
        new_pos = ball.get_position_at_time(3.0, ndim=2, gravity=True)
        dt = 2.0
        expected_pos = np.array([
            1.0 + 0.5 * dt,  # x unchanged by gravity
            2.0 + (-0.3) * dt + (-0.5) * dt * dt  # y affected by gravity (g=1)
        ])
        
        np.testing.assert_array_almost_equal(new_pos, expected_pos)
    
    def test_get_position_at_time_with_gravity_3d(self):
        """Test position calculation with gravity in 3D."""
        position = np.array([1.0, 2.0, 3.0])
        velocity = np.array([0.5, -0.3, 0.2])
        ball = Ball(position, velocity, 0.1, 0, (1, 2, 3), time=0.0)
        
        # Calculate position 1 second later with gravity
        new_pos = ball.get_position_at_time(1.0, ndim=3, gravity=True)
        expected_pos = np.array([
            1.0 + 0.5,  # x unchanged by gravity
            2.0 + (-0.3) + (-0.5),  # y affected by gravity (g=1)
            3.0 + 0.2   # z unchanged by gravity
        ])
        
        np.testing.assert_array_almost_equal(new_pos, expected_pos)
    
    def test_get_position_at_time_past_time_error(self):
        """Test that getting position at past time raises error."""
        ball = Ball(np.array([1.0, 2.0]), np.array([0.5, -0.3]), 0.1, 0, (1, 2), time=5.0)
        
        with pytest.raises(ValueError, match="Cannot get position at time 3.0 before ball's current time 5.0"):
            ball.get_position_at_time(3.0, ndim=2)
    
    def test_update_to_time_no_gravity(self):
        """Test updating ball to future time without gravity."""
        position = np.array([1.0, 2.0])
        velocity = np.array([0.5, -0.3])
        ball = Ball(position, velocity, 0.1, 0, (1, 2), time=1.0)
        
        ball.update_to_time(3.0, ndim=2, gravity=False)
        
        expected_pos = np.array([1.0 + 0.5 * 2, 2.0 + (-0.3) * 2])
        np.testing.assert_array_almost_equal(ball.position, expected_pos)
        np.testing.assert_array_almost_equal(ball.velocity, velocity)  # velocity unchanged
        assert ball.time == 3.0
    
    def test_update_to_time_with_gravity(self):
        """Test updating ball to future time with gravity."""
        position = np.array([1.0, 2.0])
        velocity = np.array([0.5, -0.3])
        ball = Ball(position, velocity, 0.1, 0, (1, 2), time=1.0)
        
        ball.update_to_time(3.0, ndim=2, gravity=True)
        
        dt = 2.0
        expected_pos = np.array([
            1.0 + 0.5 * dt,
            2.0 + (-0.3) * dt + (-0.5) * dt * dt
        ])
        expected_vel = np.array([
            0.5,  # x velocity unchanged
            -0.3 + (-dt)  # y velocity affected by gravity
        ])
        
        np.testing.assert_array_almost_equal(ball.position, expected_pos)
        np.testing.assert_array_almost_equal(ball.velocity, expected_vel)
        assert ball.time == 3.0
    
    def test_update_to_time_past_time_error(self):
        """Test that updating to past time raises error."""
        ball = Ball(np.array([1.0, 2.0]), np.array([0.5, -0.3]), 0.1, 0, (1, 2), time=5.0)
        
        with pytest.raises(ValueError, match="Cannot update to time 3.0 before ball's current time 5.0"):
            ball.update_to_time(3.0, ndim=2)
    
    def test_invalidate_all_events(self):
        """Test event invalidation."""
        ball = Ball(np.array([1.0, 2.0]), np.array([0.5, -0.3]), 0.1, 0, (1, 2))
        
        # Create mock events
        class MockEvent:
            def __init__(self):
                self.valid = True
        
        event1 = MockEvent()
        event2 = MockEvent()
        
        ball.events = [event1, event2]
        
        ball.invalidate_all_events()
        
        assert not event1.valid
        assert not event2.valid
        assert len(ball.events) == 0
    
    def test_add_event(self):
        """Test adding events to ball."""
        ball = Ball(np.array([1.0, 2.0]), np.array([0.5, -0.3]), 0.1, 0, (1, 2))
        
        class MockEvent:
            pass
        
        event = MockEvent()
        ball.add_event(event)
        
        assert len(ball.events) == 1
        assert ball.events[0] is event
    
    def test_repr(self):
        """Test string representation of ball."""
        position = np.array([1.0, 2.0])
        velocity = np.array([0.5, -0.3])
        ball = Ball(position, velocity, 0.1, 0, (1, 2), time=1.5)
        
        repr_str = repr(ball)
        assert "Ball(" in repr_str
        assert "pos=[1. 2.]" in repr_str
        assert "vel=[ 0.5 -0.3]" in repr_str
        assert "r=0.1" in repr_str
        assert "i=0" in repr_str
        assert "cell=(1, 2)" in repr_str
        assert "t=1.5" in repr_str