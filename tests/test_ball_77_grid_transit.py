"""
Test case for ball 77 grid transit event at t=1.978.
Verifies that the next grid transit should be y=2 crossing, not x=30 crossing.
"""

import numpy as np
import pytest
from src.ball import Ball
from src.physics import calculate_ball_grid_transit_time


def test_ball_77_next_grid_transit_should_be_y2_not_x30():
    """
    Test that ball 77 at t=1.978 should find y=2 crossing next, not x=30.
    
    Ball 77 data from actual event log at t=1.978:
    - Position: [27.892522024767487, 1.9493552007879855] at t=0.903
    - Velocity: [1.0299902633547728, 0.6188638938455122]
    - Cell transition at t=1.978: (28,2) -> (29,2) via x=29 crossing
    
    The next grid transit should be y=2 crossing (downward), not x=30 crossing.
    """
    # Ball 77 data from the event log
    ball_index = 77
    initial_position = np.array([27.892522024767487, 1.9493552007879855])
    velocity = np.array([1.0299902633547728, 0.6188638938455122])
    initial_time = 0.9028911084157597
    
    # Event time when we're processing the grid transit (after x=29 crossing)
    event_time = 1.9781226078139742
    
    # At event time, ball is now in cell (29, 2) after crossing x=29 boundary
    current_cell = (29, 2)
    
    # Calculate where the ball actually is at event_time
    dt = event_time - initial_time
    pos_at_event = initial_position + velocity * dt
    pos_at_event[1] -= 0.5 * 1.0 * dt**2  # gravity effect on y
    
    # Create ball object at initial time, then update to event time
    ball = Ball(initial_position, velocity, radius=0.45, index=ball_index, cell=(28, 2))
    ball.time = initial_time
    
    # Update ball to event time (this correctly updates position and velocity)
    ball.update_to_time(event_time, ndim=2, gravity=True)
    ball.cell = current_cell  # Update cell after the grid transit
    
    # Calculate next grid transit
    result = calculate_ball_grid_transit_time(ball, event_time, ndim=2, cell_size=1.0, gravity=True)
    
    assert result is not None, "Should find a next grid transit"
    
    next_transit_time, next_cell = result
    
    # Calculate expected y=2 crossing time manually
    # First get the correct velocity at event time (after gravity has acted)
    pos_check, vel_at_event = ball.get_position_and_velocity_at_time(event_time, ndim=2, gravity=True)
    
    # y(t) = y0 + vy*(t-t0) - 0.5*g*(t-t0)^2 = 2.0
    # -0.5*(t-t0)^2 + vy*(t-t0) + (y0 - 2.0) = 0
    y0 = pos_at_event[1]
    vy = vel_at_event[1]  # Use velocity at event time, not original velocity
    
    a = -0.5
    b = vy
    c = y0 - 2.0
    
    discriminant = b*b - 4*a*c
    assert discriminant >= 0, "Should have real solutions for y=2 crossing"
    
    sqrt_discriminant = np.sqrt(discriminant)
    dt1 = (-b + sqrt_discriminant) / (2*a)
    dt2 = (-b - sqrt_discriminant) / (2*a)
    
    # Find future y=2 crossings
    future_y2_times = []
    for dt in [dt1, dt2]:
        if dt > 1e-12:  # Future crossing
            t_y2 = event_time + dt
            future_y2_times.append(t_y2)
    
    assert len(future_y2_times) > 0, "Should find at least one future y=2 crossing"
    
    earliest_y2_time = min(future_y2_times)
    
    # Calculate expected x=30 crossing time
    # x(t) = x0 + vx*(t-t0) = 30.0
    x0 = pos_at_event[0]
    vx = vel_at_event[0]  # Use velocity at event time
    dt_x30 = (30.0 - x0) / vx
    t_x30 = event_time + dt_x30
    
    # Y=2 crossing should happen BEFORE x=30 crossing
    assert earliest_y2_time < t_x30, f"Y=2 crossing ({earliest_y2_time:.6f}) should happen before x=30 crossing ({t_x30:.6f})"
    
    # The next grid transit should be the y=2 crossing (downward motion)
    expected_next_cell = (29, 1)  # Moving down from (29,2) to (29,1)
    
    # Check that we found the correct next transit
    time_tolerance = 1e-6
    assert abs(next_transit_time - earliest_y2_time) < time_tolerance, \
        f"Expected y=2 crossing at t={earliest_y2_time:.6f}, but got t={next_transit_time:.6f}"
    
    assert next_cell == expected_next_cell, \
        f"Expected cell transition to {expected_next_cell}, but got {next_cell}"
    
    # Additional verification: the found transit should NOT be x=30 crossing
    assert abs(next_transit_time - t_x30) > time_tolerance, \
        f"Found x=30 crossing (t={t_x30:.6f}) instead of y=2 crossing (t={earliest_y2_time:.6f})"
    
    assert next_cell != (30, 2), \
        f"Found x=30 cell transition {next_cell} instead of y=2 transition {expected_next_cell}"


def test_ball_77_position_at_event_time():
    """
    Verify that ball 77's calculated position at event time is correct.
    """
    # Ball 77 data
    initial_position = np.array([27.892522024767487, 1.9493552007879855])
    velocity = np.array([1.0299902633547728, 0.6188638938455122])
    initial_time = 0.9028911084157597
    event_time = 1.9781226078139742
    
    # Calculate position at event time
    dt = event_time - initial_time
    pos_at_event = initial_position + velocity * dt
    pos_at_event[1] -= 0.5 * 1.0 * dt**2  # gravity effect
    
    # Should be very close to x=29 boundary (since it just crossed)
    assert abs(pos_at_event[0] - 29.0) < 1e-6, f"X position should be ~29.0, got {pos_at_event[0]}"
    
    # Y position should be above 2.0 (ball went up due to positive vy)
    assert pos_at_event[1] > 2.0, f"Y position should be above 2.0, got {pos_at_event[1]}"