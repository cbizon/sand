import numpy as np
import pytest
from src.ball import Ball
from src.wall import Wall
from src.physics import (
    calculate_ball_ball_collision_time,
    calculate_ball_wall_collision_time,
    calculate_ball_grid_transit_time,
    perform_ball_ball_collision,
    perform_ball_wall_collision
)


def validate_collision_physics(ball1_before, ball2_before, ball1_after, ball2_after, restitution=1.0):
    """
    Validate physics constraints for ball-ball collision.
    
    Args:
        ball1_before, ball2_before: Ball objects before collision
        ball1_after, ball2_after: Ball objects after collision  
        restitution: coefficient of restitution (1.0 for elastic)
    """
    # Calculate relative position and velocity vectors
    r = ball1_before.position - ball2_before.position
    v_rel_before = ball1_before.velocity - ball2_before.velocity
    v_rel_after = ball1_after.velocity - ball2_after.velocity
    
    # Normalize relative position vector
    r_unit = r / np.linalg.norm(r)
    
    # Pre-collision check: r·v < 0 (balls approaching)
    r_dot_v_before = np.dot(r_unit, v_rel_before)
    assert r_dot_v_before < 0, f"Pre-collision: balls not approaching, r·v = {r_dot_v_before:.6f}"
    
    # Post-collision check: r·v > 0 (balls separating) 
    r_dot_v_after = np.dot(r_unit, v_rel_after)
    assert r_dot_v_after > 0, f"Post-collision: balls not separating, r·v = {r_dot_v_after:.6f}"
    
    # Conservation of momentum (vector-wise)
    momentum_before = ball1_before.velocity + ball2_before.velocity
    momentum_after = ball1_after.velocity + ball2_after.velocity
    momentum_diff = np.linalg.norm(momentum_after - momentum_before)
    assert momentum_diff < 1e-10, f"Momentum not conserved: diff = {momentum_diff:.2e}"
    
    # Conservation of energy (for e=1)
    if abs(restitution - 1.0) < 1e-10:
        ke_before = 0.5 * (np.dot(ball1_before.velocity, ball1_before.velocity) + 
                          np.dot(ball2_before.velocity, ball2_before.velocity))
        ke_after = 0.5 * (np.dot(ball1_after.velocity, ball1_after.velocity) + 
                         np.dot(ball2_after.velocity, ball2_after.velocity))
        energy_diff = abs(ke_after - ke_before)
        assert energy_diff < 1e-10, f"Energy not conserved: diff = {energy_diff:.2e}"


class TestCalculateBallBallCollisionTime:
    def test_head_on_collision_2d(self):
        """Test head-on collision between two balls."""
        ball1 = Ball(np.array([1.0, 2.0]), np.array([1.0, 0.0]), 0.5, 0, (1, 2), time=0.0)
        ball2 = Ball(np.array([4.0, 2.0]), np.array([-1.0, 0.0]), 0.5, 1, (4, 2), time=0.0)
        
        collision_time = calculate_ball_ball_collision_time(ball1, ball2, 0.0, 2, gravity=False)
        
        # Distance between centers: 3.0, touch distance: 1.0, approach speed: 2.0
        # Time to collision: (3.0 - 1.0) / 2.0 = 1.0
        assert collision_time == pytest.approx(1.0)
    
    def test_balls_moving_apart(self):
        """Test balls moving away from each other."""
        ball1 = Ball(np.array([1.0, 2.0]), np.array([-1.0, 0.0]), 0.5, 0, (1, 2), time=0.0)
        ball2 = Ball(np.array([4.0, 2.0]), np.array([1.0, 0.0]), 0.5, 1, (4, 2), time=0.0)
        
        collision_time = calculate_ball_ball_collision_time(ball1, ball2, 0.0, 2, gravity=False)
        
        # Balls moving apart - no collision
        assert collision_time is None
    
    def test_parallel_motion_no_collision(self):
        """Test balls moving in parallel - no collision."""
        ball1 = Ball(np.array([1.0, 2.0]), np.array([1.0, 0.0]), 0.5, 0, (1, 2), time=0.0)
        ball2 = Ball(np.array([1.0, 4.0]), np.array([1.0, 0.0]), 0.5, 1, (1, 4), time=0.0)
        
        collision_time = calculate_ball_ball_collision_time(ball1, ball2, 0.0, 2, gravity=False)
        
        # Parallel motion, separated by 2.0, touch distance 1.0 - no collision
        assert collision_time is None
    
    def test_stationary_balls(self):
        """Test stationary balls."""
        ball1 = Ball(np.array([1.0, 2.0]), np.array([0.0, 0.0]), 0.5, 0, (1, 2), time=0.0)
        ball2 = Ball(np.array([4.0, 2.0]), np.array([0.0, 0.0]), 0.5, 1, (4, 2), time=0.0)
        
        collision_time = calculate_ball_ball_collision_time(ball1, ball2, 0.0, 2, gravity=False)
        
        # No relative motion - no collision
        assert collision_time is None
    
    def test_glancing_collision(self):
        """Test glancing collision at an angle."""
        ball1 = Ball(np.array([0.0, 0.0]), np.array([1.0, 0.5]), 0.5, 0, (0, 0), time=0.0)
        ball2 = Ball(np.array([2.0, 0.5]), np.array([-0.5, 0.0]), 0.5, 1, (2, 0), time=0.0)
        
        collision_time = calculate_ball_ball_collision_time(ball1, ball2, 0.0, 2, gravity=False)
        
        # Should find a collision time
        assert collision_time is not None
        assert collision_time > 0
    
    
    def test_with_gravity_2d(self):
        """Test collision calculation with gravity."""
        ball1 = Ball(np.array([1.0, 2.0]), np.array([1.0, 0.0]), 0.5, 0, (1, 2), time=0.0)
        ball2 = Ball(np.array([4.0, 2.0]), np.array([-1.0, 0.0]), 0.5, 1, (4, 2), time=0.0)
        
        collision_time = calculate_ball_ball_collision_time(ball1, ball2, 0.0, 2, gravity=True)
        
        # Gravity affects position but cancels in relative motion - same result as no gravity
        assert collision_time == pytest.approx(1.0)


class TestCalculateBallWallCollisionTime:
    def test_ball_approaching_vertical_wall(self):
        """Test ball approaching a vertical wall."""
        ball = Ball(np.array([1.0, 2.0]), np.array([2.0, 0.0]), 0.3, 0, (1, 2), time=0.0)
        wall = Wall(0, 5.0, restitution=1.0)  # x-normal wall at x=5.0
        
        collision_time = calculate_ball_wall_collision_time(ball, wall, 0.0, 2, gravity=False)
        
        # Ball at x=1.0, wall at x=5.0, radius=0.3, velocity=2.0
        # Collision when ball surface reaches wall: 5.0 - 0.3 = 4.7
        # Time: (4.7 - 1.0) / 2.0 = 1.85
        assert collision_time == pytest.approx(1.85)
    
    def test_ball_moving_away_from_wall(self):
        """Test ball moving away from wall."""
        ball = Ball(np.array([6.0, 2.0]), np.array([1.0, 0.0]), 0.3, 0, (6, 2), time=0.0)
        wall = Wall(0, 5.0, restitution=1.0)  # x-normal wall at x=5.0
        
        collision_time = calculate_ball_wall_collision_time(ball, wall, 0.0, 2, gravity=False)
        
        # Ball moving away from wall - no collision
        assert collision_time is None
    
    def test_ball_parallel_to_wall(self):
        """Test ball moving parallel to wall."""
        ball = Ball(np.array([3.0, 2.0]), np.array([0.0, 1.0]), 0.3, 0, (3, 2), time=0.0)
        wall = Wall(0, 5.0, restitution=1.0)  # x-normal wall at x=5.0
        
        collision_time = calculate_ball_wall_collision_time(ball, wall, 0.0, 2, gravity=False)
        
        # No motion toward wall - no collision
        assert collision_time is None
    
    
    def test_horizontal_wall_with_gravity(self):
        """Test ball falling onto horizontal wall with gravity."""
        ball = Ball(np.array([2.0, 5.0]), np.array([0.0, -1.0]), 0.4, 0, (2, 5), time=0.0)
        wall = Wall(1, 2.0, restitution=1.0)  # y-normal wall at y=2.0
        
        collision_time = calculate_ball_wall_collision_time(ball, wall, 0.0, 2, gravity=True)
        
        # Ball at y=5.0, moving down with initial velocity -1.0, with gravity
        # Collision occurs when ball surface touches wall at y=2.0
        # Since ball is above wall, collision_coord = wall.coordinate + ball.radius = 2.0 + 0.4 = 2.4
        # Equation: 5.0 + (-1.0)*t - 0.5*t^2 = 2.4
        # Rearranging: 0.5*t^2 + t - 2.6 = 0
        # Using quadratic formula: t = (-1 + sqrt(1 + 5.2)) / 1
        expected_time = (-1 + np.sqrt(1 + 5.2))
        assert collision_time == pytest.approx(expected_time)
    
    def test_ball_hitting_ceiling_upward_motion(self):
        """Test ball moving upward hitting ceiling (top wall)."""
        # Ball below ceiling, moving upward, should hit ceiling
        ball = Ball(np.array([2.0, 3.0]), np.array([0.0, 2.0]), 0.3, 0, (2, 3), time=0.0)
        wall = Wall(1, 5.0, restitution=1.0)  # ceiling at y=5.0
        
        collision_time = calculate_ball_wall_collision_time(ball, wall, 0.0, 2, gravity=True)
        
        # Ball at y=3.0, moving up with velocity +2.0, ceiling at y=5.0, radius=0.3
        # Ball center must reach y = 5.0 - 0.3 = 4.7 to touch ceiling
        # Equation: 3.0 + 2.0*t - 0.5*t^2 = 4.7
        # Rearrange: 0.5*t^2 - 2.0*t + (4.7 - 3.0) = 0
        # 0.5*t^2 - 2.0*t + 1.7 = 0
        # t = (2 ± sqrt(4 - 3.4))/1 = (2 ± sqrt(0.6))/1
        # Take smaller positive solution: t = 2 - sqrt(0.6)
        expected_time = 2 - np.sqrt(0.6)
        assert collision_time == pytest.approx(expected_time)
    
    def test_ball_missing_ceiling_due_to_gravity(self):
        """Test ball that would hit ceiling but gravity makes it fall back down."""
        # Ball below ceiling, weak upward velocity, gravity pulls it back down before hitting
        ball = Ball(np.array([2.0, 3.0]), np.array([0.0, 0.5]), 0.3, 0, (2, 3), time=0.0)
        wall = Wall(1, 5.0, restitution=1.0)  # ceiling at y=5.0
        
        collision_time = calculate_ball_wall_collision_time(ball, wall, 0.0, 2, gravity=True)
        
        # Ball at y=3.0, moving up with weak velocity +0.5, ceiling at y=5.0, radius=0.3
        # Ball center must reach y = 5.0 - 0.3 = 4.7 to touch ceiling
        # Equation: 3.0 + 0.5*t - 0.5*t^2 = 4.7
        # 0.5*t^2 - 0.5*t + (3.0 - 4.7) = 0
        # 0.5*t^2 - 0.5*t - 1.7 = 0
        # t^2 - t - 3.4 = 0
        # Discriminant = 1 + 13.6 = 14.6 > 0, so there should be solutions
        # But let's check: max height = 3.0 + (0.5^2)/(2*0.5) = 3.0 + 0.25 = 3.25
        # Since 3.25 < 4.7, ball never reaches ceiling
        assert collision_time is None
    
    def test_vertical_wall_with_gravity(self):
        """Test ball hitting vertical wall with gravity (gravity doesn't affect x-motion)."""
        ball = Ball(np.array([1.0, 3.0]), np.array([1.5, 0.0]), 0.2, 0, (1, 3), time=0.0)
        wall = Wall(0, 4.0, restitution=1.0)  # x-normal wall at x=4.0
        
        collision_time = calculate_ball_wall_collision_time(ball, wall, 0.0, 2, gravity=True)
        
        # Gravity doesn't affect x-motion: (4.0 - 0.2 - 1.0) / 1.5 = 2.8 / 1.5
        assert collision_time == pytest.approx(2.8 / 1.5)


class TestCalculateBallGridTransitTime:
    def test_ball_crossing_cell_boundary_x(self):
        """Test ball crossing cell boundary in x-direction."""
        ball = Ball(np.array([1.8, 2.3]), np.array([0.5, 0.0]), 0.1, 0, (1, 2), time=0.0)
        
        result = calculate_ball_grid_transit_time(ball, 0.0, 2, cell_size=1.0, gravity=False)
        
        # Ball at x=1.8, moving right at 0.5, cell boundary at x=2.0
        # Time: (2.0 - 1.8) / 0.5 = 0.4
        assert result is not None
        collision_time, new_cell = result
        assert collision_time == pytest.approx(0.4)
        assert new_cell == (2, 2)
    
    def test_ball_crossing_cell_boundary_y(self):
        """Test ball crossing cell boundary in y-direction."""
        ball = Ball(np.array([1.3, 2.7]), np.array([0.0, 0.6]), 0.1, 0, (1, 2), time=0.0)
        
        result = calculate_ball_grid_transit_time(ball, 0.0, 2, cell_size=1.0, gravity=False)
        
        # Ball at y=2.7, moving up at 0.6, cell boundary at y=3.0
        # Time: (3.0 - 2.7) / 0.6 = 0.5
        assert result is not None
        collision_time, new_cell = result
        assert collision_time == pytest.approx(0.5)
        assert new_cell == (1, 3)
    
    def test_ball_stationary(self):
        """Test stationary ball."""
        ball = Ball(np.array([1.5, 2.5]), np.array([0.0, 0.0]), 0.1, 0, (1, 2), time=0.0)
        
        result = calculate_ball_grid_transit_time(ball, 0.0, 2, cell_size=1.0, gravity=False)
        
        # No motion - no transit
        assert result is None
    
    def test_ball_moving_left_boundary(self):
        """Test ball crossing left cell boundary."""
        ball = Ball(np.array([1.2, 2.5]), np.array([-0.8, 0.0]), 0.1, 0, (1, 2), time=0.0)
        
        result = calculate_ball_grid_transit_time(ball, 0.0, 2, cell_size=1.0, gravity=False)
        
        # Ball at x=1.2, moving left at -0.8, left boundary at x=1.0
        # Time: (1.0 - 1.2) / (-0.8) = 0.25
        assert result is not None
        collision_time, new_cell = result
        assert collision_time == pytest.approx(0.25)
        assert new_cell == (0, 2)
    
    def test_earliest_boundary_crossing(self):
        """Test ball crossing multiple boundaries - should return earliest."""
        ball = Ball(np.array([1.9, 2.8]), np.array([1.0, 2.0]), 0.1, 0, (1, 2), time=0.0)
        
        result = calculate_ball_grid_transit_time(ball, 0.0, 2, cell_size=1.0, gravity=False)
        
        # x-boundary at t = (2.0 - 1.9) / 1.0 = 0.1
        # y-boundary at t = (3.0 - 2.8) / 2.0 = 0.1
        # Both equal, but x should be processed first
        assert result is not None
        collision_time, new_cell = result
        assert collision_time == pytest.approx(0.1)
        # Should be x-direction (first processed)
        assert new_cell == (2, 2)
    
    def test_gravity_affecting_y_motion(self):
        """Test ball crossing y-boundary with gravity."""
        ball = Ball(np.array([1.5, 2.8]), np.array([0.0, 0.5]), 0.1, 0, (1, 2), time=0.0)
        
        result = calculate_ball_grid_transit_time(ball, 0.0, 2, cell_size=1.0, gravity=True)
        
        # Solve: 2.8 + 0.5*t - 0.5*t^2 = 3.0
        # 0.5*t^2 - 0.5*t + 0.2 = 0
        # t^2 - t + 0.4 = 0
        # t = (1 ± sqrt(1 - 1.6)) / 2 - discriminant negative means no real solution
        # But let's check if ball might cross downward boundary
        # 2.8 + 0.5*t - 0.5*t^2 = 2.0
        # 0.5*t^2 - 0.5*t - 0.8 = 0
        # t^2 - t - 1.6 = 0
        # t = (1 + sqrt(1 + 6.4)) / 2 = (1 + sqrt(7.4)) / 2
        assert result is not None
        collision_time, new_cell = result
        expected_time = (1 + np.sqrt(7.4)) / 2
        assert collision_time == pytest.approx(expected_time)
        assert new_cell == (1, 1)  # Moving down to cell below


class TestPerformBallBallCollision:
    def test_head_on_elastic_collision(self):
        """Test head-on elastic collision."""
        ball1 = Ball(np.array([2.0, 2.0]), np.array([1.0, 0.0]), 0.5, 0, (2, 2), time=1.0)
        ball2 = Ball(np.array([3.0, 2.0]), np.array([-1.0, 0.0]), 0.5, 1, (3, 2), time=1.0)
        
        # Store original states for validation
        ball1_before = Ball(ball1.position.copy(), ball1.velocity.copy(), ball1.radius, ball1.index, ball1.cell)
        ball2_before = Ball(ball2.position.copy(), ball2.velocity.copy(), ball2.radius, ball2.index, ball2.cell)
        
        perform_ball_ball_collision(ball1, ball2, restitution=1.0)
        
        # Perfect elastic collision - velocities should be exchanged
        np.testing.assert_array_almost_equal(ball1.velocity, [-1.0, 0.0])
        np.testing.assert_array_almost_equal(ball2.velocity, [1.0, 0.0])
        
        # Validate physics
        validate_collision_physics(ball1_before, ball2_before, ball1, ball2, restitution=1.0)
    
    def test_inelastic_collision(self):
        """Test inelastic collision."""
        ball1 = Ball(np.array([2.0, 2.0]), np.array([2.0, 0.0]), 0.5, 0, (2, 2), time=1.0)
        ball2 = Ball(np.array([3.0, 2.0]), np.array([0.0, 0.0]), 0.5, 1, (3, 2), time=1.0)
        
        perform_ball_ball_collision(ball1, ball2, restitution=0.5)
        
        # Relative velocity should be reduced by restitution factor
        # Initial relative velocity: -2.0 (ball2 - ball1)
        # Final relative velocity: 2.0 * 0.5 = 1.0
        # So ball2.vel - ball1.vel = 1.0
        relative_vel = ball2.velocity[0] - ball1.velocity[0]
        assert relative_vel == pytest.approx(1.0)
    
    def test_balls_separating_no_collision(self):
        """Test balls that are separating - no collision should occur."""
        ball1 = Ball(np.array([2.0, 2.0]), np.array([-1.0, 0.0]), 0.5, 0, (2, 2), time=1.0)
        ball2 = Ball(np.array([3.0, 2.0]), np.array([1.0, 0.0]), 0.5, 1, (3, 2), time=1.0)
        
        original_vel1 = ball1.velocity.copy()
        original_vel2 = ball2.velocity.copy()
        
        perform_ball_ball_collision(ball1, ball2, restitution=1.0)
        
        # Velocities should be unchanged (balls separating)
        np.testing.assert_array_almost_equal(ball1.velocity, original_vel1)
        np.testing.assert_array_almost_equal(ball2.velocity, original_vel2)
    
    def test_angled_collision(self):
        """Test collision at an angle."""
        ball1 = Ball(np.array([2.0, 2.0]), np.array([1.0, 1.0]), 0.5, 0, (2, 2), time=1.0)
        ball2 = Ball(np.array([2.5, 2.5]), np.array([-0.5, -0.5]), 0.5, 1, (2, 2), time=1.0)
        
        # Store original states for validation
        ball1_before = Ball(ball1.position.copy(), ball1.velocity.copy(), ball1.radius, ball1.index, ball1.cell)
        ball2_before = Ball(ball2.position.copy(), ball2.velocity.copy(), ball2.radius, ball2.index, ball2.cell)
        
        perform_ball_ball_collision(ball1, ball2, restitution=1.0)
        
        # Conservation of momentum: total momentum should be unchanged
        # Initial total momentum: [1.0 - 0.5, 1.0 - 0.5] = [0.5, 0.5]
        total_momentum = ball1.velocity + ball2.velocity
        np.testing.assert_array_almost_equal(total_momentum, [0.5, 0.5])
        
        # Validate physics
        validate_collision_physics(ball1_before, ball2_before, ball1, ball2, restitution=1.0)
    
    def test_problematic_collision_from_simulation_logs(self):
        """Test a realistic collision scenario with approaching balls."""
        # Create a scenario where balls are actually approaching each other
        ball1 = Ball(
            position=np.array([1.0, 1.0]),
            velocity=np.array([1.0, 0.0]),
            radius=0.45,
            index=0,
            cell=(1, 1)
        )
        
        ball2 = Ball(
            position=np.array([2.0, 1.0]),
            velocity=np.array([-1.0, 0.0]),
            radius=0.45,
            index=1,
            cell=(2, 1)
        )
        
        # Check if balls are actually overlapping/touching
        distance = np.linalg.norm(ball1.position - ball2.position)
        sum_radii = ball1.radius + ball2.radius
        print(f"Distance between ball centers: {distance:.6f}")
        print(f"Sum of radii: {sum_radii:.6f}")
        print(f"Overlap: {distance < sum_radii}")
        
        # Store original states for validation
        ball1_before = Ball(ball1.position.copy(), ball1.velocity.copy(), ball1.radius, ball1.index, ball1.cell)
        ball2_before = Ball(ball2.position.copy(), ball2.velocity.copy(), ball2.radius, ball2.index, ball2.cell)
        
        # Perform collision
        perform_ball_ball_collision(ball1, ball2, restitution=1.0)
        
        print(f"Ball 1 velocity before: {ball1_before.velocity}")
        print(f"Ball 1 velocity after:  {ball1.velocity}")
        print(f"Ball 2 velocity before: {ball2_before.velocity}")
        print(f"Ball 2 velocity after:  {ball2.velocity}")
        
        # Validate physics - this should catch the issue if collision didn't work properly
        validate_collision_physics(ball1_before, ball2_before, ball1, ball2, restitution=1.0)


class TestPerformBallWallCollision:
    def test_ball_hitting_vertical_wall(self):
        """Test ball hitting vertical wall."""
        ball = Ball(np.array([4.7, 2.5]), np.array([2.0, 1.0]), 0.3, 0, (4, 2), time=1.0)
        wall = Wall(0, 5.0, restitution=1.0)  # x-normal wall at x=5.0
        
        perform_ball_wall_collision(ball, wall, 1.0)
        
        # x-component should be reflected, y-component unchanged
        np.testing.assert_array_almost_equal(ball.velocity, [-2.0, 1.0])
    
    def test_ball_hitting_horizontal_wall(self):
        """Test ball hitting horizontal wall."""
        ball = Ball(np.array([2.5, 2.4]), np.array([1.0, -3.0]), 0.4, 0, (2, 2), time=1.0)
        wall = Wall(1, 2.0, restitution=1.0)  # y-normal wall at y=2.0
        
        # Ball at y=2.4, wall at y=2.0, so ball is above wall
        # Ball moving down toward wall (velocity y=-3.0)
        # After collision, y-velocity should be reflected to +3.0
        
        perform_ball_wall_collision(ball, wall, 1.0)
        
        # y-component should be reflected, x-component unchanged
        np.testing.assert_array_almost_equal(ball.velocity, [1.0, 3.0])
    
    def test_inelastic_wall_collision(self):
        """Test inelastic collision with wall."""
        ball = Ball(np.array([4.8, 2.5]), np.array([2.0, 1.0]), 0.2, 0, (4, 2), time=1.0)
        wall = Wall(0, 5.0, restitution=0.6)  # x-normal wall at x=5.0
        
        perform_ball_wall_collision(ball, wall, 0.6)
        
        # x-component should be: -0.6 * 2.0 = -1.2, y-component unchanged
        np.testing.assert_array_almost_equal(ball.velocity, [-1.2, 1.0])
    
    def test_ball_moving_away_from_wall(self):
        """Test ball moving away from wall - no collision."""
        ball = Ball(np.array([5.2, 2.5]), np.array([1.0, 1.0]), 0.2, 0, (5, 2), time=1.0)
        wall = Wall(0, 5.0, restitution=1.0)  # x-normal wall at x=5.0
        
        original_velocity = ball.velocity.copy()
        
        perform_ball_wall_collision(ball, wall, 1.0)
        
        # Velocity should be unchanged (moving away from wall)
        np.testing.assert_array_almost_equal(ball.velocity, original_velocity)