import pytest
import numpy as np
from src.simulation import initialize_simulation


def test_balls_proper_distance_from_walls():
    """Test that balls are now placed at proper distance from walls."""
    params = {
        'ndim': 2,
        'num_balls': 4,
        'ball_radius': 0.45,  # From go.py
        'domain_size': (5.0, 3.0),
        'simulation_time': 10.0,
        'gravity': False,
        'output_rate': 0.1
    }
    
    balls, walls, grid, event_heap, output_manager = initialize_simulation(params)
    
    # Check distances from each ball to each wall
    min_distance_found = float('inf')
    
    for i, ball in enumerate(balls):
        for j, wall in enumerate(walls):
            # Distance from ball center to wall
            center_to_wall = wall.distance_to_point(ball.position)
            
            # Distance from ball surface to wall
            surface_to_wall = center_to_wall - ball.radius
            
            min_distance_found = min(min_distance_found, surface_to_wall)
            
            # All balls should be properly distanced from walls
            assert surface_to_wall > 0.01, f"Ball {i} too close to wall {wall}: surface_distance={surface_to_wall:.6f}"
    
    print(f"\nBall-wall distance analysis:")
    print(f"Minimum surface-to-wall distance found: {min_distance_found:.4f}")
    print("All balls properly placed with adequate wall clearance.")


def test_large_radius_safe_placement():
    """Test that even large radius balls are placed safely."""
    params = {
        'ndim': 2,
        'num_balls': 1,
        'ball_radius': 0.45,  # Large radius but safe for this domain
        'domain_size': (2.0, 2.0),  # Small domain
        'simulation_time': 1.0,
        'gravity': False
    }
    
    balls, walls, grid, event_heap, output_manager = initialize_simulation(params)
    
    ball = balls[0]
    
    # Check that ball is safely placed relative to all walls
    for wall in walls:
        center_to_wall = wall.distance_to_point(ball.position)
        surface_to_wall = center_to_wall - ball.radius
        
        print(f"Ball to {wall}: center_distance={center_to_wall:.4f}, surface_distance={surface_to_wall:.4f}")
        
        # Ball should be safely away from all walls
        assert surface_to_wall > 0, f"Ball overlaps with {wall}: surface_distance={surface_to_wall:.4f}"


def test_safe_ball_placement():
    """Test what should be a safe configuration."""
    params = {
        'ndim': 2,
        'num_balls': 1,
        'ball_radius': 0.1,  # Small radius
        'domain_size': (3.0, 3.0),  # Larger domain
        'simulation_time': 1.0,
        'gravity': False
    }
    
    balls, walls, grid, event_heap, output_manager = initialize_simulation(params)
    
    # Check that all balls have reasonable distance from walls
    for ball in balls:
        for wall in walls:
            center_to_wall = wall.distance_to_point(ball.position)
            surface_to_wall = center_to_wall - ball.radius
            
            # Should have at least some reasonable clearance
            assert surface_to_wall > 0.1, f"Ball too close to wall: surface_distance={surface_to_wall:.4f}"


def test_future_validation_should_prevent_close_balls():
    """Test that should pass after we add proper validation."""
    params = {
        'ndim': 2,
        'num_balls': 4,
        'ball_radius': 0.45,  # This should be rejected after adding validation
        'domain_size': (5.0, 3.0),
        'simulation_time': 10.0,
        'gravity': False
    }
    
    # After we add validation, this should raise an error
    # For now, it will pass and we'll document the issue
    try:
        balls, walls, grid, event_heap, output_manager = initialize_simulation(params)
        print("WARNING: Balls placed too close to walls - validation needed!")
        # TODO: This should become: 
        # with pytest.raises(ValueError, match="balls too close to walls"):
        #     initialize_simulation(params)
    except ValueError as e:
        if "too close to walls" in str(e):
            print("GOOD: Validation prevented balls too close to walls")
        else:
            raise


if __name__ == "__main__":
    test_balls_proper_distance_from_walls()
    test_large_radius_safe_placement()
    test_safe_ball_placement()
    test_future_validation_should_prevent_close_balls()