import pytest
import numpy as np
from src.simulation import initialize_simulation


def test_overlapping_balls_from_go_py():
    """Test that reproduces the overlapping balls issue from go.py parameters."""
    params = {
        'ndim': 2,
        'num_balls': 4,
        'ball_radius': 0.45,
        'domain_size': (5.0, 3.0),
        'simulation_time': 10.0,
        'gravity': False,
        'output_rate': 0.1
    }
    
    balls, walls, grid, event_heap, output_manager = initialize_simulation(params)
    
    # Print ball positions and distances for debugging
    print(f"Ball positions and radii (radius={params['ball_radius']}):")
    for i, ball in enumerate(balls):
        print(f"  Ball {i}: position={ball.position}, radius={ball.radius}")
    
    print("\nDistances between ball centers:")
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            ball_i = balls[i]
            ball_j = balls[j]
            distance = np.linalg.norm(ball_i.position - ball_j.position)
            sum_radii = ball_i.radius + ball_j.radius
            print(f"  Balls {i}-{j}: distance={distance:.3f}, sum_radii={sum_radii:.3f}, overlap={distance < sum_radii}")


def test_overlapping_balls_from_simulation_py_example():
    """Test with the problematic parameters that were in simulation.py example (radius 0.9)."""
    params = {
        'ndim': 2,
        'num_balls': 4,
        'ball_radius': 0.9,  # This should cause overlaps
        'domain_size': (5.0, 3.0),
        'simulation_time': 10.0,
        'gravity': False,
        'output_rate': 1.0
    }
    
    # This should now raise an error due to improved validation
    with pytest.raises(ValueError, match="balls would overlap"):
        initialize_simulation(params)


def test_overlapping_balls_with_large_radius():
    """Test that validation prevents overlapping balls with radius 0.6."""
    params = {
        'ndim': 2,
        'num_balls': 4,
        'ball_radius': 0.6,  # This should be rejected by validation
        'domain_size': (5.0, 3.0),
        'simulation_time': 10.0,
        'gravity': False,
        'output_rate': 0.1
    }
    
    # Should raise ValueError due to overlap prevention
    with pytest.raises(ValueError, match="balls would overlap"):
        initialize_simulation(params)


if __name__ == "__main__":
    test_overlapping_balls_from_go_py()
    test_overlapping_balls_with_large_radius()