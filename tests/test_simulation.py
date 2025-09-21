import os
import tempfile
import shutil
import pytest
from src.simulation import run_simulation, validate_simulation_parameters, initialize_simulation


class TestSimulation:
    
    def test_validate_simulation_parameters_valid(self):
        params = {
            'ndim': 2,
            'num_balls': 4,
            'ball_radius': 0.3,
            'domain_size': (3.0, 2.0),
            'simulation_time': 1.0
        }
        # Should not raise
        validate_simulation_parameters(params)
    
    def test_validate_simulation_parameters_missing(self):
        params = {
            'ndim': 2,
            'num_balls': 4,
            # Missing ball_radius
            'domain_size': (3.0, 2.0),
            'simulation_time': 1.0
        }
        with pytest.raises(ValueError, match="Missing required parameter"):
            validate_simulation_parameters(params)
    
    def test_validate_simulation_parameters_invalid_ndim(self):
        params = {
            'ndim': 1,  # Invalid
            'num_balls': 4,
            'ball_radius': 0.1,
            'domain_size': (3.0,),
            'simulation_time': 1.0
        }
        with pytest.raises(ValueError, match="ndim must be 2 or 3"):
            validate_simulation_parameters(params)
    
    def test_validate_simulation_parameters_invalid_ball_radius(self):
        params = {
            'ndim': 2,
            'num_balls': 4,
            'ball_radius': 1.5,  # Larger than cell size
            'domain_size': (3.0, 2.0),
            'simulation_time': 1.0
        }
        with pytest.raises(ValueError, match="ball_radius must be smaller than cell size"):
            validate_simulation_parameters(params)
    
    def test_initialize_simulation_2d(self):
        params = {
            'ndim': 2,
            'num_balls': 2,
            'ball_radius': 0.3,
            'domain_size': (3.0, 2.0),
            'simulation_time': 1.0
        }
        
        balls, walls, grid, event_heap, output_manager = initialize_simulation(params)
        
        assert len(balls) == 2
        assert len(walls) == 4  # 2D box has 4 walls
        assert grid.ndim == 2
        assert grid.domain_size == (3.0, 2.0)
        assert event_heap.is_empty()
        assert output_manager is not None
        
        # Check balls are placed correctly
        assert all(ball.radius == 0.3 for ball in balls)
        assert all(len(ball.position) == 2 for ball in balls)
        assert all(len(ball.velocity) == 2 for ball in balls)
    
    def test_initialize_simulation_3d(self):
        params = {
            'ndim': 3,
            'num_balls': 2,
            'ball_radius': 0.3,
            'domain_size': (2.0, 2.0, 2.0),
            'simulation_time': 1.0
        }
        
        balls, walls, grid, event_heap, output_manager = initialize_simulation(params)
        
        assert len(balls) == 2
        assert len(walls) == 6  # 3D box has 6 walls
        assert grid.ndim == 3
        assert grid.domain_size == (2.0, 2.0, 2.0)
        
        # Check balls are placed correctly
        assert all(len(ball.position) == 3 for ball in balls)
        assert all(len(ball.velocity) == 3 for ball in balls)
    
    def test_run_simulation_short(self):
        """Test a very short simulation to ensure it runs without errors."""
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            params = {
                'ndim': 2,
                'num_balls': 2,
                'ball_radius': 0.3,
                'domain_size': (3.0, 2.0),
                'simulation_time': 0.1,  # Very short
                'gravity': False,
                'ball_restitution': 1.0,
                'wall_restitution': 1.0,
                'output_rate': 0.05,
                'run_name': 'test_run',
                'output_dir': temp_dir
            }
            
            # Should run without errors
            run_simulation(params)
            
            # Check that output files were created in the run subdirectory
            run_dir = os.path.join(temp_dir, 'test_run')
            assert os.path.exists(run_dir)
            output_files = [f for f in os.listdir(run_dir) if f.startswith('frame_')]
            assert len(output_files) >= 1
    
    def test_too_many_balls_error(self):
        """Test that too many balls for domain size raises error."""
        params = {
            'ndim': 2,
            'num_balls': 100,  # Too many for 2x2 domain
            'ball_radius': 0.3,
            'domain_size': (2.0, 2.0),
            'simulation_time': 1.0
        }
        
        # Should raise error during initialization
        with pytest.raises(ValueError):
            initialize_simulation(params)