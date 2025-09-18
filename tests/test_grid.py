import numpy as np
import pytest
from src.grid import Grid


class TestGrid:
    
    def test_grid_2d_initialization(self):
        grid = Grid(2, (5.0, 3.0))
        assert grid.ndim == 2
        assert grid.domain_size == (5.0, 3.0)
        assert grid.cell_size == 1.0
        assert grid.num_cells == (5, 3)
        assert len(grid.cells) == 5
        assert len(grid.cells[0]) == 3
    
    def test_grid_3d_initialization(self):
        grid = Grid(3, (4.0, 6.0, 2.0))
        assert grid.ndim == 3
        assert grid.domain_size == (4.0, 6.0, 2.0)
        assert grid.cell_size == 1.0
        assert grid.num_cells == (4, 6, 2)
        assert len(grid.cells) == 4
        assert len(grid.cells[0]) == 6
        assert len(grid.cells[0][0]) == 2
    
    def test_position_to_cell_2d(self):
        grid = Grid(2, (5.0, 3.0))
        
        # Test various positions
        assert grid.position_to_cell(np.array([0.5, 1.2])) == (0, 1)
        assert grid.position_to_cell(np.array([2.9, 0.1])) == (2, 0)
        assert grid.position_to_cell(np.array([4.9, 2.9])) == (4, 2)
        
        # Test boundary clamping
        assert grid.position_to_cell(np.array([-0.5, 1.0])) == (0, 1)
        assert grid.position_to_cell(np.array([6.0, 1.0])) == (4, 1)
        assert grid.position_to_cell(np.array([2.0, -1.0])) == (2, 0)
        assert grid.position_to_cell(np.array([2.0, 5.0])) == (2, 2)
    
    def test_position_to_cell_3d(self):
        grid = Grid(3, (3.0, 2.0, 4.0))
        
        # Test various positions
        assert grid.position_to_cell(np.array([1.5, 0.8, 2.3])) == (1, 0, 2)
        assert grid.position_to_cell(np.array([0.1, 1.9, 0.1])) == (0, 1, 0)
        
        # Test boundary clamping
        assert grid.position_to_cell(np.array([-1.0, 0.5, 1.0])) == (0, 0, 1)
        assert grid.position_to_cell(np.array([5.0, 0.5, 1.0])) == (2, 0, 1)
    
    def test_add_remove_ball_2d(self):
        grid = Grid(2, (3.0, 2.0))
        
        # Add balls to different cells
        grid.add_ball(0, (1, 0))
        grid.add_ball(1, (1, 0))
        grid.add_ball(2, (2, 1))
        
        assert 0 in grid.cells[1][0]
        assert 1 in grid.cells[1][0]
        assert 2 in grid.cells[2][1]
        assert len(grid.cells[1][0]) == 2
        
        # Remove a ball
        grid.remove_ball(0, (1, 0))
        assert 0 not in grid.cells[1][0]
        assert 1 in grid.cells[1][0]
        assert len(grid.cells[1][0]) == 1
    
    def test_add_remove_ball_3d(self):
        grid = Grid(3, (2.0, 2.0, 2.0))
        
        # Add balls to different cells
        grid.add_ball(0, (0, 1, 0))
        grid.add_ball(1, (1, 1, 1))
        
        assert 0 in grid.cells[0][1][0]
        assert 1 in grid.cells[1][1][1]
        
        # Remove a ball
        grid.remove_ball(0, (0, 1, 0))
        assert 0 not in grid.cells[0][1][0]
    
    def test_move_ball_2d(self):
        grid = Grid(2, (3.0, 2.0))
        
        # Add ball and move it
        grid.add_ball(0, (1, 0))
        assert 0 in grid.cells[1][0]
        
        grid.move_ball(0, (1, 0), (2, 1))
        assert 0 not in grid.cells[1][0]
        assert 0 in grid.cells[2][1]
    
    def test_get_balls_in_neighboring_cells_2d(self):
        grid = Grid(2, (4.0, 3.0))
        
        # Add balls in various cells
        grid.add_ball(0, (1, 1))  # center
        grid.add_ball(1, (0, 1))  # left
        grid.add_ball(2, (2, 1))  # right
        grid.add_ball(3, (1, 0))  # bottom
        grid.add_ball(4, (1, 2))  # top
        grid.add_ball(5, (0, 0))  # bottom-left
        grid.add_ball(6, (3, 2))  # far away
        
        neighbors = grid.get_balls_in_neighboring_cells((1, 1))
        expected = {0, 1, 2, 3, 4, 5}  # ball 6 should not be included
        assert set(neighbors) == expected
    
    def test_get_balls_in_neighboring_cells_3d(self):
        grid = Grid(3, (3.0, 3.0, 3.0))
        
        # Add balls in center and adjacent cells
        grid.add_ball(0, (1, 1, 1))  # center
        grid.add_ball(1, (0, 1, 1))  # left
        grid.add_ball(2, (2, 1, 1))  # right
        grid.add_ball(3, (1, 0, 1))  # front
        grid.add_ball(4, (1, 2, 1))  # back
        grid.add_ball(5, (1, 1, 0))  # below
        grid.add_ball(6, (1, 1, 2))  # above
        
        neighbors = grid.get_balls_in_neighboring_cells((1, 1, 1))
        expected = {0, 1, 2, 3, 4, 5, 6}
        assert set(neighbors) == expected
    
    def test_get_balls_in_new_neighbor_cells_2d(self):
        grid = Grid(2, (5.0, 3.0))
        
        # Add balls in cells that would be newly adjacent when moving from (1,1) to (2,1)
        grid.add_ball(0, (3, 0))  # newly adjacent
        grid.add_ball(1, (3, 1))  # newly adjacent
        grid.add_ball(2, (3, 2))  # newly adjacent
        grid.add_ball(3, (1, 1))  # was already adjacent
        
        new_neighbors = grid.get_balls_in_new_neighbor_cells((1, 1), (2, 1))
        expected = {0, 1, 2}  # ball 3 should not be included
        assert set(new_neighbors) == expected
    
    def test_get_balls_in_new_neighbor_cells_3d(self):
        grid = Grid(3, (4.0, 3.0, 3.0))
        
        # Add balls in cells that would be newly adjacent when moving from (1,1,1) to (2,1,1)
        grid.add_ball(0, (3, 0, 0))  # newly adjacent
        grid.add_ball(1, (3, 1, 1))  # newly adjacent
        grid.add_ball(2, (3, 2, 2))  # newly adjacent
        grid.add_ball(3, (1, 1, 1))  # was already adjacent
        
        new_neighbors = grid.get_balls_in_new_neighbor_cells((1, 1, 1), (2, 1, 1))
        expected = {0, 1, 2}  # ball 3 should not be included
        assert set(new_neighbors) == expected
    
    def test_boundary_cases_2d(self):
        grid = Grid(2, (3.0, 2.0))
        
        # Test corner cell neighbors
        grid.add_ball(0, (0, 0))
        grid.add_ball(1, (1, 0))
        grid.add_ball(2, (0, 1))
        grid.add_ball(3, (1, 1))
        
        # Corner cell (0,0) should only see valid neighbors
        neighbors = grid.get_balls_in_neighboring_cells((0, 0))
        expected = {0, 1, 2, 3}
        assert set(neighbors) == expected
    
    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            Grid(1, (5.0,))  # 1D not supported
        
        with pytest.raises(ValueError):
            Grid(4, (5.0, 3.0, 2.0, 1.0))  # 4D not supported