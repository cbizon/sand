import numpy as np
import pytest
from src.wall import Wall, create_box_walls


class TestWall:
    def test_wall_initialization(self):
        """Test basic wall initialization."""
        bounds = ((1.0, 5.0),)
        wall = Wall(normal_axis=1, coordinate=2.0, bounds=bounds, restitution=0.9)
        
        assert wall.normal_axis == 1
        assert wall.coordinate == 2.0
        assert wall.bounds == bounds
        assert wall.restitution == 0.9
    
    def test_get_normal_vector_2d(self):
        """Test normal vector calculation in 2D."""
        # Wall perpendicular to x-axis (normal_axis=0)
        wall_x = Wall(0, 1.0, ((0.0, 2.0),))
        normal_x = wall_x.get_normal_vector(2)
        np.testing.assert_array_equal(normal_x, [1.0, 0.0])
        
        # Wall perpendicular to y-axis (normal_axis=1)
        wall_y = Wall(1, 1.0, ((0.0, 2.0),))
        normal_y = wall_y.get_normal_vector(2)
        np.testing.assert_array_equal(normal_y, [0.0, 1.0])
    
    def test_get_normal_vector_3d(self):
        """Test normal vector calculation in 3D."""
        # Wall perpendicular to x-axis (normal_axis=0)
        wall_x = Wall(0, 1.0, ((0.0, 2.0), (0.0, 3.0)))
        normal_x = wall_x.get_normal_vector(3)
        np.testing.assert_array_equal(normal_x, [1.0, 0.0, 0.0])
        
        # Wall perpendicular to y-axis (normal_axis=1)
        wall_y = Wall(1, 1.0, ((0.0, 2.0), (0.0, 3.0)))
        normal_y = wall_y.get_normal_vector(3)
        np.testing.assert_array_equal(normal_y, [0.0, 1.0, 0.0])
        
        # Wall perpendicular to z-axis (normal_axis=2)
        wall_z = Wall(2, 1.0, ((0.0, 2.0), (0.0, 3.0)))
        normal_z = wall_z.get_normal_vector(3)
        np.testing.assert_array_equal(normal_z, [0.0, 0.0, 1.0])
    
    def test_distance_to_point(self):
        """Test distance calculation to wall."""
        # Wall perpendicular to y-axis at y=2
        wall = Wall(1, 2.0, ((0.0, 5.0),))
        
        # Point at (1, 3) should be distance 1 from wall at y=2
        point = np.array([1.0, 3.0])
        distance = wall.distance_to_point(point)
        assert distance == 1.0
        
        # Point at (1, 1) should be distance 1 from wall at y=2
        point = np.array([1.0, 1.0])
        distance = wall.distance_to_point(point)
        assert distance == 1.0
        
        # Point at (1, 2) should be distance 0 from wall at y=2
        point = np.array([1.0, 2.0])
        distance = wall.distance_to_point(point)
        assert distance == 0.0
    
    def test_is_point_in_bounds_2d(self):
        """Test point bounds checking in 2D."""
        # Wall perpendicular to y-axis with x-bounds (1, 4)
        wall = Wall(1, 2.0, ((1.0, 4.0),))
        
        # Point within bounds
        assert wall.is_point_in_bounds(np.array([2.0, 3.0])) is True
        assert wall.is_point_in_bounds(np.array([1.0, 3.0])) is True
        assert wall.is_point_in_bounds(np.array([4.0, 3.0])) is True
        
        # Point outside bounds
        assert wall.is_point_in_bounds(np.array([0.5, 3.0])) is False
        assert wall.is_point_in_bounds(np.array([4.5, 3.0])) is False
    
    def test_is_point_in_bounds_3d(self):
        """Test point bounds checking in 3D."""
        # Wall perpendicular to y-axis with x-bounds (1, 4) and z-bounds (2, 5)
        wall = Wall(1, 2.0, ((1.0, 4.0), (2.0, 5.0)))
        
        # Point within bounds
        assert wall.is_point_in_bounds(np.array([2.0, 3.0, 3.0])) is True
        assert wall.is_point_in_bounds(np.array([1.0, 3.0, 2.0])) is True
        assert wall.is_point_in_bounds(np.array([4.0, 3.0, 5.0])) is True
        
        # Point outside x-bounds
        assert wall.is_point_in_bounds(np.array([0.5, 3.0, 3.0])) is False
        assert wall.is_point_in_bounds(np.array([4.5, 3.0, 3.0])) is False
        
        # Point outside z-bounds
        assert wall.is_point_in_bounds(np.array([2.0, 3.0, 1.5])) is False
        assert wall.is_point_in_bounds(np.array([2.0, 3.0, 5.5])) is False
    
    def test_repr(self):
        """Test string representation."""
        wall = Wall(1, 2.0, ((1.0, 4.0),), 0.8)
        repr_str = repr(wall)
        
        assert "Wall(y-normal" in repr_str
        assert "coord=2.0" in repr_str
        assert "bounds=((1.0, 4.0),)" in repr_str
        assert "e=0.8" in repr_str


class TestCreateBoxWalls:
    def test_create_box_walls_2d(self):
        """Test creating 2D box walls."""
        walls = create_box_walls(ndim=2, box_size=(10.0, 8.0), inset=0.5, restitution=0.9)
        
        # Should have 4 walls in 2D
        assert len(walls) == 4
        
        # Check wall properties
        wall_coords = [(w.normal_axis, w.coordinate) for w in walls]
        expected_coords = [
            (1, 0.5),      # bottom (y-normal)
            (1, 7.5),      # top (y-normal)
            (0, 0.5),      # left (x-normal)
            (0, 9.5)       # right (x-normal)
        ]
        
        for expected in expected_coords:
            assert expected in wall_coords
        
        # Check all walls have correct restitution
        for wall in walls:
            assert wall.restitution == 0.9
    
    def test_create_box_walls_3d(self):
        """Test creating 3D box walls."""
        walls = create_box_walls(ndim=3, box_size=(10.0, 8.0, 6.0), inset=0.2, restitution=1.0)
        
        # Should have 6 walls in 3D
        assert len(walls) == 6
        
        # Check wall properties
        wall_coords = [(w.normal_axis, w.coordinate) for w in walls]
        expected_coords = [
            (1, 0.2),      # bottom (y-normal)
            (1, 7.8),      # top (y-normal)
            (0, 0.2),      # left (x-normal)
            (0, 9.8),      # right (x-normal)
            (2, 0.2),      # front (z-normal)
            (2, 5.8)       # back (z-normal)
        ]
        
        for expected in expected_coords:
            assert expected in wall_coords
        
        # Check bounds for y-normal walls (bottom and top)
        y_normal_walls = [w for w in walls if w.normal_axis == 1]
        assert len(y_normal_walls) == 2
        for wall in y_normal_walls:
            # Should have x-bounds and z-bounds
            assert len(wall.bounds) == 2
            assert wall.bounds[0] == (0.2, 9.8)  # x bounds
            assert wall.bounds[1] == (0.2, 5.8)  # z bounds
        
        # Check bounds for x-normal walls (left and right)
        x_normal_walls = [w for w in walls if w.normal_axis == 0]
        assert len(x_normal_walls) == 2
        for wall in x_normal_walls:
            # Should have y-bounds and z-bounds
            assert len(wall.bounds) == 2
            assert wall.bounds[0] == (0.2, 7.8)  # y bounds
            assert wall.bounds[1] == (0.2, 5.8)  # z bounds
        
        # Check bounds for z-normal walls (front and back)
        z_normal_walls = [w for w in walls if w.normal_axis == 2]
        assert len(z_normal_walls) == 2
        for wall in z_normal_walls:
            # Should have x-bounds and y-bounds
            assert len(wall.bounds) == 2
            assert wall.bounds[0] == (0.2, 9.8)  # x bounds
            assert wall.bounds[1] == (0.2, 7.8)  # y bounds
    
    def test_create_box_walls_default_params(self):
        """Test creating box walls with default parameters."""
        walls = create_box_walls(ndim=2, box_size=(5.0, 4.0))
        
        assert len(walls) == 4
        
        # Check default inset (0.01)
        expected_coords = [
            (1, 0.01),     # bottom
            (1, 3.99),     # top
            (0, 0.01),     # left
            (0, 4.99)      # right
        ]
        
        wall_coords = [(w.normal_axis, w.coordinate) for w in walls]
        for expected in expected_coords:
            assert expected in wall_coords
        
        # Check default restitution (1.0)
        for wall in walls:
            assert wall.restitution == 1.0
    
    def test_create_box_walls_invalid_ndim(self):
        """Test that invalid dimensions raise error."""
        with pytest.raises(ValueError, match="Unsupported number of dimensions: 4"):
            create_box_walls(ndim=4, box_size=(1, 1, 1, 1))