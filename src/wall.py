import numpy as np
from typing import Tuple


class Wall:
    """
    Represents a wall boundary in the simulation.
    
    Walls are planar boundaries perpendicular to coordinate axes.
    Defined by:
    - normal_axis: which coordinate axis the wall is perpendicular to (0=x, 1=y, 2=z)
    - coordinate: the position along that axis where the wall lies
    - bounds: the extent of the wall in the other dimensions
    """
    
    def __init__(self, normal_axis: int, coordinate: float, 
                 bounds: Tuple[Tuple[float, float], ...], restitution: float = 1.0):
        """
        Initialize a Wall.
        
        Args:
            normal_axis: axis perpendicular to wall (0=x, 1=y, 2=z)
            coordinate: position along normal axis where wall is located
            bounds: bounds for other axes as ((min1, max1), (min2, max2), ...)
                   For 2D with normal_axis=1: ((x_min, x_max),)
                   For 3D with normal_axis=1: ((x_min, x_max), (z_min, z_max))
            restitution: coefficient of restitution (1.0 = perfectly elastic)
        """
        self.normal_axis = normal_axis
        self.coordinate = coordinate
        self.bounds = bounds
        self.restitution = restitution
    
    def get_normal_vector(self, ndim: int) -> np.ndarray:
        """
        Get the outward normal vector for this wall.
        
        Args:
            ndim: number of dimensions (2 or 3)
            
        Returns:
            unit normal vector pointing away from simulation domain
        """
        normal = np.zeros(ndim)
        normal[self.normal_axis] = 1.0
        return normal
    
    def distance_to_point(self, point: np.ndarray) -> float:
        """
        Calculate distance from point to wall.
        
        Args:
            point: position vector
            
        Returns:
            distance to wall (always positive)
        """
        return abs(point[self.normal_axis] - self.coordinate)
    
    def is_point_in_bounds(self, point: np.ndarray) -> bool:
        """
        Check if point projection is within wall bounds.
        
        Args:
            point: position vector
            
        Returns:
            True if point projection is within wall bounds
        """
        # Check bounds for all axes except the normal axis
        bound_idx = 0
        for axis in range(len(point)):
            if axis != self.normal_axis:
                if bound_idx >= len(self.bounds):
                    return False
                min_bound, max_bound = self.bounds[bound_idx]
                if not (min_bound <= point[axis] <= max_bound):
                    return False
                bound_idx += 1
        return True
    
    def __repr__(self):
        axis_names = ['x', 'y', 'z']
        axis_name = axis_names[self.normal_axis] if self.normal_axis < 3 else f'axis{self.normal_axis}'
        return f"Wall({axis_name}-normal, coord={self.coordinate}, bounds={self.bounds}, e={self.restitution})"


def create_box_walls(ndim: int, box_size: Tuple[float, ...], inset: float = 0.01, 
                     restitution: float = 1.0) -> list:
    """
    Create walls defining a rectangular box boundary.
    
    Args:
        ndim: number of dimensions (2 or 3)
        box_size: (width, height) for 2D or (width, height, depth) for 3D
        inset: distance to inset walls from box edges
        restitution: coefficient of restitution for all walls
        
    Returns:
        list of Wall objects defining the box boundary
    """
    walls = []
    
    if ndim == 2:
        width, height = box_size[:2]
        
        # Bottom and top walls (y-normal, axis=1)
        x_bounds = ((inset, width - inset),)
        walls.extend([
            Wall(1, inset, x_bounds, restitution),  # bottom
            Wall(1, height - inset, x_bounds, restitution)  # top
        ])
        
        # Left and right walls (x-normal, axis=0) 
        y_bounds = ((inset, height - inset),)
        walls.extend([
            Wall(0, inset, y_bounds, restitution),  # left
            Wall(0, width - inset, y_bounds, restitution)  # right
        ])
        
    elif ndim == 3:
        width, height, depth = box_size[:3]
        
        # Bottom and top walls (y-normal, axis=1)
        xz_bounds = ((inset, width - inset), (inset, depth - inset))
        walls.extend([
            Wall(1, inset, xz_bounds, restitution),  # bottom
            Wall(1, height - inset, xz_bounds, restitution)  # top
        ])
        
        # Left and right walls (x-normal, axis=0)
        yz_bounds = ((inset, height - inset), (inset, depth - inset))
        walls.extend([
            Wall(0, inset, yz_bounds, restitution),  # left
            Wall(0, width - inset, yz_bounds, restitution)  # right
        ])
        
        # Front and back walls (z-normal, axis=2)
        xy_bounds = ((inset, width - inset), (inset, height - inset))
        walls.extend([
            Wall(2, inset, xy_bounds, restitution),  # front
            Wall(2, depth - inset, xy_bounds, restitution)  # back
        ])
        
    else:
        raise ValueError(f"Unsupported number of dimensions: {ndim}")
    
    return walls