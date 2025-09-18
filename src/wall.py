import numpy as np
from typing import Tuple


class Wall:
    """
    Represents a wall boundary in the simulation.
    
    Walls are planar boundaries perpendicular to coordinate axes.
    Defined by:
    - normal_axis: which coordinate axis the wall is perpendicular to (0=x, 1=y, 2=z)
    - coordinate: the position along that axis where the wall lies
    """
    
    def __init__(self, normal_axis: int, coordinate: float, restitution: float = 1.0):
        """
        Initialize a Wall.
        
        Args:
            normal_axis: axis perpendicular to wall (0=x, 1=y, 2=z)
            coordinate: position along normal axis where wall is located
            restitution: coefficient of restitution (1.0 = perfectly elastic)
        """
        self.normal_axis = normal_axis
        self.coordinate = coordinate
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
    
    
    def __repr__(self):
        axis_names = ['x', 'y', 'z']
        axis_name = axis_names[self.normal_axis] if self.normal_axis < 3 else f'axis{self.normal_axis}'
        return f"Wall({axis_name}-normal, coord={self.coordinate}, e={self.restitution})"


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
        
        walls.extend([
            Wall(1, inset, restitution),  # bottom
            Wall(1, height - inset, restitution),  # top
            Wall(0, inset, restitution),  # left
            Wall(0, width - inset, restitution)  # right
        ])
        
    elif ndim == 3:
        width, height, depth = box_size[:3]
        
        walls.extend([
            Wall(1, inset, restitution),  # bottom
            Wall(1, height - inset, restitution),  # top
            Wall(0, inset, restitution),  # left
            Wall(0, width - inset, restitution),  # right
            Wall(2, inset, restitution),  # front
            Wall(2, depth - inset, restitution)  # back
        ])
        
    else:
        raise ValueError(f"Unsupported number of dimensions: {ndim}")
    
    return walls