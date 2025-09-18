import numpy as np
from typing import List, Tuple


class Grid:
    """
    Grid class for spatial partitioning of balls in molecular dynamics simulation.
    
    Divides simulation volume into cells with dimension 1 on each edge.
    Each cell contains a list of ball indices whose centers are in that cell.
    """
    
    def __init__(self, ndim: int, domain_size: Tuple[float, ...]):
        """
        Initialize grid.
        
        Args:
            ndim: number of dimensions (2 or 3)
            domain_size: (width, height) for 2D or (width, height, depth) for 3D
        """
        self.ndim = ndim
        self.domain_size = domain_size[:ndim]
        self.cell_size = 1.0  # Fixed cell size sets length scale
        
        # Calculate number of cells in each dimension
        self.num_cells = tuple(int(np.ceil(size)) for size in self.domain_size)
        
        # Create grid as nested lists - each cell contains list of ball indices
        if ndim == 2:
            self.cells = [[[] for _ in range(self.num_cells[1])] 
                         for _ in range(self.num_cells[0])]
        elif ndim == 3:
            self.cells = [[[[] for _ in range(self.num_cells[2])] 
                          for _ in range(self.num_cells[1])]
                         for _ in range(self.num_cells[0])]
        else:
            raise ValueError(f"Unsupported number of dimensions: {ndim}")
    
    def position_to_cell(self, position: np.ndarray) -> Tuple[int, ...]:
        """
        Convert position to cell coordinates.
        
        Args:
            position: position vector
            
        Returns:
            tuple of cell indices
        """
        cell_coords = []
        for i in range(self.ndim):
            cell_coord = int(position[i])
            # Clamp to valid range
            cell_coord = max(0, min(cell_coord, self.num_cells[i] - 1))
            cell_coords.append(cell_coord)
        return tuple(cell_coords)
    
    def add_ball(self, ball_index: int, cell: Tuple[int, ...]):
        """Add ball to specified cell."""
        if self.ndim == 2:
            self.cells[cell[0]][cell[1]].append(ball_index)
        elif self.ndim == 3:
            self.cells[cell[0]][cell[1]][cell[2]].append(ball_index)
    
    def remove_ball(self, ball_index: int, cell: Tuple[int, ...]):
        """Remove ball from specified cell."""
        if self.ndim == 2:
            self.cells[cell[0]][cell[1]].remove(ball_index)
        elif self.ndim == 3:
            self.cells[cell[0]][cell[1]][cell[2]].remove(ball_index)
    
    def move_ball(self, ball_index: int, old_cell: Tuple[int, ...], new_cell: Tuple[int, ...]):
        """Move ball from old cell to new cell."""
        self.remove_ball(ball_index, old_cell)
        self.add_ball(ball_index, new_cell)
    
    def get_balls_in_neighboring_cells(self, cell: Tuple[int, ...]) -> List[int]:
        """Get all ball indices in neighboring cells (including the cell itself)."""
        ball_indices = []
        
        if self.ndim == 2:
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    neighbor = (cell[0] + di, cell[1] + dj)
                    if self._is_valid_cell(neighbor):
                        if self.ndim == 2:
                            ball_indices.extend(self.cells[neighbor[0]][neighbor[1]])
        elif self.ndim == 3:
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    for dk in [-1, 0, 1]:
                        neighbor = (cell[0] + di, cell[1] + dj, cell[2] + dk)
                        if self._is_valid_cell(neighbor):
                            ball_indices.extend(self.cells[neighbor[0]][neighbor[1]][neighbor[2]])
        
        return ball_indices
    
    def get_balls_in_new_neighbor_cells(self, old_cell: Tuple[int, ...], new_cell: Tuple[int, ...]) -> List[int]:
        """Get ball indices in newly adjacent cells when moving from old_cell to new_cell."""
        # Calculate movement direction
        movement = tuple(new_cell[i] - old_cell[i] for i in range(self.ndim))
        
        ball_indices = []
        
        if self.ndim == 2:
            # In 2D, there are 3 newly adjacent cells in the direction of movement
            if movement[0] != 0:  # Moving in x direction
                x_new = new_cell[0] + movement[0]
                for dy in [-1, 0, 1]:
                    neighbor = (x_new, new_cell[1] + dy)
                    if self._is_valid_cell(neighbor):
                        ball_indices.extend(self.cells[neighbor[0]][neighbor[1]])
            
            if movement[1] != 0:  # Moving in y direction
                y_new = new_cell[1] + movement[1]
                for dx in [-1, 0, 1]:
                    neighbor = (new_cell[0] + dx, y_new)
                    if self._is_valid_cell(neighbor):
                        ball_indices.extend(self.cells[neighbor[0]][neighbor[1]])
        
        elif self.ndim == 3:
            # In 3D, there are 9 newly adjacent cells in the plane of movement
            if movement[0] != 0:  # Moving in x direction
                x_new = new_cell[0] + movement[0]
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        neighbor = (x_new, new_cell[1] + dy, new_cell[2] + dz)
                        if self._is_valid_cell(neighbor):
                            ball_indices.extend(self.cells[neighbor[0]][neighbor[1]][neighbor[2]])
            
            if movement[1] != 0:  # Moving in y direction
                y_new = new_cell[1] + movement[1]
                for dx in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        neighbor = (new_cell[0] + dx, y_new, new_cell[2] + dz)
                        if self._is_valid_cell(neighbor):
                            ball_indices.extend(self.cells[neighbor[0]][neighbor[1]][neighbor[2]])
            
            if movement[2] != 0:  # Moving in z direction
                z_new = new_cell[2] + movement[2]
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        neighbor = (new_cell[0] + dx, new_cell[1] + dy, z_new)
                        if self._is_valid_cell(neighbor):
                            ball_indices.extend(self.cells[neighbor[0]][neighbor[1]][neighbor[2]])
        
        return ball_indices
    
    def _is_valid_cell(self, cell: Tuple[int, ...]) -> bool:
        """Check if cell coordinates are valid."""
        for i in range(self.ndim):
            if cell[i] < 0 or cell[i] >= self.num_cells[i]:
                return False
        return True