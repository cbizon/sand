import numpy as np
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .events import Event


class Ball:
    def __init__(self, position: np.ndarray, velocity: np.ndarray, radius: float, index: int, 
                 cell: Tuple[int, ...], time: float = 0.0):
        """
        Initialize a Ball object.
        
        Args:
            position: numpy array of position coordinates (2D or 3D)
            velocity: numpy array of velocity components (2D or 3D)
            radius: ball radius
            index: index in the balls list
            cell: tuple of cell indices (i, j) for 2D or (i, j, k) for 3D
            time: time of most recent collision (default 0.0)
        """
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.radius = radius
        self.index = index
        self.cell = cell
        self.time = time
        self.events: List['Event'] = []
    
    def get_position_at_time(self, t: float, ndim: int, gravity: bool = False) -> np.ndarray:
        """
        Calculate ball position at given time, accounting for gravity.
        
        Args:
            t: target time
            ndim: number of dimensions (2 or 3) - world property
            gravity: whether gravity is enabled (g=1 if True, scaled time units)
            
        Returns:
            position at time t
            
        Note: In right-handed coordinate system:
        - 2D: x (horizontal right), y (vertical up)
        - 3D: x (horizontal right), y (vertical up), z (out of screen toward viewer)
        - Gravity acts in negative y direction (downward)
        """
        dt = t - self.time
        if dt < 0:
            raise ValueError(f"Cannot get position at time {t} before ball's current time {self.time}")
        
        # Position = initial_position + velocity * dt + 0.5 * gravity * dt^2
        new_position = self.position + self.velocity * dt
        
        if gravity and ndim >= 2:
            # Add gravitational displacement (gravity acts in negative y direction, g=1 in scaled time)
            gravity_displacement = np.zeros_like(self.position)
            gravity_displacement[1] = -0.5 * dt * dt  # g=1 in scaled time units
            new_position += gravity_displacement
            
        return new_position
    
    def update_to_time(self, t: float, ndim: int, gravity: bool = False):
        """
        Update ball's position and velocity to given time.
        
        Args:
            t: target time
            ndim: number of dimensions (2 or 3) - world property
            gravity: whether gravity is enabled (g=1 in scaled time units)
        """
        if t < self.time:
            raise ValueError(f"Cannot update to time {t} before ball's current time {self.time}")
        
        dt = t - self.time
        
        # Update position
        self.position = self.get_position_at_time(t, ndim, gravity)
        
        # Update velocity (gravity affects velocity, g=1 in scaled time units)
        if gravity and ndim >= 2:
            gravity_velocity = np.zeros_like(self.velocity)
            gravity_velocity[1] = -dt  # g=1 in scaled time units
            self.velocity += gravity_velocity
        
        # Update time
        self.time = t
    
    def invalidate_all_events(self):
        """Mark all events involving this ball as invalid."""
        for event in self.events:
            event.valid = False
        self.events.clear()
    
    def add_event(self, event: 'Event'):
        """Add an event to this ball's event list."""
        self.events.append(event)
    
    def __repr__(self):
        return f"Ball(pos={self.position}, vel={self.velocity}, r={self.radius}, i={self.index}, cell={self.cell}, t={self.time})"