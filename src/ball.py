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
    
    def get_position_and_velocity_at_time(self, t: float, ndim: int, gravity: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate ball position and velocity at given time, accounting for gravity.
        
        Args:
            t: target time
            ndim: number of dimensions (2 or 3) - world property
            gravity: whether gravity is enabled (g=1 if True, scaled time units)
            
        Returns:
            tuple of (position, velocity) at time t
        """
        dt = t - self.time
        if dt < 0:
            raise ValueError(f"Cannot get position/velocity at time {t} before ball's current time {self.time}")
        
        # Calculate position
        position = self.position + self.velocity * dt
        
        if gravity and ndim >= 2:
            # Add gravitational displacement (gravity acts in negative y direction, g=1 in scaled time)
            gravity_displacement = np.zeros_like(self.position)
            gravity_displacement[1] = -0.5 * dt * dt  # g=1 in scaled time units
            position += gravity_displacement
        
        # Calculate velocity
        velocity = self.velocity.copy()
        if gravity and ndim >= 2:
            # Velocity changes due to gravity: v = v0 + g*t (g acts in negative y direction)
            gravity_velocity = np.zeros_like(self.velocity)
            gravity_velocity[1] = -dt  # g=1 in scaled time units
            velocity += gravity_velocity
        
        return position, velocity
    
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
        
        # Update position and velocity
        self.position, self.velocity = self.get_position_and_velocity_at_time(t, ndim, gravity)
        
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