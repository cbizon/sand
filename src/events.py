from abc import ABC, abstractmethod
from typing import List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .ball import Ball
    from .wall import Wall


class Event(ABC):
    """Base class for all simulation events."""
    
    def __init__(self, time: float):
        """
        Initialize event.
        
        Args:
            time: time when event occurs
        """
        self.time = time
        self.valid = True
    
    @abstractmethod
    def get_participants(self) -> List[Union['Ball', 'Wall']]:
        """Return list of balls and walls participating in this event."""
        pass
    
    @abstractmethod
    def process(self, **kwargs) -> None:
        """Process this event - update participants and simulation state."""
        pass
    
    def __lt__(self, other: 'Event') -> bool:
        """Enable heap ordering by time."""
        return self.time < other.time
    
    def __repr__(self):
        return f"{self.__class__.__name__}(t={self.time}, valid={self.valid})"


class BallBallCollision(Event):
    """Event for collision between two balls."""
    
    def __init__(self, time: float, ball1: 'Ball', ball2: 'Ball'):
        """
        Initialize ball-ball collision event.
        
        Args:
            time: time when collision occurs
            ball1: first ball in collision
            ball2: second ball in collision
        """
        super().__init__(time)
        self.ball1 = ball1
        self.ball2 = ball2
        
        # Add this event to both balls' event lists
        ball1.add_event(self)
        ball2.add_event(self)
    
    def get_participants(self) -> List['Ball']:
        """Return the two balls involved in collision."""
        return [self.ball1, self.ball2]
    
    def process(self, **kwargs) -> None:
        """
        Process ball-ball collision.
        
        Required kwargs:
            ndim: number of dimensions
            gravity: whether gravity is enabled
            ball_restitution: coefficient of restitution for balls
            balls: list of all balls in simulation
            walls: list of all walls
            grid: the spatial grid
            event_heap: heap to add new events to
        """
        from .physics import perform_ball_ball_collision
        
        ndim = kwargs['ndim']
        gravity = kwargs['gravity']
        ball_restitution = kwargs['ball_restitution']
        event_heap = kwargs['event_heap']
        
        print(f"PROCESSING BallBallCollision: Ball {self.ball1.index} and Ball {self.ball2.index} at t={self.time:.6f}")
        print(f"  Ball {self.ball1.index} position before: {self.ball1.position}, velocity: {self.ball1.velocity}")
        print(f"  Ball {self.ball2.index} position before: {self.ball2.position}, velocity: {self.ball2.velocity}")
        
        # Update both balls to collision time
        self.ball1.update_to_time(self.time, ndim, gravity)
        self.ball2.update_to_time(self.time, ndim, gravity)
        print(f"  Ball {self.ball1.index} position at collision: {self.ball1.position}")
        print(f"  Ball {self.ball2.index} position at collision: {self.ball2.position}")
        
        # Perform collision (updates velocities)
        perform_ball_ball_collision(self.ball1, self.ball2, ball_restitution)
        print(f"  Ball {self.ball1.index} velocity after: {self.ball1.velocity}")
        print(f"  Ball {self.ball2.index} velocity after: {self.ball2.velocity}")
        
        # Invalidate all events for both balls (velocities changed)
        events1 = len(self.ball1.events)
        events2 = len(self.ball2.events)
        self.ball1.invalidate_all_events()
        self.ball2.invalidate_all_events()
        print(f"  Invalidated {events1} events for ball {self.ball1.index}, {events2} events for ball {self.ball2.index}")
        
        # Generate new events for both balls (ball-ball, ball-wall, ball-grid)
        from .event_generation import generate_events_for_ball
        balls = kwargs['balls']
        walls = kwargs['walls']
        grid = kwargs['grid']
        
        new_events = generate_events_for_ball(self.ball1, balls, walls, grid, self.time, ndim, gravity)
        new_events.extend(generate_events_for_ball(self.ball2, balls, walls, grid, self.time, ndim, gravity))
        print(f"  Generated {len(new_events)} new events total")
        
        # Add new events to heap
        for event in new_events:
            event_heap.add_event(event)


class BallWallCollision(Event):
    """Event for collision between a ball and a wall."""
    
    def __init__(self, time: float, ball: 'Ball', wall: 'Wall'):
        """
        Initialize ball-wall collision event.
        
        Args:
            time: time when collision occurs
            ball: ball in collision
            wall: wall in collision
        """
        super().__init__(time)
        self.ball = ball
        self.wall = wall
        
        # Add this event to ball's event list
        ball.add_event(self)
    
    def get_participants(self) -> List[Union['Ball', 'Wall']]:
        """Return ball and wall involved in collision."""
        return [self.ball, self.wall]
    
    def process(self, **kwargs) -> None:
        """
        Process ball-wall collision.
        
        Required kwargs:
            ndim: number of dimensions
            gravity: whether gravity is enabled
            wall_restitution: coefficient of restitution for walls
            balls: list of all balls in simulation
            walls: list of all walls
            grid: the spatial grid
            event_heap: heap to add new events to
        """
        from .physics import perform_ball_wall_collision
        
        ndim = kwargs['ndim']
        gravity = kwargs['gravity']
        wall_restitution = kwargs['wall_restitution']
        event_heap = kwargs['event_heap']
        
        print(f"PROCESSING BallWallCollision: Ball {self.ball.index} hits {self.wall} at t={self.time:.6f}")
        print(f"  Ball position before: {self.ball.position}")
        print(f"  Ball velocity before: {self.ball.velocity}")
        
        # Update ball to collision time
        self.ball.update_to_time(self.time, ndim, gravity)
        print(f"  Ball position at collision: {self.ball.position}")
        
        # Perform collision (updates velocity)
        perform_ball_wall_collision(self.ball, self.wall, wall_restitution)
        print(f"  Ball velocity after collision: {self.ball.velocity}")
        
        # Invalidate all events for the ball (velocity changed)
        events_invalidated = len(self.ball.events)
        self.ball.invalidate_all_events()
        print(f"  Invalidated {events_invalidated} events for ball {self.ball.index}")
        
        # Generate new events for the ball (ball-ball, ball-wall, ball-grid)
        from .event_generation import generate_events_for_ball
        balls = kwargs['balls']
        walls = kwargs['walls']
        grid = kwargs['grid']
        
        new_events = generate_events_for_ball(self.ball, balls, walls, grid, self.time, ndim, gravity)
        print(f"  Generated {len(new_events)} new events for ball {self.ball.index}:")
        for event in new_events:
            print(f"    {event}")
        
        # Add new events to heap
        for event in new_events:
            event_heap.add_event(event)


class BallGridTransit(Event):
    """Event for ball moving from one grid cell to another."""
    
    def __init__(self, time: float, ball: 'Ball', new_cell: tuple):
        """
        Initialize ball grid transit event.
        
        Args:
            time: time when ball crosses cell boundary
            ball: ball changing cells
            new_cell: tuple of new cell indices
        """
        super().__init__(time)
        self.ball = ball
        self.new_cell = new_cell
        
        # Add this event to ball's event list
        ball.add_event(self)
    
    def get_participants(self) -> List['Ball']:
        """Return ball involved in grid transit."""
        return [self.ball]
    
    def process(self, **kwargs) -> None:
        """
        Process ball grid transit.
        
        Required kwargs:
            grid: Grid object to update cell memberships
            balls: list of all balls in simulation
            ndim: number of dimensions
            gravity: whether gravity is enabled
            event_heap: heap to add new events to
        """
        grid = kwargs['grid']
        event_heap = kwargs['event_heap']
        
        print(f"PROCESSING BallGridTransit: Ball {self.ball.index} moving from {self.ball.cell} to {self.new_cell} at t={self.time:.6f}")
        print(f"  Ball position: {self.ball.position}")
        print(f"  Ball velocity: {self.ball.velocity}")
        
        # Update ball's cell (position/velocity/time unchanged)
        old_cell = self.ball.cell
        self.ball.cell = self.new_cell
        
        # Update grid cell memberships
        grid.move_ball(self.ball.index, old_cell, self.new_cell)
        print(f"  Updated grid: ball {self.ball.index} moved from {old_cell} to {self.new_cell}")
        
        # Note: No event invalidation needed - velocity unchanged
        
        # Generate new ball-ball events for newly adjacent cells
        from .event_generation import generate_ball_ball_events_for_new_cell, generate_ball_grid_event
        balls = kwargs['balls']
        ndim = kwargs['ndim']
        gravity = kwargs['gravity']
        current_time = self.time  # Use event time as current time
        
        new_events = generate_ball_ball_events_for_new_cell(self.ball, old_cell, balls, grid, current_time, ndim, gravity)
        print(f"  Generated {len(new_events)} new ball-ball events for newly adjacent cells")
        
        # Generate new ball-grid transit event for continued movement in new cell
        grid_events = generate_ball_grid_event(self.ball, current_time, ndim, gravity)
        new_events.extend(grid_events)
        print(f"  Generated {len(grid_events)} new ball-grid events for continued movement")
        
        for event in new_events:
            print(f"    {event}")
        
        # Add new events to heap
        for event in new_events:
            event_heap.add_event(event)


class ExportEvent(Event):
    """Event for outputting simulation state to file."""
    
    def __init__(self, time: float):
        """
        Initialize export event.
        
        Args:
            time: time when export should occur
        """
        super().__init__(time)
    
    def get_participants(self) -> List:
        """Export events don't involve specific balls or walls."""
        return []
    
    def process(self, **kwargs) -> None:
        """
        Process export event.
        
        Required kwargs:
            balls: list of all balls
            output_manager: object to handle file output
            ndim: number of dimensions
            gravity: whether gravity is enabled
        """
        balls = kwargs['balls']
        output_manager = kwargs['output_manager']
        ndim = kwargs['ndim']
        gravity = kwargs['gravity']
        
        # Calculate current positions for all balls without updating them
        current_positions = []
        current_velocities = []
        
        for ball in balls:
            pos = ball.get_position_at_time(self.time, ndim, gravity)
            current_positions.append(pos)
            current_velocities.append(ball.velocity.copy())
        
        # Write to output
        output_manager.write_frame(self.time, current_positions, current_velocities)


class EndEvent(Event):
    """Event to signal simulation end."""
    
    def __init__(self, time: float):
        """
        Initialize end event.
        
        Args:
            time: simulation end time
        """
        super().__init__(time)
    
    def get_participants(self) -> List:
        """End events don't involve specific balls or walls."""
        return []
    
    def process(self, **kwargs) -> None:
        """Process end event - signal simulation to stop."""
        kwargs['simulation_state']['should_end'] = True