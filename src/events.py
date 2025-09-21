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
        
        import json
        
        log_entry = {
            "event_type": "BallBallCollision",
            "time": self.time,
            "ball1": {
                "index": self.ball1.index,
                "position_before": self.ball1.position.tolist(),
                "velocity_before": self.ball1.velocity.tolist(),
                "cell": self.ball1.cell,
                "time": self.ball1.time
            },
            "ball2": {
                "index": self.ball2.index, 
                "position_before": self.ball2.position.tolist(),
                "velocity_before": self.ball2.velocity.tolist(),
                "cell": self.ball2.cell,
                "time": self.ball2.time
            }
        }
        
        # Update both balls to collision time
        self.ball1.update_to_time(self.time, ndim, gravity)
        self.ball2.update_to_time(self.time, ndim, gravity)
        
        # Add collision positions to log
        log_entry["ball1"]["position_at_collision"] = self.ball1.position.tolist()
        log_entry["ball2"]["position_at_collision"] = self.ball2.position.tolist()
        
        # Perform collision (updates velocities)
        perform_ball_ball_collision(self.ball1, self.ball2, ball_restitution)
        
        # Add final velocities to log
        log_entry["ball1"]["velocity_after"] = self.ball1.velocity.tolist()
        log_entry["ball2"]["velocity_after"] = self.ball2.velocity.tolist()
        
        # Print updated log entry
        print(json.dumps(log_entry))
        
        # Invalidate all events for both balls (velocities changed)
        events1 = len(self.ball1.events)
        events2 = len(self.ball2.events)
        self.ball1.invalidate_all_events()
        self.ball2.invalidate_all_events()
        
        log_entry["events_invalidated"] = {
            "ball1": events1,
            "ball2": events2
        }
        
        # Generate new events for both balls (ball-ball, ball-wall, ball-grid)
        from .event_generation import generate_events_for_ball
        balls = kwargs['balls']
        walls = kwargs['walls']
        grid = kwargs['grid']
        
        # Generate new events for both balls (suppress individual event logging)
        import sys, io
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        new_events = generate_events_for_ball(self.ball1, balls, walls, grid, self.time, ndim, gravity)
        new_events.extend(generate_events_for_ball(self.ball2, balls, walls, grid, self.time, ndim, gravity))
        
        # Restore stdout
        sys.stdout = original_stdout
        
        # Add event generation summary to collision log with detailed event info
        event_details = []
        for event in new_events:
            event_info = {"type": event.__class__.__name__, "time": event.time}
            if hasattr(event, 'ball1') and hasattr(event, 'ball2'):
                event_info["ball1"] = event.ball1.index
                event_info["ball2"] = event.ball2.index
            elif hasattr(event, 'ball'):
                event_info["ball"] = event.ball.index
                if hasattr(event, 'wall'):
                    event_info["wall"] = str(event.wall)
                elif hasattr(event, 'new_cell'):
                    event_info["new_cell"] = list(event.new_cell)
            event_details.append(event_info)
        
        log_entry["new_events_generated"] = {
            "count": len(new_events),
            "events": event_details
        }
        
        # Final log output
        print(json.dumps(log_entry))
        
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
        
        import json
        
        log_entry = {
            "event_type": "BallWallCollision",
            "time": self.time,
            "ball": {
                "index": self.ball.index,
                "position_before": self.ball.position.tolist(),
                "velocity_before": self.ball.velocity.tolist(),
                "cell": self.ball.cell,
                "time": self.ball.time
            },
            "wall": str(self.wall)
        }
        
        # Update ball to collision time
        self.ball.update_to_time(self.time, ndim, gravity)
        log_entry["ball"]["position_at_collision"] = self.ball.position.tolist()
        
        # Perform collision (updates velocity)
        perform_ball_wall_collision(self.ball, self.wall, wall_restitution)
        log_entry["ball"]["velocity_after"] = self.ball.velocity.tolist()
        
        # Invalidate all events for the ball (velocity changed)
        events_invalidated = len(self.ball.events)
        self.ball.invalidate_all_events()
        log_entry["events_invalidated"] = events_invalidated
        
        # Generate new events for the ball (suppress individual event logging)
        from .event_generation import generate_events_for_ball
        balls = kwargs['balls']
        walls = kwargs['walls']
        grid = kwargs['grid']
        
        # Suppress individual event generation logging during collision processing
        import sys, io
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        new_events = generate_events_for_ball(self.ball, balls, walls, grid, self.time, ndim, gravity)
        
        # Restore stdout
        sys.stdout = original_stdout
        
        # Add event generation summary to collision log with detailed event info
        event_details = []
        for event in new_events:
            event_info = {"type": event.__class__.__name__, "time": event.time}
            if hasattr(event, 'ball1') and hasattr(event, 'ball2'):
                event_info["ball1"] = event.ball1.index
                event_info["ball2"] = event.ball2.index
            elif hasattr(event, 'ball'):
                event_info["ball"] = event.ball.index
                if hasattr(event, 'wall'):
                    event_info["wall"] = str(event.wall)
                elif hasattr(event, 'new_cell'):
                    event_info["new_cell"] = list(event.new_cell)
            event_details.append(event_info)
        
        log_entry["new_events_generated"] = {
            "count": len(new_events),
            "events": event_details
        }
        
        # Print complete collision log with nested event generation
        print(json.dumps(log_entry))
        
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
        
        import json
        
        log_entry = {
            "event_type": "BallGridTransit", 
            "time": self.time,
            "ball": {
                "index": self.ball.index,
                "position": self.ball.position.tolist(),
                "velocity": self.ball.velocity.tolist(),
                "old_cell": self.ball.cell,
                "new_cell": self.new_cell,
                "time": self.ball.time
            }
        }
        
        # Update ball's cell (position/velocity/time unchanged)
        old_cell = self.ball.cell
        self.ball.cell = self.new_cell
        
        # Update grid cell memberships
        grid.move_ball(self.ball.index, old_cell, self.new_cell)
        
        # Generate new ball-ball events for newly adjacent cells
        from .event_generation import generate_ball_ball_events_for_new_cell, generate_ball_grid_event
        balls = kwargs['balls']
        ndim = kwargs['ndim']
        gravity = kwargs['gravity']
        current_time = self.time  # Use event time as current time
        
        # Generate new events (suppress individual event logging)
        import sys, io
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        new_events = generate_ball_ball_events_for_new_cell(self.ball, old_cell, balls, grid, current_time, ndim, gravity)
        
        # Generate new ball-grid transit event for continued movement in new cell
        grid_events = generate_ball_grid_event(self.ball, current_time, ndim, gravity)
        new_events.extend(grid_events)
        
        # Restore stdout
        sys.stdout = original_stdout
        
        # Add event generation summary to transit log with detailed event info
        event_details = []
        for event in new_events:
            event_info = {"type": event.__class__.__name__, "time": event.time}
            if hasattr(event, 'ball1') and hasattr(event, 'ball2'):
                event_info["ball1"] = event.ball1.index
                event_info["ball2"] = event.ball2.index
            elif hasattr(event, 'ball'):
                event_info["ball"] = event.ball.index
                if hasattr(event, 'wall'):
                    event_info["wall"] = str(event.wall)
                elif hasattr(event, 'new_cell'):
                    event_info["new_cell"] = list(event.new_cell)
            event_details.append(event_info)
        
        log_entry["new_events_generated"] = {
            "ball_ball_events": len(new_events) - len(grid_events),
            "grid_events": len(grid_events),
            "total_count": len(new_events),
            "events": event_details
        }
        
        # Print JSON log
        print(json.dumps(log_entry))
        
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
        
        # Calculate current positions and velocities for all balls without updating them
        current_positions = []
        current_velocities = []
        
        for ball in balls:
            pos, vel = ball.get_position_and_velocity_at_time(self.time, ndim, gravity)
            current_positions.append(pos)
            current_velocities.append(vel)
        
        # Check for overlapping balls
        import numpy as np
        import math
        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                ball_i = balls[i]
                ball_j = balls[j]
                pos_i = current_positions[i]
                pos_j = current_positions[j]
                
                # Calculate distance between ball centers
                distance_vec = pos_j - pos_i
                distance = math.sqrt(np.dot(distance_vec, distance_vec))
                min_distance = ball_i.radius + ball_j.radius
                
                if distance < min_distance:
                    error_msg = f"OVERLAP DETECTED at t={self.time:.6f}: Ball {ball_i.index} and Ball {ball_j.index}"
                    error_msg += f"\n  Ball {ball_i.index}: position={pos_i}, radius={ball_i.radius}"
                    error_msg += f"\n  Ball {ball_j.index}: position={pos_j}, radius={ball_j.radius}"
                    error_msg += f"\n  Distance={distance:.6f}, Min distance={min_distance:.6f}"
                    error_msg += f"\n  Overlap amount={min_distance - distance:.6f}"
                    print(error_msg)
                    
                    # Kill the simulation
                    raise RuntimeError(f"Ball overlap detected: {error_msg}")
        
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