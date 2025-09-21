from typing import List, TYPE_CHECKING
from .events import BallBallCollision, BallWallCollision, BallGridTransit
from .physics import (
    calculate_ball_ball_collision_time,
    calculate_ball_wall_collision_time, 
    calculate_ball_grid_transit_time
)

if TYPE_CHECKING:
    from .ball import Ball
    from .wall import Wall
    from .grid import Grid


def generate_ball_ball_events(ball: 'Ball', other_balls: List['Ball'], 
                             current_time: float, ndim: int, 
                             gravity: bool = False, log_events: bool = True) -> List[BallBallCollision]:
    """
    Generate ball-ball collision events for a ball against a list of other balls.
    
    Args:
        ball: the ball to generate events for
        other_balls: list of other balls to check collisions against
        current_time: current simulation time
        ndim: number of dimensions
        gravity: whether gravity is enabled
        
    Returns:
        list of BallBallCollision events
    """
    events = []
    
    for other_ball in other_balls:
        # Skip self-collision
        if ball.index == other_ball.index:
            continue
            
        collision_time = calculate_ball_ball_collision_time(
            ball, other_ball, current_time, ndim, gravity
        )
        
        if collision_time is not None:
            event = BallBallCollision(collision_time, ball, other_ball)
            events.append(event)
            
            import json
            log_entry = {
                "event_type": "EventCreated",
                "created_event": "BallBallCollision", 
                "time": collision_time,
                "ball1": ball.index,
                "ball2": other_ball.index
            }
            print(json.dumps(log_entry))
        else:
            import json
            log_entry = {
                "event_type": "CollisionCheck",
                "result": "no_collision",
                "ball1": ball.index, 
                "ball2": other_ball.index,
                "current_time": current_time
            }
            print(json.dumps(log_entry))
    
    return events


def generate_ball_wall_events(ball: 'Ball', walls: List['Wall'],
                             current_time: float, ndim: int,
                             gravity: bool = False) -> List[BallWallCollision]:
    """
    Generate ball-wall collision events for a ball against all walls.
    
    Args:
        ball: the ball to generate events for
        walls: list of walls to check collisions against
        current_time: current simulation time
        ndim: number of dimensions
        gravity: whether gravity is enabled
        
    Returns:
        list of BallWallCollision events
    """
    events = []
    
    for wall in walls:
        collision_time = calculate_ball_wall_collision_time(
            ball, wall, current_time, ndim, gravity
        )
        
        if collision_time is not None:
            event = BallWallCollision(collision_time, ball, wall)
            events.append(event)
            
            import json
            log_entry = {
                "event_type": "EventCreated",
                "created_event": "BallWallCollision",
                "time": collision_time,
                "ball": ball.index,
                "wall": str(wall)
            }
            print(json.dumps(log_entry))
    
    return events


def generate_ball_grid_event(ball: 'Ball', current_time: float, ndim: int,
                            gravity: bool = False) -> List[BallGridTransit]:
    """
    Generate ball-grid transit event for when a ball crosses cell boundaries.
    
    Args:
        ball: the ball to generate event for
        current_time: current simulation time
        ndim: number of dimensions
        gravity: whether gravity is enabled
        
    Returns:
        list containing BallGridTransit event (empty if no transit)
    """
    events = []
    
    result = calculate_ball_grid_transit_time(ball, current_time, ndim, 1.0, gravity)
    
    if result is not None:
        transit_time, new_cell = result
        event = BallGridTransit(transit_time, ball, new_cell)
        events.append(event)
        
        import json
        log_entry = {
            "event_type": "EventCreated",
            "created_event": "BallGridTransit",
            "time": transit_time,
            "ball": ball.index,
            "from_cell": ball.cell,
            "to_cell": new_cell
        }
        print(json.dumps(log_entry))
    
    return events


def generate_events_for_ball(ball: 'Ball', balls: List['Ball'], walls: List['Wall'],
                           grid: 'Grid', current_time: float, ndim: int,
                           gravity: bool = False) -> List:
    """
    Generate all events (ball-ball, ball-wall, ball-grid) for a single ball.
    
    Args:
        ball: the ball to generate events for
        balls: list of all balls in simulation
        walls: list of all walls
        grid: the spatial grid
        current_time: current simulation time
        ndim: number of dimensions
        gravity: whether gravity is enabled
        
    Returns:
        list of all events involving this ball
    """
    events = []
    
    # Ball-ball events with neighboring balls
    neighbor_ball_indices = grid.get_balls_in_neighboring_cells(ball.cell)
    neighbor_balls = [balls[i] for i in neighbor_ball_indices]
    
    # Generate neighbor cell info for JSON log
    neighbor_cells_info = {}
    if ndim == 2:
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                neighbor_cell = (ball.cell[0] + di, ball.cell[1] + dj)
                if (0 <= neighbor_cell[0] < grid.num_cells[0] and 
                    0 <= neighbor_cell[1] < grid.num_cells[1]):
                    balls_in_cell = grid.cells[neighbor_cell[0]][neighbor_cell[1]]
                    neighbor_cells_info[str(neighbor_cell)] = balls_in_cell
                else:
                    neighbor_cells_info[str(neighbor_cell)] = "OUT_OF_BOUNDS"
    
    import json
    log_entry = {
        "event_type": "EventGeneration",
        "ball": ball.index,
        "cell": ball.cell,
        "current_time": current_time,
        "neighbor_cells": neighbor_cells_info,
        "neighbor_balls": [b.index for b in neighbor_balls]
    }
    print(json.dumps(log_entry))
    events.extend(generate_ball_ball_events(ball, neighbor_balls, current_time, ndim, gravity))
    
    # Ball-wall events
    events.extend(generate_ball_wall_events(ball, walls, current_time, ndim, gravity))
    
    # Ball-grid transit event
    events.extend(generate_ball_grid_event(ball, current_time, ndim, gravity))
    
    return events


def generate_ball_ball_events_for_new_cell(ball: 'Ball', old_cell: tuple, 
                                          balls: List['Ball'], grid: 'Grid',
                                          current_time: float, ndim: int,
                                          gravity: bool = False) -> List[BallBallCollision]:
    """
    Generate ball-ball collision events for newly adjacent cells after grid transit.
    
    Args:
        ball: the ball that moved cells
        old_cell: the previous cell coordinates
        balls: list of all balls in simulation
        grid: the spatial grid
        current_time: current simulation time
        ndim: number of dimensions
        gravity: whether gravity is enabled
        
    Returns:
        list of BallBallCollision events with balls in newly adjacent cells
    """
    # Get balls in newly adjacent cells
    new_neighbor_indices = grid.get_balls_in_new_neighbor_cells(old_cell, ball.cell)
    new_neighbor_balls = [balls[i] for i in new_neighbor_indices]
    
    # Generate collision events with these newly adjacent balls
    return generate_ball_ball_events(ball, new_neighbor_balls, current_time, ndim, gravity)