import numpy as np
import math
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .ball import Ball
    from .wall import Wall


def calculate_ball_ball_collision_time(ball1: 'Ball', ball2: 'Ball', 
                                       current_time: float, ndim: int, 
                                       gravity: bool = False) -> Optional[float]:
    """
    Calculate when two balls will collide, if at all.
    
    Args:
        ball1: first ball
        ball2: second ball 
        current_time: current simulation time
        ndim: number of dimensions
        gravity: whether gravity is enabled
        
    Returns:
        collision time if collision occurs in future, None otherwise
    """
    # Get positions at current time
    pos1 = ball1.get_position_at_time(current_time, ndim, gravity)
    pos2 = ball2.get_position_at_time(current_time, ndim, gravity)
    
    # Relative position and velocity
    rel_pos = pos2 - pos1
    rel_vel = ball2.velocity - ball1.velocity
    
    # CHEAP TEST FIRST: Check if balls are moving apart
    # If rel_pos · rel_vel > 0, balls are moving away from each other
    pos_dot_vel = np.dot(rel_pos, rel_vel)
    if pos_dot_vel > 0:
        return None  # Moving apart, no collision
    
    # CHEAP TEST: Check if relative velocity is zero
    rel_vel_sq = np.dot(rel_vel, rel_vel)  # |rel_vel|^2
    if rel_vel_sq < 1e-24:  # Effectively zero relative velocity
        return None
    
    # Now do the expensive calculations
    rel_pos_sq = np.dot(rel_pos, rel_pos)  # |rel_pos|^2
    touch_distance_sq = (ball1.radius + ball2.radius)**2
    
    # Solve quadratic: |rel_pos + rel_vel * t|^2 = touch_distance^2
    # a*t^2 + b*t + c = 0 where:
    a = rel_vel_sq  # |rel_vel|^2
    b = 2 * pos_dot_vel  # 2*(rel_pos·rel_vel) - already computed!
    c = rel_pos_sq - touch_distance_sq  # |rel_pos|^2 - r^2
    
    # Check discriminant before expensive sqrt
    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return None
    
    # Only compute sqrt if we need it
    sqrt_discriminant = math.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2*a)
    t2 = (-b + sqrt_discriminant) / (2*a)
    
    # We want the earliest positive time
    collision_time = None
    if t1 > 1e-12:
        collision_time = t1
    if t2 > 1e-12 and (collision_time is None or t2 < collision_time):
        collision_time = t2
    
    if collision_time is None:
        return None
    
    return current_time + collision_time


def calculate_ball_wall_collision_time(ball: 'Ball', wall: 'Wall', 
                                       current_time: float, ndim: int,
                                       gravity: bool = False) -> Optional[float]:
    """
    Calculate when a ball will collide with a wall, if at all.
    
    Args:
        ball: the ball
        wall: the wall
        current_time: current simulation time
        ndim: number of dimensions
        gravity: whether gravity is enabled
        
    Returns:
        collision time if collision occurs in future, None otherwise
    """
    # Get ball position at current time
    pos = ball.get_position_at_time(current_time, ndim, gravity)
    vel = ball.velocity
    
    wall_axis = wall.normal_axis
    
    # Distance from ball center to wall
    ball_to_wall = wall.coordinate - pos[wall_axis]
    
    # Determine collision coordinate (where ball surface touches wall)
    if ball_to_wall > 0:
        # Ball is on negative side of wall - collision when ball surface reaches wall
        collision_coord = wall.coordinate - ball.radius
    else:
        # Ball is on positive side of wall - collision when ball surface reaches wall
        collision_coord = wall.coordinate + ball.radius
    
    velocity_component = vel[wall_axis]
    
    if wall_axis == 1 and gravity:
        # Y-direction with gravity - solve quadratic
        
        # Solve: pos_y + vel_y*t - 0.5*t^2 = collision_coord
        # Rearrange: 0.5*t^2 - vel_y*t + (pos_y - collision_coord) = 0
        a = 0.5
        b = -vel[1]
        c = pos[1] - collision_coord
        
        # Check discriminant before sqrt
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return None
            
        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-b + sqrt_discriminant) / (2*a)
        t2 = (-b - sqrt_discriminant) / (2*a)
        
        collision_time = None
        if t1 > 1e-12:
            collision_time = t1
        if t2 > 1e-12 and (collision_time is None or t2 < collision_time):
            collision_time = t2
    else:
        # Linear motion - CHEAP!
        if abs(velocity_component) < 1e-12:
            return None  # Not moving toward wall
        
        collision_distance = collision_coord - pos[wall_axis]
        collision_time = collision_distance / velocity_component
        if collision_time <= 1e-12:
            return None  # Collision in past
    
    if collision_time is None:
        return None
    
    return current_time + collision_time


def calculate_ball_grid_transit_time(ball: 'Ball', current_time: float, ndim: int,
                                     cell_size: float, gravity: bool = False) -> Optional[tuple]:
    """
    Calculate when a ball will cross a cell boundary, if at all.
    
    Args:
        ball: the ball
        current_time: current simulation time
        ndim: number of dimensions
        cell_size: size of grid cells
        gravity: whether gravity is enabled
        
    Returns:
        (collision_time, new_cell) if transit occurs, None otherwise
    """
    pos = ball.get_position_at_time(current_time, ndim, gravity)
    vel = ball.velocity
    
    earliest_time = None
    new_cell = None
    
    # Check each dimension for cell boundary crossings
    for axis in range(ndim):
        current_cell_coord = ball.cell[axis]
        velocity_component = vel[axis]
        
        # CHEAP TEST: Skip if not moving in this direction
        if abs(velocity_component) < 1e-12 and not (axis == 1 and gravity):
            continue
        
        # Calculate boundaries
        left_boundary = current_cell_coord * cell_size
        right_boundary = (current_cell_coord + 1) * cell_size
        
        if axis == 1 and gravity:
            # Y-direction with gravity - quadratic
            for boundary, new_coord in [(left_boundary, current_cell_coord - 1),
                                       (right_boundary, current_cell_coord + 1)]:
                # -0.5*t^2 + vel_y*t + (pos_y - boundary) = 0
                a = -0.5
                b = vel[1]
                c = pos[1] - boundary
                
                discriminant = b*b - 4*a*c
                if discriminant >= 0:
                    sqrt_discriminant = math.sqrt(discriminant)
                    t1 = (-b + sqrt_discriminant) / (2*a)
                    t2 = (-b - sqrt_discriminant) / (2*a)
                    
                    for t in [t1, t2]:
                        if t > 1e-12 and (earliest_time is None or t < earliest_time):
                            earliest_time = t
                            new_cell_coords = list(ball.cell)
                            new_cell_coords[axis] = new_coord
                            new_cell = tuple(new_cell_coords)
        else:
            # Linear motion - CHEAP!
            for boundary, new_coord in [(left_boundary, current_cell_coord - 1),
                                       (right_boundary, current_cell_coord + 1)]:
                t = (boundary - pos[axis]) / velocity_component
                
                if t > 1e-12 and (earliest_time is None or t < earliest_time):
                    earliest_time = t
                    new_cell_coords = list(ball.cell)
                    new_cell_coords[axis] = new_coord
                    new_cell = tuple(new_cell_coords)
    
    if earliest_time is None:
        return None
    
    return current_time + earliest_time, new_cell


def perform_ball_ball_collision(ball1: 'Ball', ball2: 'Ball', restitution: float):
    """
    Perform collision between two balls, updating their velocities.
    
    Args:
        ball1: first ball (modified in place)
        ball2: second ball (modified in place) 
        restitution: coefficient of restitution
    """
    # Calculate collision normal
    rel_pos = ball2.position - ball1.position
    distance_sq = np.dot(rel_pos, rel_pos)
    
    # CHEAP TEST: avoid expensive sqrt
    if distance_sq < 1e-24:
        # Balls at same position - arbitrary normal
        normal = np.zeros_like(ball1.position)
        normal[0] = 1.0
    else:
        distance = math.sqrt(distance_sq)  # Only sqrt when needed
        normal = rel_pos / distance
    
    # Relative velocity
    rel_vel = ball2.velocity - ball1.velocity
    vel_along_normal = np.dot(rel_vel, normal)
    
    # CHEAP TEST: if separating, no collision needed
    if vel_along_normal > 0:
        return
    
    # Update velocities (equal mass case)
    delta_vel = -(1 + restitution) * vel_along_normal
    velocity_change = (delta_vel / 2.0) * normal
    
    ball1.velocity -= velocity_change
    ball2.velocity += velocity_change


def perform_ball_wall_collision(ball: 'Ball', wall: 'Wall', restitution: float):
    """
    Perform collision between ball and wall, updating ball velocity.
    
    Args:
        ball: the ball (modified in place)
        wall: the wall
        restitution: coefficient of restitution
    """
    # Get wall normal pointing away from wall surface toward ball
    normal = np.zeros(len(ball.position))
    
    # Determine normal direction based on which side of wall the ball is on
    ball_coordinate = ball.position[wall.normal_axis]
    if ball_coordinate < wall.coordinate:
        # Ball is on negative side - normal points in negative direction
        normal[wall.normal_axis] = -1.0
    else:
        # Ball is on positive side - normal points in positive direction  
        normal[wall.normal_axis] = 1.0
    
    # Velocity component along normal (toward wall is negative)
    vel_along_normal = np.dot(ball.velocity, normal)
    
    # CHEAP TEST: if moving away from wall, no collision
    if vel_along_normal >= 0:
        return
    
    # Reflect velocity component along normal
    ball.velocity -= (1 + restitution) * vel_along_normal * normal