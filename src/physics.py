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
    
    Uses exact equations of motion accounting for different reference times:
    Ball 1: x1(t) = x0 + v0(t - t0) + 0.5*g*(t - t0)^2
    Ball 2: x2(t) = x1 + v1(t - t1) + 0.5*g*(t - t1)^2
    
    Args:
        ball1: first ball with position, velocity, and time
        ball2: second ball with position, velocity, and time
        current_time: current simulation time (collision must be after this)
        ndim: number of dimensions
        gravity: whether gravity is enabled
        
    Returns:
        collision time if collision occurs in future, None otherwise
    """
    # Ball parameters
    x0, v0, t0 = ball1.position, ball1.velocity, ball1.time
    x1, v1, t1 = ball2.position, ball2.velocity, ball2.time
    r = ball1.radius + ball2.radius
    
    if not gravity or ndim < 2:
        # Without gravity, use the simple linear approach but with correct reference times
        # Get positions and velocities at current time
        pos1, vel1 = ball1.get_position_and_velocity_at_time(current_time, ndim, gravity)
        pos2, vel2 = ball2.get_position_and_velocity_at_time(current_time, ndim, gravity)
        
        # Relative position and velocity
        rel_pos = pos2 - pos1
        rel_vel = vel2 - vel1
        
        # CHEAP TEST FIRST: Check if balls are moving apart
        pos_dot_vel = np.dot(rel_pos, rel_vel)
        if pos_dot_vel > 0:
            return None  # Moving apart, no collision
        
        # CHEAP TEST: Check if relative velocity is zero
        rel_vel_sq = np.dot(rel_vel, rel_vel)
        if rel_vel_sq < 1e-24:
            return None
        
        # Solve quadratic: |rel_pos + rel_vel * dt|^2 = r^2
        rel_pos_sq = np.dot(rel_pos, rel_pos)
        a = rel_vel_sq
        b = 2 * pos_dot_vel
        c = rel_pos_sq - r*r
        
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return None
        
        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2*a)
        t2 = (-b + sqrt_discriminant) / (2*a)
        
        # Find earliest positive time
        collision_time = None
        if t1 > 1e-12:
            collision_time = t1
        if t2 > 1e-12 and (collision_time is None or t2 < collision_time):
            collision_time = t2
        
        if collision_time is None:
            return None
        
        return current_time + collision_time
    
    else:
        # With gravity, use exact equations of motion
        # Position difference: x2(t) - x1(t) = distance_vector(t)
        # |distance_vector(t)| = r when collision occurs
        
        # Set up the equation: |x1 + v1(t-t1) + 0.5*g*(t-t1)^2 - x0 - v0(t-t0) - 0.5*g*(t-t0)^2|^2 = r^2
        
        # Let's expand this step by step:
        # dx = x1 - x0
        # dv = v1 - v0  
        # dt_offset = t1 - t0
        
        dx = x1 - x0
        dv = v1 - v0
        dt_offset = t1 - t0
        
        # For the gravity term: 0.5*g*[(t-t1)^2 - (t-t0)^2]
        # = 0.5*g*[t^2 - 2*t*t1 + t1^2 - t^2 + 2*t*t0 - t0^2]
        # = 0.5*g*[2*t*(t0-t1) + t1^2 - t0^2]
        # = g*t*(t0-t1) + 0.5*g*(t1^2 - t0^2)
        
        # So distance_vector(t) = dx + dv*t - dv*t0 + v1*t0 - v1*t1 + g*t*(t0-t1) + 0.5*g*(t1^2 - t0^2)
        # Simplifying: distance_vector(t) = [dx + dv*(-t0) + v1*(t0-t1) + 0.5*g*(t1^2-t0^2)] + t*[dv + g*(t0-t1)]
        
        # Set up gravity vector g = [0, -1] in 2D
        g = np.zeros_like(x0)
        g[1] = -1.0  # gravity in negative y direction
        
        # Constant term
        const_term = dx + dv*(-t0) + v1*(t0-t1) + 0.5*g*(t1*t1 - t0*t0)
        
        # Linear coefficient 
        linear_coeff = dv + g*(t0-t1)
        
        # Gravity coefficient (coefficient of t^2 term is zero because gravity affects both balls equally)
        # Wait, let me recalculate this properly...
        
        # Actually, let's be more systematic. The distance vector at time t is:
        # d(t) = [x1 + v1*(t-t1) + 0.5*g*(t-t1)^2] - [x0 + v0*(t-t0) + 0.5*g*(t-t0)^2]
        
        # Expanding:
        # d(t) = x1 - x0 + v1*t - v1*t1 - v0*t + v0*t0 + 0.5*g*[(t-t1)^2 - (t-t0)^2]
        # d(t) = (x1-x0) + t*(v1-v0) + (v0*t0 - v1*t1) + 0.5*g*[(t-t1)^2 - (t-t0)^2]
        
        # For the gravity term:
        # (t-t1)^2 - (t-t0)^2 = t^2 - 2*t*t1 + t1^2 - t^2 + 2*t*t0 - t0^2
        #                     = 2*t*(t0-t1) + (t1^2 - t0^2)
        
        # So: d(t) = (x1-x0) + (v0*t0-v1*t1) + 0.5*g*(t1^2-t0^2) + t*[(v1-v0) + g*(t0-t1)]
        
        A = dx + v0*t0 - v1*t1 + 0.5*g*(t1*t1 - t0*t0)  # constant term
        B = dv + g*(t0-t1)  # linear term coefficient
        
        # Now we need to solve |A + B*t|^2 = r^2
        # (A + B*t) · (A + B*t) = r^2
        # A·A + 2*A·B*t + B·B*t^2 = r^2
        # B·B*t^2 + 2*A·B*t + (A·A - r^2) = 0
        
        a_coeff = np.dot(B, B)  # coefficient of t^2
        b_coeff = 2 * np.dot(A, B)  # coefficient of t
        c_coeff = np.dot(A, A) - r*r  # constant term
        
        # Check if this is actually a quadratic (a_coeff != 0)
        if abs(a_coeff) < 1e-24:
            # Linear case: 2*A·B*t + (A·A - r^2) = 0
            if abs(b_coeff) < 1e-24:
                return None  # No solution
            t_collision = -c_coeff / b_coeff
            if t_collision > current_time + 1e-12:
                return t_collision
            else:
                return None
        
        # Solve quadratic
        discriminant = b_coeff*b_coeff - 4*a_coeff*c_coeff
        if discriminant < 0:
            return None
        
        sqrt_discriminant = math.sqrt(discriminant)
        t1_sol = (-b_coeff - sqrt_discriminant) / (2*a_coeff)
        t2_sol = (-b_coeff + sqrt_discriminant) / (2*a_coeff)
        
        # Find earliest time that's in the future
        collision_time = None
        if t1_sol > current_time + 1e-12:
            collision_time = t1_sol
        if t2_sol > current_time + 1e-12 and (collision_time is None or t2_sol < collision_time):
            collision_time = t2_sol
        
        return collision_time


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
    # Get ball position and velocity at current time
    pos, vel = ball.get_position_and_velocity_at_time(current_time, ndim, gravity)
    
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
        # Rearrange: 0.5*t^2 + (-vel_y)*t + (collision_coord - pos_y) = 0
        a = 0.5
        b = -vel[1]  # -vel[1] = -(-1.0) = +1.0 for downward motion
        c = collision_coord - pos[1]  # 2.4 - 5.0 = -2.6
        
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
    pos, vel = ball.get_position_and_velocity_at_time(current_time, ndim, gravity)
    
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