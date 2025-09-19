import numpy as np
import os
from typing import List, Tuple, Dict, Any
from .ball import Ball
from .wall import create_box_walls
from .grid import Grid
from .event_heap import EventHeap
from .events import ExportEvent, EndEvent
from .event_generation import generate_events_for_ball


class OutputManager:
    """Manages output file writing for simulation data."""
    
    def __init__(self, output_dir: str = "runs"):
        """
        Initialize output manager.
        
        Args:
            output_dir: directory to write output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.frame_count = 0
        
    def write_frame(self, time: float, positions: List[np.ndarray], velocities: List[np.ndarray]):
        """
        Write simulation frame to file.
        
        Args:
            time: simulation time
            positions: list of ball positions
            velocities: list of ball velocities
        """
        filename = os.path.join(self.output_dir, f"frame_{self.frame_count:06d}.txt")
        
        with open(filename, 'w') as f:
            f.write(f"# Time: {time}\n")
            f.write(f"# Balls: {len(positions)}\n")
            
            for i, (pos, vel) in enumerate(zip(positions, velocities)):
                if len(pos) == 2:
                    f.write(f"{i} {pos[0]} {pos[1]} {vel[0]} {vel[1]}\n")
                elif len(pos) == 3:
                    f.write(f"{i} {pos[0]} {pos[1]} {pos[2]} {vel[0]} {vel[1]} {vel[2]}\n")
        
        self.frame_count += 1


def validate_simulation_parameters(params: Dict[str, Any]) -> None:
    """
    Validate simulation parameters.
    
    Args:
        params: dictionary of simulation parameters
        
    Raises:
        ValueError: if parameters are invalid
    """
    required_params = ['ndim', 'num_balls', 'ball_radius', 'domain_size', 'simulation_time']
    for param in required_params:
        if param not in params:
            raise ValueError(f"Missing required parameter: {param}")
    
    if params['ndim'] not in [2, 3]:
        raise ValueError("ndim must be 2 or 3")
    
    if params['num_balls'] <= 0:
        raise ValueError("num_balls must be positive")
    
    if params['ball_radius'] <= 0:
        raise ValueError("ball_radius must be positive")
    
    if params['ball_radius'] >= 1.0:
        raise ValueError("ball_radius must be smaller than cell size (1.0)")
    
    # Check if balls would overlap with current placement strategy (center of cells)
    # Minimum distance between ball centers is 1.0 (adjacent cells)
    # So maximum radius is 0.5 to avoid overlaps
    if params['ball_radius'] > 0.5:
        raise ValueError(f"ball_radius ({params['ball_radius']}) too large - balls would overlap. Maximum radius is 0.5 for current placement strategy.")
    
    
    if len(params['domain_size']) != params['ndim']:
        raise ValueError("domain_size must match ndim")
    
    if any(size <= 0 for size in params['domain_size']):
        raise ValueError("all domain_size values must be positive")
    
    if params['simulation_time'] <= 0:
        raise ValueError("simulation_time must be positive")


def initialize_simulation(params: Dict[str, Any]) -> Tuple[List[Ball], List, Grid, EventHeap, OutputManager]:
    """
    Initialize simulation components.
    
    Args:
        params: simulation parameters
        
    Returns:
        tuple of (balls, walls, grid, event_heap, output_manager)
    """
    validate_simulation_parameters(params)
    
    ndim = params['ndim']
    num_balls = params['num_balls']
    ball_radius = params['ball_radius']
    domain_size = params['domain_size']
    wall_restitution = params.get('wall_restitution', 1.0)
    
    # Create walls
    walls = create_box_walls(ndim, domain_size, inset=0.01, restitution=wall_restitution)
    
    # Create grid
    grid = Grid(ndim, domain_size)
    
    # Check if we have enough cells for all balls
    total_cells = 1
    for size in domain_size:
        total_cells *= int(size)
    
    if num_balls > total_cells:
        raise ValueError(f"Too many balls ({num_balls}) for domain size {domain_size} ({total_cells} cells)")
    
    # Set random seed for reproducible results
    random_seed = params.get('random_seed', 100)
    np.random.seed(random_seed)
    
    # Create balls with non-overlapping positions
    balls = []
    for i in range(num_balls):
        # Simple placement: center each ball in its own cell
        if ndim == 2:
            cell_x = i % int(domain_size[0])
            cell_y = (i // int(domain_size[0])) % int(domain_size[1])
            position = np.array([cell_x + 0.5, cell_y + 0.5])
            cell = (cell_x, cell_y)
        else:  # ndim == 3
            cells_per_layer = int(domain_size[0]) * int(domain_size[1])
            cell_x = i % int(domain_size[0])
            cell_y = (i // int(domain_size[0])) % int(domain_size[1])
            cell_z = i // cells_per_layer
            position = np.array([cell_x + 0.5, cell_y + 0.5, cell_z + 0.5])
            cell = (cell_x, cell_y, cell_z)
        
        # Random velocity from gaussian
        velocity = np.random.normal(0.0, 1.0, ndim)
        
        ball = Ball(position, velocity, ball_radius, i, cell)
        balls.append(ball)
        
        # Add ball to grid
        grid.add_ball(i, cell)
    
    # Create event heap
    event_heap = EventHeap()
    
    # Create output manager
    output_manager = OutputManager(params.get('output_dir', 'runs'))
    
    return balls, walls, grid, event_heap, output_manager


def run_simulation(params: Dict[str, Any]) -> None:
    """
    Run the molecular dynamics simulation.
    
    Args:
        params: simulation parameters including:
            - ndim: number of dimensions (2 or 3)
            - num_balls: number of balls
            - ball_radius: radius of balls
            - domain_size: (width, height) for 2D or (width, height, depth) for 3D
            - simulation_time: total simulation time
            - gravity: whether gravity is enabled (default False)
            - ball_restitution: coefficient of restitution for balls (default 1.0)
            - wall_restitution: coefficient of restitution for walls (default 1.0)
            - output_rate: time interval for output (default 1.0)
            - output_dir: directory for output files (default 'runs')
    """
    # Initialize simulation
    balls, walls, grid, event_heap, output_manager = initialize_simulation(params)
    
    # Extract parameters
    ndim = params['ndim']
    simulation_time = params['simulation_time']
    gravity = params.get('gravity', False)
    ball_restitution = params.get('ball_restitution', 1.0)
    wall_restitution = params.get('wall_restitution', 1.0)
    output_rate = params.get('output_rate', 1.0)
    
    current_time = 0.0
    
    # Generate initial events for all balls
    print(f"Initializing events for {len(balls)} balls...")
    for ball in balls:
        # For initialization, only pass higher-indexed balls to prevent duplicates
        higher_indexed_balls = [b for b in balls if b.index > ball.index]
        
        # Get neighbor ball indices from grid
        neighbor_ball_indices = grid.get_balls_in_neighboring_cells(ball.cell)
        # Filter to only higher-indexed neighboring balls
        higher_neighbor_balls = [balls[i] for i in neighbor_ball_indices if i > ball.index]
        
        print(f"  Ball {ball.index} in cell {ball.cell}: checking {len(higher_neighbor_balls)} higher-indexed neighboring balls {[b.index for b in higher_neighbor_balls]}")
        
        # Generate ball-ball events (only with higher-indexed neighbors)
        from .event_generation import generate_ball_ball_events, generate_ball_wall_events, generate_ball_grid_event
        events = []
        events.extend(generate_ball_ball_events(ball, higher_neighbor_balls, current_time, ndim, gravity))
        
        # Generate ball-wall events (no duplication issue)
        events.extend(generate_ball_wall_events(ball, walls, current_time, ndim, gravity))
        
        # Generate ball-grid transit event (no duplication issue)
        events.extend(generate_ball_grid_event(ball, current_time, ndim, gravity))
        
        for event in events:
            event_heap.add_event(event)
    
    # Add export events (including initial condition at t=0)
    event_heap.add_event(ExportEvent(0.0))  # Initial condition
    export_time = output_rate
    while export_time <= simulation_time:
        event_heap.add_event(ExportEvent(export_time))
        export_time += output_rate
    
    # Add end event
    event_heap.add_event(EndEvent(simulation_time))
    
    # Simulation state
    simulation_state = {'should_end': False}
    
    print(f"Starting simulation (t=0 to {simulation_time})...")
    event_count = 0
    
    # Main event loop
    while not event_heap.is_empty() and not simulation_state['should_end']:
        event = event_heap.get_next_event()
        
        if event is None:
            break
        
        current_time = event.time
        event_count += 1
        
        if event_count % 1000 == 0:
            print(f"Processing event {event_count}, time={current_time:.3f}")
        
        # Process event based on type
        event.process(
            ndim=ndim,
            gravity=gravity,
            ball_restitution=ball_restitution,
            wall_restitution=wall_restitution,
            grid=grid,
            balls=balls,
            walls=walls,
            output_manager=output_manager,
            simulation_state=simulation_state,
            event_heap=event_heap
        )
    
    print(f"Simulation completed. Processed {event_count} events.")
    print(f"Final time: {current_time}")

