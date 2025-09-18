# Molecular Dynamics Simulation of Granular Media

An event-driven molecular dynamics simulation of circular/spherical particles (balls) in 2D or 3D rectangular domains.

## Overview

This simulation implements an event-driven approach where time advances by jumping between collision events rather than using fixed time steps. The system tracks:

- **Ball-ball collisions** between particles
- **Ball-wall collisions** with domain boundaries  
- **Grid transit events** when particles move between spatial cells
- **Export events** for periodic data output

## Key Features

- **2D and 3D support** - simulate in either 2 or 3 dimensions
- **Event-driven algorithm** - efficient, exact collision timing
- **Spatial partitioning** - grid-based neighbor finding for O(N) scaling
- **Elastic collisions** - configurable restitution coefficients
- **Gravity support** - optional gravitational acceleration
- **Periodic output** - configurable data export intervals

## Installation

Requires Python 3.7+ with numpy and pytest:

```bash
pip install numpy pytest
```

## Quick Start

Run a simple 2D simulation:

```python
from src.simulation import run_simulation

params = {
    'ndim': 2,
    'num_balls': 4,
    'ball_radius': 0.9,
    'domain_size': (5.0, 3.0),
    'simulation_time': 10.0,
    'gravity': False,
    'output_rate': 1.0
}

run_simulation(params)
```

## Simulation Parameters

### Required Parameters

- `ndim` (int): Number of dimensions (2 or 3)
- `num_balls` (int): Number of particles to simulate
- `ball_radius` (float): Radius of all particles (must be < 1.0)
- `domain_size` (tuple): Domain dimensions (width, height) for 2D or (width, height, depth) for 3D
- `simulation_time` (float): Total simulation time

### Optional Parameters

- `gravity` (bool): Enable gravity in negative y-direction (default: False)
- `ball_restitution` (float): Coefficient of restitution for ball-ball collisions (default: 1.0)
- `wall_restitution` (float): Coefficient of restitution for ball-wall collisions (default: 1.0)
- `output_rate` (float): Time interval between data exports (default: 1.0)
- `output_dir` (str): Directory for output files (default: 'runs')

## Coordinate System

The simulation uses a right-handed coordinate system:

**2D System:**
- **x-axis**: Horizontal, positive direction points RIGHT
- **y-axis**: Vertical, positive direction points UP
- **Origin (0,0)**: Bottom-left corner

**3D System:**
- **x-axis**: Horizontal, positive direction points RIGHT
- **y-axis**: Vertical, positive direction points UP  
- **z-axis**: Depth, positive direction points OUT toward viewer
- **Origin (0,0,0)**: Bottom-left-front corner

**Gravity** (when enabled) acts in the negative y-direction with magnitude g=1.

## Output Format

The simulation writes frame files to the output directory with format:
```
# Time: 5.0
# Balls: 4
0 1.2 2.3 0.5 -0.2
1 3.1 1.8 -0.8 0.6
2 2.5 0.9 0.0 1.2
3 4.2 2.7 -0.3 -0.5
```

Each line contains: `ball_index x y [z] vx vy [vz]`

## Examples

### Basic 2D Simulation
```python
from src.simulation import run_simulation

# Simple 2D gas simulation
params = {
    'ndim': 2,
    'num_balls': 6,
    'ball_radius': 0.9,
    'domain_size': (4.0, 3.0),
    'simulation_time': 5.0,
    'gravity': False,
    'ball_restitution': 1.0,  # Perfectly elastic
    'output_rate': 0.5
}

run_simulation(params)
```

### 3D Simulation with Gravity
```python
# 3D granular media with gravity
params = {
    'ndim': 3,
    'num_balls': 8,
    'ball_radius': 0.9,
    'domain_size': (3.0, 4.0, 3.0),
    'simulation_time': 10.0,
    'gravity': True,  # Gravity enabled
    'ball_restitution': 0.9,  # Slightly inelastic
    'wall_restitution': 0.8,
    'output_rate': 0.2
}

run_simulation(params)
```

### Command Line Usage
```bash
# Run the example simulation
python src/simulation.py

# Or create your own script
python my_simulation.py
```

## Algorithm Details

The simulation implements the event-driven molecular dynamics algorithm:

1. **Initialization**: Place balls in grid cells with random velocities
2. **Event Generation**: Calculate collision times for all ball pairs, walls, and grid transits
3. **Event Processing**: Process events chronologically:
   - Update particle positions and velocities
   - Invalidate affected events
   - Generate new events for updated particles
4. **Output**: Periodically write particle states to files

## Performance

The spatial grid limits collision detection to neighboring cells, giving approximately O(N) scaling for uniform distributions. Typical performance:

- **2D**: ~1000 balls for interactive speeds
- **3D**: ~500 balls for interactive speeds

Performance depends on collision frequency and domain density.

## Testing

Run the test suite:
```bash
# Run all tests
python -m pytest

# Run specific test modules
python -m pytest tests/test_simulation.py -v
python -m pytest tests/test_physics.py -v
```

## Project Structure

```
src/
├── ball.py              # Ball class with position, velocity, time
├── wall.py              # Wall boundaries and collision detection
├── events.py            # Event classes (collisions, transits, output)
├── physics.py           # Collision time calculations and physics
├── grid.py              # Spatial partitioning grid
├── event_heap.py        # Priority queue for events
├── event_generation.py  # Functions to generate events
└── simulation.py        # Main simulation loop and initialization

tests/
├── test_ball.py
├── test_wall.py  
├── test_events.py
├── test_physics.py
├── test_grid.py
├── test_event_heap.py
├── test_event_generation.py
└── test_simulation.py
```

## Limitations

- Ball radius must be smaller than grid cell size (1.0)
- Maximum one ball per grid cell during initialization
- Perfectly spherical particles only
- Rectangular domain boundaries only

## License

This project is part of a molecular dynamics simulation framework for granular media research.