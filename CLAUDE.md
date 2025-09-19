# Molecular Simulations of Granular Media

## Goal

This is an implementation of a molecular dynamics simulation of granular media.

## Basic Setup

* github: This project has a github repo at https://github.com/cbizon/sand
* uv: we are using uv for package and environment management
* tests: we are using pytest, and want to maintain high code coverage

## Key Dependencies

- numpy: numerical computations
- pytest: testing framework  
- Flask: web application for visualization
- matplotlib: plotting and visualization

## Coordinate System

We use a right-handed coordinate system with the following conventions:

**2D System:**
- **x-axis**: Horizontal, positive direction points RIGHT
- **y-axis**: Vertical, positive direction points UP
- **Origin (0,0)**: Bottom-left corner of simulation domain

**3D System:**
- **x-axis**: Horizontal, positive direction points RIGHT  
- **y-axis**: Vertical, positive direction points UP
- **z-axis**: Depth, positive direction points OUT toward viewer (right-handed rule)
- **Origin (0,0,0)**: Bottom-left-front corner of simulation domain

**Gravity:**
- When enabled, gravity acts in the **negative y-direction** (downward)
- Magnitude g=1 in scaled time units

**Simulation Domain:**
- Rectangular box with walls inset by 0.01 from domain boundaries
- Domain spans from (0,0) to (width, height) in 2D
- Domain spans from (0,0,0) to (width, height, depth) in 3D

## Basic Workflow

We will construct a simulation of circular grains interacting via collision. This will be an event-driven simulation.  In an event driven simulation, we don't have a "time step".  Instead we calculate when events will occur, and advance to that point.

## Data Structures

Each ball is either a circle or sphere.  We have a ball class.  it has a position and a velocity. It also has a "time" which is the time of its most recent collision

Our balls are in a list, and we index to them based on their order in the list.
That index is also a member of the ball instance.

We will have an Event class.  There will be several subclasses:
BallBallCollision
BallWallCollision
BallGridTransit
ExportEvent

Each Ball will also contain a list of events that it participates in.

Our simulation volume is a 2D or 3D grid. We divide it into cells with dimension 1 on each edge.  We are going to make a 2 or 3 dimensional array of cells. Each cell holds a list of the ball indices whose centers are in the cell.  These should probably be numpy arrays.

We have Wall elements as well to store positions and paremeters of each wall. Walls currently are either horizontal or vertical.  Our initial setup will be elastic walls 0.01 inset from the bounds of the grid.

Each Event contains 1. refs to the elements (balls and walls) that participate in the event.  2. Time at which the event occurs. 3. A valid/invalid flag. True = valid.

There is also a heap to keep track of events, sorted such that the next event to process in the 

## Algorithm

The basic idea is that for each ball, we figure out what ball it will hit next.  We can use the laws of mechanics to determine when that will occur. There's no point in simulating the time between now and the collision. So we just jump to that time point.  Of course that is happening for all of the balls. So we will need to calculate for all balls and advance to the next (global) collision.  When a collision occurs, we don't actually need to update anything except the balls that are members of the collision. A collision across the cell is unchanged.  To limit the potentially N-squared nature of potential collisions, we use the grid to limit which balls we need to check for collisions.

1. Initialize. We will have parameters defining a run.  In particular we will have (to start): 
- Number of dimensions (2 or 3)
- Number of balls
- Ball radius
- gravity (true (g=1) or false (g=0))
- ball coefficient of restitution (1.0 - perfectly elastic for initial implementation, treat as elastic gas) 
- wall coefficient of restitution (1.0 - perfectly elastic for initial implementation)
- cell size
- output rate
- simulation time
- random_seed (default 100 for reproducible results)

The Data structures described above are initialized. balls must not overlap one another or walls. Ball radius must be smaller than cell size - if not, abort at startup (larger balls would require checking non-adjacent cells for collisions). Maximum ball radius is 0.5 for current placement strategy to avoid overlaps. If our balls are smaller than a cell, then the easiest way to handle this is by putting each ball in its own cell at the center. Velocities are chosen from an n-dimensional zero-centered, sd=1 gaussian using the specified random seed.

Cell memberships are updated with the intial positions.

Now we walk over every ball.  Based on its cell, we look at neighboring cells to find the set of balls that this ball might hit.  Based on the position, velocity, and radius of the balls, we calculate when the balls would collide.  Note that gravity does not matter for ball-ball collisions because the 1/2 g t ** 2 terms in the positions of each ball cancel out.  For each collision we find that will occur in the future, we create a BallBallCollision and put it on the heap.

**DUPLICATE PREVENTION**: During initialization only, to avoid creating duplicate ball-ball collision events, each ball should only generate events with higher-indexed balls in its neighborhood. During simulation after collisions, generate events with all neighboring balls.

We also calculate when the ball would leave its current cell, and make a BallGridEvent (gravity affects this timing)

We also calcualte when the ball would hit a wall, and make a BallWallEvent (gravity affects this timing).

We also add events for output events at the specified delta (including an initial t=0 export), and an end event

2. In a loop, until we hit the time limit,  chose the next event from the heap.  If it is invalid, discard it.  if it is valid, process it.

3. To process a BallBallCollision, 
- update the two balls positions based on their times, the current time, and their velocities.  This should advance the balls to the collision time.  
- Perform the collision, updating the velocities of the balls
- Update the times of the balls to the collision time
- Invalidate all events for each ball (velocity changed, so all existing events are invalid)
- based on the new ball position, time, and velocities, calculate a new round of ball/ball, ball/wall, and ball/grid events and add them to the heap.

4. To process a BallWallCollision
- update the ball position based on its times, the current time, and its velocities.  This should advance the ball to the collision time.  
- Perform the collision, updating the velocity of the ball
- Update the time of the ball to the collision time
- Invalidate all events for the ball (velocity changed, so all existing events are invalid)
- based on the new ball position, time, and velocities, calculate a new round of ball/ball, ball/wall, and ball/grid events and add them to the heap.

5. To process a BallGridTransit
- The ball position, time, and velocity remain the same (no velocity change, so existing events remain valid)
- Update the ball's cell
- Update the cell memberships
- Generate new ball-ball events for balls in newly adjacent cells (only in direction of movement)
- **CRITICAL**: Generate a new BallGridTransit event for the ball's continued movement in its new cell

6. To process an output event, calculate and write the positions and velocities of all balls into a file. There is no need to update the internal positions or times of the balls

7, To proocess the end item, end.

## Project structure
src/
├── ball.py              # Ball class with position, velocity, radius
├── wall.py              # Wall classes for boundary collisions
├── grid.py              # Spatial grid for collision optimization
├── events.py            # Event classes (BallBallCollision, BallWallCollision, BallGridTransit, ExportEvent, EndEvent)
├── event_heap.py        # Priority queue for events
├── event_generation.py  # Functions to generate collision events
├── physics.py           # Physics calculations (collision times, collision responses)
├── simulation.py        # Main simulation logic and initialization
tests/                   # Comprehensive test suite
runs/                    # Output directory for simulation frames
app.py                   # Flask web application for visualization
go.py                    # Example simulation parameters


## ***RULES OF THE ROAD***

- Don't use mocks. They obscure problems

- Ask clarifying questions

- Don't make classes just to group code. It is non-pythonic and hard to test.

- Do not implement bandaids - treat the root cause of problems

- Don't use try/except as a way to hide problems.  It is often good just to let something fail and figure out why.

- Once we have a test, do not delete it without explicit permission.  

- Do not return made up results if an API fails.  Let it fail.

- When changing code, don't make duplicate functions - just change the function. We can always roll back changes if needed.

- Keep the directories clean, don't leave a bunch of junk laying around.

- When making pull requests, NEVER ever mention a `co-authored-by` or similar aspects. In particular, never mention the tool used to create the commit message or PR.

- Check git status before commits

## Key Implementation Details

### Event Generation and Duplicate Prevention
- During initialization: Each ball generates events only with higher-indexed neighboring balls to prevent duplicates
- During simulation: After velocity changes (collisions), generate events with all neighboring balls
- The `generate_ball_ball_events()` function creates events between one ball and all balls passed to it
- Duplicate prevention is controlled by filtering the input balls, not within the function itself

### Grid Transit Events
- When a ball crosses a cell boundary (BallGridTransit), it must generate:
  1. New ball-ball collision events with newly adjacent balls
  2. **A new BallGridTransit event for continued movement in the new cell**
- Failure to generate new grid transit events causes balls to become "stuck" in wrong cells

### Physics Validation
- Ball-ball collisions must satisfy:
  - Pre-collision: r·v < 0 (balls approaching)
  - Post-collision: r·v > 0 (balls separating)  
  - Momentum conservation (vector-wise)
  - Energy conservation (for e=1)

