#!/usr/bin/env python3
"""
Plot x vs t and y vs t for ball 77 during grid transit.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Ball data from the log
ball_index = 77
initial_position = [27.892522024767487, 1.9493552007879855]
velocity = [1.0299902633547728, 0.6188638938455122]
initial_time = 0.9028911084157597
old_cell = [28, 2]
new_cell = [29, 2]

# Ball crossed from cell (28,2) to cell (29,2), so it crossed the x=29 boundary
# The grid cells have size 1.0, so boundaries are at integer values

print(f"Ball {ball_index} trajectory analysis:")
print(f"Initial position: {initial_position}")
print(f"Velocity: {velocity}")
print(f"Initial time: {initial_time}")
print(f"Cell transition: {old_cell} -> {new_cell}")
print(f"Crossed boundary at x = 29.0")
print()

# Time range for plotting
t_start = initial_time
t_end = initial_time + 2.0  # Plot for 2 seconds after initial time
t = np.linspace(t_start, t_end, 1000)

# Calculate positions over time
# With gravity: x(t) = x0 + vx*(t-t0), y(t) = y0 + vy*(t-t0) - 0.5*g*(t-t0)^2
# where g = 1.0 (gravity acceleration)
dt = t - initial_time

# Position trajectories
x_pos = initial_position[0] + velocity[0] * dt
y_pos = initial_position[1] + velocity[1] * dt - 0.5 * 1.0 * dt**2  # gravity g=1

# Find when ball crossed x=29 boundary
# x0 + vx*dt = 29
# dt = (29 - x0) / vx
if velocity[0] != 0:
    dt_x_crossing = (29.0 - initial_position[0]) / velocity[0]
    t_x_crossing = initial_time + dt_x_crossing
    x_crossing = 29.0
    y_at_x_crossing = initial_position[1] + velocity[1] * dt_x_crossing - 0.5 * 1.0 * dt_x_crossing**2
    
    print(f"X-boundary crossing time: t = {t_x_crossing:.6f}")
    print(f"X-crossing position: ({x_crossing:.6f}, {y_at_x_crossing:.6f})")

# Find when ball might cross y boundaries (if it does)
# y0 + vy*dt - 0.5*g*dt^2 = y_boundary
# -0.5*dt^2 + vy*dt + (y0 - y_boundary) = 0
y_boundaries = [1.0, 2.0, 3.0]  # Check common boundaries
y_crossings = []

for y_boundary in y_boundaries:
    # Quadratic: -0.5*dt^2 + vy*dt + (y0 - y_boundary) = 0
    a = -0.5
    b = velocity[1]
    c = initial_position[1] - y_boundary
    
    discriminant = b*b - 4*a*c
    if discriminant >= 0:
        sqrt_discriminant = np.sqrt(discriminant)
        dt1 = (-b + sqrt_discriminant) / (2*a)
        dt2 = (-b - sqrt_discriminant) / (2*a)
        
        for dt_y in [dt1, dt2]:
            if dt_y > 1e-12:  # Future crossing
                t_y_crossing = initial_time + dt_y
                x_at_y_crossing = initial_position[0] + velocity[0] * dt_y
                y_crossings.append((t_y_crossing, x_at_y_crossing, y_boundary))

# Find all y=2 crossings specifically
y2_crossings = [crossing for crossing in y_crossings if abs(crossing[2] - 2.0) < 1e-6]
y2_crossings.sort(key=lambda x: x[0])  # Sort by time

print(f"Y=2 boundary crossings:")
for i, (t_y, x_y, y_boundary) in enumerate(y2_crossings):
    print(f"  Crossing {i+1}: t = {t_y:.6f}, position = ({x_y:.6f}, {y_boundary:.6f})")

# Find the earliest future y-crossing of any boundary
earliest_y_crossing = None
if y_crossings:
    earliest_y_crossing = min(y_crossings, key=lambda x: x[0])
    print(f"Earliest Y-boundary crossing: t = {earliest_y_crossing[0]:.6f}, y = {earliest_y_crossing[2]:.1f}")

print()

# Create plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# X vs T plot
ax1.plot(t, x_pos, 'b-', linewidth=2, label=f'Ball {ball_index} x-position')
ax1.axhline(y=28.0, color='gray', linestyle='--', alpha=0.7, label='Cell boundaries')
ax1.axhline(y=29.0, color='gray', linestyle='--', alpha=0.7)
ax1.axhline(y=30.0, color='gray', linestyle='--', alpha=0.7)

if velocity[0] != 0:
    ax1.axvline(x=t_x_crossing, color='red', linestyle=':', alpha=0.8, label=f'X-crossing time t={t_x_crossing:.3f}')
    ax1.plot(t_x_crossing, x_crossing, 'ro', markersize=8, label=f'X-crossing point')

ax1.set_xlabel('Time (t)')
ax1.set_ylabel('X Position')
ax1.set_title(f'Ball {ball_index}: X Position vs Time')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Y vs T plot  
ax2.plot(t, y_pos, 'g-', linewidth=2, label=f'Ball {ball_index} y-position')
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Cell boundaries')
ax2.axhline(y=2.0, color='gray', linestyle='--', alpha=0.7)
ax2.axhline(y=3.0, color='gray', linestyle='--', alpha=0.7)

# Plot all y=2 crossings
if y2_crossings:
    colors = ['red', 'orange']
    for i, (t_y, x_y, y_boundary) in enumerate(y2_crossings):
        color = colors[i % len(colors)]
        label = f'Y=2 crossing {i+1}: t={t_y:.3f}'
        ax2.axvline(x=t_y, color=color, linestyle=':', alpha=0.8, label=label)
        ax2.plot(t_y, y_boundary, 'o', color=color, markersize=8)

ax2.set_xlabel('Time (t)')
ax2.set_ylabel('Y Position')
ax2.set_title(f'Ball {ball_index}: Y Position vs Time (with gravity)')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('ball_77_trajectory.png', dpi=150, bbox_inches='tight')
print("Plot saved as ball_77_trajectory.png")

# Print some analysis
print("Analysis:")
print(f"Ball starts in cell ({old_cell[0]}, {old_cell[1]}) at position ({initial_position[0]:.3f}, {initial_position[1]:.3f})")
print(f"Ball moves with velocity ({velocity[0]:.3f}, {velocity[1]:.3f})")
print(f"X-motion is linear: x(t) = {initial_position[0]:.3f} + {velocity[0]:.3f} * (t - {initial_time:.3f})")
print(f"Y-motion has gravity: y(t) = {initial_position[1]:.3f} + {velocity[1]:.3f} * (t - {initial_time:.3f}) - 0.5 * (t - {initial_time:.3f})²")
print(f"Ball crosses x=29 boundary to enter cell ({new_cell[0]}, {new_cell[1]})")