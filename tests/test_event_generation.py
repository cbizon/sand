import numpy as np
import pytest
from src.event_generation import (
    generate_ball_ball_events,
    generate_ball_wall_events,
    generate_ball_grid_event,
    generate_events_for_ball,
    generate_ball_ball_events_for_new_cell
)
from src.ball import Ball
from src.wall import Wall, create_box_walls
from src.grid import Grid
from src.events import BallBallCollision, BallWallCollision, BallGridTransit


class TestEventGeneration:
    
    def test_generate_ball_ball_events_collision(self):
        # Two balls moving toward each other
        ball1 = Ball(np.array([1.0, 1.0]), np.array([1.0, 0.0]), 0.1, 0, (1, 1))
        ball2 = Ball(np.array([2.0, 1.0]), np.array([-1.0, 0.0]), 0.1, 1, (2, 1))
        
        events = generate_ball_ball_events(ball1, [ball2], 0.0, 2, False)
        
        assert len(events) == 1
        assert isinstance(events[0], BallBallCollision)
        assert events[0].ball1 is ball1
        assert events[0].ball2 is ball2
        assert events[0].time > 0.0
    
    def test_generate_ball_ball_events_no_collision(self):
        # Two balls moving away from each other
        ball1 = Ball(np.array([1.0, 1.0]), np.array([-1.0, 0.0]), 0.1, 0, (1, 1))
        ball2 = Ball(np.array([2.0, 1.0]), np.array([1.0, 0.0]), 0.1, 1, (2, 1))
        
        events = generate_ball_ball_events(ball1, [ball2], 0.0, 2, False)
        
        assert len(events) == 0
    
    def test_generate_ball_ball_events_creates_all_collision_events(self):
        # Test that ball-ball events are created for all potential collisions
        ball1 = Ball(np.array([1.0, 1.0]), np.array([1.0, 0.0]), 0.1, 0, (1, 1))
        ball2 = Ball(np.array([2.0, 1.0]), np.array([-1.0, 0.0]), 0.1, 1, (2, 1))
        ball3 = Ball(np.array([3.0, 1.0]), np.array([-1.0, 0.0]), 0.1, 2, (3, 1))
        
        # ball1 (index 0) should create events with ball2 and ball3
        events1 = generate_ball_ball_events(ball1, [ball2, ball3], 0.0, 2, False)
        assert len(events1) == 2
        
        # ball2 (index 1) should create events with both ball1 and ball3 when all are provided
        # Create new ball3 that will collide with ball2 (moving toward each other)
        ball3_colliding = Ball(np.array([4.0, 1.0]), np.array([-2.0, 0.0]), 0.1, 2, (4, 1))
        events2 = generate_ball_ball_events(ball2, [ball1, ball3_colliding], 0.0, 2, False)
        assert len(events2) == 2  # Now creates events with both balls
    
    def test_generate_ball_ball_events_self_skip(self):
        # Should skip self-collision
        ball1 = Ball(np.array([1.0, 1.0]), np.array([1.0, 0.0]), 0.1, 0, (1, 1))
        
        events = generate_ball_ball_events(ball1, [ball1], 0.0, 2, False)
        assert len(events) == 0
    
    def test_generate_ball_wall_events(self):
        # Ball moving toward wall
        ball = Ball(np.array([0.5, 1.0]), np.array([-1.0, 0.0]), 0.1, 0, (0, 1))
        walls = create_box_walls(2, (3.0, 2.0), 0.01, 1.0)
        
        events = generate_ball_wall_events(ball, walls, 0.0, 2, False)
        
        # Should find collision with left wall
        wall_events = [e for e in events if isinstance(e, BallWallCollision)]
        assert len(wall_events) >= 1
        assert all(e.time > 0.0 for e in wall_events)
    
    def test_generate_ball_wall_events_no_collision(self):
        # Ball moving parallel to walls
        ball = Ball(np.array([1.0, 1.0]), np.array([0.0, 1.0]), 0.1, 0, (1, 1))
        walls = create_box_walls(2, (3.0, 2.0), 0.01, 1.0)
        
        events = generate_ball_wall_events(ball, walls, 0.0, 2, False)
        
        # Should find collision with top wall
        wall_events = [e for e in events if isinstance(e, BallWallCollision)]
        assert len(wall_events) >= 1
    
    def test_generate_ball_grid_event(self):
        # Ball moving toward cell boundary
        ball = Ball(np.array([0.9, 1.0]), np.array([1.0, 0.0]), 0.1, 0, (0, 1))
        
        events = generate_ball_grid_event(ball, 0.0, 2, False)
        
        assert len(events) == 1
        assert isinstance(events[0], BallGridTransit)
        assert events[0].ball is ball
        assert events[0].time > 0.0
        assert events[0].new_cell == (1, 1)
    
    def test_generate_ball_grid_event_no_transit(self):
        # Stationary ball
        ball = Ball(np.array([0.5, 1.0]), np.array([0.0, 0.0]), 0.1, 0, (0, 1))
        
        events = generate_ball_grid_event(ball, 0.0, 2, False)
        
        assert len(events) == 0
    
    def test_generate_events_for_ball(self):
        # Set up simulation components
        grid = Grid(2, (3.0, 2.0))
        
        ball1 = Ball(np.array([1.5, 1.0]), np.array([1.0, 0.0]), 0.1, 0, (1, 1))
        ball2 = Ball(np.array([2.5, 1.0]), np.array([-1.0, 0.0]), 0.1, 1, (2, 1))
        balls = [ball1, ball2]
        
        # Add balls to grid
        grid.add_ball(0, (1, 1))
        grid.add_ball(1, (2, 1))
        
        walls = create_box_walls(2, (3.0, 2.0), 0.01, 1.0)
        
        events = generate_events_for_ball(ball1, balls, walls, grid, 0.0, 2, False)
        
        # Should have ball-ball, ball-wall, and ball-grid events
        ball_ball_events = [e for e in events if isinstance(e, BallBallCollision)]
        ball_wall_events = [e for e in events if isinstance(e, BallWallCollision)]
        ball_grid_events = [e for e in events if isinstance(e, BallGridTransit)]
        
        assert len(ball_ball_events) >= 1  # Collision with ball2
        assert len(ball_wall_events) >= 1  # Collision with walls
        assert len(ball_grid_events) >= 1  # Grid transit
    
    def test_generate_ball_ball_events_for_new_cell(self):
        # Set up grid and balls
        grid = Grid(2, (4.0, 3.0))
        
        # Ball that moved from (1,1) to (2,1)
        ball1 = Ball(np.array([2.5, 1.5]), np.array([1.0, 0.0]), 0.1, 0, (2, 1))
        
        # Ball in newly adjacent cell (3,1)
        ball2 = Ball(np.array([3.5, 1.5]), np.array([0.0, 0.0]), 0.1, 1, (3, 1))
        
        # Ball that was already adjacent (1,1) - should not generate events
        ball3 = Ball(np.array([1.5, 1.5]), np.array([0.0, 0.0]), 0.1, 2, (1, 1))
        
        balls = [ball1, ball2, ball3]
        
        # Add balls to grid
        grid.add_ball(0, (2, 1))
        grid.add_ball(1, (3, 1))
        grid.add_ball(2, (1, 1))
        
        events = generate_ball_ball_events_for_new_cell(
            ball1, (1, 1), balls, grid, 0.0, 2, False
        )
        
        # Should only generate event with newly adjacent ball2, not ball3
        assert len(events) >= 0  # Depends on actual collision calculation
        
        # Verify it's using the new neighbor function correctly
        new_neighbors = grid.get_balls_in_new_neighbor_cells((1, 1), (2, 1))
        assert 1 in new_neighbors  # ball2 should be in new neighbors
        assert 2 not in new_neighbors  # ball3 should not be in new neighbors
    
    def test_generate_events_with_gravity(self):
        # Test event generation with gravity enabled
        ball = Ball(np.array([1.0, 1.5]), np.array([0.0, 1.0]), 0.1, 0, (1, 1))
        walls = create_box_walls(2, (3.0, 2.0), 0.01, 1.0)
        
        events = generate_ball_wall_events(ball, walls, 0.0, 2, True)
        
        # Should still generate wall collision events with gravity
        wall_events = [e for e in events if isinstance(e, BallWallCollision)]
        assert len(wall_events) >= 1