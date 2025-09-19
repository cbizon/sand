import os
import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, jsonify, send_file
import io
import base64
from typing import List, Tuple, Optional

app = Flask(__name__)

class FrameReader:
    """Reads and parses simulation frame files."""
    
    def __init__(self, frames_dir: str = "runs"):
        self.frames_dir = frames_dir
        self.frames = []
        self.current_frame = 0
        self.load_frames()
    
    def load_frames(self):
        """Load all frame files from directory."""
        if not os.path.exists(self.frames_dir):
            print(f"Warning: Frames directory '{self.frames_dir}' does not exist")
            return
        
        # Find all frame files
        frame_files = []
        for filename in os.listdir(self.frames_dir):
            if filename.startswith('frame_') and filename.endswith('.txt'):
                match = re.match(r'frame_(\d+)\.txt', filename)
                if match:
                    frame_num = int(match.group(1))
                    frame_files.append((frame_num, filename))
        
        # Sort by frame number
        frame_files.sort(key=lambda x: x[0])
        
        # Load each frame
        self.frames = []
        for frame_num, filename in frame_files:
            filepath = os.path.join(self.frames_dir, filename)
            frame_data = self.parse_frame_file(filepath)
            if frame_data:
                self.frames.append(frame_data)
        
        print(f"Loaded {len(self.frames)} frames")
    
    def parse_frame_file(self, filepath: str) -> Optional[dict]:
        """Parse a single frame file."""
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                return None
            
            # Parse header
            time_line = lines[0].strip()
            balls_line = lines[1].strip()
            
            time = float(time_line.split(': ')[1])
            num_balls = int(balls_line.split(': ')[1])
            
            # Parse ball data
            positions = []
            velocities = []
            
            for i in range(2, min(2 + num_balls, len(lines))):
                parts = lines[i].strip().split()
                if len(parts) >= 5:  # ball_id x y vx vy [z vz]
                    if len(parts) == 5:  # 2D
                        pos = np.array([float(parts[1]), float(parts[2])])
                        vel = np.array([float(parts[3]), float(parts[4])])
                    else:  # 3D
                        pos = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                        vel = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
                    
                    positions.append(pos)
                    velocities.append(vel)
            
            return {
                'time': time,
                'num_balls': num_balls,
                'positions': positions,
                'velocities': velocities,
                'ndim': len(positions[0]) if positions else 2
            }
        
        except Exception as e:
            print(f"Error parsing frame file {filepath}: {e}")
            return None
    
    def get_frame(self, index: int) -> Optional[dict]:
        """Get frame data by index."""
        if 0 <= index < len(self.frames):
            return self.frames[index]
        return None
    
    def get_num_frames(self) -> int:
        """Get total number of frames."""
        return len(self.frames)


class FrameVisualizer:
    """Creates visualizations of simulation frames."""
    
    def __init__(self, ball_radius: float = 0.45):
        self.ball_radius = ball_radius
        
    def create_frame_image(self, frame_data: dict, domain_size: Tuple[float, ...] = (5.0, 3.0)) -> str:
        """Create a visualization of a frame and return as base64 encoded image."""
        if not frame_data or not frame_data['positions']:
            return self._create_empty_image()
        
        positions = frame_data['positions']
        ndim = frame_data['ndim']
        time = frame_data['time']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if ndim == 2:
            # Plot 2D simulation
            positions = frame_data['positions']
            velocities = frame_data['velocities']
            
            # Plot balls as circles and velocity vectors
            for pos, vel in zip(positions, velocities):
                x, y = pos[0], pos[1]
                vx, vy = vel[0], vel[1]
                
                # Draw ball
                circle = plt.Circle((x, y), self.ball_radius, color='blue', alpha=0.7)
                ax.add_patch(circle)
                
                # Draw velocity vector
                # Scale velocity for visibility (adjust scale factor as needed)
                scale_factor = 0.2
                ax.arrow(x, y, vx * scale_factor, vy * scale_factor, 
                        head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.8)
            
            # Set domain bounds
            ax.set_xlim(0, domain_size[0])
            ax.set_ylim(0, domain_size[1] if len(domain_size) > 1 else domain_size[0])
            
            # Draw walls (inset by 0.01)
            wall_inset = 0.01
            wall_color = 'red'
            wall_width = 3
            
            # Bottom wall
            ax.axhline(y=wall_inset, color=wall_color, linewidth=wall_width)
            # Top wall
            ax.axhline(y=domain_size[1] - wall_inset, color=wall_color, linewidth=wall_width)
            # Left wall
            ax.axvline(x=wall_inset, color=wall_color, linewidth=wall_width)
            # Right wall
            ax.axvline(x=domain_size[0] - wall_inset, color=wall_color, linewidth=wall_width)
            
        else:
            # Plot 3D simulation (2D projection to x-y plane)
            positions = frame_data['positions']
            velocities = frame_data['velocities']
            
            # Plot balls as circles and velocity vectors (projected to x-y plane)
            for pos, vel in zip(positions, velocities):
                x, y = pos[0], pos[1]  # Project to x-y plane
                vx, vy = vel[0], vel[1]  # Project velocity to x-y plane
                
                # Draw ball
                circle = plt.Circle((x, y), self.ball_radius, color='blue', alpha=0.7)
                ax.add_patch(circle)
                
                # Draw velocity vector (x-y projection)
                # Scale velocity for visibility
                scale_factor = 0.2
                ax.arrow(x, y, vx * scale_factor, vy * scale_factor, 
                        head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.8)
            
            # Set domain bounds
            ax.set_xlim(0, domain_size[0])
            ax.set_ylim(0, domain_size[1] if len(domain_size) > 1 else domain_size[0])
        
        ax.set_aspect('equal')
        ax.set_title(f'Granular Media Simulation - Time: {time:.3f}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        # Remove grid lines
        
        # Convert to base64 image
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        return img_base64
    
    def _create_empty_image(self) -> str:
        """Create an empty placeholder image."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No frame data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        return img_base64


# Global instances
frame_reader = FrameReader()
visualizer = FrameVisualizer()
current_frame_index = 0

@app.route('/')
def index():
    """Main page with frame viewer."""
    return render_template('viewer.html')

@app.route('/api/frame/<int:frame_index>')
def get_frame(frame_index):
    """API endpoint to get frame data and image."""
    global current_frame_index
    
    if frame_index < 0 or frame_index >= frame_reader.get_num_frames():
        return jsonify({'error': 'Invalid frame index'}), 400
    
    current_frame_index = frame_index
    frame_data = frame_reader.get_frame(frame_index)
    
    if not frame_data:
        return jsonify({'error': 'Frame data not available'}), 404
    
    # Create visualization
    image_base64 = visualizer.create_frame_image(frame_data)
    
    return jsonify({
        'frame_index': frame_index,
        'total_frames': frame_reader.get_num_frames(),
        'time': frame_data['time'],
        'num_balls': frame_data['num_balls'],
        'image': image_base64
    })

@app.route('/api/next')
def next_frame():
    """Go to next frame."""
    global current_frame_index
    next_index = min(current_frame_index + 1, frame_reader.get_num_frames() - 1)
    return get_frame(next_index)

@app.route('/api/prev')
def prev_frame():
    """Go to previous frame."""
    global current_frame_index
    prev_index = max(current_frame_index - 1, 0)
    return get_frame(prev_index)

@app.route('/api/info')
def get_info():
    """Get simulation info."""
    return jsonify({
        'total_frames': frame_reader.get_num_frames(),
        'current_frame': current_frame_index
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5055)
