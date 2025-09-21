import os
import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, jsonify, send_file, request, redirect, url_for
import io
import base64
from typing import List, Tuple, Optional

app = Flask(__name__)

class FrameReader:
    """Reads and parses simulation frame files."""
    
    def __init__(self, base_dir: str = "../runs", run_name: str = None):
        self.base_dir = base_dir
        self.run_name = run_name
        self.frames_dir = os.path.join(base_dir, run_name) if run_name else base_dir
        self.frames = []
        self.current_frame = 0
        self.parameters = {}
        self.load_parameters()
        self.load_frames()
    
    def get_available_runs(self) -> List[str]:
        """Get list of available simulation runs."""
        if not os.path.exists(self.base_dir):
            return []
        
        runs = []
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path):
                # Check if it contains frame files
                has_frames = any(f.startswith('frame_') and f.endswith('.txt') 
                               for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f)))
                if has_frames:
                    runs.append(item)
        
        return sorted(runs)
    
    def load_run(self, run_name: str):
        """Load a specific simulation run."""
        self.run_name = run_name
        self.frames_dir = os.path.join(self.base_dir, run_name)
        self.frames = []
        self.current_frame = 0
        self.parameters = {}
        self.load_parameters()
        self.load_frames()
    
    def load_parameters(self):
        """Load simulation parameters from parameters.json file."""
        import json
        params_file = os.path.join(self.frames_dir, "parameters.json")
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r') as f:
                    self.parameters = json.load(f)
                print(f"Loaded parameters: {self.parameters}")
            except Exception as e:
                print(f"Warning: Could not load parameters from {params_file}: {e}")
                # Set defaults
                self.parameters = {
                    'domain_size': (5.0, 3.0),
                    'ball_radius': 0.45,
                    'ndim': 2
                }
        else:
            print(f"Warning: Parameters file {params_file} not found, using defaults")
            # Set defaults
            self.parameters = {
                'domain_size': (5.0, 3.0),
                'ball_radius': 0.45,
                'ndim': 2
            }
    
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
        
    def create_frame_image(self, frame_data: dict, domain_size: Tuple[float, ...] = (5.0, 3.0), highlight_balls: List[int] = None, show_grid: bool = False) -> str:
        """Create a visualization of a frame and return as base64 encoded image."""
        if not frame_data or not frame_data['positions']:
            return self._create_empty_image()
        
        positions = frame_data['positions']
        ndim = frame_data['ndim']
        time = frame_data['time']
        highlight_balls = highlight_balls or []
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if ndim == 2:
            # Plot 2D simulation
            positions = frame_data['positions']
            velocities = frame_data['velocities']
            
            # Plot balls as circles and velocity vectors
            for i, (pos, vel) in enumerate(zip(positions, velocities)):
                x, y = pos[0], pos[1]
                vx, vy = vel[0], vel[1]
                
                # Choose color based on whether ball is highlighted
                ball_color = 'orange' if i in highlight_balls else 'blue'
                
                # Draw ball
                circle = plt.Circle((x, y), self.ball_radius, color=ball_color, alpha=0.7)
                ax.add_patch(circle)
                
                # Draw velocity vector
                # Scale velocity for visibility (adjust scale factor as needed)
                scale_factor = 0.2
                ax.arrow(x, y, vx * scale_factor, vy * scale_factor, 
                        head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.8)
            
            # Set domain bounds
            ax.set_xlim(0, domain_size[0])
            ax.set_ylim(0, domain_size[1] if len(domain_size) > 1 else domain_size[0])
            
            # Draw grid if requested
            if show_grid:
                grid_color = 'lightgray'
                grid_width = 0.5
                grid_alpha = 0.7
                
                # Draw vertical grid lines at integer boundaries
                for x in range(int(domain_size[0]) + 1):
                    ax.axvline(x=x, color=grid_color, linewidth=grid_width, alpha=grid_alpha)
                
                # Draw horizontal grid lines at integer boundaries  
                for y in range(int(domain_size[1]) + 1):
                    ax.axhline(y=y, color=grid_color, linewidth=grid_width, alpha=grid_alpha)
            
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
            for i, (pos, vel) in enumerate(zip(positions, velocities)):
                x, y = pos[0], pos[1]  # Project to x-y plane
                vx, vy = vel[0], vel[1]  # Project velocity to x-y plane
                
                # Choose color based on whether ball is highlighted
                ball_color = 'orange' if i in highlight_balls else 'blue'
                
                # Draw ball
                circle = plt.Circle((x, y), self.ball_radius, color=ball_color, alpha=0.7)
                ax.add_patch(circle)
                
                # Draw velocity vector (x-y projection)
                # Scale velocity for visibility
                scale_factor = 0.2
                ax.arrow(x, y, vx * scale_factor, vy * scale_factor, 
                        head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.8)
            
            # Set domain bounds
            ax.set_xlim(0, domain_size[0])
            ax.set_ylim(0, domain_size[1] if len(domain_size) > 1 else domain_size[0])
            
            # Draw grid if requested (3D case - x-y projection)
            if show_grid:
                grid_color = 'lightgray'
                grid_width = 0.5
                grid_alpha = 0.7
                
                # Draw vertical grid lines at integer boundaries
                for x in range(int(domain_size[0]) + 1):
                    ax.axvline(x=x, color=grid_color, linewidth=grid_width, alpha=grid_alpha)
                
                # Draw horizontal grid lines at integer boundaries  
                for y in range(int(domain_size[1]) + 1):
                    ax.axhline(y=y, color=grid_color, linewidth=grid_width, alpha=grid_alpha)
        
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

@app.route('/api/runs')
def get_runs():
    """API endpoint to get available simulation runs."""
    runs = frame_reader.get_available_runs()
    return jsonify({
        'runs': runs,
        'current_run': frame_reader.run_name
    })

@app.route('/api/load_run/<run_name>')
def load_run(run_name):
    """API endpoint to load a specific simulation run."""
    global current_frame_index
    
    available_runs = frame_reader.get_available_runs()
    if run_name not in available_runs:
        return jsonify({'error': f'Run "{run_name}" not found'}), 404
    
    try:
        frame_reader.load_run(run_name)
        current_frame_index = 0
        return jsonify({
            'success': True,
            'run_name': run_name,
            'total_frames': frame_reader.get_num_frames()
        })
    except Exception as e:
        return jsonify({'error': f'Failed to load run: {str(e)}'}), 500

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
    
    # Get highlighted balls from query parameter
    highlight_balls = []
    highlight_param = request.args.get('highlight', '')
    if highlight_param:
        try:
            highlight_balls = [int(x.strip()) for x in highlight_param.split(',') if x.strip().isdigit()]
        except ValueError:
            pass  # Ignore invalid ball indices
    
    # Get grid display option from query parameter
    show_grid = request.args.get('grid', '').lower() in ['true', '1', 'on']
    
    # Create visualization using parameters from frame reader
    domain_size = tuple(frame_reader.parameters.get('domain_size', (5.0, 3.0)))
    ball_radius = frame_reader.parameters.get('ball_radius', 0.45)
    
    # Update visualizer with correct ball radius
    visualizer.ball_radius = ball_radius
    image_base64 = visualizer.create_frame_image(frame_data, domain_size, highlight_balls, show_grid)
    
    return jsonify({
        'frame_index': frame_index,
        'total_frames': frame_reader.get_num_frames(),
        'time': frame_data['time'],
        'num_balls': frame_data['num_balls'],
        'image': image_base64,
        'highlighted_balls': highlight_balls,
        'show_grid': show_grid
    })

@app.route('/api/next')
def next_frame():
    """Go to next frame."""
    global current_frame_index
    next_index = min(current_frame_index + 1, frame_reader.get_num_frames() - 1)
    # Preserve parameters if present
    params = {}
    if request.args.get('highlight'):
        params['highlight'] = request.args.get('highlight')
    if request.args.get('grid'):
        params['grid'] = request.args.get('grid')
    
    return redirect(url_for('get_frame', frame_index=next_index, **params))

@app.route('/api/prev')
def prev_frame():
    """Go to previous frame."""
    global current_frame_index
    prev_index = max(current_frame_index - 1, 0)
    # Preserve parameters if present
    params = {}
    if request.args.get('highlight'):
        params['highlight'] = request.args.get('highlight')
    if request.args.get('grid'):
        params['grid'] = request.args.get('grid')
    
    return redirect(url_for('get_frame', frame_index=prev_index, **params))

@app.route('/api/info')
def get_info():
    """Get simulation info."""
    return jsonify({
        'total_frames': frame_reader.get_num_frames(),
        'current_frame': current_frame_index
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5055)
