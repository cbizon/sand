from src.simulation import run_simulation

params = {
    'ndim': 2,
    'num_balls': 100,
    'ball_radius': 0.45,
    'domain_size': (50.0, 20.0),
    'simulation_time': 10.0,
    'gravity': True,
    'output_rate': 0.1,
    'run_name': 'example_gravity_sim'
}

run_simulation(params)

# To visualize the results, run the Flask app:
# Option 1: uv run python run_app.py
# Option 2: cd src && uv run python app.py
# Then visit http://localhost:5055 in your browser
# Use the run selector in the web interface to choose between different simulation runs
