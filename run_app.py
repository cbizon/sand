#!/usr/bin/env python3
"""
Launcher script for the Flask visualization app.
This script changes to src directory and runs the Flask app with uv.
"""

import os
import subprocess
import sys

def main():
    # Change to src directory
    src_dir = os.path.join(os.path.dirname(__file__), 'src')
    os.chdir(src_dir)
    
    print("Starting Flask visualization app...")
    print("Visit http://localhost:5055 in your browser")
    
    # Run the Flask app using uv
    try:
        subprocess.run(['uv', 'run', 'python', 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\nFlask app stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error running Flask app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()