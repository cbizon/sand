#!/usr/bin/env python3
"""
Script to extract and analyze JSON log entries related to specific balls.
"""

import sys
import json
from typing import Set, List, Dict, Any


def extract_ball_logs_json(log_content: str, ball_ids: Set[int]) -> List[Dict[str, Any]]:
    """
    Extract all JSON log entries that mention any of the specified ball IDs.
    
    Args:
        log_content: Full simulation log content (JSONL format)
        ball_ids: Set of ball IDs to search for
        
    Returns:
        List of relevant log entries as dictionaries
    """
    relevant_entries = []
    lines = log_content.strip().split('\n')
    
    for line in lines:
        if not line.strip():
            continue
            
        try:
            entry = json.loads(line)
            
            # Check if this entry involves any of our target balls
            involves_target_ball = False
            
            # Check different event types and their ball references
            if entry.get("event_type") == "BallBallCollision":
                ball1_idx = entry.get("ball1", {}).get("index")
                ball2_idx = entry.get("ball2", {}).get("index")
                if ball1_idx in ball_ids or ball2_idx in ball_ids:
                    involves_target_ball = True
                    
            elif entry.get("event_type") == "BallWallCollision":
                ball_idx = entry.get("ball", {}).get("index")
                if ball_idx in ball_ids:
                    involves_target_ball = True
                    
            elif entry.get("event_type") == "BallGridTransit":
                ball_idx = entry.get("ball", {}).get("index")
                if ball_idx in ball_ids:
                    involves_target_ball = True
                    
            elif entry.get("event_type") == "EventDiscarded":
                # Check ball1, ball2, or ball fields
                ball1 = entry.get("ball1")
                ball2 = entry.get("ball2") 
                ball = entry.get("ball")
                if (ball1 in ball_ids or ball2 in ball_ids or ball in ball_ids):
                    involves_target_ball = True
                    
            elif entry.get("event_type") == "SimulationStart":
                # Check if any of our target balls appear in initial_balls
                initial_balls = entry.get("initial_balls", [])
                relevant_balls = []
                for ball_data in initial_balls:
                    if ball_data.get("ball") in ball_ids:
                        involves_target_ball = True
                        relevant_balls.append(ball_data)
                
                # If we found relevant balls, create a filtered version of the entry
                if involves_target_ball:
                    filtered_entry = entry.copy()
                    filtered_entry["initial_balls"] = relevant_balls
                    relevant_entries.append(filtered_entry)
                    continue  # Skip the normal append at the end
            
            if involves_target_ball:
                relevant_entries.append(entry)
                
        except json.JSONDecodeError:
            # Skip non-JSON lines
            continue
    
    return relevant_entries


def analyze_ball_interactions(entries: List[Dict[str, Any]], ball_ids: Set[int]) -> Dict[str, Any]:
    """
    Analyze the interactions between the specified balls.
    
    Args:
        entries: List of log entries
        ball_ids: Set of ball IDs to analyze
        
    Returns:
        Analysis summary
    """
    analysis = {
        "total_entries": len(entries),
        "event_types": {},
        "ball_ball_collisions": [],
        "direct_interactions": 0,
        "timeline": []
    }
    
    for entry in entries:
        event_type = entry.get("event_type", "unknown")
        analysis["event_types"][event_type] = analysis["event_types"].get(event_type, 0) + 1
        
        # Track timeline
        time = entry.get("time")
        if time is not None:
            analysis["timeline"].append({
                "time": time,
                "event": event_type,
                "entry": entry
            })
        
        # Look for direct ball-ball interactions between our target balls
        if event_type == "BallBallCollision":
            ball1_idx = entry.get("ball1", {}).get("index")
            ball2_idx = entry.get("ball2", {}).get("index") 
            if ball1_idx in ball_ids and ball2_idx in ball_ids:
                analysis["direct_interactions"] += 1
                analysis["ball_ball_collisions"].append(entry)
    
    # Sort timeline by time
    analysis["timeline"].sort(key=lambda x: x.get("time", 0))
    
    return analysis


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_ball_logs_json.py <ball_id1> <ball_id2> [log_file]")
        print("Example: python extract_ball_logs_json.py 29 47")
        print("If no log_file specified, reads from stdin")
        sys.exit(1)
    
    try:
        ball_id1 = int(sys.argv[1])
        ball_id2 = int(sys.argv[2])
        ball_ids = {ball_id1, ball_id2}
    except ValueError:
        print("Error: Ball IDs must be integers")
        sys.exit(1)
    
    # Read log content
    if len(sys.argv) > 3:
        log_file = sys.argv[3]
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
        except FileNotFoundError:
            print(f"Error: Log file '{log_file}' not found")
            sys.exit(1)
    else:
        # Read from stdin
        log_content = sys.stdin.read()
    
    # Extract relevant entries
    relevant_entries = extract_ball_logs_json(log_content, ball_ids)
    
    # Analyze interactions
    analysis = analyze_ball_interactions(relevant_entries, ball_ids)
    
    # Print analysis summary
    print(f"=== Analysis for balls {ball_id1} and {ball_id2} ===")
    print(f"Total relevant log entries: {analysis['total_entries']}")
    print(f"Direct ball-ball interactions: {analysis['direct_interactions']}")
    print(f"Event type breakdown: {analysis['event_types']}")
    print()
    
    if analysis['ball_ball_collisions']:
        print("Direct ball-ball collision events:")
        for collision in analysis['ball_ball_collisions']:
            print(f"  Time {collision.get('time', 'unknown')}:")
            print(json.dumps(collision, indent=4))
        print()
    else:
        print("No direct ball-ball collisions found between these balls.")
        print()
    
    # Show all events with proper indentation
    print("All relevant events:")
    for i, event in enumerate(analysis['timeline']):
        print(f"{i+1}. t={event['time']:.6f}: {event['event']}")
        print(json.dumps(event['entry'], indent=2))
        print()


if __name__ == "__main__":
    main()