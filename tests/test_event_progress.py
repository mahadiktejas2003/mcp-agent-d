"""Test event progress conversion from log events."""
import json
import subprocess
from pathlib import Path

def test_event_conversion():
    """Test conversion of log events to progress events using gold master approach."""
    # Get the paths
    log_file = str(Path(__file__).parent / "fixture" / "mcp_basic_agent_20250130_215534.jsonl")
    expected_output_file = Path(__file__).parent / "fixture" / "expected_output.txt"
    
    # Run the event_summary script to get current output
    result = subprocess.run(
        ["python3", "scripts/event_summary.py", log_file],
        capture_output=True,
        text=True
    )
    current_output = result.stdout
    
    # Load expected output
    with open(expected_output_file) as f:
        expected_output = f.read()
    
    # Compare outputs
    assert current_output.strip() == expected_output.strip(), \
        "Event summary output does not match expected output"

if __name__ == "__main__":
    test_event_conversion()