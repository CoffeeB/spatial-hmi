"""
Root entry point for Gestura / Spatial HMI server.
Forwards to scripts/run_hmi_server.py.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.run_hmi_server import main

if __name__ == "__main__":
    main()
