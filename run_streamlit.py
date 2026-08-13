#!/usr/bin/env python3
"""
Streamlit Agriculture App Launcher
Simpler alternative to Flask with no template path issues
"""

import subprocess
import sys
import os

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Streamlit command
cmd = [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    os.path.join(PROJECT_ROOT, "app_streamlit.py"),
    "--logger.level=debug"
]

print(f"""
🌾 Starting Telangana Agriculture AI (Streamlit Version)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📂 Project root: {PROJECT_ROOT}
🌐 Opening browser at: http://localhost:8501
⏹️  Press Ctrl+C to stop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# Run streamlit
try:
    subprocess.run(cmd, cwd=PROJECT_ROOT)
except KeyboardInterrupt:
    print("\n✅ Application stopped")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nInstall Streamlit with: pip install streamlit folium streamlit-folium")
