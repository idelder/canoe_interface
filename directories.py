import os
import sys
from platformdirs import PlatformDirs

def is_bundled() -> bool:
    """Check if we're running as a bundled executable"""
    return getattr(sys, 'frozen', False)

def get_resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if is_bundled():
        # Running as compiled executable - get path from _MEIPASS
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    else:
        # Running as script - use current directory
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, relative_path)

LOG_DIR: str
"""File location of logs"""
ASSETS_DIR: str
"""File location of assets like images"""
SCHEMA_FILE: str
"""File location of SQL schema"""
CONFIG_FILE: str
"""File location for saved configuration of UI options"""

# Set up directories based on whether we're running as exe or not
if is_bundled():
    # Running as compiled executable - use platform-specific dirs
    dirs = PlatformDirs("RSSSelector", "CANOE")
    INPUT_DIR = dirs.user_config_path
    LOG_DIR = dirs.user_log_path
    ASSETS_DIR = get_resource_path("assets")
else:
    # Running as Python script - use local dirs
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.join(BASE_DIR, "input")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# File paths
CONFIG_FILE = os.path.join(INPUT_DIR, "config.json")
SCHEMA_FILE = os.path.join(ASSETS_DIR, "schema.sql")