from pathlib import Path

__version__ = "0.1.1"

filename = Path(__file__).parent / "VERSION"

with open(filename) as version_file:
    version = version_file.read().strip()
    __version__ = version
