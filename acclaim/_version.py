"""Package version, resolved from installed distribution metadata."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("acclaim")
except PackageNotFoundError:
    __version__ = "unknown"
