from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("microcorrelate")
except PackageNotFoundError:
    __version__ = "unknown"
