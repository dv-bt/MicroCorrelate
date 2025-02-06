"""
This module is a placeholder for the utils module of the microcorrelate package.
"""

from pathlib import Path
import re


def round_up_multiple(num: int, base: int) -> int:
    """Round up a number to a multiple of base"""
    return num + (base - num) % base


def extract_integers(
    base_path: Path, regex_pattern: str, glob_pattern: str, recursive: bool = False
) -> list[int]:
    """Extract integers matching the group in a regex pattern from filenames matching
    the glob pattern
    """
    try:
        matches = [
            int(re.search(regex_pattern, str(i)).group(1))
            for i in base_path.rglob(glob_pattern)
        ]
    except AttributeError:
        matches = []
    return matches
