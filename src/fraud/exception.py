"""Custom exception that records which file and line an error came from.

Usage:
    try:
        ...
    except Exception as e:
        raise FraudException(e) from e
"""
import os
import sys
import traceback

PACKAGE_DIR = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))


class FraudException(Exception):
    def __init__(self, error: Exception):
        frames = traceback.extract_tb(sys.exc_info()[2])
        # Point at the last line in OUR code (src/fraud), not deep inside pandas
        ours = [f for f in frames if os.path.normcase(f.filename).startswith(PACKAGE_DIR)]
        frame = (ours or frames)[-1] if frames else None
        where = f" in {os.path.basename(frame.filename)} line {frame.lineno}" if frame else ""
        super().__init__(f"{type(error).__name__}: {error}{where}")
