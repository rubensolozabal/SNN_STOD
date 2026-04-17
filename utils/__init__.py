"""Useful utils
"""
import datetime
import os
import sys

from .misc import *
try:
    from .logger import *
except ModuleNotFoundError:
    pass
# from .visualize import *
from .eval import *

# progress bar
sys.path.append(os.path.join(os.path.dirname(__file__), "progress"))
try:
    from progress.bar import Bar as Bar
except ModuleNotFoundError:
    class Bar:  # pragma: no cover - lightweight fallback when progress isn't installed.
        def __init__(self, message, max=0):
            self.message = message
            self.max = max
            self.index = 0
            self.suffix = ""
            self.elapsed_td = datetime.timedelta(0)
            self.eta_td = datetime.timedelta(0)

        def next(self):
            self.index += 1

        def finish(self):
            pass
