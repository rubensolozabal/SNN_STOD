import numpy as np


# Restore deprecated NumPy scalar aliases expected by older dependencies.
if not hasattr(np, "int"):
    np.int = int
