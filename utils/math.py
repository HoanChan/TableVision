import numpy as np

def min_with_default(arr, default = 0):
    try: return np.min(arr)
    except: return default

def max_with_default(arr, default = 0):
    try: return np.max(arr)
    except: return default