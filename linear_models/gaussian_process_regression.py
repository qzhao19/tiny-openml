# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 23:32:51 2020

@author: qizhao
"""

import numpy as np
from scipy import linalg as sp_ln

def zero_mean(X):
    """Return a zero-mean function
    """
    return np.zero((X.shape[1], 1), dtype=X.dtype)


