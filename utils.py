# Code taken from https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks

import numpy as np
import tensorflow as tf

# helper methods to print nice table (taken from CGT code)
def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim==0
        x = x.item()
    if isinstance(x, float): rep = "%g"%x
    else: rep = str(x)
    return " "*(l - len(rep)) + rep

def fmt_row(width, row):
    out = " | ".join(fmt_item(x, width) for x in row)
    return out
