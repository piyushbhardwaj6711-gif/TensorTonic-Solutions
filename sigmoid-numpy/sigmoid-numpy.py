import numpy as np

def sigmoid(x):
    arr = np.asarray(x, dtype=float)

    # Numerically stable: only ever exponentiate non-positive values
    out = np.where(
        arr >= 0,
        1.0 / (1.0 + np.exp(-np.abs(arr))),
        np.exp(-np.abs(arr)) / (1.0 + np.exp(-np.abs(arr))),
    )

    # Scalar in -> Python float out
    if arr.ndim == 0:
        return float(out)
    return out