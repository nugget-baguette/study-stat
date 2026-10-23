import numpy as np

def calculate_acf(y, max_lag):
    if len(y) - max_lag <= 0:
        raise ValueError("Max lag must be shorter than original y length")
    y = np.asarray(y)
    N = len(y)

    yc = y - y.mean()
    gamma_0 = np.sum(yc ** 2) / N

    acf_values = np.zeros(max_lag + 1)
    acf_values[0] = 1.

    for k in range(1, max_lag + 1):
        gamma_k = np.sum(yc[:-k] * yc[k:]) / N
        acf_values[k] = gamma_k / gamma_0

    return acf_values