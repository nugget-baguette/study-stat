import numpy as np

def fft_scratch(x):
    """
    Cooley Tukey FFT
    x: input size must be 2 ** n
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    # terminate recursive function loop
    if N <= 1:
        return x

    # separate into even index set and odd index set
    # the number off list x[0::2] and x[1::2] are N/2
    even = fft_scratch(x[0::2])
    odd  = fft_scratch(x[1::2])
    
    # Twiddle factor
    # 0 to N/2 - 1, ammounted to N/2 factors
    # due to symmetricity of trigonometric function
    factor = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    
    # Butterfly operation: different calculation according to even or odd
    # k < N/2 -> even + factor * odd
    # k >= N/2 -> even - factor * odd
    return np.concatenate([even + factor * odd,
                           even - factor * odd])

if __name__ == '__main__':
    N = 2 ** 10
    x = np.random.random(N)

    X_scratch = fft_scratch(x)
    X_numpy = np.fft.fft(x)

    print(np.allclose(X_scratch, X_numpy))