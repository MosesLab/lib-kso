import numpy as np

import matplotlib.pyplot as plt

def fourier_series(C, k, phi, M=1024):

    C = np.expand_dims(C,-1)
    k = np.expand_dims(k, -1)
    phi = np.expand_dims(phi, -1)

    x = np.arange(0,1,1/M,dtype=np.float32)
    x = np.expand_dims(x, 0)

    f = C * np.sin(k * x + phi)

    return np.sum(f,axis=0)

def rand_fourier_series(N=20, k_max=1000):

    k = k_max * np.random.random([N])
    C = np.random.random([N]) / k
    phi = 2 * np.pi * np.random.random([N])

    return fourier_series(C,k,phi)

def main():

    I = rand_fourier_series()

    plt.plot(I)
    plt.show()



if __name__ == "__main__":
    main()