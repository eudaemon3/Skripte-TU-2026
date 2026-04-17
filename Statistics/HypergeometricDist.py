import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt

def hypergeometric(N,K,N_a, K_a):
    return comb(N_a, K_a) * comb(N-N_a, K-K_a) / comb(N,K)

N = 100
K = 40
N_a = 30

v = np.arange(N_a + 1)
h = hypergeometric(N, K, N_a, v)

print(hypergeometric(30,10,8,4))
print(comb(8,4)/3**4 *(2/3)**4)
print(hypergeometric(30,10,4,4))
print(1/3**4)
print(1/3 * 1/3 * 2/3)
print(20*19*10 / (30*29*28))