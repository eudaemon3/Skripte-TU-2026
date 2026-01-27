import numpy as np

K1 = 9.38e-22
K2 = 2.41
K3 = 66.29

x_CO = lambda K : -K/2 + np.sqrt(K**2 / 4 + K)
print(f'{1/ x_CO(K1) - 1:.2e}')
print(1/x_CO(K2) - 1)
print(1/ x_CO(K3) - 1)