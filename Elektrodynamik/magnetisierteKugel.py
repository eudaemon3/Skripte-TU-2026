import numpy as np
import matplotlib.pyplot as plt

# parameter
mu_r = 5
B0 = 1.0
R = 1.0
N_grid = 3
mu_0 = 4 * np.pi * 1e-7

# grid points
x = np.linspace(-N_grid*R, N_grid*R, 200)
z = np.linspace(-N_grid*R, N_grid*R, 200)
X, Z = np.meshgrid(x, z)

r = np.sqrt(X**2 + Z**2)
r_safe = np.where(r == 0, 1, r)

M = 3 * B0 / mu_0 * (mu_r - 1) / (mu_r + 2)

# calculate cartesian components
Bx_out = mu_0 * M * R**3 * X * Z / r_safe**5

Bz_out = B0 + mu_0 * M * R**3 * (
    Z**2 / r_safe**5 - 1 / (3 * r_safe**3)
)

B_in = 3 * mu_r / (mu_r + 2) * B0

# create B asd H fields
innen = r <= R

B_x = np.where(innen, 0, Bx_out)
B_z = np.where(innen, B_in, Bz_out)

H_x = np.where(innen, 0, Bx_out / mu_0)
H_z = np.where(innen, B_in / (mu_0 * mu_r), Bz_out / mu_0)

# plot
plt.figure(figsize=(7, 7))
plt.streamplot(X, Z, B_x, B_z, density=1.2)

circ = plt.Circle((0, 0), R, fill=False, color="black", linewidth=2)
plt.gca().add_patch(circ)

plt.xlabel("x")
plt.ylabel("z")
plt.xlim(-N_grid*R, N_grid*R)
plt.ylim(-N_grid*R, N_grid*R)
plt.grid()

plt.show()
