#------------------------------------------------------------------
# author: eudaemon
# 
# Description: 
#   Simulation of the time evolution of a gauss packet in an 
#   infinite square well potential using Schrödinger equation.
#
# Links: https://en.wikipedia.org/wiki/Quantum_revival
# -----------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import scienceplots
plt.style.use('science')
plt.rcParams.update({'font.size': 14})

from InfiniteSquareWell import InfSquareWell

a = 1            # Breite des Topfs
x_0 = 0          # Startposition
s_0 = a*0.05     # Breite der Gauss-Kurve
N = 500          # Anzahl berechneter Basisvektoren

rx = 1001        # x Achse sampling
tx = 801        # Zeit Achste sampling

tau = 2 * s_0**2
k_0 = 5 * a / tau

inf_sqare_well = InfSquareWell(N, a, 2*tau, x_0, s_0, k_0)
psi, x, t = inf_sqare_well.get_psi(rx, tx)
mu_ind = inf_sqare_well.get_mu(psi)

fig, ax = plt.subplots(figsize=(6, 4))
line, = ax.plot([], [], 'b-', lw=1)
# exp_value, = ax.plot([], [], 'k--', lw=1)

ax.set_xlim(-a/2, a/2)
max_psi = np.max(np.abs(psi[:, 0])**2) 
max_E = inf_sqare_well._E_n(4) 
scale = max_psi * 1.5 / max_E  

ax.set_ylim(0, max_psi * 1.1)
ax.set_xlabel("x", fontsize=16)
ax.set_ylabel(r"$|\psi(x,t)|^2$", fontsize=16)
ax.grid(True, 'minor', linestyle='--', alpha=0.5)
ax.grid(True, 'major', linestyle='-', alpha=0.9)
ax.set_title('Gauss packet in infinite square well', fontsize=18)

for n in range(1, 4):  # n from 1 to 3
    E_n = inf_sqare_well._E_n(n)
    phi_n = inf_sqare_well._phi_n(n, x) / 2

    ax.plot(x, E_n * scale * np.ones_like(x), 'r-', alpha=0.5, linewidth=0.8)
    ax.plot(x, phi_n + E_n * scale, 'k-', alpha=0.5, linewidth=0.8)

def update(i):
    # exp_value.set_data((x[mu_ind[i]], x[mu_ind[i]]), (0, np.abs(psi[mu_ind[i], i])**2) )
    line.set_data(x, np.abs(psi[:, i])**2)
    return line

fig.tight_layout()

ani = FuncAnimation(fig, update, frames=range(0,len(t),1), interval=20, blit=False)
ani.save('anim/inf_sqare_well.gif', writer='pillow', fps=30, dpi=80)
plt.show()