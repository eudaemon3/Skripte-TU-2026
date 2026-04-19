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
s_0 = a/10       # Breite der Gauss-Kurve
N = 100          # Anzahl berechneter Basisvektoren

rx = 2000        # x Achse sampling
tx = 1000        # Zeit Achste sampling

kn = lambda n: np.pi/a * n
k_0 = kn(5)

tau = a / k_0
tmax = 10*tau

psi, x, t = InfSquareWell(N, a, tmax, x_0, s_0, k_0).get_psi(rx, tx)

fig, ax = plt.subplots(figsize=(6, 4))
line, = ax.plot([], [], 'b-', lw=2)
ax.set_xlim(-a/2, a/2)
ax.set_ylim(0, np.max(np.abs(psi)**2) * 1.1)
ax.set_xlabel("x")
ax.set_ylabel(r"$|\psi(x,t)|^2$")
ax.grid(True, 'minor', linestyle='--', alpha=0.5)
ax.grid(True, 'major', linestyle='-', alpha=0.9)
ax.set_title(r'$x_0 = 0, \quad \sigma_0 = a/10,\quad k_0 = 5\pi/a$')

def update(i):
    line.set_data(x, np.abs(psi[:, i])**2)
    return line

fig.tight_layout()
ani = FuncAnimation(fig, update, frames=range(0,len(t),1), interval=20, blit=False)
ani.save('anim/anim02.gif', writer='pillow', fps=30, dpi=80)
plt.show()