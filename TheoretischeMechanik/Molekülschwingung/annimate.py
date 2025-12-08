import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from molecule_lib import equations_of_motion, create_initial_conditions, solve_with_K, Modes
import molecule_lib as ml

# Define Parameters
#------------------------------------------------------------------

# 'breathing', 'rotation', 'mode5', 'mode6'
MODE = Modes.mode6
T_SPAN = (0, 20)
T_RES = 1000
USE_MATRIX_METHOD = True

#------------------------------------------------------------------

r_eq = ml.R_EQ
init_amp_rec = ml.ANIM_PARAMS[MODE]

positions_0, velocities_0 = create_initial_conditions(mode=MODE, amplitude=init_amp_rec)
y0 = np.concatenate([positions_0.flatten(), velocities_0.flatten()])

t_span = T_SPAN
t_eval = np.linspace(min(t_span), max(t_span), T_RES)

if USE_MATRIX_METHOD:
    scale_factor = 6
    positions_t = solve_with_K(y0=y0, t_eval=t_eval*scale_factor)
else:
    solution = solve_ivp(equations_of_motion, t_span, y0, 
        t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-11
    )
    positions_t = solution.y[:6].T.reshape(-1, 3, 2)

colors = ['crimson', 'royalblue', 'rebeccapurple']

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlabel('x', fontsize=16)
ax.set_ylabel('y', fontsize=16)
ax.set_xlim(-1.25, 1.75)
ax.set_ylim(-1.5,1.5)
ax.grid(True)

for i in range(3):
    ax.plot(r_eq[i, 0], r_eq[i, 1], 'kx', markersize=8)

atoms = []
trails = []
for i in range(3):
    atom, = ax.plot([], [], 'o', color=colors[i], markersize=12, label=f'Atom {i+1}')
    trail, = ax.plot([], [], '-', color=colors[i], alpha=0.4, linewidth=1)
    atoms.append(atom)
    trails.append(trail)

lines = []
for i in range(3):
    j = (i + 1) % 3
    line, = ax.plot([], [], 'k-', linewidth=1.2, alpha=0.3)
    lines.append(line)

ax.legend(loc='upper right', fontsize=16)


trail_length = 150

def init():
    for atom in atoms:
        atom.set_data([], [])
    for trail in trails:
        trail.set_data([], [])
    for line in lines:
        line.set_data([], [])

    return atoms + trails + lines

def animate(frame):
    for i, atom in enumerate(atoms):
        atom.set_data([positions_t[frame, i, 0]], [positions_t[frame, i, 1]])
    
    start = max(0, frame - trail_length)
    for i, trail in enumerate(trails):
        trail.set_data(positions_t[start:frame+1, i, 0], 
                      positions_t[start:frame+1, i, 1])
    
    for i, line in enumerate(lines):
        j = (i + 1) % 3
        line.set_data([positions_t[frame, i, 0], positions_t[frame, j, 0]],
                     [positions_t[frame, i, 1], positions_t[frame, j, 1]])
    
    return atoms + trails + lines 

anim = FuncAnimation(fig, animate, init_func=init, frames=T_RES, 
                    interval=30, blit=True, repeat=True)
plt.show()