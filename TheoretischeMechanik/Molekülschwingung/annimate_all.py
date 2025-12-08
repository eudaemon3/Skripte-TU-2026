import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import molecule_lib as ml
from molecule_lib import create_initial_conditions, equations_of_motion, Modes
import scienceplots
plt.style.use('science')
plt.rcParams.update({'font.size': 12})

T_SPAN = (0, 20)
T_RES = 500 
colors = ['crimson', 'royalblue', 'rebeccapurple']
modes_list = [Modes.rotation, Modes.breathing, Modes.mode5, Modes.mode6]
names = ['rotation', 'breathing', 'mode5', 'mode6']
trail_length = 80

r_eq = ml.R_EQ
t_eval = np.linspace(T_SPAN[0], T_SPAN[1], T_RES)

fig, axs = plt.subplots(2,2, figsize=(10,10)) 
axs[0][0].set_ylabel('y', fontsize=16) 
axs[1][0].set_ylabel('y', fontsize=16) 
axs[1][0].set_xlabel('x', fontsize=16) 
axs[1][1].set_xlabel('x', fontsize=16) 
i = 0 
for line in axs: 
    for ax in line: 
        ax.set_xlim(-1.25, 1.75) 
        ax.set_ylim(-1.5,1.5) 
        ax.set_title(f'Mode: {names[i]}', fontsize=18, fontweight=10) 
        ax.grid(True) 
        i+=1

positions_list = []
for MODE in modes_list:
    init_amp = ml.ANIM_PARAMS[MODE]
    positions_0, velocities_0 = create_initial_conditions(mode=MODE, amplitude=init_amp)
    y0 = np.concatenate([positions_0.flatten(), velocities_0.flatten()])
    solution = solve_ivp(equations_of_motion, T_SPAN, y0, 
        t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-11)
    positions_t = solution.y[:6].T.reshape(-1, 3, 2)
    positions_list.append(positions_t)

atoms_list = []
trails_list = []
lines_list = []

axs_flat = axs.flatten()
for ax in axs_flat:
    atoms = []
    trails = []
    lines = []
    for i in range(3):
        ax.plot(r_eq[i,0], r_eq[i,1], 'kx', markersize=8)
    for i in range(3):
        atom, = ax.plot([], [], 'o', color=colors[i], markersize=10)
        trail, = ax.plot([], [], '-', color=colors[i], alpha=0.4, linewidth=1)
        atoms.append(atom)
        trails.append(trail)
    for i in range(3):
        line, = ax.plot([], [], 'k-', alpha=0.3, linewidth=1.2)
        lines.append(line)
    atoms_list.append(atoms)
    trails_list.append(trails)
    lines_list.append(lines)

def init():
    for atoms, trails, lines in zip(atoms_list, trails_list, lines_list):
        for atom in atoms: atom.set_data([], [])
        for trail in trails: trail.set_data([], [])
        for line in lines: line.set_data([], [])
    return [obj for group in (atoms_list, trails_list, lines_list) for sublist in group for obj in sublist]

def animate(frame):
    for ax_idx, positions_t in enumerate(positions_list):
        atoms = atoms_list[ax_idx]
        trails = trails_list[ax_idx]
        lines = lines_list[ax_idx]
        for i, atom in enumerate(atoms):
            atom.set_data([positions_t[frame,i,0]], [positions_t[frame,i,1]])
        start = max(0, frame - trail_length)
        for i, trail in enumerate(trails):
            trail.set_data(positions_t[start:frame+1,i,0], positions_t[start:frame+1,i,1])
        for i, line in enumerate(lines):
            j = (i+1) % 3
            line.set_data([positions_t[frame,i,0], positions_t[frame,j,0]],
                          [positions_t[frame,i,1], positions_t[frame,j,1]])
    return [obj for group in (atoms_list, trails_list, lines_list) for sublist in group for obj in sublist]

handles = [plt.Line2D([0],[0], color=colors[i], marker='o', linestyle='') for i in range(3)]
labels = [f'Atom {i+1}' for i in range(3)]
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=16, frameon=True)

plt.tight_layout(rect=[0,0.05,1,1])
anim = FuncAnimation(fig, animate, init_func=init, frames=T_RES,
                     interval=30, blit=True, repeat=True)
anim.save('Molekül_Schwingung.gif', writer='pillow', fps=30, dpi=80)
plt.show()