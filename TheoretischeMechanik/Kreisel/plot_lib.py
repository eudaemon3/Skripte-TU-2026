#------------------------------------------------------------------
# Functions to plot and simulate the spinning top problem
#
# author: eudaemon
# -----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D

def animate_asym_spinner(omega_vec, e3_inertial, interval=40):
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle("Asymmetric spinning top", fontsize=18)

    # ---------- LEFT: omega space ----------
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title(r'Angular velocity in $KS$ system', fontsize=14)

    lim = np.max(np.abs(omega_vec))
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_zlim(-lim, lim)
    ax1.set_box_aspect((1, 1, 1))

    ax1.set_xlabel(r'$\omega_1$', fontsize=16)
    ax1.set_ylabel(r'$\omega_2$', fontsize=16)
    ax1.set_zlabel(r'$\omega_3$', fontsize=16)

    ax1.xaxis.set_ticklabels([]) 
    ax1.yaxis.set_ticklabels([]) 
    ax1.zaxis.set_ticklabels([])

    # dashed coordinate axes
    ax1.plot([-lim, lim], [0, 0], [0, 0], '--', color='black', linewidth=0.8)
    ax1.plot([0, 0], [-lim, lim], [0, 0], '--', color='black', linewidth=0.8)
    ax1.plot([0, 0], [0, 0], [-lim, lim], '--', color='black', linewidth=0.8)

    omega_line = Line3D([], [], [], lw=1, color='blue')
    ax1.add_line(omega_line)
    omega_vec_arrow = ax1.quiver(0, 0, 0, 0, 0, 0)

    # ---------- RIGHT: inertial frame ----------
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(r'body axis in inertial system', fontsize=14)

    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_zlim(-1.1, 1.1)
    ax2.set_box_aspect((1, 1, 1))

    ax2.set_xlabel('x', fontsize=16)
    ax2.set_ylabel('y', fontsize=16)
    ax2.set_zlabel('z', fontsize=16)

    ax2.xaxis.set_ticklabels([]) 
    ax2.yaxis.set_ticklabels([]) 
    ax2.zaxis.set_ticklabels([]) 

    # unit sphere
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    U, V = np.meshgrid(u, v)
    ax2.plot_wireframe(
        np.cos(U)*np.sin(V),
        np.sin(U)*np.sin(V),
        np.cos(V),
        color='gray',
        alpha=0.3,
        linewidth=0.5
    )

    # inertial axes
    ax2.plot([-1, 1], [0, 0], [0, 0], '--', color='black', linewidth=0.8)
    ax2.plot([0, 0], [-1, 1], [0, 0], '--', color='black', linewidth=0.8)
    ax2.plot([0, 0], [0, 0], [-1, 1], '--', color='black', linewidth=0.8)

    body_traj = Line3D([], [], [], lw=1, color='crimson')
    ax2.add_line(body_traj)
    body_axis = ax2.quiver(0, 0, 0, 0, 0, 0, linewidth=1.5)
    plt.subplots_adjust(top=0.8)
    # ---------- animation update ----------
    def update(i):
        nonlocal omega_vec_arrow, body_axis

        # omega space
        omega_vec_arrow.remove()
        omega_vec_arrow = ax1.quiver(
            0, 0, 0,
            omega_vec[i, 0],
            omega_vec[i, 1],
            omega_vec[i, 2],
            normalize=False,
            color='black'
        )

        omega_line.set_data(omega_vec[:i, 0], omega_vec[:i, 1])
        omega_line.set_3d_properties(omega_vec[:i, 2])

        # inertial frame
        body_axis.remove()
        body_axis = ax2.quiver(
            0, 0, 0,
            e3_inertial[i, 0],
            e3_inertial[i, 1],
            e3_inertial[i, 2],
            color='black',
            linewidth=2
        )

        body_traj.set_data(e3_inertial[:i, 0], e3_inertial[:i, 1])
        body_traj.set_3d_properties(e3_inertial[:i, 2])

        return omega_line, omega_vec_arrow, body_traj, body_axis

    ani = FuncAnimation(
        fig,
        update,
        frames=min(len(omega_vec), len(e3_inertial)),
        interval=interval,
        blit=False
    )

    return ani


def animate_symmetric_spinner(x,y,z, t_eval):
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(projection='3d')

    # set axis properties
    lim = 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_zlabel('z', fontsize=14)

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    # draw sphere
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    U, V = np.meshgrid(u, v)
    X = np.cos(U) * np.sin(V)
    Y = np.sin(U) * np.sin(V)
    Z = np.cos(V)
    ax.plot_wireframe(X, Y, Z, color='gray', alpha=0.3, linewidth=0.5)

    # draw coordinate system
    ax.plot([-1,1],[0,0],[0,0],'--',color='black',linewidth=0.8)
    ax.plot([0,0],[-1,1],[0,0],'--',color='black',linewidth=0.8)
    ax.plot([0,0],[0,0],[-1,1],'--',color='black',linewidth=0.8)

    ax.plot(x, y, z, 'k-', alpha=0.8, linewidth=0.8)
    # trail, = ax.plot(x[:1], y[:1], z[:1], 'g-', linewidth=1, alpha=0.7)
    axis = ax.quiver(0, 0, 0, 0, 0, 0)
    trail_length = 200
    
    def update(frame):
        nonlocal axis
        axis.remove()
        axis = ax.quiver(
            0, 0, 0, x[frame], y[frame], z[frame],
            color='black', linewidth=2
        )
        
        # start = max(0, frame - trail_length)
        # trail.set_data_3d(x[start:frame+1], y[start:frame+1], z[start:frame+1])
        
        return [axis]
        return axis, trail

    anim = FuncAnimation(fig, update, frames=len(t_eval), interval=20, blit=False)
    return anim

def plot_effective_potential(theta_range, U_eff, E_tilde, turning_points):
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    mask = abs(U_eff) <= 6*E_tilde 

    ax1.plot(np.degrees(theta_range)[mask], U_eff[mask], 'b-', linewidth=1.2, label='$U_{eff}(θ)$')
    ax1.axhline(E_tilde, color='crimson', linestyle='-', linewidth=1.2, label=fr'$\tilde{{E}}$ = {E_tilde:.2f}')
    for i, tp in enumerate(turning_points[:2]):
        ax1.axvline(np.degrees(tp), color='k', linestyle='--', alpha=0.7, label=fr'$θ_{i+1}$ = {np.degrees(tp):.1f}°')
    ax1.set_xlabel(r'Theta $\theta$ / $^\circ$', fontsize=16)
    ax1.set_ylabel(r'Energy $E$ / a.u', fontsize=16)
    ax1.legend(fontsize=14, frameon=True)

    ax1.grid(True, alpha=0.8)
    ax1.set_xlim([0, 90])
    
    plt.tight_layout()
    return fig
