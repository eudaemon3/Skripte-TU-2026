#------------------------------------------------------------------
# Functions to plot and simulate the spinning top problem
#
# author: eudaemon
# -----------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D

def plot_omega(omega_vec):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    lim = np.max(np.abs(omega_vec))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel(r'$\omega_1$')
    ax.set_ylabel(r'$\omega_2$')
    ax.set_zlabel(r'$\omega_3$')

    ax.zaxis.set_ticklabels([])
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    ax.plot([-lim, lim], [0, 0], [0, 0], '--', color='black')
    ax.plot([0, 0], [-lim, lim], [0, 0], '--', color='black')
    ax.plot([0, 0], [0, 0], [-lim, lim], '--', color='black')
    # ax.view_init(elev=35, azim=155, roll=-105)
    line = Line3D([], [], [], lw=1)
    ax.add_line(line)

    quiver = ax.quiver(0, 0, 0, 0, 0, 0)

    def update(i):
        nonlocal quiver
        quiver.remove()
        quiver = ax.quiver(
            0, 0, 0,
            omega_vec[i,0],
            omega_vec[i,1],
            omega_vec[i,2],
            normalize=False
        )
        line.set_data(omega_vec[:i,0], omega_vec[:i,1])
        line.set_3d_properties(omega_vec[:i,2])
        return line, quiver

    ani = FuncAnimation(
        fig,
        update,
        frames=len(omega_vec),
        interval=40,
        blit=False
    )

    return ani

def animate_body_axis(e3_inertial, interval=40):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

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

    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    U, V = np.meshgrid(u, v)
    X = np.cos(U) * np.sin(V)
    Y = np.sin(U) * np.sin(V)
    Z = np.cos(V)
    ax.plot_wireframe(X, Y, Z, color='gray', alpha=0.3, linewidth=0.5)

    ax.plot([-1,1],[0,0],[0,0],'--',color='black',linewidth=0.8)
    ax.plot([0,0],[-1,1],[0,0],'--',color='black',linewidth=0.8)
    ax.plot([0,0],[0,0],[-1,1],'--',color='black',linewidth=0.8)

    traj, = ax.plot([], [], [], lw=1)
    body_axis = ax.quiver(0, 0, 0, 0, 0, 0)

    def update(i):
        nonlocal body_axis
        body_axis.remove()
        body_axis = ax.quiver(
            0, 0, 0,
            e3_inertial[i,0],
            e3_inertial[i,1],
            e3_inertial[i,2],
            color='black',
            linewidth=2
        )

        traj.set_data(
            e3_inertial[:i,0],
            e3_inertial[:i,1]
        )
        traj.set_3d_properties( #type:ignore
            e3_inertial[:i,2]
        )

        return traj, body_axis

    ani = FuncAnimation(
        fig,
        update,
        frames=len(e3_inertial),
        interval=interval,
        blit=False
    )

    return ani
