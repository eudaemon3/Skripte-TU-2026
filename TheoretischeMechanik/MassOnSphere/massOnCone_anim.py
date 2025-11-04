"""
mass_on_cone.py

Simulation und Animation eines Massenpunktes, der sich reibungsfrei auf einer
konischen Fläche bewegt (Kegel mit Öffnungswinkel theta).

Benutzung:
    - Script direkt ausführen: berechnet die Trajektorie für die in den
      Anfangsbedingungen (q0, q_dot0, phi0, phi_dot0) angegebenen Werte und
      speichert eine animierte GIF-Datei 'massOnCone.gif'.

Bedeutung der Anfangsbedingungen (p0 = (q0, q_dot0, phi0, phi_dot0)):
    q0        : radialer Abstand entlang der Mantellinie des Kegels
    q_dot0    : Anfangsgeschwindigkeit in radialer Richtung 
    phi0      : Anfangswinkel
    phi_dot0  : Anfangswinkelgeschwindigkeit

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import animation
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import Axes3D

# define parameters
m = 1
g = 9.81

theta = np.pi/8

# initial conditions
q0 = 1
q_dot0 = 0
phi0 = 0
phi_dot0 = 5
tmax = 2.15

def generalTrjectory(p0, tmax):
    q0, q_dot0, phi0, phi_dot0 = p0
    p1 = m * q0**2 * (np.sin(theta)**2) * phi_dot0

    def dSdt(t, S0):
        q, q_dot, phi = S0

        phi_dot = p1 / (m * q**2 * np.sin(theta)**2)
        q_ddot = q*np.sin(theta)**2 * phi_dot**2 - g*np.cos(theta)
        return [q_dot, q_ddot, phi_dot]

    t_span = (0.0, tmax)
    t_eval = np.linspace(t_span[0], t_span[1], 800)
    S0 = [q0, q_dot0, phi0]
    sol = solve_ivp(dSdt, t_span, S0, t_eval=t_eval, rtol=1e-9, atol=1e-12)

    q = sol.y[0]
    phi = sol.y[2]
    time = sol.t

    x = q*np.sin(theta)*np.cos(phi)
    y = q*np.sin(theta)*np.sin(phi)
    z = q*np.cos(theta)
    return time, x, y, z

p02 = (q0, q_dot0, phi0, phi_dot0)

time, x, y, z = generalTrjectory(p02, tmax)

# create axis and figure
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r"Massenpunkt auf einer Kegeloberfläche", fontsize=18)
ax.set_box_aspect([1,1,1])

# annotate initial conditions 
ic_text = (rf"$q_0={p02[0]:.2f}$, "
           rf"$v_0={p02[1]:.2f}$, "
           rf"$\phi_0={p02[2]:.2f}$, "
           rf"$\omega_0={p02[3]:.2f}$")
ax.text2D(0.2, 0.95, ic_text, transform=ax.transAxes,
          fontsize=12, color="black")

# qmin and qmax rings 
q = np.sqrt(x**2 + y**2 + z**2)
qmin = q.min()
qmax = q.max()
phi_ring = np.linspace(0, 2*np.pi, 400)
for q_val in (qmin, qmax):
    z_ring = q_val * np.cos(theta)
    r_ring = q_val * np.sin(theta)
    x_ring = r_ring * np.cos(phi_ring)
    y_ring = r_ring * np.sin(phi_ring)
    ax.plot(x_ring, y_ring, np.full_like(phi_ring, z_ring),
            color='forestgreen', linewidth=1, alpha=0.9)

# draw cone 
phi = np.linspace(0,2*np.pi,36)
r = np.linspace(0,qmax,18)
Phi,R = np.meshgrid(phi,r)
Xs = R*np.cos(Phi)*np.sin(theta)
Ys = R*np.sin(Phi)*np.sin(theta)
Zs = R*np.cos(theta)
ax.plot_surface(Xs, Ys, Zs, color='black', alpha=0.12, linewidth=0)

ax.plot(x, y, z, color='red', linewidth=1, alpha=0.5, label='Trajektorie (vollständig)')

point, = ax.plot([x[0]],[y[0]],[z[0]], marker='o', color='black', markersize=7)

def init():
    point.set_data([x[0]],[y[0]])
    point.set_3d_properties([z[0]])
    return (point,)

def update(i):
    point.set_data([x[i]],[y[i]])
    point.set_3d_properties([z[i]])

    azim = (i * 360/len(time)) % 360
    elev = 20
    ax.view_init(elev=elev, azim=azim)

    return (point,)

fig.tight_layout()

# create animation
ani = animation.FuncAnimation(fig, update, frames=range(0,len(time),2),
                              init_func=init, interval=20, blit=False)
ani.save('massOnSphere.gif', writer='pillow', fps=15, dpi=60)
plt.show()
