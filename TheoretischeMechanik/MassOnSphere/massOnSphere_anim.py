import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

R = 1
m = 1
g = 9.81
# 0 -> dont show at all
# 1 -> show full trail
# 2 -> show trail for 150 iterations
show_trail = 2

def stationaryTrajectory(p0, tmax):
    theta_bar, _, phi0, _ = p0

    def dphidt(t, phi):
        return np.sqrt(-g/(R*np.cos(theta_bar)))

    t = np.linspace(0, tmax, 200)
    sol_m2 = solve_ivp(dphidt,t_span=(0,max(t)),y0=[phi0],t_eval=t)

    phi = sol_m2.y[0]
    time = sol_m2.t

    x = R*np.sin(theta_bar)*np.cos(phi)
    y = R*np.sin(theta_bar)*np.sin(phi)
    z = R*np.cos(theta_bar)*np.ones_like(phi)

    return time, x,y, z

def generalTrjectory(p0, tmax):
    theta0, theta_dot0, phi0, phi_dot0 = p0
    
    p1 = m * R**2 * (np.sin(theta0)**2) * phi_dot0

    def dSdt(t, S0):
        theta, thdot, phi = S0

        s = np.sin(theta)
        if abs(s) < 1e-8: s = 1e-8
        phi_dot = p1 / (m * R**2 * s**2)

        theta_ddot = np.sin(theta)*np.cos(theta) * phi_dot**2 + (g/R)*np.sin(theta)
        return [thdot, theta_ddot, phi_dot]

    t_span = (0.0, tmax)
    t_eval = np.linspace(t_span[0], t_span[1], 800)
    S0 = [theta0, theta_dot0, phi0]
    sol = solve_ivp(dSdt, t_span, S0, t_eval=t_eval, rtol=1e-9, atol=1e-12)

    theta = sol.y[0]
    phi = sol.y[2]
    time = sol.t

    x = R*np.sin(theta)*np.cos(phi)
    y = R*np.sin(theta)*np.sin(phi)
    z = R*np.cos(theta)
    return time, x,y,z

p01 = (np.pi-np.pi/4, 0,0,0)
p02 = (np.pi/8,0,0,1)

time, x,y,z = generalTrjectory(p02, 8.0)
# time, x,y,z = stationaryTrajectory(p01, 8.0)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r"constraint Trajectory $\theta_0 = \pi/8$, $\dot \phi_0 = 1$")
ax.set_box_aspect([1,1,1])
ax.set_xlim(-1.1*R,1.1*R)
ax.set_ylim(-1.1*R,1.1*R)
ax.set_zlim(-1.1*R,1.1*R)

# draw sphere
u = np.linspace(0,2*np.pi,36)
v = np.linspace(0,np.pi,18)
U,V = np.meshgrid(u,v)
Xs = R*np.cos(U)*np.sin(V)
Ys = R*np.sin(U)*np.sin(V)
Zs = R*np.cos(V)
ax.plot_surface(Xs, Ys, Zs, color='black', alpha=0.12, linewidth=0)

point, = ax.plot([x[0]],[y[0]],[z[0]], marker='o', color='royalblue')
trail, = ax.plot([], [], [], lw=1, color='crimson')

def init():
    point.set_data([x[0]],[y[0]])
    point.set_3d_properties([z[0]])
    trail.set_data([], [])
    trail.set_3d_properties([])
    return point, trail

def update(i):
    point.set_data([x[i]],[y[i]])
    point.set_3d_properties([z[i]])
    match show_trail:
        case 1:
            istart = 0
            trail.set_data(x[istart:i], y[istart:i])
            trail.set_3d_properties(z[istart:i])
        case 2: 
            istart = max(0, i-150) # remove trail
            trail.set_data(x[istart:i], y[istart:i])
            trail.set_3d_properties(z[istart:i])
        case _: pass
    return point, trail

fig.tight_layout()
ani = animation.FuncAnimation(fig, update, frames=range(0,len(time)), init_func=init, interval=20, blit=False)
ani.save('massOnSphere.gif', writer='pillow', fps=30, dpi=80)
plt.show()