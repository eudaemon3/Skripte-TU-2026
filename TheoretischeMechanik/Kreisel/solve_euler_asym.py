#------------------------------------------------------------------
# Solving the euler equations for an asymmetric spinning top
#
# author: eudaemon
# -----------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from plot_lib import animate_asym_spinner

Theta1, Theta2, Theta3 = 1.0, 1.0, 3.0
omega0 = [0.1, 0.1, 2.0]
e_angle0 = [0.0, np.pi/6, 0.0]
time = np.linspace(0, 20, 500)


def omega_dot(t, w):
    w1, w2, w3 = w
    return [
        (Theta3 - Theta2)/Theta1 * w2*w3,
        (Theta1 - Theta3)/Theta2 * w3*w1,
        (Theta2 - Theta1)/Theta3 * w1*w2
    ]

sol_omega = solve_ivp(omega_dot, (0, time[-1]), omega0, t_eval=time)
omega_vec = sol_omega.y.T

def euler_dot(t, e_angles):
    phi, theta, psi = e_angles

    w1 = np.interp(t, sol_omega.t, sol_omega.y[0])
    w2 = np.interp(t, sol_omega.t, sol_omega.y[1])
    w3 = np.interp(t, sol_omega.t, sol_omega.y[2])

    s = np.sin(theta)
    c = np.cos(theta)

    return [
        (w1*np.sin(psi) + w2*np.cos(psi)) / s,
        w1*np.cos(psi) - w2*np.sin(psi),
        w3 - c*(w1*np.sin(psi) + w2*np.cos(psi)) / s
    ]

sol_angles = solve_ivp(euler_dot, (0, time[-1]), e_angle0, t_eval=time)

phi, theta, psi = sol_angles.y

def R(phi, theta, psi):
    c1, s1 = np.cos(phi), np.sin(phi)
    c2, s2 = np.cos(theta), np.sin(theta)
    c3, s3 = np.cos(psi), np.sin(psi)
    return np.array([
        [c1*c3 - s1*c2*s3, -c1*s3 - s1*c2*c3, s1*s2],
        [s1*c3 + c1*c2*s3, -s1*s3 + c1*c2*c3, -c1*s2],
        [s2*s3, s2*c3, c2]
    ])

e3_inertial = np.array([R(phi[i], theta[i], psi[i]) @ np.array([0,0,1]) for i in range(len(phi))])

ani = animate_asym_spinner(omega_vec, e3_inertial)
ani.save("TheoretischeMechanik/Kreisel/plots/spin_asymetric_C01_anim.gif", writer="pillow", fps=20)
plt.show()