#------------------------------------------------------------------
# Solving the euler equations for an asymmetric spinning top
#
# author: eudaemon
# -----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from plot_lib import animate_symmetric_spinner, plot_effective_potential

Theta_1 = 1.0
Theta_3 = 0.5
M = 1
g = 9.81
S = 0.5

def equations_of_motion(t, y, p_phi, p_psi):
    theta, phi, psi, theta_dot = y
    sin_theta = np.sin(theta)
    if np.abs(sin_theta) < 1e-6:
        sin_theta = np.sign(sin_theta) * 1e-6
    cos_theta = np.cos(theta)
    phi_dot = (p_phi - p_psi * cos_theta) / (Theta_1 * sin_theta**2)
    psi_dot = p_psi / Theta_3 - phi_dot * cos_theta
    term1 = (phi_dot * sin_theta)**2 * np.sin(2*theta) / 2
    term2 = -(phi_dot * cos_theta + psi_dot) * phi_dot * sin_theta * Theta_3 / Theta_1
    term3 = M * g * S * sin_theta / Theta_1
    theta_ddot = term1 + term2 + term3
    return [theta_dot, phi_dot, psi_dot, theta_ddot]

def p_k(theta, phi_dot, psi_dot):
    p_psi = Theta_3 * (phi_dot * np.cos(theta) + psi_dot)
    p_phi = Theta_1 * phi_dot * np.sin(theta)**2 + p_psi * np.cos(theta)
    return p_phi, p_psi

def euler_to_cartesian(theta, phi, psi, length=1.0):
    x = length * np.sin(theta) * np.sin(phi)
    y = -length * np.sin(theta) * np.cos(phi)
    z = length * np.cos(theta)
    return x, y, z

def effective_potential(theta, p_phi, p_psi):
    sin_theta = np.sin(theta)
    if np.abs(sin_theta) < 1e-10:
        return np.inf
    return (p_phi - p_psi * np.cos(theta))**2 / (Theta_1 * sin_theta**2) + M * g * S * np.cos(theta)

def total_energy(theta, theta_dot, phi_dot, psi_dot):
    kin = (Theta_1/2) * theta_dot**2 + (Theta_1/2) * (phi_dot * np.sin(theta))**2 + (Theta_3/2) * (phi_dot * np.cos(theta) + psi_dot)**2
    pot = M * g * S * np.cos(theta)
    return kin + pot

def find_turning_points(p_phi, p_psi, E_0):
    theta_range = np.linspace(0.01, np.pi - 0.01, 2000)
    E_tilde = E_0 - p_psi**2 / (2 * Theta_3)
    U_eff = np.array([effective_potential(th, p_phi, p_psi) for th in theta_range])
    diff = E_tilde - U_eff
    turning_points = []
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0:
            theta_turn = theta_range[i] - diff[i] * (theta_range[i+1] - theta_range[i]) / (diff[i+1] - diff[i])
            turning_points.append(theta_turn)
    return turning_points, theta_range, U_eff, E_tilde

theta_0 = np.pi/4
theta_dot_0 = 0
phi_0 = 0
psi_0 = 0
phi_dot_0 = 2
psi_dot_0 = 25.0

p_phi, p_psi = p_k(theta_0, phi_dot_0, psi_dot_0)
E_0 = total_energy(theta_0, theta_dot_0, phi_dot_0, psi_dot_0)

y0 = [theta_0, phi_0, psi_0, theta_dot_0]
time = np.linspace(0, 10.0, 500)

solution = solve_ivp(
    equations_of_motion,
    (0, max(time)),
    y0,
    args=(p_phi, p_psi),
    t_eval=time,
    method='RK45',
    rtol=1e-8,
    atol=1e-10
)

theta = solution.y[0]
phi = solution.y[1]
psi = solution.y[2]
x, y, z = euler_to_cartesian(theta, phi, psi)

turning_points, theta_range, U_eff, E_tilde = find_turning_points(p_phi, p_psi, E_0)

p = plot_effective_potential(theta_range, U_eff, E_tilde, turning_points)
ani1 = animate_symmetric_spinner(x, y, z, time)
# ani1.save("TheoretischeMechanik/Kreisel/plots/spin_syymetric_C03_anim.gif", writer="pillow", fps=20)
plt.show()