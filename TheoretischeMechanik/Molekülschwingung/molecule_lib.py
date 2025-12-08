import numpy as np
from enum import Enum

class Modes(Enum):
    breathing = 1
    rotation = 2
    mode5 = 3
    mode6 = 4
    rotK = 5

root3 = np.sqrt(3)
r0_eq = root3
sigma = r0_eq / (2**(1/6))
epsilon = 1.0
MASSES = [1,1,1]

k = 1.0 
K = k/4 * np.array([
        [6,0,-3, root3, -3,-root3],
        [0,2, root3,-1, -root3,-1],
        [-3,root3,3,-root3,0,0],
        [root3,-1,-root3,5,0,-4],
        [-3,-root3,0,0,3,root3],
        [-root3,-1,0,-4,root3,5]
    ], dtype=np.float64)

ANIM_PARAMS = {
    Modes.breathing : 0.14,
    Modes.rotation : 0.25,
    Modes.mode5 : 0.1,
    Modes.mode6 : 0.1,
    Modes.rotK : 0.1
}

R_EQ = np.array([
    [1, 0],
    [-1/2, root3/2],
    [-1/2, -root3/2]
])

def F_LJ(r_vec, epsilon=epsilon, sigma=sigma):
    r = np.linalg.norm(r_vec)
    if r < 1e-10:
        return np.zeros(2)
    return 24 * epsilon * (2*(sigma/r)**13 - (sigma/r)**7) * (r_vec / r)

def compute_acceleration(positions, masses=MASSES):
    x1, y1, x2, y2, x3, y3 = positions
    m1, m2, m3 = masses
    
    r12_vec = np.array([x1-x2, y1-y2])
    r13_vec = np.array([x1-x3, y1-y3])
    r23_vec = np.array([x2-x3, y2-y3])
    
    F12_vec = F_LJ(r12_vec)
    F13_vec = F_LJ(r13_vec)
    F23_vec = F_LJ(r23_vec)
    
    forces = np.array([
        (F12_vec[0] + F13_vec[0]) / m1,
        (F12_vec[1] + F13_vec[1]) / m1,
        (-F12_vec[0] + F23_vec[0]) / m2,
        (-F12_vec[1] + F23_vec[1]) / m2,
        (-F13_vec[0] - F23_vec[0]) / m3,
        (-F13_vec[1] - F23_vec[1]) / m3
    ])
    
    return forces

def equations_of_motion(t, y):
    pos = y[:6]
    vel = y[6:]
    
    a = compute_acceleration(pos)
    dydt = np.concatenate([vel, a]) 
    return dydt

def create_initial_conditions(r_eq=R_EQ, mode='breathing', amplitude=0.05):
    positions_0 = r_eq.copy()
    velocities_0 = np.zeros_like(positions_0)
    
    if mode == Modes.breathing:
        u4 = np.array([2, 0, -1, root3, -1, -root3]) / np.sqrt(12)
        positions_0 = r_eq + amplitude * u4.reshape(3, 2)
        
    elif mode == Modes.rotation:
        omega = amplitude * 2
        for i in range(3):
            tangent = np.array([-r_eq[i, 1], r_eq[i, 0]])
            tangent = tangent / np.linalg.norm(tangent)
            velocities_0[i] = omega * tangent
            
    elif mode == Modes.mode5:
        u5 = np.array([2, 0, -1, -root3, -1, root3]) / np.sqrt(12)
        positions_0 = r_eq + amplitude * u5.reshape(3, 2)
        
    elif mode == Modes.mode6:
        u6 = np.array([-1, root3, 2, 0, -1, -root3]) / np.sqrt(12)
        positions_0 = r_eq + amplitude * u6.reshape(3, 2)

    elif mode == Modes.rotK:
        u6 = np.array([0, 2,-root3, -1, root3, -1]) / np.sqrt(12)
        positions_0 = r_eq + amplitude * u6.reshape(3, 2)
        
    return positions_0, velocities_0

def solve_with_K(y0, t_eval, K=K, r_eq=R_EQ):
    # Verschiebung vom Gleichgewicht
    u0 = y0[:6] - r_eq.flatten()
    v0 = y0[6:]

    eigvals, eigvecs = np.linalg.eigh(K)
    omegas = np.sqrt(np.abs(eigvals))

    # Projektion auf Eigenmoden
    A = eigvecs.T @ u0
    B = np.zeros_like(A)
    nonzero = omegas > 1e-12
    B[nonzero] = (eigvecs.T @ v0)[nonzero] / omegas[nonzero]

    # Zeitentwicklung
    U = np.zeros((len(A), len(t_eval)))
    for i, w in enumerate(omegas):
        if w < 1e-12:
            U[i,:] = A[i] + B[i]*t_eval
        else:
            U[i,:] = A[i]*np.cos(w*t_eval) + B[i]*np.sin(w*t_eval)

    # zurücktransformieren
    u_t = (eigvecs @ U).T
    positions_t = u_t.reshape(-1, 3, 2) + r_eq  # Gleichgewicht wieder hinzufügen
    return positions_t
