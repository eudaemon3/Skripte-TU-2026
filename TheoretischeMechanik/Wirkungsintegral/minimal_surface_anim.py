import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import scienceplots
plt.style.use("science")

R = 1
D_crit = 1.3257
D_start = D_crit - 1.1
D_end = D_crit*0.99

frames = 100  # (5 s bei 20 fps)
Ds = np.linspace(D_start, D_end, frames)

# Bisection function
def bisect(f, a, b, tol=1e-10, max_iter=200):
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        return None
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol:
            return m
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)

# Lambda für Seifenhautform
y_fun = lambda x, k: k * np.cosh(x / k)

# Animation vorbereiten
fig, ax = plt.subplots(figsize=(6,4))

line_surface, = ax.plot([], [], color='royalblue')
line_left, = ax.plot([], [], color='black')
line_right, = ax.plot([], [], color='black')

ax.set_xlim(-D_crit/2, D_crit/2)
ax.set_ylim(0, 1.1*R)
ax.set_xlabel(r'$x$', fontsize=14)
ax.set_ylabel(r'$y(x)$', fontsize=14)
ax.set_title('Annimation: Minimale Oberfläche einer Seifenhaut', fontsize=16)
ax.grid(True)

def init():
    line_surface.set_data([], [])
    line_left.set_data([], [])
    line_right.set_data([], [])
    return line_surface, line_left, line_right

def animate(i):
    D = Ds[i]
    alpha = R*2 / D
    x_vals = np.linspace(-D/2, D/2, 1000)
    
    def f(xi):
        return np.cosh(xi) - alpha * xi
    
    # Bisection auf xi-Intervall
    xis = np.linspace(1e-6, 20, 5000)
    roots = []
    for j in range(len(xis)-1):
        a, b = xis[j], xis[j+1]
        if f(a)*f(b) < 0:
            root = bisect(f, a, b)
            if root is not None:
                roots.append(root)
    
    if roots:
        xi = roots[0]  # kleinster positive Root
        k = D / (2*xi)
        y_vals = y_fun(x_vals, k)
    else:
        y_vals = np.zeros_like(x_vals)
    
    # Linien aktualisieren
    line_surface.set_data(x_vals, y_vals)
    line_left.set_data([-D/2, -D/2], [0, R])
    line_right.set_data([D/2, D/2], [0, R])
    
    return line_surface, line_left, line_right

anim = FuncAnimation(fig, animate, frames=frames, init_func=init, blit=True)
anim.save('soap_surface.gif', writer=PillowWriter(fps=20))

plt.show()