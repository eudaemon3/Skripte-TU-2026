import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import scienceplots
from scipy.special import factorial, hermite
plt.style.use('science')
plt.rcParams.update({'font.size': 14})

n_lim_plot = 6
x_lim = 7

m = 1.0
hbar = 1.0

a_init = 1
b_init = 0

color_theme = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

def phi_n_0(n, x, alpha):
    N = np.sqrt(alpha / (np.sqrt(np.pi) * 2**n * factorial(n)))
    return N * np.exp(-alpha**2 * x**2 / 2) * hermite(n)(alpha * x)

def E_n(n, a, b, omega):
    return hbar * omega * (n + 0.5) - b**2 / (4 * a)

def V(x, a, b):
    return a * x**2 + b * x


x = np.linspace(-x_lim, x_lim, 1000)

# ── Layout: make room for two sliders at the bottom ──────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
plt.subplots_adjust(left=0.1, right=0.75, bottom=0.25)

# ── Slider axes ───────────────────────────────────────────────────────────────
ax_slider_a = fig.add_axes([0.1, 0.12, 0.6, 0.03])
ax_slider_b = fig.add_axes([0.1, 0.06, 0.6, 0.03])

slider_a = Slider(ax_slider_a, r'$a$', 0.1, 2.0, valinit=a_init, valstep=0.01, color='tab:blue')
slider_b = Slider(ax_slider_b, r'$b$', -5.0, 5.0, valinit=b_init, valstep=0.01, color='tab:orange')


def draw(a, b):
    ax.cla()
    omega = np.sqrt(2 * a / m)
    alpha = np.sqrt(m * omega / hbar)

    ax.plot(x, V(x, a, b), color='black', linewidth=1.5, label=r'$V(x)$')

    for n in range(n_lim_plot + 1):
        color = color_theme[n % len(color_theme)]
        en = E_n(n, a, b, omega)
        ax.plot(x, np.ones_like(x) * en,
                color=color, linestyle='--', alpha=0.5)
        ax.plot(x, en + phi_n_0(n, x + b / (2 * a), alpha),
                color=color, alpha=1.0, label=f'$n={n}$')

    # Axis limits & labels
    dist = x_lim + 0.5
    low = E_n(0, a, b, omega) - 1
    high = E_n(n_lim_plot, a, b, omega) + 1

    ax.set_ylim(low, high)
    ax.set_xlim(-dist, dist)
    ax.set_xlabel(r'$x$', fontsize=16)
    ax.set_ylabel(r'Energy  $E\,/\,\hbar\omega$', fontsize=16)
    ax.set_title('Moved Harmonic Oscillator', fontsize=20)

    
    ax.plot([-dist, dist], [0,0], color='black')
    ax.plot([0,0], [low, high], color='black')

    ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1), fontsize=16)
    ax.grid(True, alpha=1)
    fig.canvas.draw_idle()

draw(a_init, b_init)

# ── Slider callbacks ──────────────────────────────────────────────────────────
def update(_):
    draw(slider_a.val, slider_b.val)

slider_a.on_changed(update)
slider_b.on_changed(update)

plt.show()