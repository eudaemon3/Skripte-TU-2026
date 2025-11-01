import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

gamma, g = 0.1, 10
x0, y0 = 0, 0
v0 = 3

angels = np.array([1/6, 1/4, 1/3]) * np.pi

def dvdt(t,S):
    x, y, vx, vy = S
    return [vx, vy, -gamma* vx, -gamma * vy - g]

def analytical_solution(t, vx0, vy0): 
    x = (vx0/gamma) * (1 - np.exp(-gamma*t)) 
    y = ((vy0 + g/gamma)/gamma) * (1 - np.exp(-gamma*t)) - (g/gamma)*t 
    return x, y

t = np.linspace(0,3,100)

fig, axs = plt.subplots(1,2, figsize=(10,4))

for alpha in angels:
    v0x = v0*np.cos(alpha) 
    v0y = v0*np.sin(alpha)
    S0 = (x0, y0, v0x, v0y)

    sol_num = solve_ivp(dvdt,t_span=(0,max(t)),y0=S0,t_eval=t) 
    x_ana, y_ana = analytical_solution(t, v0x, v0y)

    x_num = sol_num.y[0]
    y_num = sol_num.y[1]

    axs[0].plot(x_num, y_num, '-', label=fr"$\alpha$={alpha:.2f} rad") 
    axs[1].plot(x_ana, y_ana, '--', label=fr"$\alpha$={alpha:.2f} rad") 

axs[0].set_xlabel("x(t)") 
axs[0].set_ylabel("y(t)") 
axs[0].set_title(fr"Numerische Lösung (g={g}, $\gamma$={gamma}, $v_0$={v0})") 
axs[0].grid(True) 
axs[0].legend() 

axs[1].set_xlabel("x(t)") 
axs[1].set_ylabel("y(t)") 
axs[1].set_title(fr"Analytische Lösung (g={g}, $\gamma$={gamma}, $v_0$={v0})") 
axs[1].grid(True) 
axs[1].legend() 
plt.show()