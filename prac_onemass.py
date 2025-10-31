import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

L = 1
m = 0.1       # mass (kg)
g = 10          # gravity (m/s^2)
k = 5          # spring constant (N/m)
c = 0.05          # damping coefficient
x0 = 0
y0 = 0

def onemass(t, p):
    
    global L, m, c, g, k, x0, y0

    x, y, xd, yd = p

    q = np.array([x,y])
    q0 = np.array([x0, y0])

    q_dot = np.array([xd,yd])
    # e = (q)/np.linalg.norm(q,2)
    e = (q-q0)/np.linalg.norm(q-q0,2)

    M = np.diag([m,m])

    Fe = np.array([0, -m*g])

    # Fd = -c*(q_dot) 
    Fd = -c*(q_dot.dot(e)) * e
    Fs = -k*(np.linalg.norm(q-q0,2) + 1e-8-L)*e

    F = Fs + Fd + Fe

    q_ddot = np.linalg.inv(M).dot(F)

    v_dot = np.concatenate((q_dot, q_ddot))
    return v_dot

t_span = (0,10)
t_eval = np.arange(0,10.05,0.05)
init = [L, 0, 0, 0]

sol = solve_ivp(onemass, t_span, init, t_eval=t_eval, atol=1e-8, rtol=1e-8)
x,y,xd,yd = sol.y
t = sol.t

plt.figure(1)
for k in range(0,len(t),5):
    plt.plot([x0, x[k]],[y0,y[k]], 'ko-', markersize=10)
    plt.axis('equal')
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.title(f"time:{t[k]:.2f} s")
    plt.pause(0.05)
    plt.clf()
plt.show()
