import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

m = 1.0
L = 1.0
g = 1.0
Ax = 0.0   # set Ax = 0 for stabilization problem
Ay = 0.2
w = 3 * np.sqrt(2 * g * L / Ay)
a = 10.0
b = 100.0

def kapitza(t,p):
    global L, m, g, Ax, Ay, w, a, b

    x,y,xd,yd = p

    beta = Ax*np.sin(w*t)
    mu = Ay*np.cos(w*t)

    beta_dot = Ax*w*np.cos(w*t)
    mu_dot = -Ay*w*np.sin(w*t)

    beta_ddot = -Ax*(w**2)*np.sin(w*t)
    mu_ddot = -Ay*(w**2)*np.cos(w*t)

    e = (x-beta)**2 + (y-mu)**2 - L**2

    J = np.array([2*(x-beta), 2*(y-mu)])

    J_dot = np.array([2*(xd-beta_dot), 2*(yd-mu_dot)])
    q_dot = np.array([xd-beta_dot,yd-mu_dot])

    Fe = np.array([0,-m*g])

    M= np.diag([m,m])

    c_ddot = np.array([beta_ddot, mu_ddot])

    lam = (J.dot(c_ddot) - J.dot(np.linalg.inv(M)).dot(Fe) -J_dot.dot(q_dot) -a*J.dot(q_dot) -b*e)*m/(4*(L**2))

    q_ddot = np.linalg.solve(M,(Fe + (J.T).dot(lam)))

    p_dot = np.concatenate(([xd,yd],q_ddot))
    return p_dot

t_span = (0,20)
t_eval = np.arange(0,20.05,0.05)
init = np.array([
    L * np.sin(np.pi / 50),
    Ay + L * np.cos(np.pi / 50),
    Ax * w,
    0.0
])

sol = solve_ivp(kapitza, t_span, init, t_eval=t_eval,atol=1e-8,rtol=1e-8)
x,y,xd,yd = sol.y
t = sol.t

beta = Ax*np.sin(w*t)
mu = Ay*np.cos(w*t)

plt.figure(1)
for k in range(0,len(t),5):
    plt.plot([beta[k], x[k]],[mu[k], y[k]], 'ko-', markersize=10)
    plt.axis('equal')
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.title(f"time:{t[k]:.2f} s")
    plt.pause(0.05)
    plt.clf()

plt.show()