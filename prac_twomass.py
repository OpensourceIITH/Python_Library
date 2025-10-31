import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
# Define constants (global-like)
m1 = 1
m2 = 1
g = 10
k1 = 400
k2 = 400
c1 = 0
c2 = 0
L1 = 0.5
L2 = 0.2
x0=0
y0=0

def twomass(t,p):

    global m1, m2, g, k1, k2, c1, c2, x0, y0, L1, L2

    x1, y1, x2 ,y2, x1d, y1d, x2d, y2d = p

    q_dot = np.array([x1d,y1d,x2d,y2d])

    M = np.diag([m1, m1, m2,m2])

    r1 = np.array([x1-x0, y1-y0])
    r1d = np.array([x1d,y1d])

    r2 = np.array([x2-x0,y2-y0])
    r2d = np.array([x2d,y2d])

    e1 = (r1)/np.linalg.norm(r1,2)
    e2 = (r2-r1)/np.linalg.norm(r2-r1,2)

    Fs1 = -k1*(np.linalg.norm(r1,2)-L1)*e1
    Fs2 = -k2*(np.linalg.norm(r2-r1,2)-L2)*e2

    Fd1 = -c1*(r1d.dot(e1))*e1
    Fd2 = -c2*((r2d-r1d).dot(e2))*e2

    Fe1 = np.array([0,-m1*g])
    Fe2 = np.array([0,-m2*g])

    F1 = Fe1 + Fs1 + Fd1 - Fs2 - Fd2
    F2 = Fe2 + Fs2 + Fd2

    F = np.concatenate((F1,F2))

    q_ddot = np.linalg.solve(M,F)

    p_dot = np.concatenate((q_dot,q_ddot))

    return p_dot

t_span = (0,20)
t_eval = np.arange(0,20.05, 0.05)
init = np.array([L1, 0, L1 + L2, 0, 0, 0, 0, 0])

sol = solve_ivp(twomass, t_span, init, t_eval=t_eval, atol=1e-8, rtol=1e-8)

x1,y1,x2,y2, x1d, y1d, x2d, y2d = sol.y
t = sol.t

# plt.figure(1)
for k in range(0,len(t),5):
    plt.plot([x0,x1[k]],[y0,y1[k]], 'ko-', markersize=10)
    plt.plot([x1[k],x2[k]], [y1[k], y2[k]],'ko-', markersize=10)
    plt.axis('equal')
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.title(f"time:{t[k]:.2f} s")
    plt.pause(0.05)
    plt.clf()

plt.show()

