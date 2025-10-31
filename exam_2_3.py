import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

L = 1
m1 = 1
m2 = 1
g = 1
alpha = 1
beta = 1
k = 1
l = 1
c = 1

def fun(t,p):
    global L, m1, m2, g, alpha, beta, k, l ,c
    x1,y1,x2,y2,x1d,y1d,x2d,y2d = p

    q_dot = np.array([x1d, y1d, x2d, y2d])

    phi1 = y1
    phi2 = (x2-x1)**2 + (y2-y1)**2 - L**2

    r1 = np.array([x1,y1])
    r2 = np.array([x2,y2])

    r1d = np.array([x1d,y1d])
    r2d = np.array([x2d,y2d])

    phi = np.array([phi1, phi2])

    J = np.array([
        [0, 1, 0, 0],
        [-2*(x2-x1), -2*(y2-y1), 2*(x2-x1), 2*(y2-y1)]
    ])

    J_dot = np.array([
        [0,0,0,0],
        [-2*(x2d-x1d), -2*(y2d-y1d), 2*(x2d-x1d), 2*(y2d-y1d)]
    ])

    M = np.diag([m1,m1,m2,m2])

    e1 = r1/np.linalg.norm(r1,2)

    F1 = -k*(np.linalg.norm(r1,2)-l)*e1 - c*(r1d.dot(e1))*e1 
    F2 = np.array([0,0])

    F = np.hstack((F1,F2))

    Fe = np.array([0, -m1*g, 0, -m2*g])
    F = F + Fe
    lam = -(np.linalg.inv(J.dot(np.linalg.inv(M)).dot(J.T))).dot(J.dot(np.linalg.inv(M)).dot(F) + J_dot.dot(q_dot) + alpha*J.dot(q_dot) + beta*phi)

    q_ddot = np.linalg.inv(M).dot(F + (J.T).dot(lam))

    return np.concatenate([q_dot, q_ddot])

t_span = (0,30)
t_eval = np.arange(0,30.05, 0.05)
init = [l,0, l,-L, 0,0,0.5,0]

sol = solve_ivp(fun,t_span, init, t_eval=t_eval,atol=1e-8, rtol=1e-8)
x1,y1,x2,y2,x1d,y1d,x2d,y2d = sol.y
t = sol.t

plt.figure(1)
for k in range(0,len(t), 5):
    plt.plot([0,x1[k]],[0,y1[k]], 'ko-', markersize=10)
    plt.plot([x1[k],x2[k]],[y1[k],y2[k]], 'ro-', markersize=10)
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.title(f"time:{t[k]:.2f}s")
    plt.pause(0.05)
    plt.clf()

plt.show()
