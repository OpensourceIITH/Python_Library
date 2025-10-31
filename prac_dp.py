import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

m1 = 1.0
m2 = 1.0
L1 = 1.0
L2 = 1.0
g = 1.0
a = 10.0
b = 100.0
x0 = 0.0
y0 = 0.0
def dp(t,p):
    global m1, m2, g, L1, L2, x0, y0,a,b

    x1,y1,x2,y2,x1d,y1d,x2d,y2d = p

    M = np.diag([m1,m1,m2,m2])

    e1 = (x1-x0)**2 + (y1-y0)**2 -L1**2
    e2 = (x2-x1)**2 + (y2-y1)**2 -L2**2

    e = np.array([e1,e2])

    J = np.array([
        [2*(x1-x0), 2*(y1-y0), 0, 0],
        [-2*(x2-x1), -2*(y2-y1), 2*(x2-x1), 2*(y2-y1)]
    ])

    J_dot = np.array([
        [2*(x1d), 2*(y1d), 0, 0],
        [-2*(x2d-x1d), -2*(y2d-y1d), 2*(x2d-x1d), 2*(y2d-y1d)]
    ])

    q_dot = np.array([x1d,y1d,x2d,y2d])

    Fe = np.array([0,-m1*g,0,-m2*g])

    lam = -(np.linalg.inv(J.dot(np.linalg.inv(M)).dot(J.T))).dot(J.dot(np.linalg.inv(M)).dot(Fe) + J_dot.dot(q_dot) + a*J.dot(q_dot) + b*e)

    q_ddot = np.linalg.solve(M, Fe + (J.T).dot(lam))

    p_dot =np.concatenate((q_dot,q_ddot))
    return p_dot

t_span = (0,20)
t_eval = np.arange(0,20.05,0.05)
init = np.array([1, 0, 2, 0, 0, 0, 0, 0])

sol = solve_ivp(dp,t_span,init, t_eval=t_eval, atol=1e-8, rtol=1e-8)
x1,y1,x2,y2,x1d,y1d,x2d,y2d = sol.y
t = sol.t

plt.figure(1)
for k in range(0,len(t), 5):
    plt.plot([x0,x1[k]],[y0,y1[k]], 'ko-', markersize=10)
    plt.plot([x1[k],x2[k]],[y1[k],y2[k]], 'ko-', markersize=10)
    plt.axis('equal')
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.title(f"time : {t[k]:.2f} s")
    plt.pause(0.05)
    plt.clf()

plt.show()