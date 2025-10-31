import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
m1 = 1.0
m2 = 1.0
m3 = 1.0
Jm1 = 1.0
Jm2 = 1.0
Jm3 = 1.0
g = 10.0
a = 100.0
b = 100.0

def R(theta):
    D = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return D

def R_dot(theta):
    D = np.array([
        [-np.sin(theta), -np.cos(theta)],
        [np.cos(theta), -np.sin(theta)]
    ])
    return D

def R_ddot(theta):
    D = np.array([
        [-np.cos(theta), np.sin(theta)],
        [-np.sin(theta), -np.cos(theta)]
    ])
    return D

p1 = np.array([0.1,0.1])
p2 = np.array([-0.1,0.1])
p3 = np.array([-0.1,-0.1])
p4 = np.array([0.1,-0.1])

u1 = np.array([0.1,0.1])
u2 = np.array([-0.1,0.1])
u3 = np.array([-0.1,-0.1])
u4 = np.array([0.1,-0.1])

w1 = np.array([0.1,0.1])
w2 = np.array([-0.1,0.1])
w3 = np.array([-0.1,-0.1])
w4 = np.array([0.1,-0.1])

def dprb(t,p):
    global m1, m2, m3, Jm1, Jm2, Jm3, g, p1,p2,p3,p4, u1,u2,u3,u4,w1,w2,w3,w4 ,a,b

    xcp, ycp, thetap, xcu, ycu, thetau, xcw, ycw, thetaw, xcpd, ycpd, thetapd, xcud, ycud, thetaud, xcwd, ycwd, thetawd = p

    M = np.diag([m1, m1, Jm1, m2, m2, Jm2, m3, m3, Jm3])
    q_dot = np.array([xcpd, ycpd, thetapd, xcud, ycud, thetaud, xcwd, ycwd, thetawd])

    rcp = np.array([xcp,ycp])
    rcu = np.array([xcu,ycu])
    rcw = np.array([xcw,ycw])

    phi1 = rcp + R(thetap).dot(p1) - p1
    phi2 = rcp + R(thetap).dot(p3) - rcu - R(thetau).dot(u2) 
    phi3 = rcu + R(thetau).dot(u3) - rcw - R(thetaw).dot(w2)

    phi = np.hstack((phi1, phi2, phi3))

    J = np.array([
        [1,0,R_dot(thetap).dot(p1)[0],0,0,0,0,0,0],
        [0,1,R_dot(thetap).dot(p1)[1],0,0,0,0,0,0],
        [1,0,R_dot(thetap).dot(p3)[0],-1,0,-R_dot(thetau).dot(u2)[0],0,0,0],
        [0,1,R_dot(thetap).dot(p3)[1],0,-1,-R_dot(thetau).dot(u2)[1],0,0,0],
        [0,0,0,1,0,R_dot(thetau).dot(u3)[0],-1,0,-R_dot(thetaw).dot(w2)[0]],
        [0,0,0,0,1,R_dot(thetau).dot(u3)[1],0,-1,-R_dot(thetaw).dot(w2)[1]],
    ])

    J_dot = np.array([
        [0,0,R_ddot(thetap).dot(p1)[0],0,0,0,0,0,0],
        [0,0,R_ddot(thetap).dot(p1)[1],0,0,0,0,0,0],
        [0,0,R_ddot(thetap).dot(p3)[0],0,0,-R_ddot(thetau).dot(u2)[0],0,0,0],
        [0,0,R_ddot(thetap).dot(p3)[1],0,0,-R_ddot(thetau).dot(u2)[1],0,0,0],
        [0,0,0,0,0,R_ddot(thetau).dot(u3)[0],0,0,-R_ddot(thetaw).dot(w2)[0]],
        [0,0,0,0,0,R_ddot(thetau).dot(u3)[1],0,0,-R_ddot(thetaw).dot(w2)[1]],
    ])

    Fe = np.array([0, -m1*g, 0, 0, -m2*g, 0, 0, -m3*g,0])

    lam = -(np.linalg.inv(J.dot(np.linalg.inv(M)).dot(J.T))).dot(J.dot(np.linalg.inv(M)).dot(Fe) + J_dot.dot(q_dot) + a*J.dot(q_dot) + b*phi)

    q_ddot = np.linalg.inv(M).dot(Fe + (J.T).dot(lam))

    p_dot = np.concatenate((q_dot, q_ddot))
    return p_dot

t_span = (0,30)
t_eval = np.arange(0, 30.05, 0.05)
init = np.array([0, 0, 0, 0, -0.2, 0, 0, -0.4, 0,0, 0, 0, 0, 0, 0, 0, 0, 0])

sol = solve_ivp(dprb, t_span, init, t_eval=t_eval, atol=1e-8, rtol=1e-8)

xcp, ycp, thetap, xcu, ycu, thetau, xcw, ycw, thetaw, xcpd, ycpd, thetapd, xcud, ycud, thetaud, xcwd, ycwd, thetawd = sol.y
t = sol.t

plt.figure(1)
for k in range(0, len(t), 5):

    rcp = np.array([xcp[k],ycp[k]])
    rcu = np.array([xcu[k],ycu[k]])
    rcw = np.array([xcw[k],ycw[k]])

    rp1 = rcp + R(thetap[k]).dot(p1)
    rp2 = rcp + R(thetap[k]).dot(p2)
    rp3 = rcp + R(thetap[k]).dot(p3)
    rp4 = rcp + R(thetap[k]).dot(p4)

    ru1 = rcu + R(thetau[k]).dot(u1)
    ru2 = rcu + R(thetau[k]).dot(u2)
    ru3 = rcu + R(thetau[k]).dot(u3)
    ru4 = rcu + R(thetau[k]).dot(u4)

    rw1 = rcw + R(thetaw[k]).dot(w1)
    rw2 = rcw + R(thetaw[k]).dot(w2)
    rw3 = rcw + R(thetaw[k]).dot(w3)
    rw4 = rcw + R(thetaw[k]).dot(w4)

    plt.plot([rp1[0],rp2[0]], [rp1[1],rp2[1]], 'b-')
    plt.plot([rp2[0],rp3[0]], [rp2[1],rp3[1]], 'r-')
    plt.plot([rp3[0],rp4[0]], [rp3[1],rp4[1]], 'g-')
    plt.plot([rp4[0],rp1[0]], [rp4[1],rp1[1]], 'k-')

    plt.plot([ru1[0],ru2[0]], [ru1[1],ru2[1]], 'b-')
    plt.plot([ru2[0],ru3[0]], [ru2[1],ru3[1]], 'r-')
    plt.plot([ru3[0],ru4[0]], [ru3[1],ru4[1]], 'g-')
    plt.plot([ru4[0],ru1[0]], [ru4[1],ru1[1]], 'k-')

    plt.plot([rw1[0],rw2[0]], [rw1[1],rw2[1]], 'b-')
    plt.plot([rw2[0],rw3[0]], [rw2[1],rw3[1]], 'r-')
    plt.plot([rw3[0],rw4[0]], [rw3[1],rw4[1]], 'g-')
    plt.plot([rw4[0],rw1[0]], [rw4[1],rw1[1]], 'k-')

    plt.axis('equal')
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.title(f"time : {t[k]:.2f} s")
    plt.pause(0.05)
    plt.clf()

plt.show()