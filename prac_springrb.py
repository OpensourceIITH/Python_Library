import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

m = 1.0
Jm = 1.0
k1 = 0.0
k2 = 0.0
c1 = 0.0
c2 = 0.0
g = 10.0

p1 = np.array([0.1, 0.1])
p2 = np.array([-0.1, 0.1])
p3 = np.array([-0.1, -0.1])
p4 = np.array([0.1, -0.1])
rg1 = np.array([0.1, 0.2])
rg2 = np.array([-0.1, 0.2])

L1 = np.linalg.norm(rg1 - p1)
L2 = np.linalg.norm(rg2 - p2)


a = 10.0
b = 10.0

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

def springrb(t,p):
    global m, Jm, g, rg1, rg2,p1,p2,p3,p4, k1,k2,c1,c2,L1,L2,a,b

    xc,yc,theta,xcd,ycd,thetad = p

    rc = np.array([xc,yc])
    vc = np.array([xcd,ycd])

    q_dot = np.array([xcd,ycd, thetad])

    rp1 = rc + R(theta).dot(p1)
    rp2 = rc + R(theta).dot(p2)

    vp1 = vc + R_dot(theta).dot(p1)*thetad
    vp2 = vc + R_dot(theta).dot(p2)*thetad

    phi = rc + R(theta).dot(p3) - p3

    J = np.array([
        [1,0,R_dot(theta).dot(p3)[0]],
        [0,1,R_dot(theta).dot(p3)[1]],
    ])

    J_dot = np.array([
        [0,0,R_ddot(theta).dot(p3)[0]],
        [0,0,R_ddot(theta).dot(p3)[1]],
    ])

    e1 = (rp1-rg1)/np.linalg.norm(rp1-rg1,2)
    e2 = (rp2-rg2)/np.linalg.norm(rp2-rg2,2)

    F1 = -k1*(np.linalg.norm(rp1-rg1)-L1)*e1 - c1*(vp1.dot(e1))*e1
    F2 = -k2*(np.linalg.norm(rp2-rg2)-L2)*e2 - c2*(vp2.dot(e2))*e2

    T1 = np.cross(np.append(R(theta).dot(p1),0), np.append(F1,0))
    T2 = np.cross(np.append(R(theta).dot(p2),0), np.append(F2,0))

    Fe = np.array([0, -m*g])
    
    rr = 1
    kk = 1
    if(t > 10):
        rr = 0
    if(kk > 20):
        kk = 0
    
    F = np.array([rr*F1[0]+ kk*F2[0] + Fe[0], rr*F1[1]+ kk*F2[1] + Fe[1], rr*T1[2]+kk*T2[2]])

    M = np.diag([m,m,Jm])

    lam = -(np.linalg.inv(J.dot(np.linalg.inv(M)).dot(J.T))).dot(J.dot(np.linalg.inv(M)).dot(F) + J_dot.dot(q_dot) + a*J.dot(q_dot) + b*phi)

    q_ddot = np.linalg.solve(M, F + (J.T).dot(lam))

    p_dot = np.concatenate((q_dot,q_ddot))
    return p_dot

t_span = (0,20)
t_eval = np.arange(0,20.05, 0.05)
init = np.array([0, 0, 0, 0, 0, 0])

sol = solve_ivp(springrb, t_span, init, t_eval=t_eval, atol=1e-8, rtol=1e-8)
t = sol.t

# q = sol.y
xc,yc,theta,xcd,ycd,thetad = sol.y

# q = sol.y.T
# xc,yc,theta,xcd,ycd,thetad = q[:,0], q[:,1], q[:,2], q[:,3], q[:,4], q[:,5]

plt.figure(1)
for k in range(0, len(t), 5):

    rc = np.array([xc[k],yc[k]])
    vc = np.array([xcd[k],ycd[k]])

    rp1 = rc + R(theta[k]).dot(p1)
    rp2 = rc + R(theta[k]).dot(p2)
    rp3 = rc + R(theta[k]).dot(p3)
    rp4 = rc + R(theta[k]).dot(p4)

    plt.plot([rp1[0],rp2[0]],[rp1[1],rp2[1]], 'b-')
    plt.plot([rp2[0],rp3[0]],[rp2[1],rp3[1]], 'r-')
    plt.plot([rp3[0],rp4[0]],[rp3[1],rp4[1]], 'g-')
    plt.plot([rp4[0],rp1[0]],[rp4[1],rp1[1]], 'k-')

    plt.title(f"time : {t[k]:.2f} s")
    plt.axis('equal')
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.pause(0.05)
    plt.clf()

plt.show()