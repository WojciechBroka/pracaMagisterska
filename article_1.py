import sympy as sm
import numpy as np
import scipy as sc
import math as m
import matplotlib.pyplot as plt

# a_ship, b_ship = sm.symbols("a_ship b_ship")
# Cu, Cv, Cr, Vw = sm.symbols("Cu Cv Cr Vw")
# d1, d2, d3, d3_ = sm.symbols("d1 d2 d3 d3_")
# k1, k2, k3, a = sm.symbols("k1 k2 k3 a")
# k1_, k2_, k3_, k4_, k5_ = sm.symbols("k1_ k2_ k3_ k4_ k5_")
# variables
# u, v, r = sm.symbols("u v r")
# U, V, Z1, Z2, Z3 = sm.symbols("U V Z1 Z2 Z3")
# x, y, z1, z2, z3 = sm.symbols("x y z1 z2 z3")
# theta = sm.symbols("theta")
# m, m3 = sm.symbols("m m3")
# tau_u, tau_r = sm.symbols("tau_u tau_r")
# x_dot, y_dot = sm.symbols("x_dot y_dot") #Blimp vel
# x, y = sm.symbols("x y") #Blimp position
# Tp, time = sm.symbols("Tp time") #Sample time and simulation time

# constant variables
a_ship = 1.2  # ship length
b_ship = 0.5  # ship width
Cu = 0.42
Cv = 0.42
Cr = 0.42
Vw = 0.1  # wind velocity
d1_ceil = 0.01
d1_floor = 0.008
#d1 = 0.009
d2_ceil = 0.029
d2_floor = 0.035
#d2 = 0.045
d3_ceil = 0.050
d3_floor = 0.035
d3 = 0.042
k1 = 0.52
k2 = 0.50
k3 = 0.35
a = 0.15
k1_ = 1.6#1.6
k2_ = 0.13
k3_ = 1.0
k4_ = 0.16
k5_ = 0.0032
theta_zero = 0.0
r_zero = 0.0
m12 = 2
m3 = 0.05
Tp = 0.33
time = 25
coeff = 10.5

#
K = np.array([[k1_,k2_,k3_,k4_,k5_]])
k_r = k1 + k2 - d3_floor
k_th = k1*k2
theta3_floor = a/((k3 - k1)*(k3 - k2))
theta3_ceil = a/(k3**2 - (d3_ceil + k_r)*k3 + k1*k2)
delta_1 = (d1_ceil - d1_floor)*0.5
delta_2 = (d2_ceil - d2_floor)*0.5
delta_3 = (theta3_ceil - theta3_floor)*0.5
Delta_1 = coeff*delta_1
Delta_2 = coeff*delta_2
Delta_3 = coeff*delta_3
d10 = (d1_ceil + d1_floor)*0.5
d20 = (d2_ceil + d2_floor)*0.5
theta30 = (theta3_ceil + theta3_floor)*0.5
d1 = d10 + Delta_1
d2 = d20 + Delta_2
#theta_3 = theta30 + Delta_3
Delta_d = d3 - d3_floor
lambda_1 = 0.5*((Delta_d + k1 + k2) + m.sqrt(Delta_d**2 + 2*Delta_d*(k1 + k2) + (k1 - k2)**2))
lambda_2 = 0.5*((Delta_d + k1 + k2) - m.sqrt(Delta_d**2 + 2*Delta_d*(k1 + k2) + (k1 - k2)**2))
theta_1 = ((lambda_2*theta_zero + r_zero)/(lambda_2 - lambda_1)) + a/((lambda_2 - lambda_1)*(k3 - lambda_1))
theta_2 = ((k1*theta_zero + r_zero)/(lambda_1 - lambda_2)) + a/((lambda_1 - lambda_2)*(k3 - lambda_2))
theta_3 = a/((lambda_1 - k3)*(lambda_2 - k3))
b = np.array([[1/m12,0,0,0,0]]).T
A10 = np.array([[-d10,0,0,0,0],[theta30*k3,-(d20 - k3),0,0,0],[1,0,0,0,0],[0,1,theta30*k3,k3,0],[0,0,1,0,0]])
D = np.array([[1,0,0,0],[0,1,1,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
E = np.array([[-delta_1,0,0,0,0],[0,-delta_2,0,0,0],[k3*delta_3,0,0,0,0],[0,0,k3*delta_3,0,0]])
DELTA = np.diag([Delta_1/delta_1,Delta_2/delta_2,Delta_3/delta_3,Delta_3/delta_3])

def r(t_):
    return -theta_1*lambda_1*m.exp(-lambda_1*t_) - theta_2*lambda_2*m.exp(-lambda_2*t_) - theta_3*k3*m.exp(-k3*t_)

def theta(t_):
    return theta_1 *m.exp(-lambda_1*t_) + theta_2*m.exp(-lambda_2*t_) + theta_3*m.exp(-k3*t_)

def alpha(t_):
    return -(theta_1*lambda_1*m.exp(-(lambda_1 - k3)*t_) + theta_2*lambda_2*m.exp(-(lambda_2 - k3)*t_))

def A2(t_):
    return np.array([[0,r(t_)*m.exp(-k3*t_),0,0,0],[alpha(t_),0,0,0,0],[0,0,0,r(t_)*m.exp(-k3*t_),0],[0,0,alpha(t_),0,0],[0,0,0,0,0]])

def F(theta_):
    return m.sqrt(a_ship ** 2 * m.sin(theta_) ** 2 + b_ship ** 2 * m.cos(theta_) ** 2)

def fu(theta_):
    return Cu * Vw ** 2 * F(theta_) * m.cos(theta_)

def fv(theta_):
    return Cv * Vw ** 2 * F(theta_) * m.sin(theta_)

def fr(theta_):
    return Cr * Vw ** 2 * F(theta_) * m.sin(2 * theta_)

def f(theta_):
    return np.array([[fu(theta_)/m12,fv(theta_)/m12,0,0,0]]).T

def f_dis(t_):
    one = Cu*Vw**2*F(theta(t_))*m.cos(theta(t_))/m12
    two = -Cv*Vw**2*m.exp(-k3*t_)*F(theta(t_))*m.sin(theta(t_))/m12
    return np.array([[one,two,0,0,0]]).T

def CalcBlimpVel(u_, v_, theta_):
    x_dot_ = u_*np.cos(theta_) - v_*np.sin(theta_)
    y_dot_ = u_*np.sin(theta_) + v_*np.cos(theta_)
    return np.array([[x_dot_, y_dot_]]).T

def CalcZ1Z2(x_, y_, theta_):
    xc = x_*m.cos(theta_) + y_*m.sin(theta_)
    yc = -x_*m.sin(theta_) - y_*m.cos(theta_)
    return np.array([[xc,yc]]).T

def DynamicSystem2(r_, rSter, fr_):
    tmp1 = -d3*r_ + (rSter+fr_)/m3
    tmp2 = r_
    return np.array([[tmp1, tmp2]]).T

def DynamicSystem1e(q_, t_):
    return (A10 - b*K + D@DELTA@E + A2(t_))@q_ + f_dis(t_)

def Controler_r(r_, theta_, t_):
    return m3*(-k_r*r_ - k_th*theta_ + a*m.exp(-k3*t_) - fr(theta_))

def Controler_u(u_, v_, z1_, z2_, z3_, t_):
    return -(k1_*u_ + k2_*v_*m.exp(-k3*t_) + k3_*z1_ + k4_*z2_*m.exp(-k3*t_) + k5_*z3_)

def VelSat(x):
    if x > 0.5:
        return 0.5
    elif x < -0.5:
        return -0.5
    else:
        return x

t_actual = 0

#results containers
t_history = []
blimpVelX_history = []
blimpVelY_history = []
blimpPosX_history = []
blimpPosY_history = []
u_u_history = []
u_r_history = []
u_history = []
v_history = []
r_history = []
theta_history = []

#conditions
u_tmp = 0.0
v_tmp = 0.0
r_tmp = 0.0
theta_tmp = theta_zero
z1_tmp = 0
z2_tmp = 0
z3_tmp = 0
x_dot_tmp = 0
y_dot_tmp = 0
u_dot_last = 0
v_dot_last = 0
r_dot_last = 0
theta_dot_last = 0
z1_last = 0
z2_last = 0
z3_last = 0
x_dot_last = 0
y_dot_last = 0
xb = 0
yb = 0.4

while t_actual < time:
    # sim first system
    X = np.array([[u_tmp, v_tmp * m.exp(-k3 * t_actual), z1_tmp, z2_tmp * m.exp(-k3 * t_actual), z3_tmp]]).T
    #u_u = Controler_u(u_tmp, v_tmp * m.exp(k3 * t_actual), z1_tmp, z2_tmp * m.exp(k3 * t_actual), z3_tmp, t_actual)
    u_u = 0
    X_dot = DynamicSystem1e(X, t_actual)
    # sim second system
    u_r = Controler_r(r_tmp, theta_tmp, t_actual)
    x_dot = DynamicSystem2(r_tmp, u_r, fr(theta_tmp))
    #
    u_dot = X_dot[0,0]
    v_dot = X_dot[1,0]
    z1_dot = X_dot[2,0]
    z2_dot = X_dot[3,0]
    z3_dot = X_dot[4,0]
    r_dot = x_dot[0,0]
    theta_dot = x_dot[1,0]

       # integration
   #  u_tmp = 0.5 * Tp * (u_dot + u_dot_last) + u_tmp
   #  v_tmp = 0.5 * Tp * (v_dot + v_dot_last) + v_tmp
   #  r_tmp = 0.5 * Tp * (r_dot + r_dot_last) + r_tmp
   #  theta_tmp = 0.5 * Tp * (theta_dot + theta_dot_last) + theta_tmp
   #  z1_tmp = 0.5 * Tp * (z1_dot + z1_last) + z1_tmp
   #  z2_tmp = 0.5 * Tp * (z2_dot + z2_last) + z2_tmp
   #  z3_tmp = 0.5 * Tp * (z3_dot + z3_last) + z3_tmp

    u_tmp = Tp * u_dot + u_tmp
    v_tmp = Tp * v_dot + v_tmp
    r_tmp = Tp * r_dot + r_tmp
    theta_tmp = Tp * theta_dot + theta_tmp
    z1_tmp = Tp * z1_dot + z1_tmp
    z2_tmp = Tp * z2_dot + z2_tmp
    z3_tmp = Tp * z3_dot + z3_tmp

    u_tmp = VelSat(u_tmp)
    v_tmp = VelSat(v_tmp)
    r_dot = VelSat(r_dot)

    #Calculate blimp velocity and position in earth fixed frame
    VelBlimp = CalcBlimpVel(u_tmp, v_tmp, theta_tmp)
    x_dot_tmp = VelBlimp[0,0]
    y_dot_tmp = VelBlimp[1,0]
    # xb = 0.5*Tp*(x_dot_tmp + x_dot_last) + xb
    # yb = 0.5 * Tp * (y_dot_tmp + y_dot_last) + yb
    xb = Tp * x_dot_tmp + xb
    yb = Tp * y_dot_tmp + yb
    # get results
    t_history.append(t_actual)
    r_history.append(r_tmp)
    u_u_history.append(u_u)
    u_r_history.append(u_r)
    u_history.append(u_tmp)
    v_history.append(v_tmp)
    theta_history.append(theta_tmp)
    blimpVelX_history.append(x_dot_tmp)
    blimpVelY_history.append(y_dot_tmp)
    blimpPosX_history.append(xb)
    blimpPosY_history.append(yb)

    # change lasts
    u_dot_last = u_tmp
    v_dot_last = v_tmp
    r_dot_last = r_tmp
    theta_dot_last = theta_tmp
    z1_last = z1_tmp
    z2_last = z2_tmp
    z3_last = z3_tmp
    x_dot_last = x_dot_tmp
    y_dot_last = y_dot_tmp
    t_actual = t_actual + Tp
#end of while loop

# c=np.array([[1,2,0,0,0]]).T
# print(DynamicSystem1e(c,1.0,1.01))
# print(CalcZ1Z2(1,1,0.5))
# t_history
# blimpVel_history
# blimpPos_history
# u_u_history
# u_r_history
# u_history
# v_history
# r_history
# theta_history

plt.plot(t_history,r_history)
plt.xlabel('czas [s]')
plt.show()

plt.plot(t_history,t_history,v_history,u_history)
plt.xlabel('czas [s]')
plt.show()

plt.plot(t_history,t_history,blimpPosX_history,blimpPosY_history)
plt.xlabel('czas [s]')
plt.show()

plt.plot(blimpPosX_history,blimpPosY_history)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()
