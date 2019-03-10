import numpy as np
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


g_0 = 9.80665  # gravitational acceleration
a = 340.294  # speed of sound in air
rho_0 = 1.225  # density of air
k = 1.4  # heat capacity ratio for dry air
R = 29.27  # gas constant (meter/degree)
rEarth = 6371.210  # Earth radius
T_0 = 288.15  # absolute temperature
gradTemp = (1/R) * (k - 1)/k  # temperature gradient

# 130 mm towed field gun M1954 (M-46)
# projectile 53-БР-482(Б), weight 33.4 kg, start speed 930 m/s;
d = 0.130  # caliber in meters
h = 0.21515  # length of projectile length
q = 33.4  # wight in kg

# initial parameters
V_0 = 930  # start speed of projectile, m/s
x_0, y_0, z_0 = 0, 0, 0  # coordinates
theta_0 = 0.785398  # trajectory angle
psi_0 = 0  # direction angle

Q_0 = [V_0, x_0, y_0, theta_0, theta_0]
t = np.arange(0, 80, 1/8)


def yDelta(y):  # approximation of delta(y) = rho/rho_0
    yKm = y / 1000
    T = T_0 + gradTemp * yKm  # temperature on 'y' height
    return pow(T/T_0, -(1/(gradTemp * R)) + 1)



def density(y):  # density of air
    yKm = y / 1000
    T = T_0 + gradTemp * yKm
    return rho_0 * (T_0 / T) * math.exp(-math.log(R * T)/(R * gradTemp) + math.log(R * T_0)/(R * gradTemp))


def gravity(y):  # gravitational acceleration on 'y' height
    yKm = y / 1000
    return g_0 * (rEarth/(rEarth + yKm))


def formCoefficient(height, caliber):  # form factor approximation calculation
    c1 = 0.0049231
    c2 = -0.043823
    c3 = 0.097576
    c4 = 0.16973
    c5 = -1.0858
    c6 = 1.9086

    return c1 * pow(height / caliber, 5) + c2 * pow(height / caliber, 4) + c3 * pow(height / caliber, 3) + \
        c4 * pow(height / caliber, 2) + c5 * (height/caliber) + c6


i = formCoefficient(h, d)


def c_x(mach):  # Siachi function
    if mach < 0.73:
        return 0.157

    if mach < 0.82:
        return 0.033 * mach + 0.133

    if mach < 0.91:
        return 0.161 + 3.9 * pow(mach - 0.823, 2)

    if mach <= 1.:
        return 1.5 * mach - 1.176

    if mach <= 1.18:
        return 0.384 - 1.6 * pow(mach - 1.176, 2)

    if mach < 1.62:
        return 0.384 * math.sin(1.85 / mach)

    if mach < 3.06:
        return 0.29 / mach + 0.172

    else:
        return 0.301 - 0.11 * mach


def windage(v, y, g):
    M = v / a  # Mach number
    return  (i * pow(d, 2)) * (density(y) / rho_0) * pow(v, 2) * 4.8014 * c_x(M) * (1/(q * 10))


def f(t, Q):  # system of diff equations

    g = gravity(Q[2])
    Q_new = np.zeros(5)

    R = windage(Q[0], Q[2], g)

    Q_new[0] = -g * math.sin(Q[3]) - R
    Q_new[1] = Q[0] * math.cos(Q[3])
    Q_new[2] = Q[0] * math.sin(Q[3])
    Q_new[3] = (-g * math.cos(Q[3]))/Q[0]

    return Q_new


solution = solve_ivp(f, [0, 80], Q_0, t_eval=t)  # solve system

plt.plot(solution.y[1], solution.y[2])  # plotting

plt.show()
