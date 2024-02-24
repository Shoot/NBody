import math
import numpy as np
import pylab as py
from matplotlib import animation


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    title_text.set_text('')
    return line1, line2, title_text


def force_es(r_):
    F = np.zeros(2)
    Fmag = GG * Me * Ms / (np.linalg.norm(r_) + 1e-20) ** 2
    theta = math.atan(np.abs(r_[1]) / (np.abs(r_[0]) + 1e-20))
    F[0] = Fmag * np.cos(theta)
    F[1] = Fmag * np.sin(theta)
    if r_[0] > 0:
        F[0] = -F[0]
    if r_[1] > 0:
        F[1] = -F[1]
    return F


def force_js(r_):
    F = np.zeros(2)
    Fmag = GG * Mj * Ms / (np.linalg.norm(r_) + 1e-20) ** 2
    theta = math.atan(np.abs(r_[1]) / (np.abs(r_[0]) + 1e-20))
    F[0] = Fmag * np.cos(theta)
    F[1] = Fmag * np.sin(theta)
    if r_[0] > 0:
        F[0] = -F[0]
    if r_[1] > 0:
        F[1] = -F[1]
    return F


def force_ej(re_, rj_):
    r_ = np.zeros(2)
    F = np.zeros(2)
    r_[0] = re_[0] - rj_[0]
    r_[1] = re_[1] - rj_[1]
    Fmag = GG * Me * Mj / (np.linalg.norm(r_) + 1e-20) ** 2
    theta = math.atan(np.abs(r_[1]) / (np.abs(r_[0]) + 1e-20))
    F[0] = Fmag * np.cos(theta)
    F[1] = Fmag * np.sin(theta)
    if r_[0] > 0:
        F[0] = -F[0]
    if r_[1] > 0:
        F[1] = -F[1]
    return F


def force(r_, planet_, ro_, vo_):
    if planet_ == 'earth':
        return force_es(r_) + force_ej(r_, ro_)
    if planet_ == 'jupiter':
        return force_js(r_) - force_ej(r_, ro_)


def dr_dt(t_, r_, v_, planet_, ro_, vo_):
    return v_


def dv_dt(t_, r_, v_, planet_, ro_, vo_):
    F = force(r_, planet_, ro_, vo_)
    if planet_ == 'earth':
        y = F / Me
    if planet_ == 'jupiter':
        y = F / Mj
    return y


# Differential equation solvers
# ===================================================================
def RK4Solver(t_, r_, v_, h_, planet, ro, vo):
    k11 = dr_dt(t_, r_, v_, planet, ro, vo)
    k21 = dv_dt(t_, r_, v_, planet, ro, vo)

    k12 = dr_dt(t_ + 0.5 * h_, r_ + 0.5 * h_ * k11, v_ + 0.5 * h_ * k21, planet, ro, vo)
    k22 = dv_dt(t_ + 0.5 * h_, r_ + 0.5 * h_ * k11, v_ + 0.5 * h_ * k21, planet, ro, vo)

    k13 = dr_dt(t_ + 0.5 * h_, r_ + 0.5 * h_ * k12, v_ + 0.5 * h_ * k22, planet, ro, vo)
    k23 = dv_dt(t_ + 0.5 * h_, r_ + 0.5 * h_ * k12, v_ + 0.5 * h_ * k22, planet, ro, vo)

    k14 = dr_dt(t_ + h_, r_ + h_ * k13, v_ + h_ * k23, planet, ro, vo)
    k24 = dv_dt(t_ + h_, r_ + h_ * k13, v_ + h_ * k23, planet, ro, vo)

    y0 = r_ + h_ * (k11 + 2. * k12 + 2. * k13 + k14) / 6.
    y1 = v_ + h_ * (k21 + 2. * k22 + 2. * k23 + k24) / 6.

    z = [y0, y1]
    return z
# =====================================================================

Me = 6e24  # Mass of Earth in kg
Ms = 2e30  # Mass of Sun in kg
Mj = 1.9e27  # Mass of Jupiter

G = 6.673e-11  # Gravitational Constant

RR = 1.496e11  # Normalizing distance in km (= 1 AU)
MM = 6e24  # Normalizing mass
TT = 365 * 24 * 60 * 60.0  # Normalizing time (1 year)

FF = (G * MM ** 2) / RR ** 2  # Unit force
EE = FF * RR  # Unit energy

GG = (MM * G * TT ** 2) / (RR ** 3)

Me = Me / MM  # Normalized mass of Earth
Ms = Ms / MM  # Normalized mass of Sun
Mj = 500 * Mj / MM  # Normalized mass of Jupiter/Super Jupiter

ti = 0  # initial time = 0
tf = 120  # final time = 120 years

N = 100 * tf  # 100 points per year
t = np.linspace(ti, tf, N)  # time array from ti to tf with N points

h = t[2] - t[1]  # time step (uniform)

# Initialization
r = np.zeros([N, 2])  # position vector of Earth
v = np.zeros([N, 2])  # velocity vector of Earth
rj = np.zeros([N, 2])  # position vector of Jupiter
vj = np.zeros([N, 2])  # velocity vector of Jupiter
rs = np.zeros([N, 2])  # position vector of Jupiter
vs = np.zeros([N, 2])  # velocity vector of Jupiter

ri = [1496e8 / RR, 0]  # initial position of earth
rji = [5.2, 0]  # initial position of Jupiter

vv = np.sqrt(Ms * GG / ri[0])  # Magnitude of Earth's initial velocity

vvj = 13.06e3 * TT / RR  # Magnitude of Jupiter's initial velocity

vi = [0, vv * 1.0]  # Initial velocity vector for Earth.Taken to be along y direction as ri is on x-axis.
vji = [0, vvj * 1.0]  # Initial velocity vector for Jupiter

t[0] = ti
r[0, :] = ri
v[0, :] = vi
rj[0, :] = rji
vj[0, :] = vji

AreaVal[0] = 0

for i in range(0, N - 1):
    [r[i + 1, :], v[i + 1, :]] = RK4Solver(t[i], r[i, :], v[i, :], h, 'earth', rj[i, :], vj[i, :])
    [rj[i + 1, :], vj[i + 1, :]] = RK4Solver(t[i], rj[i, :], vj[i, :], h, 'jupiter', r[i, :], v[i, :])


def animate(i_):
    earth_trail = 500
    jupiter_trail = 500
    title_text.set_text(f'Прошло {round(float(t[i_]), 2)} лет')
    line1.set_data(r[i_:max(1, i_ - earth_trail):-1, 0], r[i_:max(1, i_ - earth_trail):-1, 1])
    line2.set_data(rj[i_:max(1, i_ - jupiter_trail):-1, 0], rj[i_:max(1, i_ - jupiter_trail):-1, 1])
    return line1, line2

fig, ax = py.subplots()
ax.axis('square')
ax.set_xlim((-10, 10)); ax.set_ylim((-10, 10))
title_text = ax.text(0.24, 1.05, '', transform=ax.transAxes, va='center')
ax.plot(0, 0, 'o', markersize=5, markerfacecolor="#FDB813", markeredgecolor="#FD7813") # 1
line1, = ax.plot([], [], 'o-', color='#d2eeff', markersize=5, markerfacecolor='#0077BE', lw=2,
                 markevery=10000)  # 2
line2, = ax.plot([], [], 'o-', color='#e3dccb', markersize=5, markerfacecolor='#f66338', lw=2,
                 markevery=10000)  # 3
SEC = 1
SPEED = 100
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=int(SPEED*SEC), interval=5, blit=True)
anim.save('orbit.mp4', fps=SPEED, dpi=500, extra_args=['-vcodec', 'libx264'])