import math
import numpy as np
import pylab as py
from matplotlib import animation



def init():
    line1.set_data([], [])
    line2.set_data([], [])
    title_text.set_text('')
    return line1,line2,title_text


def force_es(r_):
    F = np.zeros(2)
    Fmag = GG*Me*Ms/(np.linalg.norm(r_)+1e-20)**2
    theta = math.atan(np.abs(r_[1])/(np.abs(r_[0])+1e-20))
    F[0] = Fmag * np.cos(theta)
    F[1] = Fmag * np.sin(theta)
    if r_[0] > 0:
        F[0] = -F[0]
    if r_[1] > 0:
        F[1] = -F[1]
    return F

def force_js(r_):
    F = np.zeros(2)
    Fmag = GG*Mj*Ms/(np.linalg.norm(r_)+1e-20)**2
    theta = math.atan(np.abs(r_[1])/(np.abs(r_[0])+1e-20))
    F[0] = Fmag * np.cos(theta)
    F[1] = Fmag * np.sin(theta)
    if r_[0] > 0:
        F[0] = -F[0]
    if r_[1] > 0:
        F[1] = -F[1]
    return F

def force_ej(re_,rj_):
    r_ = np.zeros(2)
    F = np.zeros(2)
    r_[0] = re_[0] - rj_[0]
    r_[1] = re_[1] - rj_[1]
    Fmag = GG*Me*Mj/(np.linalg.norm(r_)+1e-20)**2
    theta = math.atan(np.abs(r_[1])/(np.abs(r_[0])+1e-20))
    F[0] = Fmag * np.cos(theta)
    F[1] = Fmag * np.sin(theta)
    if r_[0] > 0:
        F[0] = -F[0]
    if r_[1] > 0:
        F[1] = -F[1]
    return F


def force(r_,planet_,ro_,vo_):
    if planet_ == 'earth':
        return force_es(r_) + force_ej(r_,ro_)
    if planet_ == 'jupiter':
        return force_js(r_) - force_ej(r_,ro_)


def dr_dt(t_,r_,v_,planet_,ro_,vo_):
    return v_


def dv_dt(t_,r_,v_,planet_,ro_,vo_):
    F = force(r_,planet_,ro_,vo_)
    y = F/planet_.m
    return y

def RK4Solver(t_,r_,v_,h_,planet,ro,vo):
    k11 = dr_dt(t_,r_,v_,planet,ro,vo)
    k21 = dv_dt(t_,r_,v_,planet,ro,vo)

    k12 = dr_dt(t_ + 0.5*h_,r_ + 0.5*h_*k11,v_ + 0.5*h_*k21,planet,ro,vo)
    k22 = dv_dt(t_ + 0.5*h_,r_ + 0.5*h_*k11,v_ + 0.5*h_*k21,planet,ro,vo)

    k13 = dr_dt(t_ + 0.5*h_,r_ + 0.5*h_*k12,v_ + 0.5*h_*k22,planet,ro,vo)
    k23 = dv_dt(t_ + 0.5*h_,r_ + 0.5*h_*k12,v_ + 0.5*h_*k22,planet,ro,vo)

    k14 = dr_dt(t_ + h_,r_ + h_*k13,v_ + h_*k23,planet,ro,vo)
    k24 = dv_dt(t_ + h_,r_ + h_*k13,v_ + h_*k23,planet,ro,vo)

    y0 = r_ + h_ * (k11 + 2.*k12 + 2.*k13 + k14) / 6.
    y1 = v_ + h_ * (k21 + 2.*k22 + 2.*k23 + k24) / 6.

    z = [y0, y1]
    return z


def KineticEnergy(v_):
    vn = np.linalg.norm(v_)
    return 0.5*EARTH_MASS*vn**2

def PotentialEnergy(r_):
    fmag = np.linalg.norm(force_es(r_))
    rmag = np.linalg.norm(r_)
    return -fmag*rmag

def AngMomentum(r_,v_):
    rn = np.linalg.norm(r_)
    vn = np.linalg.norm(v_)
    r_ = r_/rn
    v_ = v_/vn
    rdotv = r_[0]*v_[0]+r_[1]*v_[1]
    theta = math.acos(rdotv)
    return EARTH_MASS*rn*vn*np.sin(theta)

_1 = Body(m=6e24) # Earth
_2 = Body(m=1.9e27) # Jupiter
_3 = Body(m=2e30) # Sun
G = 6.673e-11
AU_KM = 1.496e11
EARTH_MASS = 6e24
YEAR_SEC = 365*24*60*60.0
FF = (G*EARTH_MASS**2)/AU_KM**2          # Unit force
EE = FF*AU_KM                    # Unit energy
GG = (EARTH_MASS*G*YEAR_SEC**2)/(AU_KM**3)
initial_time = 0
final_time = 120
points = 100*final_time
t = np.linspace(initial_time,final_time,points)
time_step = t[2]-t[1] # time step (uniform)

AreaVal = np.zeros(points)
r = np.zeros([points,2])         # position vector of Earth
v = np.zeros([points,2])         # velocity vector of Earth
rj = np.zeros([points,2])        # position vector of Jupiter
vj = np.zeros([points,2])        # velocity vector of Jupiter
ri = [1496e8/AU_KM,0]          # initial position of earth
rji = [5.2,0]               # initial position of Jupiter




vv = np.sqrt(Ms*GG/ri[0])         # Magnitude of Earth's initial velocity

vvj = 13.06e3 * YEAR_SEC/AU_KM             # Magnitude of Jupiter's initial velocity

vi = [0, vv*1.0]                  # Initial velocity vector for Earth.Taken to be along y direction as ri is on x-axis.
vji = [0, vvj*1.0]                # Initial velocity vector for Jupiter




t[0] = initial_time
r[0,:] = ri
v[0,:] = vi
rj[0,:] = rji
vj[0,:] = vji



for i in range(0,points-1):
    r[i+1,:],v[i+1,:]=RK4Solver(t[i],r[i,:],v[i,:],time_step,'earth',rj[i,:],vj[i,:])
    rj[i+1,:],vj[i+1,:]=RK4Solver(t[i],rj[i,:],vj[i,:],time_step,'jupiter',r[i,:],v[i,:])



def animate(i_):
    earth_trail = 200
    jupiter_trail = 200
    title_text.set_text(f'Прошло {t[i_] : .2} лет')
    line1.set_data(r[i_:max(1,i_-earth_trail):-1,0], r[i_:max(1,i_-earth_trail):-1,1])
    line2.set_data(rj[i_:max(1,i_-jupiter_trail):-1,0], rj[i_:max(1,i_-jupiter_trail):-1,1])
    return line1,line2



fig, ax = py.subplots()
ax.axis('square')
ax.set_xlim(( -7.2, 7.2)) # !
ax.set_ylim((-7.2, 7.2)) # !

ax.plot(0, 0, 'o', markersize = 9, markerfacecolor = "#FDB813", markeredgecolor ="#FD7813")
line1, = ax.plot([], [], 'o-',color = '#d2eeff', markersize = 8, markerfacecolor = '#0077BE',lw=2, markevery=10000)   # line for Earth
line2, = ax.plot([], [], 'o-',color = '#e3dccb', markersize = 8, markerfacecolor = '#f66338',lw=2,markevery=10000)   # line for Jupiter


ax.plot([-6,-5],[6.5,6.5],'r-')
ax.text(-4.5,6.3,r'1 Астр Ед = $1.496 \times 10^8$ км')

ax.plot(-6,-6.2,'o', color = '#d2eeff', markerfacecolor = '#0077BE')
ax.text(-5.5,-6.4,'Земля')

ax.plot(-3.3,-6.2,'o', color = '#e3dccb',markersize = 8, markerfacecolor = '#f66338')
ax.text(-2.9,-6.4,'Супер Юпитер (500x масса)')

ax.plot(5,-6.2,'o', markersize = 9, markerfacecolor = "#FDB813",markeredgecolor ="#FD7813")
ax.text(5.5,-6.4,'Солнышко')
title_text = ax.text(0.24, 1.05, '', transform = ax.transAxes, va='center')

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=10, interval=5, blit=True)

anim.save('orbit.mp4', fps=30,dpi = 500, extra_args=['-vcodec', 'libx264'])
