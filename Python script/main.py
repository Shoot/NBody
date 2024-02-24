import scipy as sci
import numpy as np
from paint import paint
G = 6.67408e-11
m_nd = 1.989e+30  # kg
r_nd = 5.326e+12  # m
v_nd = 30000  # m/s
t_nd = 79.91 * 365.25 * 24 * 3600  # s
K1 = G * t_nd * m_nd / (r_nd ** 2 * v_nd)
K2 = v_nd * t_nd / r_nd

m1 = 1.1  # 1.m
m2 = 0.907  # 2.m
m3 = 1.425  # 3.m
m4 = 2.425  # 4.m

r1 = [-0.5, 1, 0]  # 1.xyz
r2 = [0.5, 0, 0.5]  # 2.xyz
r3 = [0.2, 1, 1.5]  # 3.xyz
r4 = [0, 0, 0]  # 4.xyz

# центр масс = (m1 * r1 + m2 * r2 + m3 * r3 + m4 * r4) / (m1 + m2 + m3 + m4)

v1 = [0.02, 0.02, 0.02]  # 1.v.xyz
v2 = [-0.05, 0, -0.1]  # 2.v.xyz
v3 = [0, -0.03, 0] # 3.v.xyz
v4 = [0, 0, 0] # 4.v.xyz

r1 = np.array(r1)
r2 = np.array(r2)
r3 = np.array(r3)
r4 = np.array(r4)
v1 = np.array(v1)
v2 = np.array(v2)
v3 = np.array(v3)
v4 = np.array(v4)

# скорость центра масс = (m1 * v1 + m2 * v2 + m3 * v3 + m4 * v4) / (m1 + m2 + m3 + m4)

def FourBodyEquations(w, t, G, m1, m2, m3):
    r1_ = w[:3]
    r2_ = w[3:6]
    r3_ = w[6:9]
    r4_ = w[9:12]
    v1_ = w[12:15]
    v2_ = w[15:18]
    v3_ = w[18:21]
    v4_ = w[21:24]

    # Find out distances between the three bodies
    r12 = sci.linalg.norm(r2_ - r1_)
    r13 = sci.linalg.norm(r3_ - r1_)
    r14 = sci.linalg.norm(r4_ - r1_)
    r23 = sci.linalg.norm(r3_ - r2_)
    r24 = sci.linalg.norm(r4_ - r2_)
    r34 = sci.linalg.norm(r4_ - r3_)

    # Define the derivatives according to the equations
    dv1bydt = K1 * m2 * (r2_ - r1_) / r12 ** 3 + K1 * m3 * (r3_ - r1_) / r13 ** 3 + K1 * m4 * (r4_ - r1_) / r14 ** 3
    dv2bydt = K1 * m1 * (r1_ - r2_) / r12 ** 3 + K1 * m3 * (r3_ - r2_) / r23 ** 3 + K1 * m4 * (r4_ - r2_) / r24 ** 3
    dv3bydt = K1 * m1 * (r1_ - r3_) / r13 ** 3 + K1 * m2 * (r2_ - r3_) / r23 ** 3 + K1 * m4 * (r4_ - r3_) / r34 ** 3
    dv4bydt = K1 * m1 * (r1_ - r4_) / r14 ** 3 + K1 * m2 * (r2_ - r4_) / r24 ** 3 + K1 * m3 * (r3_ - r4_) / r34 ** 3
    dr1bydt = K2 * v1_
    dr2bydt = K2 * v2_
    dr3bydt = K2 * v3_
    dr4bydt = K2 * v4_

    # Package the derivatives into one final size-24 array
    r12_derivs = np.concatenate((dr1bydt, dr2bydt))
    r123_derivs = np.concatenate((r12_derivs, dr3bydt))
    r_derivs = np.concatenate((r123_derivs, dr4bydt))
    v12_derivs = np.concatenate((dv1bydt, dv2bydt))
    v123_derivs = np.concatenate((v12_derivs, dv3bydt))
    v_derivs = np.concatenate((v123_derivs, dv4bydt))
    derivs = np.concatenate((r_derivs, v_derivs))
    return derivs

init_params = np.array([r1, r2, r3, r4, v1, v2, v3, v4])  # Package initial parameters into one size-24 array
init_params = init_params.flatten()  # Flatten the array to make it 1D
time_span = np.linspace(0, 20, 1000)  # Time span is 20 orbital years and 1000 points

import scipy.integrate
three_body_sol = sci.integrate.odeint(FourBodyEquations, init_params, time_span, args=(G, m1, m2, m3))

r1_sol = three_body_sol[:, :3]
r2_sol = three_body_sol[:, 3:6]
r3_sol = three_body_sol[:, 6:9]
r4_sol = three_body_sol[:, 9:12]
print(r1_sol, end = '\n\n')
print(r2_sol, end = '\n\n')
print(r3_sol, end = '\n\n')
print(r4_sol, end = '\n\n')

paint(r1_sol, r2_sol, r3_sol, r4_sol)