from Casadi_NMPC_Crist import *
import matplotlib.pyplot as plt
import time
import math
import numpy as np
import time

# Modelo

from Sulf_Potass_Modelo import *

plant = ODEModel(dt=dt, x=X, u=U, dx=dx,
                  J=J, y=Y)
plant.get_equations(intg='idas')

## Otimizador

opts = {
    'warn_initial_bounds': False, 'print_time': False,
    'ipopt': {'print_level': 1}
}


## Condições iniciais

Mu0guess = 1478
Mu1guess = 22996
Mu2guess = 1800863
Mu3guess = 248516167
Cguess = 0.159
Tguess = 40

xguess = [Mu0guess, Mu1guess, Mu2guess, Mu3guess, Cguess]
uguess = [Tguess]

## Fronteiras - Condição dos estados

lbMu0 = 0  # 0.1
ubMu0 = 1e20
lbMu1 = 0  # 0.1
ubMu1 = 1e20
lbMu2 = 0
ubMu2 = 1e20
lbMu3 = 0
ubMu3 = 1e20
lbC = 0
ubC = 0.5

lbx = [lbMu0, lbMu1, lbMu2, lbMu3, lbC]
ubx = [ubMu0, ubMu1, ubMu2, ubMu3, ubC]

## Fronteiras - Condição das variáveis manipuladas

lbT = 0
upT = 40

lbu = [lbT]
ubu = [upT]

## NMPC

N = 5
M = 1
Q = np.diag([10, 55])
W = np.diag([0.002])
lbdT = -1
ubdT = 1
lbdu = [lbdT]
ubdu = [ubdT]

nmpc = NMPC_SS(dt=dt, N=N, M=M, Q=Q, W=W, x=X, u=U, c=C, dx=dx,
               xguess=xguess, uguess=uguess, lbx=lbx, ubx=ubx, lbu=lbu,
               ubu=ubu, lbdu=lbdu, ubdu=ubdu, opts={})

# Inicialização
t = 1  # contador
tsim = 55  # horas
nopt = 11  # Tempo para cada rodada de otimização
niter = math.ceil(tsim / (dt * nopt))

xf = [1478.00986666666, 22995.8230590611, 1800863.24079725, 248516167.940593, 0.15861523304]
uf = [39.86]

xhat = copy.deepcopy(xf)

sp_L = 11
sp_CV = 1.2

ysim1 = np.zeros([niter * nopt, 5])
usim1 = np.zeros([niter * nopt, 1])
spsim1 = np.zeros([niter * nopt, 2])

ysim = np.zeros([niter * nopt + 1, 5])
usim = np.zeros([niter * nopt + 1, 1])
spsim = np.zeros([niter * nopt + 1, 2])

ysim[0, :] = xf
usim[0, :] = uf
spsim[0, :] = np.array([sp_L, sp_CV])

cpu_time = []

start_time = time.time()

## Simulação

for ksim in range(0, niter):
    start = time.time()  # comp time
    n = ksim / niter
    opts = {
        'warn_initial_bounds': False, 'print_time': False,
        'ipopt': {'print_level': 1, 'constr_viol_tol': 1e-10,
                  'tol': 1e-10}
    }

    sp = np.array([sp_L, sp_CV]).reshape(2, 1)

    for i in range(0, nopt):
        # NMPC
        ctrl = nmpc.calc_control_actions(ksim=ksim + 1, x0=xhat, u0=uf, sp=sp,)
        uf = ctrl['U']

        # Dados da planta
        sim = plant.simulate_step(xf=xf, uf=uf)
        xf = sim['x'].ravel()
        ysim1[t - 1, :] = xf
        usim1[t - 1, :] = sim['u']
        spsim1[t - 1, :] = np.reshape(sp, [1, len(sp)])
        xhat = sim['x']

        ysim[t, :] = xf
        usim[t, :] = sim['u']
        spsim[t, :] = np.reshape(sp, [1, len(sp)])

        t += 1
    end = time.time()
    cpu_time += [end - start]

exec_time = time.time() - start_time

print(f'Tempo de execução = {exec_time} s')

avg_time = np.mean(cpu_time)  # avg time spent at each opt cycle

# Plot
time = np.linspace(5, tsim + 5, niter * nopt + 1)

print(len(time))
print(time)

# Plotagem dos resultados.
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.35)

ax[0].plot(time, usim[:, 0], color='black')
ax[0].set_xlabel("Time (min)", fontsize=15)
ax[0].set_ylabel("T", fontsize=15)

ax[1].plot(time, ysim[:, 1]/ysim[:, 0], color='black')
ax[1].plot(time, spsim[:, 0], linestyle='--')
ax[1].set_xlabel("Time (min)", fontsize=15)
ax[1].set_ylabel(r"L ${\mu}m$/#", fontsize=15)

plt.show()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.35)

ax[0].plot(time, usim[:, 0], color='black')
ax[0].set_xlabel("Time (min)", fontsize=15)
ax[0].set_ylabel("T", fontsize=15)

ax[1].plot(time, (ysim[:, 0]*ysim[:, 2]/ysim[:, 1]**2 - 1)**0.5, color='black')
ax[1].plot(time, spsim[:, 1], linestyle='--')
ax[1].set_xlabel("Time (min)", fontsize=15)
ax[1].set_ylabel('CV', fontsize=15)

plt.show()

print(usim)


resultados = np.zeros([len(time), 13])

for i in range(len(time)):
    resultados[i][0] = time[i]
    resultados[i][1] = usim[i, 0]
    resultados[i][2] = ysim[i, 0]
    resultados[i][3] = ysim[i, 1]
    resultados[i][4] = ysim[i, 2]
    resultados[i][5] = ysim[i, 3]
    resultados[i][6] = ysim[i, 4]
    resultados[i][7] = (-686.2686 + 3.579165 * (usim[i, 0] + 273.15) - 0.00292874 * (usim[i, 0] + 273.15) ** 2)*1e-3
    resultados[i][8] = ysim[i, 4]*1e3/(-686.2686 + 3.579165*(usim[i, 0] + 273.15) - 0.00292874*(usim[i, 0] + 273.15)**2)
    resultados[i][9] = ysim[i, 1]/ysim[i, 0]
    resultados[i][10] = spsim[i, 0]
    resultados[i][11] = (ysim[i, 0]*ysim[i, 2]/ysim[i, 1]**2 - 1)**0.5
    resultados[i][12] = spsim[i, 1]

# np.savetxt('MPC_PB_artigo_controle_CV_Comp_Lmaior.csv', resultados, delimiter=",")

