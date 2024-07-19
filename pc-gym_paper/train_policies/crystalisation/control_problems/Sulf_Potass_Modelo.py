from casadi import *
import sys


# Valores dos parâmetros usados
ka = 0.923714966
kb = -6754.878558
kc = 0.92229965554
kd = 1.341205945
kg = 48.07514464
k1 = -4921.261419
k2 = 1.871281405
a = 0.50523693
b = 7.271241375
alfa = 7.510905767
ro = 2.658  # [roc] = g/cm^3

#Valores de entrada para mu0, mu1, mu2, mu3 e concentração
mu0 = MX.sym('Mu0', 1)
mu1 = MX.sym('Mu1', 1)
mu2 = MX.sym('Mu2', 1)
mu3 = MX.sym('Mu3', 1)
conc = MX.sym('C', 1)

X = vertcat(mu0, mu1, mu2, mu3, conc)
C = vertcat(mu1/mu0, (mu2*mu0/(mu1**2) - 1)**0.5)      # vetor das variáveis controladas

# Saídas do modelo
Y = vertcat(mu0, mu1, mu2, mu3, conc)

# Entradas do modelo
T = MX.sym('T', 1)
U = vertcat(T)

Ceq = -686.2686 + 3.579165*(T + 273.15) - 0.00292874*(T + 273.15)**2 #g/L

S = conc*1e3 - Ceq #g/L

B0 = ka*np.exp(kb/(T + 273.15))*(S**2)**(kc/2)*((mu3)**2)**(kd/2) # # /(cm^3*min)

Ginf = kg*np.exp(k1/(T + 273.15))*(S**2)**(k2/2) # [G] = [Ginf] = cm / min

dmi0dt = B0
dmi1dt = Ginf*(a*mu0 + b*mu1*1e-4)*1e4
dmi2dt = 2*Ginf*(a*mu1*1e-4 + b*mu2*1e-8)*1e8
dmi3dt = 3*Ginf*(a*mu2*1e-8 + b*mu3*1e-12)*1e12
dcdt = -0.5*ro*alfa*Ginf*(a*mu2*1e-8 + b*mu3*1e-12)

dx = vertcat(dmi0dt, dmi1dt, dmi2dt, dmi3dt, dcdt)

#cv = (mu2*mu0/(mu1**2) - 1)**0.5

J = -(mu1/mu0 + mu3/mu2 + conc)

dt = 1
