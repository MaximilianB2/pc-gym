import numpy as np

'''
This file contains the case studies models, action normalisation and default disturbance values.
'''
#Define the action normalisation for each case study
action_norms = {  'cstr_ode':{'high': 303, 'low': 297},
                 }
#Default values for disturbance variables
default_values = {  'cstr_ode':{'0': 350, '1': 1}, }

#Add default values for other case studies here
                    # 'first_order_system_ode':,
                    # 'second_order_system_ode': ,
                    # 'large_scale_ode': ,
                    # 'cstr_series_recycle_ode': ,
                    # 'cstr_series_recycle_two_ode': ,
                    # 'distillation_ode': ,
                    # 'multistage_extraction_ode': ,
                    # 'multistage_extraction_reactive_ode': ,
                    # 'heat_ex_ode': ,
                    # 'biofilm_reactor_ode': ,
                    # 'polymerisation_ode': ,
                    # 'four_tank_ode': 



# Define the case studies

def cstr_ode(x,u):
    #defines cstr ode model 

    ###Parameters### (May want to pass these externally)
    q = 100 #m3/s
    V = 100 #m3
    rho = 1000 #kg/m3
    C = 0.239 #Joules/kg K
    deltaHr = -5e4 #Joules/kg K
    EA_over_R = 8750 #K 
    k0 = 7.2e10 #1/sec
    UA = 5e4 # W/K
    Ti = 350 #K
    caf = 1

    ###Model equations###
    ca,T = x[0],x[1]
    
    if u.shape == (1,1):
        Tc = u[0]
    else:
        Tc,Ti,caf = u[0],u[1],u[2]

  
    rA = k0 * np.exp(-EA_over_R/T)*ca
    dxdt = [
        q/V*(caf-ca) - rA,
        q/V*(Ti-T) + ((-deltaHr)*rA)*(1/(rho*C)) + UA*(Tc-T)*(1/(rho*C*V))]

    return dxdt

def first_order_system_ode(x,u):

    #Parameters
    K = 1
    tau = 0.5

    #Model Equations
    x = x[0]
    u = u[0]

    dxdt = [(K*u - x)*1/tau]

    return dxdt

def second_order_system_ode(x,u):

    #Parameters
    K = 1
    tau = 0.5
    zeta = 0.25

    #Model Equations 
    x, dxdt = x[0], x[1]

    dzdt = [
        dxdt,
        (K*u - x - 2*zeta*tau*dxdt) * (1/(tau**2))  
        ]

    return dzdt

def large_scale_ode(x, u):

    #Section 3.2 (Example 2) from https://www.sciencedirect.com/science/article/pii/S0098135404001784
    #This is a challenging control problem as the system can exhibit a severe snowball effect (Luyben, Tyr ́eus, & Luyben, 1999) if

    #Parameters
    rho = 1 #Liquid density
    alpha_1 = 90 #Volatility see: http://www.separationprocesses.com/Distillation/DT_Chp01-3.htm#:~:text=VLE%3A%20Relative%20Volatility&text=In%20order%20to%20separate%20a,is%20termed%20the%20relative%20volatility.
    k_1 = 0.0167 #Rate constant
    k_2 = 0.0167 #Rate constant
    A_R = 5 #Vessel area
    A_M = 10 #Vessel area
    A_B = 5 #Vessel area  
    x1_O = 1.00

    ###Model Equations###

    ##States##
    #H_R - Liquid level of reactor
    #x1_R - Molar liquid fraction of component 1 in reactor 
    #x2_R - Molar liquid fraction of component 2 in reactor 
    #x3_R - Molar liquid fraction of component 3 in reactor 
    #H_M - Liquid level of storage tank
    #x1_M - Molar liquid fraction of component 1 in storage tank 
    #x2_M - Molar liquid fraction of component 2 in storage tank
    #x3_M - Molar liquid fraction of component 3 in storage tank 
    #H_B - Liquid level of flash tank
    #x1_B - Molar liquid fraction of component 1 in bottoms
    #x2_B - Molar liquid fraction of component 2 in bottoms
    #x3_B - #Molar liquid fraction of component 3 in bottoms

    H_R, x1_R, x2_R, x3_R, H_M, x1_M, x2_M, x3_M, H_B, x1_B, x2_B, x3_B = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11]

    ##Inputs##
    #F_O - Reactor input flowrate
    #F_R - Reactor outlet flowrate
    #F_M - Storage tank outlet flowrate
    #B - Bottoms flowrate
    #D - Distillate flowrate

    F_O, F_R, F_M, B, D = u[0], u[1], u[2], u[3], u[4]

    #Calculate distillate composition (only component 1 and 2 are volatile and component 3 is not
    #while component 1 is 90 times more volatile than component 2)

    x1_D = ((x1_B * alpha_1) / (1 - x1_B + x1_B * alpha_1))
    x2_D = 1 - x1_D

    dxdt = [
        (1/(rho*A_R)) * (F_O + D - F_R),
        ((F_O*(x1_O - x1_R) + D*(x1_D - x1_R))/(rho*A_R*H_R)) - k_1 * x1_R,
        ((-F_O * x2_R + D * (x2_D - x2_R))/(rho*A_R*H_R)) + k_1 * x1_R - k_2 * x2_R,
        ((-x3_R*(F_O + D))/(rho*A_R*H_R)) + k_2 * x2_R,
        (1/(rho*A_M)) * (F_R - F_M),
        ((F_R)/(rho*A_M*H_M))*(x1_R - x1_M),
        ((F_R)/(rho*A_M*H_M))*(x2_R - x2_M),
        ((F_R)/(rho*A_M*H_M))*(x3_R - x3_M),
        (1/(rho*A_B))*(F_M - B - D),
        (1/(rho*A_B*H_B))*(F_M*(x1_M-x1_B) - D*(x1_D - x1_B)),
        (1/(rho*A_B*H_B))*(F_M*(x2_M-x2_B) - D*(x2_D - x2_B)),
        (1/(rho*A_B*H_B))*(F_M*(x3_M-x3_B) + D*(x3_B))
        ]

    return dxdt

def cstr_series_recycle_ode(x, u):
    #https://pubs.acs.org/doi/pdf/10.1021/jp9621585
    #both temporal and spatial instabilities far from equilibrium
    #Approximating the dynamics of the oscillatory BZ reaction
    #http://www.scholarpedia.org/article/Belousov-Zhabotinsky_reaction
    #http://www.scholarpedia.org/article/Belousov-Zhabotinsky_reaction
    #https://en.wikipedia.org/wiki/Belousov%E2%80%93Zhabotinsky_reaction

    #Parameters
    V1 = 400 
    V2 = 300 
    f = 1
    s = 77.27
    q = 8.375-6
    psi = 32.2
    XO = 1.0
    YO = 10.0
    ZO = 1.0
    R = 10

    ###Model Equations###

    ##States##
    #X1 - Reactant X in reactor 1 
    #Y1 - Reactant Y in reactor 1
    #Z1 - Reactant Z in reactor 1
    #X2 - Reactant X in reactor 2
    #Y2 - Reactant Y in reactor 2
    #Z2 - Reactant Z in reactor 2

    X1, Y1, Z1, X2, Y2, Z2 = x[0], x[1], x[2], x[3], x[4], x[5]

    ##Inputs##
    Q = u[0]

    tau_1 = V1/Q
    tau_2 = V2/Q

    dxdt = [
        ((XO - R*X1 + (R-1)*X2)/tau_1) + s*(Y1 - X1*Y1 + X1 - q*(X1**2)),
        ((YO - R*Y1 + (R-1)*Y2)/tau_1) + (1/s)*(Y1 - X1*Y1 + f*Z1),
        ((ZO - R*Z1 + (R-1)*Z2)/tau_1) + psi*(X1 - Z1),
        ((R*(X1 - X2))/tau_2) + s*(Y2 - X2*Y2 + X2 - q*(X2**2)),
        ((R*(Y1 - Y2))/tau_2) + (1/s)*(-Y2 - X2*Y2 + f*Z2),
        ((R*(Z1 - Z2))/tau_2) + psi*(X2 - Z2)
        ]

    return dxdt

def cstr_series_recycle_two_ode(x, u):
    #Use this instead for CSTR in series and recycle
    #Based on this paper: https://www.jstage.jst.go.jp/article/jcej1968/17/5/17_5_497/_pdf/-char/ja
    #Same challenges as cstr_series_recycle_ode: temporal and spatial instabilities far from equilibrium, bifurcation, chaos

    #Parameters
    C_O = 97.35 #mol/m3
    T_O = 298 #K
    V1 = 1e-3 #m3
    V2 = 2e-3 #m3
    U1A1 = 0.461 #kJ/s*K
    U2A2 = 0.732 #kJ/s*K
    rho = 1.05e3 #kg/m3
    cp = 3.766 #kJ/kg*K
    k = 3.118e5 #s-1
    E = 46.14 #kJ/mol
    deltaH = 58.41 #kJ/mol
    R = 8.3145e-3 #kJ/mol K

    ###Model Equations###

    ##States##
    #C1 - Reactant 1 in reactor 1 
    #T1 - Temp in reactor 1
    #C2 - Reactant 1 in reactor 2
    #T2 - Temp in reactor 2

    C1, T1, C2, T2 = x[0], x[1], x[2], x[3]

    ##Inputs##
    #F - Feed flowrate
    #L - Recyle flowrate
    #Tc1 - Cooling water temperature of reactor 1
    #Tc2 - Cooling water temperature of reactor 2
    F, L, Tc1, Tc2 = u[0], u[1], u[2], u[3]

    dxdt = [
        (C_O/V1)*F + (1/V1)*L*C2 - (1/V1) * (F+L) * C1 - k*C1*np.exp((-E/(R*T1))),
        (T_O/V1)*F + (1/V1)*L*T2 - ((U1A1)/(V1*rho*cp))*(T1 - Tc1) - (1/V1)*(F+L)*T1 + ((k*(-deltaH))/(rho*cp))*C1*np.exp((-E/(R*T1))),
        (1/V2) * (F+L) * (C1-C2) - k*C2*np.exp((-E/(R*T2))),
        (1/V2) * (F+L) * (T1-T2) - ((U2A2)/(V2*rho*cp))*(T2 - Tc2) + ((k*(-deltaH))/(rho*cp))*C2*np.exp((-E/(R*T2)))
        ]

    return dxdt


def distillation_ode(x, u):
    #Based on Ingham book Pg 519 (497)
    #7 plate distillation column with feedplate is at position 3 so 3 enriching plates above
    #and 3 stripping plates below

    #Parameters
    D = 100 #kmol/hr
    q = 1 #Feed quality (q=1 is saturated liquid)
    alpha = 5 #Relative volatility of more volatile component i.e. 10 times more volatile
    X_feed = 0.2
    M0 = Mb = M = 2000


    ###Model Equations###

    ##States##
    #X0 - Liquid phase mole fraction of reflux drum
    #Xi - Liquid phase mole fraction of plate i
    #Xf - Liquid phase mole fraction of feedplate
    #Xb - Liquid phase mole fraction of reboiler

    X0, X1, X2, X3, Xf, X4, X5, X6, Xb = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]

    ##Inputs##
    #R - Reflux ratio
    #F - Feed rate

    R, F = u[0], u[1]

    L = R*D
    V = (R+1)*D
    L_dash = L + q*F
    V_dash = V + (1-q)*F
    W = F - D
    Y1 = ((alpha * X1)/(1 + (alpha - 1)*X1))
    Y2 = ((alpha * X2)/(1 + (alpha - 1)*X2))
    Y3 = ((alpha * X3)/(1 + (alpha - 1)*X3))
    Yf = ((alpha * Xf)/(1 + (alpha - 1)*Xf))
    Y4 = ((alpha * X4)/(1 + (alpha - 1)*X4))
    Y5 = ((alpha * X5)/(1 + (alpha - 1)*X5))
    Y6 = ((alpha * X6)/(1 + (alpha - 1)*X6))
    Yb = ((alpha * Xb)/(1 + (alpha - 1)*Xb))


    dxdt = [
        (1/M0) * ((V * Y1) - (L+D)*X0), #reflux drum
        (1/M) * (L*(X0-X1) + V*(Y2-Y1)), #Plate 1
        (1/M) * (L*(X1-X2) + V*(Y3-Y2)), #Plate 2
        (1/M) * (L*(X2-X3) + V*(Yf-Y3)), #Plate 3
        (1/M) * (L*X3 - L_dash*Xf + V_dash*Y4 - V*Yf +F*X_feed), #Feed plate
        (1/M) * (L_dash*(Xf-X4) + V_dash*(Y5-Y4)), #Plate 4
        (1/M) * (L_dash*(X4-X5) + V_dash*(Y6-Y5)), #Plate 5
        (1/M) * (L_dash*(X5-X6) + V_dash*(Yb-Y6)), #Plate 6
        (1/Mb) * (L_dash*X6 - W*Xb - V_dash*Yb) #Reboiler 
        ]

    return dxdt

def multistage_extraction_ode(x, u):
    #Based on Ingham pg 471 (449)
    #Solute and solvent extraction in 5 stage countercurrent extraction process
    #Maybe also change to include backmixing as per pg 475 (453)

    #Parameters
    Vl = 5 #Liquid volume in each stage
    Vg = 5 #Gas volume in each stage
    m = 1 #Equilibrium constant [-]
    Kla = 5 #Mass transfer capacity constant 1/hr 
    eq_exponent = 2 #Change the nonlinearity of the equilibrium relationship
    X0 = 0.6 #Feed concentration of liquid
    Y6 = 0.05 #Feed conc of gas


    ###Model Equations###

    ##States##
    #Xn - Concentration of solute in liquid pase of stage n [kg/m3]
    #Yn - Concentration of solute in gas phase of stage n [kg/m3]

    X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]

    ##Inputs##
    #L - Liquid flowrate m3/hr
    #G - Gas flowrate m3/hr

    L, G = u[0], u[1]

    X1_eq = ((Y1**eq_exponent)/m)
    X2_eq = ((Y2**eq_exponent)/m)
    X3_eq = ((Y3**eq_exponent)/m)
    X4_eq = ((Y4**eq_exponent)/m)
    X5_eq = ((Y5**eq_exponent)/m)

    Q1 = Kla*(X1 - X1_eq)*Vl
    Q2 = Kla*(X2 - X2_eq)*Vl
    Q3 = Kla*(X3 - X3_eq)*Vl
    Q4 = Kla*(X4 - X4_eq)*Vl
    Q5 = Kla*(X5 - X5_eq)*Vl

    dxdt = [
        (1/Vl)*(L * (X0 - X1) - Q1),
        (1/Vg)*(G * (Y2 - Y1) + Q1),
        (1/Vl)*(L * (X1 - X2) - Q2),
        (1/Vg)*(G * (Y3 - Y2) + Q2),
        (1/Vl)*(L * (X2 - X3) - Q3),
        (1/Vg)*(G * (Y4 - Y3) + Q3),
        (1/Vl)*(L * (X3 - X4) - Q4),
        (1/Vg)*(G * (Y5 - Y4) + Q4),
        (1/Vl)*(L * (X4 - X5) - Q5),
        (1/Vg)*(G * (Y6 - Y5) + Q5), 
        ]

    return dxdt

def multistage_extraction_reactive_ode(x, u):
    #Based on Ingham pg 161 (139)
    #Solute and solvent extraction in 5 stage countercurrent extraction process with reaction A+B -->C
    #The reaction takes place between a solute A in the L-phase, which is transferred
    #to the G-phase by the process of mass transfer, where it then reacts with
    #a second component, B, to form an inert product, C, such that A, B and C are
    #all present in the G-phase.
    #Components B and C are both immiscible in phase L and remain in phase G

    #Parameters
    Vl = 5 #Liquid volume in each stage
    Vg = 5 #Gas volume in each stage
    m = 1 #Equilibrium constant [-]
    Kla = 0.01 #Mass transfer capacity constant 1/hr 
    k = 0.1 #reaction equilibrium constant 
    eq_exponent = 2 #Change the nonlinearity of the equilibrium relationship
    XA0 = 2.00 #Feed concentration of component A in liquid phase
    YA6 = 0.00 #Feed conc of component A in gas phase
    YB6 = 2.00 #Feed conc of component B in gas phase
    YC6 = 0.00 #Feed conc of component C in gas phase


    ###Model Equations###

    ##States##
    #XAn - Concentration of solute (A) in liquid pase of stage n [kg/m3]
    #YAn - Concentration of solute (A) in gas phase of stage n [kg/m3]
    #YBn - Concentration of B in gas phase of stage n [kg/m3]
    #YCn - Concentration of C in gas phase of stage n [kg/m3]

    XA1, YA1, YB1, YC1, XA2, YA2, YB2, YC2, XA3, YA3, YB3, YC3, XA4, YA4, YB4, YC4, XA5, YA5, YB5, YC5 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11]

    ##Inputs##
    #L - Liquid flowrate m3/hr
    #G - Gas flowrate m3/hr

    L, G = u[0], u[1]

    XA1_eq = ((YA1**eq_exponent)/m)
    XA2_eq = ((YA2**eq_exponent)/m)
    XA3_eq = ((YA3**eq_exponent)/m)
    XA4_eq = ((YA4**eq_exponent)/m)
    XA5_eq = ((YA5**eq_exponent)/m)

    Q1 = Kla*(XA1 - XA1_eq)*Vl
    Q2 = Kla*(XA2 - XA2_eq)*Vl
    Q3 = Kla*(XA3 - XA3_eq)*Vl
    Q4 = Kla*(XA4 - XA4_eq)*Vl
    Q5 = Kla*(XA5 - XA5_eq)*Vl

    r1 = k*YA1*YB1
    r2 = k*YA2*YB2
    r3 = k*YA3*YB3
    r4 = k*YA4*YB4
    r5 = k*YA5*YB5

    dxdt = [
        (1/Vl)*(L * (XA0 - XA1) - Q1),
        (1/Vg)*(G * (YA2 - YA1) + Q1 - r1*Vg),
        (1/Vg)*(G * (YB2 - YB1) - r1*Vg),
        (1/Vg)*(G * (YC2 - YC1) + r1*Vg),
        (1/Vl)*(L * (XA1 - XA2) - Q2),
        (1/Vg)*(G * (YA3 - YA2) + Q2 - r2*Vg),
        (1/Vg)*(G * (YB3 - YB2) - r2*Vg),
        (1/Vg)*(G * (YC3 - YC2) + r2*Vg),
        (1/Vl)*(L * (XA2 - XA3) - Q3),
        (1/Vg)*(G * (YA4 - YA3) + Q3 - r3*Vg),
        (1/Vg)*(G * (YB4 - YB3) - r3*Vg),
        (1/Vg)*(G * (YC4 - YC3) + r3*Vg),
        (1/Vl)*(L * (XA3 - XA4) - Q4),
        (1/Vg)*(G * (YA5 - YA4) + Q4 - r4*Vg),
        (1/Vg)*(G * (YB5 - YB4) - r4*Vg),
        (1/Vg)*(G * (YC5 - YC4) + r4*Vg),
        (1/Vl)*(L * (XA4 - XA5) - Q5),
        (1/Vg)*(G * (YA6 - YA5) + Q5 - r5*Vg),
        (1/Vg)*(G * (YB6 - YB5) - r5*Vg),
        (1/Vg)*(G * (YC6 - YC5) + r5*Vg),
        ]

    return dxdt


def heat_ex_ode(x, u):
    #Based on Ingham pg 533 (511)
    #Use finite difference method with 8 sections
    #Heating fluid on tube side and fluid to be heated on shell side


    #Parameters
    Utm = 1 #Tube-metal overall heat transfer coefficient [kW/m2 K]
    Usm = 1 #Shell-metal overall heat transfer coefficient [kW/m2 K]
    L = 1 #Length segment of each stage
    Dt = 1 #Internal diameter of tube wall [m]
    Dm = 2 #Outside diameter of metal wall [m]
    Ds = 3 #Shell wall diameter [m]
    cpt = 1 #Heat capacity of tube side fluid [kJ/kg K]
    cpm = 1 #Heat capacity of metal wall [kJ/kg K]
    cps = 1 #Heat capacity of shell side fluid [kJ/kg K]
    rhot = 1 #Densit of tube side fluid [kg/m3]
    rhom = 1 #Density of metal [kg/m3]
    rhos = 1 #Density of shell side fluid [kg/m3]




    ###Model Equations###

    ##States##
    #Ttn - Temperature of tube side fluid of section n [K]
    #Tmn - Temperature of the metal wall of section n [K]
    #Tsn - Temperature of the shell side fluid of section n [K]

    Tt1, Tm1, Ts1, Tt2, Tm2, Ts2, Tt3, Tm3, Ts3, Tt4, Tm4, Ts4, Tt5, Tm5, Ts5, Tt6, Tm6, Ts6, Tt7, Tm7, Ts7, Tt8, Tm8, Ts8 = \
        x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19], x[20], x[21], x[22], x[23],

    ##Inputs##
    #Ft - Flowrate of tubeside fluid [m3/s]
    #Fs - Flowrate of shellside fluid [m3/s]
    #Ts9 - Inlet temperature of shellside fluid [K]
    #Tt0 - Inlet temperature of tubeside fluid [K]

    Ft, Fs, Tt0, Ts9 = u[0], u[1], u[2], u[3]

    Vt = L*np.pi*Dt**2
    At = L*np.pi*Dt

    Vm = L*np.pi*(Dm**2-Dt**2)
    Am = L*np.pi*Dm

    Vs = L*np.pi*(Ds**2-Dm**2)

    Qt1 = Utm * At * (Tt1 - Tm1)
    Qm1 = Usm * Am * (Tm1 - Ts1)
    Qt2 = Utm * At * (Tt2 - Tm2)
    Qm2 = Usm * Am * (Tm2 - Ts2)
    Qt3 = Utm * At * (Tt3 - Tm3)
    Qm3 = Usm * Am * (Tm3 - Ts3)
    Qt4 = Utm * At * (Tt4 - Tm4)
    Qm4 = Usm * Am * (Tm4 - Ts4)
    Qt5 = Utm * At * (Tt5 - Tm5)
    Qm5 = Usm * Am * (Tm5 - Ts5)
    Qt6 = Utm * At * (Tt6 - Tm6)
    Qm6 = Usm * Am * (Tm6 - Ts6)
    Qt7 = Utm * At * (Tt7 - Tm7)
    Qm7 = Usm * Am * (Tm7 - Ts7)
    Qt8 = Utm * At * (Tt8 - Tm8)
    Qm8 = Usm * Am * (Tm8 - Ts8)

    dxdt = [
        (1/(cpt*rhot*Vt)) * (Ft*cpt*(Tt0 - Tt1) - Qt1),
        (1/(cpm*rhom*Vm)) * (Qt1 - Qm1),
        (1/(cps*rhos*Vs)) * (Fs*cps*(Ts2 - Ts1) + Qm1),
        (1/(cpt*rhot*Vt)) * (Ft*cpt*(Tt1 - Tt2) - Qt2),
        (1/(cpm*rhom*Vm)) * (Qt2 - Qm2),
        (1/(cps*rhos*Vs)) * (Fs*cps*(Ts3 - Ts2) + Qm2),
        (1/(cpt*rhot*Vt)) * (Ft*cpt*(Tt2 - Tt3) - Qt3),
        (1/(cpm*rhom*Vm)) * (Qt3 - Qm3),
        (1/(cps*rhos*Vs)) * (Fs*cps*(Ts4 - Ts3) + Qm3),
        (1/(cpt*rhot*Vt)) * (Ft*cpt*(Tt3 - Tt4) - Qt4),
        (1/(cpm*rhom*Vm)) * (Qt4 - Qm4),
        (1/(cps*rhos*Vs)) * (Fs*cps*(Ts5 - Ts4) + Qm4),
        (1/(cpt*rhot*Vt)) * (Ft*cpt*(Tt4 - Tt5) - Qt5),
        (1/(cpm*rhom*Vm)) * (Qt5 - Qm5),
        (1/(cps*rhos*Vs)) * (Fs*cps*(Ts6 - Ts5) + Qm5),
        (1/(cpt*rhot*Vt)) * (Ft*cpt*(Tt5 - Tt6) - Qt6),
        (1/(cpm*rhom*Vm)) * (Qt6 - Qm6),
        (1/(cps*rhos*Vs)) * (Fs*cps*(Ts7 - Ts6) + Qm6),
        (1/(cpt*rhot*Vt)) * (Ft*cpt*(Tt6 - Tt7) - Qt7),
        (1/(cpm*rhom*Vm)) * (Qt7 - Qm7),
        (1/(cps*rhos*Vs)) * (Fs*cps*(Ts8 - Ts7) + Qm7),
        (1/(cpt*rhot*Vt)) * (Ft*cpt*(Tt7 - Tt8) - Qt8),
        (1/(cpm*rhom*Vm)) * (Qt8 - Qm8),
        (1/(cps*rhos*Vs)) * (Fs*cps*(Ts9 - Ts8) + Qm8),
        ]

    return dxdt


def biofilm_reactor_ode(x, u):
    #Based on Ingham pg 569 (547)
    #a fluidised biofilm sand bed reactor for nitrification, as investigated by Tanaka et al. (1981), 
    #is modelled as three tanks-in-series with a recycle loop (Fig. 1).
    #More info in pages
    #https://onlinelibrary.wiley.com/doi/epdf/10.1002/bit.260230803
    #Biofilm reactor split into three stages
    ###IMPORTANT: I think the equations in the book are wrong for the component balances
    #I think it should be, according to the reactions, -r1 for component 1 which gets conumed
    #then +r1 -r2 for component 2 which gets prodced in reaction 1 but consumed in reaction 2
    #then +r2 for component 3 which gets produced in reaction 2
    ###Now that I think about it, I do think the book is right, they just mean for e.g. that rs2_1 = +r1_1 - r2_1
    #Which is quite nice notation to use for the paper


    #Parameters
    V = 1 #Volume of one reactor stage [L]
    Va = 1 #Volume of absorber tank [L]
    Kla = 1 #Transfer coefficient [hr]
    m = 0.5 #Equilibrium constant [-]
    eq_exponent = 1
    O_air = 1 #Concentration of oxygen in air [mg/L] https://www.quora.com/What-is-the-number-of-moles-of-oxygen-in-one-litre-of-air-containing-21-of-oxygen-by-volume-under-standard-conditions
    vm_1 = 1 #Maximum velocity through fluidized bed for reaction 1 [mg/L hr]
    vm_2 = 1 #Maximum velocity through fluidized bed for reaction 2[mg/L hr]
    K1 = 1 #Equilibrium constant for reaction 1 (Saturation constant for ammonia in reaction 1) [mg/L]
    K2 = 1 #Equilibrium constant for reaction 2 (Saturation constant for ammonia in reaction 2) [mg/L]
    KO_1 = 1 #Equilibrium constant for oxygen in reaction 1 (saturation constant for oxygen) [mg/L]
    KO_2 = 1 # Equilibrium consrant for oxygne in reactikn 2 (saturation constant for oxygen) [mg/L]


    ###Model Equations###

    ##States##
    #Si_n - Nitrogen compound i (i = 1 (Ammonia), 2 (Nitrite), 3 (Nitrate)) in stage n of fluidized bed reactor [mg/L]
    #O_n - Dissolved oxygen in stage n of fluidized bed reactor [mg/L]
    #Si_A - Nitrogen compound i in Absorber, A, [mg/L]
    #O_A - Dissolved oxygen in Absorber, A, [mg/L]
    S1_1, S2_1, S3_1, O_1, S1_2, S2_2, S3_2, O_2, S1_3, S2_3, S3_3, O_3, S1_A, S2_A, S3_A, O_A = \
        x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15]

    ##Inputs##
    #F - Feed and effluent flow rate [L/hr]
    #Fr - Recycle flow rate [L/hr]
    #Si_F - Nitrogen  compound i in feed stream, F, to absorber
    F, Fr, S1_F, S2_F, S3_F = u[0], u[1], u[2], u[3], u[4]

    #Reaction rates and stoichiometry 
    r1_1 = ((vm_1 * S1_1)/(K1 + S1_1)) * ((O_1)/(KO_1 + O_1))
    r2_1 = ((vm_2 * S2_1)/(K2 + S2_1)) * ((O_1)/(KO_2 + O_1))
    ro_1 = -r1_1 * 3.5 - r2_1 *1.1

    r1_2 = ((vm_1 * S1_2)/(K1 + S1_2)) * ((O_2)/(KO_1 + O_2))
    r2_2 = ((vm_2 * S2_2)/(K2 + S2_2)) * ((O_2)/(KO_2 + O_2))
    ro_2 = -r1_2 * 3.5 - r2_2 *1.1

    r1_3 = ((vm_1 * S1_3)/(K1 + S1_3)) * ((O_3)/(KO_1 + O_3))
    r2_3 = ((vm_2 * S2_3)/(K2 + S2_3)) * ((O_3)/(KO_2 + O_3))
    ro_3 = -r1_3 * 3.5 - r2_3 *1.1

    #TODO:correct this
    rs1_1 = -r1_1
    rs2_1 = +r1_1 -r2_1
    rs3_1 = r2_1

    rs1_2 = -r1_2
    rs2_2 = +r1_2 -r2_2
    rs3_2 = r2_2

    rs1_3 = -r1_3
    rs2_3 = +r1_3 -r2_3
    rs3_3 = r2_3

    O_Aeq = ((O_air**eq_exponent)/m) #What should this equation be? I think its the oxygen dissolved in the water in equilibrium with air so use henrys law?

    dxdt = [
        (Fr/V) * (S1_A - S1_1) - rs1_1,
        (Fr/V) * (S2_A - S2_1) - rs2_1,
        (Fr/V) * (S3_A - S3_1) - rs3_1,
        (Fr/V) * (O_A - O_1) - ro_1,
        (Fr/V) * (S1_1 - S1_2) - rs1_2,
        (Fr/V) * (S2_1 - S2_2) - rs2_2,
        (Fr/V) * (S3_1 - S3_2) - rs3_2,
        (Fr/V) * (O_1 - O_2) - ro_2,
        (Fr/V) * (S1_2 - S1_3) - rs1_3,
        (Fr/V) * (S2_2 - S2_3) - rs2_3,
        (Fr/V) * (S3_2 - S3_3) - rs3_3,
        (Fr/V) * (O_2 - O_3) - ro_3,
        (Fr/Va) * (S1_3 - S1_A) + (F/Va) * (S1_F - S1_A),
        (Fr/Va) * (S2_3 - S2_A) + (F/Va) * (S2_F - S2_A),
        (Fr/Va) * (S3_3 - S3_A) + (F/Va) * (S3_F - S3_A),
        (Fr/Va) * (O_3 - O_A) + Kla * (O_Aeq - O_A)
        ]

    return dxdt


def polymerisation_ode(x, u):
    #We are using a simplified model as per Ingham pg 332 (310) but 
    #based on https://reader.elsevier.com/reader/sd/pii/0009250981801741?token=FB9B6D3024D079EF7764FE6C1D6DD504F88AC920171DAC207B160DE945D789A9F8255D9466C6709844990B5902DE52D5&originRegion=eu-west-1&originCreation=20220926101218
    #and the reference in Ingham and the main reason for interest is that its typically used as a benchmark problem for
    #large scale systems: https://reader.elsevier.com/reader/sd/pii/S0098135414001938?token=686A3316355AB1A4C0BA7325C0B8F0FA4A5FF45AAA4E0485CC7AFA9EAAFA26A76D0DD7D8BA11D0D46861E7524A800905&originRegion=eu-west-1&originCreation=20220926101053
    #due to the possible chain length
    #From paper: considering the number of the chain length, there are 7385 process states. Therefore, there are a total of 7385 state functions in the process dynamic model.
    #We just calculate average chain length
    #Values and rate constants are from this paper:
    #https://pubs.acs.org/doi/pdf/10.1021/i260023a018?casa_token=D3FEgtOnOigAAAAA:mgavEJEHPTcS-qCPxI0Nv0OPfvkT0dBMRsd_xh7s2HmAP3gxX0CGjAqKbzwO7Qj6Je3vhOjBClbgZ00f


    #The interest is in finding the effect of changes in the reactor inlet conditions on the degree of polymerisation obtained

    #Parameters
    Ap = 6e10 #Pre-exponential factor for step p [1/sec] - From 1967 paper
    Ad = 4e10 #Pre-exponential factor for step d [1/sec] - From 1967 paper
    At = 9e10 #Pre-exponential factor for step t [1/sec] - From 1967 paper
    Ep_over_R = 7750 #Activation energy over R for step p [K] - From 1967 paper (1/2 Et)
    Ed_over_R = 8500 #Activation energy over R for step d [K] - From 1967 paper (30kcal/mol)
    Et_over_R = 8250 #Activation energy over R for step t [K] - From 1967 paper (random)
    f = 0.5 #Reactivity fraction for free radicals [-]
    V = 1 #Reactor volume [m3]
    deltaHp = -3e4 #Heat of reaction per monomer unit [kJ/kmol] https://nvlpubs.nist.gov/nistpubs/jres/44/jresv44n3p221_A1b.pdf
    rho = 1200 #Density of input fluid mixture [kg/m3]
    cp = 2 #Heat capacity of fluid mixture [kj/kg K]

    ###Model Equations###

    ##States##
    #T - Temperature of reactor [K]
    #M - Monomer concentration [kmol/m3]
    #I - Initiator concetration [kmol/m3]
    #X - Degree of polymerisation or average chain length [-]
    #We are ignorning the solvent 

    T, M, I = x[0], x[1], x[2]

    ##Inputs##
    #F - Feed flowrate [m3/s]
    #Tf - Feed temperature [K]
    #Mf - Monomer feed concetration [kmol/m3]
    #If - Initiator feed concetration [kmol/m3]

    F, Tf, Mf, If = u[0], u[1], u[2], u[3]


    kp = Ap * np.exp(-Ep_over_R/T)
    kd = Ad * np.exp(-Ed_over_R/T)
    kt = At * np.exp(-Et_over_R/T)

    ri = 2*f*kd*I
    rp = kp*((f*kd*I)/(kt))**0.5 #Based on assumption rt = rp - see page 333 and assuming Arrhenius temperature dependence of the rate constants, see pg 4 of original 1967 paper (equation 11)

    #X = ((kp*M)/(f*kd*kt*I))**0.5 #See definition in book and this is the basis of the system being large scale see comment above and see in original 1967 paper (equation 12)


    dxdt = [
        (F/V)*(Tf - T) + ((-deltaHp)/(rho*cp)) * rp,
        (F/V)*(Mf - M) - rp,
        (F/V)*(If - I) - ri
        ]

    return dxdt


def four_tank_ode(x, u):
    #From: https://people.kth.se/~kallej/papers/quadtank_acc98.pdf
    #and https://reader.elsevier.com/reader/sd/pii/S0959152406001387?token=9D60090E495B278D7ADDE8460B6915DE8486B836754DF82E8ACC7E42ECBDF2A2E7A5A3AD00E3771E49C9FAA110DE3885&originRegion=eu-west-1&originCreation=20220927075107

    #Assume all tanks are 1m3 - 1 m2 cross section and 1m height

    #Parameters
    g = 9.81 #Acceleration due to gravity [m/s2]
    gamma_1 = 0.2 #Fraction bypassed by valve to tank 1 [-]
    gamma_2 = 0.2 #Fraction bypassed by valve to tank 2 [-]
    k1 = 0.00085 #1st pump gain [m3/Volts S]
    k2 = 0.00095 #2nd pump gain [m3/Volts S]
    a1 = 0.0035 #Cross sectional area of outlet of tank 1 [m2]
    a2 = 0.0030 #Cross sectional area of outlet of tank 2 [m2]
    a3 = 0.0020 #Cross sectional area of outlet of tank 3 [m2]
    a4 = 0.0025 #Cross sectional area of outlet of tank 4 [m2]
    A1 = 1 #Cross sectional area of tank 1 [m2]
    A2 = 1 #Cross sectional area of tank 2 [m2]
    A3 = 1 #Cross sectional area of tank 3 [m2]
    A4 = 1 #Cross sectional area of tank 4 [m2]

    ###Model Equations###

    ##States##
    #h1 - Height of level in tank 1 [m]
    #h2 - Height of level in tank 2 [m]
    #h3 - Height of level in tank 3 [m]
    #h4 - Height of level in tank 4 [m]

    h1, h2, h3, h4 = x[0], x[1], x[2], x[3]

    ##Inputs##
    #v1 - Voltage applied to pump 1 [V]
    #v2 - Voltage applied to pump 2 [V]

    v1, v2 = u[0], u[1]


    dxdt = [
        (-a1/A1)*np.sqrt(2*g*h1) + (a3/A1)*np.sqrt(2*g*h3) + ((gamma_1*k1)/(A1))*v1,
        (-a2/A2)*np.sqrt(2*g*h2) + (a4/A2)*np.sqrt(2*g*h4) + ((gamma_2*k2)/(A2))*v2,
        (-a3/A3)*np.sqrt(2*g*h3) + (((1-gamma_2)*k2)/(A3))*v2,
        (-a4/A4)*np.sqrt(2*g*h4) + (((1-gamma_1)*k1)/(A4))*v1,
        ]

    return dxdt