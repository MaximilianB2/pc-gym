import numpy as np
import scipy.integrate as scp
from scipy.integrate import odeint 
import gym
from gym import spaces, logger


class ReactorTest(gym.Env):
    '''
    Simple CSTR Control Problem to test the Open AI gym interface.
    
    Observation: 
        Type: Continuous 
        0: Outlet concentration of A 
        1: Reactor Temperature 
    
    Action:
        Type: Continuous
        0: Reactor Jacket temperature
        
    Reward:
        ISE of the setpoint tracking
    
    Starting state:
        Ca = 0.87725294608097
        T  = 324.475443431599
        Tj = 295
        
    Episode Termination:
        User-determined simulation length
    
    '''
    def __init__(self):
        self.Tf  = 350
        
    def cstr(self,x,t,u,Tf,Caf):
        Tc = u # Temperature of cooling jacket (K)
        # Tf = Feed Temperature (K)
        # Caf = Feed Concentration (mol/m^3)

    
        Ca = x[0] # Concentration of A in CSTR (mol/m^3)
        T  = x[1] # Temperature in CSTR (K)

    
        q      = 100    # Volumetric Flowrate (m^3/sec)
        V      = 100    # Volume of CSTR (m^3)
        rho    = 1000   # Density of A-B Mixture (kg/m^3)
        Cp     = 0.239  # Heat capacity of A-B Mixture (J/kg-K)
        mdelH  = 5e4    # Heat of reaction for A->B (J/mol)
        EoverR = 8750   # E -Activation energy (J/mol), R -Constant = 8.31451 J/mol-K
        k0     = 7.2e10 # Pre-exponential factor (1/sec)
        UA     = 5e4    # U -Heat Transfer Coefficient (W/m^2-K) A -Area - (m^2)
        rA     = k0*np.exp(-EoverR/T)*Ca # reaction rate
        dCadt  = q/V*(Caf - Ca) - rA     # Calculate concentration derivative
        dTdt   = q/V*(Tf - T) \
                + mdelH/(rho*Cp)*rA \
                + UA/V/rho/Cp*(Tc-T)   # Calculate temperature derivative

        # == Return xdot == #
        xdot    = np.zeros(2)
        xdot[0] = dCadt
        xdot[1] = dTdt
        return xdot
    
    def _step(self,action):
       
        next_state = odeint(cstr,self.x0,self.ts,args=(action,self.Tf,self.Caf))
        reward = 1
        done = False 
        info = None
        return next_state, reward, done,info
    
    
    def _reset(self):
        pass
    
    def step(self, action):
        return self._step(action)
    

    

       