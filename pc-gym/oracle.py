
import numpy as np
from casadi import SX, vertcat, nlpsol, inf, Function
from Integrator import integration_engine

class oracle():
    '''
    Oracle Class - Class to solve the optimal control problem with perfect 
    knowledge of the environment.

    Inputs: Env

    Outputs: Optimal control and state trajectories
    '''

    def __init__(self,env,env_params):  
      self.env = env(env_params)
      self.env_params = env_params
      
    
       


       