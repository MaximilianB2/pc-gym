import numpy as np
from casadi import *
import matplotlib.pyplot as plt



class integration_engine():
    '''
    Integration class
    Contains both the casadi and JAX integration wrappers.

    Inputs: Environment, x0, dt,u_t

    Output: x+  
    '''

    def __init__(self,Models_env,env_params):
        self.env = Models_env(env_params)
        integration_method = env_params['integration_method']
        
        assert integration_method in ['jax', 'casadi'], "integration_method must be either 'jax' or 'casadi'"
        
        if integration_method == 'casadi':
            # Generate casadi model
            self.sym_x = self.gen_casadi_variable(self.env.Nx, "x")
            self.sym_u = self.gen_casadi_variable(self.env.Nu, "u")    
            self.casadi_sym_model = self.casadify(self.env.model, self.sym_x, self.sym_u)
            self.casadi_model_func = self.gen_casadi_function([self.sym_x, self.sym_u],[self.casadi_sym_model],
                                                            "model_func", ["x","u"], ["model_rhs"])
            
        if integration_method == 'jax':
            pass

        

    def casadi_step(self,state,uk):

        '''
        Integrate one time step with casadi.

        input: x0, uk
        output: x+
        '''
        plant_func = self.casadi_model_func
        discretised_plant = self.discretise_model(plant_func, self.env.dt)
      
        xk = state[:self.env.Nx]
        Fk = discretised_plant(x0=xk, p=uk)
        return Fk

    def casadify(self, model, sym_x, sym_u):
        """
        Given a model with Nx states and Nu inputs and returns rhs of ode,
        return casadi symbolic model (Not function!)
        
        Inputs:
            model - model to be casidified i.e. a list of ode rhs of size Nx
            
        Outputs:
            dxdt - casadi symbolic model of size Nx of rhs of ode
        """

        dxdt = model(sym_x, sym_u)
        dxdt = vertcat(*dxdt) #Return casadi list of size Nx

        return dxdt



    def gen_casadi_variable(self, n_dim, name = "x"):
        """
        Generates casadi symbolic variable given n_dim and name for variable
        
        Inputs:
            n_dim - symbolic variable dimension
            name - name for symbolic variable
            
        Outputs:
            var - symbolic version of variable
        """

        var = SX.sym(name, n_dim)

        return var

    def gen_casadi_function(self, casadi_input, casadi_output, name, input_name=[], output_name=[]):
        """
        Generates a casadi function which maps inputs (casadi symbolic inputs) to outputs (casadi symbolic outputs)
        
        Inputs:
            casadi_input - list of casadi symbolics constituting inputs
            casadi_output - list of casadi symbolic output of function
            name - name of function
            input_name - list of names for each input
            output_name - list of names for each output
        
        Outputs:
            casadi function mapping [inputs] -> [outputs]
        
        """

        function = Function(name, casadi_input, casadi_output, input_name, output_name)

        return function
    
    def discretise_model(self, casadi_func, delta_t):
        """
        Input:
            casadi_func to be discretised
        
        Output:
            discretised casadi func
        """
        x = SX.sym("x", self.env.Nx)
        u = SX.sym("u", self.env.Nu)
        xdot = casadi_func(x, u)

        dae = {'x':x, 'p':u, 'ode':xdot} 
        t0 = 0
        tf = delta_t
        discrete_model = integrator('discrete_model', 'cvodes', dae,t0,tf)

        return discrete_model
    