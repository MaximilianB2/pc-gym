
import numpy as np
from casadi import*
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
      self.T = 10
      self.N = 6
      self.model_info = self.env.model.info()
    def model_gen(self):
      '''
      Generates a model for the given environment.

      Returns:
      f: A casadi function that can be used to solve the differential equations defined by the model.
      '''
      self.u  = MX.sym('u',self.env.Nu)
      
      self.x = MX.sym('x',self.env.Nx)
      dxdt = self.env.model(self.x,self.u)
      dxdt = vertcat(*dxdt)
      f = Function('f',[self.x,self.u],[dxdt],['x','u'],['dxdt'])
      return f
    
    def integrator_gen(self):
      '''
      Generates an integrator object for the given model.

      Returns:
      F: A casadi function that can be used to integrate the model over a given time horizon.
      '''
      f = self.model_gen()
      intg_options = {'tf': T/N, 'simplify': True, 'number_of_finite_elements': 4}
      dae = {'x': self.x, 'p': self.u, 'ode': f(self.x, self.u)}
      intg = integrator('intg', 'rk', dae, intg_options)
      res = intg(x0=self.x, p=self.u)
      x_next = res['xf']
      F = Function('F',[self.x,self.u],[x_next],['x','u'],['x_next'])
      return F
    def disturbance_index(self):
      
      '''
      Generates the indices of when the disturbance value changes.
      
      Inputs: self

      Returns: index of when the disturbance value changes.
      
      '''
      for key in self.env_params['disturbances']:
        disturbance = self.env_params['disturbances'][key]
        index = []
        for i in range(disturbance.shape[0]-1):
          if disturbance[i] != disturbance[i+1]:
            index.append(i+1)
        return index
       
    def ocp(self,t_step):
      """
      Solves an optimal control problem (OCP) using the IPOPT solver.

      Returns:
      - M: A function that takes current state x_0 (p) and returns the optimal control input u.

      """
      opti = Opti()
      F = self.integrator_gen()
      x = opti.variable(self.env.Nx, self.N+1)
      # Define the control variable

  
      u = opti.variable(self.env.Nu, self.N)
           
      p = opti.parameter(self.env.Nx, 1)
     
      
      setpoint = opti.parameter(1, self.N+1)
      cost = sum1(sum2((x[0,:] - setpoint)**2)) #TO DO: Remember to change this when custom rewards are implemented
      opti.minimize(cost)

      for k in range(self.N):
        opti.subject_to(x[:, k+1] == F(x[:, k], u[:, k]))

      opti.subject_to(u[0,:] >= self.env_params['a_space']['low'])
      opti.subject_to(u[0,:] <= self.env_params['a_space']['high'])
      
      if self.env_params['disturbances'] is not None:
        for i, k in enumerate(self.env.model.info()['disturbances'], start=0):
          if k in self.env.disturbances.keys():
            print(self.env.disturbances[k][t_step])
            opti.subject_to(u[self.env.Nu-len(self.env.model.info()['disturbances'])+i,:] == self.env.disturbances[k][t_step]) # Add disturbance to control vector
            opti.set_initial(u[self.env.Nu-len(self.env.model.info()['disturbances'])+i,:], self.env.disturbances[k][t_step])
          else:
            opti.subject_to(u[self.env.Nu-len(self.env.model.info()['disturbances'])+i,:] == self.model_info['parameters'][str(k)]) # if there is no disturbance at this timestep, use the default value
            opti.set_initial(u[self.env.Nu-len(self.env.model.info()['disturbances'])+i,:], self.model_info['parameters'][str(k)])
       
      opti.subject_to(x[:, 0] == p )
      # TODO: Add user-defined constraints  

      opti.solver('ipopt')
      setpoint_value = np.array([[0.9]*(self.N+1)])  # TODO: Make agnostic to model & setpoint variation
      
      opti.set_value(setpoint, setpoint_value)
      opti.set_value(p, [0.8,330]) # TODO: Make agnostic to model
      initial_x_values = np.zeros((2, self.N+1))
      initial_x_values[0, :] = 0.8 # TODO: Make agnostic to model
      initial_x_values[1, :] = 330 # TODO: Make agnostic to model
      opti.set_initial(x, initial_x_values)

     
     
      opti.set_initial(u[0,:], 297) # TODO: Make agnostic to model
     

      opts = {
        'ipopt': {
          'print_level': 0,
        },
        'print_time': 0
      }

    
      opti.solver('ipopt', opts)
      M = opti.to_function('M', [p], [u[:,1]], ['p'], ['u'])
      return M
    
    def mpc(self):
      '''
      Solves a model predictive control problem (MPC) using the optimal control problem (OCP) solver.

      Returns:
      - x_opt: Optimal state trajectory
      - u_opt: Optimal control trajectory
      '''
      regen_index = self.disturbance_index()
      print(regen_index)
      M = self.ocp(t_step=0)
      F = self.integrator_gen()
      self.ts = 100
      u_log = np.zeros((self.env.Nu,self.ts))
      x_log = np.zeros((self.env.Nx,self.ts))
      x = np.array(self.env_params['x0'])
      for i in range(self.ts):
          # print(i)
          if i in regen_index:
            M = self.ocp(t_step=i)
            # print('reformulating ocp')
          
          try:
              x_log[:,i] = x
          except:
              x_log[:,i] = x.reshape(-1)
          if self.env_params['noise'] is True:
            noise_percentage = self.env_params['noise_percentage']
            try:
              x += np.random.normal(0,1,(self.env.Nx)) * x * noise_percentage
            except:
              x += np.random.normal(0,1,(self.env.Nx,1)) * x * noise_percentage
          u = M(x).full()
          #print(u)
          u_log[:,i] = u[0]
          x = F(x,u).full() 
      return x_log, u_log
