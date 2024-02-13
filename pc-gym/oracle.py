from casadi import *
import numpy as np

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
      self.x0 = env_params['x0']
      self.T = self.env.tsim
      self.N = 10
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
      tf = self.env.dt
      t0 = 0
      dae = {'x': self.x, 'p': self.u, 'ode': f(self.x, self.u)}
      opts = {'simplify': True, 'number_of_finite_elements': 4}
      intg = integrator('intg', 'rk', dae, t0,tf,opts)
      res = intg(x0=self.x, p=self.u)
      x_next = res['xf']
      F = Function('F',[self.x,self.u],[x_next],['x','u'],['x_next'])
      return F
    
    def disturbance_index(self):
      
      '''
      Generates the indices of when the disturbance or setpoint value changes.
      
      Inputs: self

      Returns: index of when either the disturbance or setpoint value changes.
      
      '''
      index = []
      if self.env_params.get('disturbances') is not None:
        for key in self.env_params['disturbances']:
          disturbance = self.env_params['disturbances'][key]
          for i in range(disturbance.shape[0]-1):
            if disturbance[i] != disturbance[i+1]:
              index.append(i+1)
      for key in self.env_params['SP']:
        SP = self.env_params['SP'][key]
        for i in range(len(SP)-1):
          if SP[i] != SP[i+1]:
            index.append(i+1)
      
      index = list(set(index))
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
     
      
      setpoint = opti.parameter(len(self.env_params['SP']), self.N+1)
      cost = 0
      for k in self.env_params['SP']:
        i  = self.model_info['states'].index(k)  
        cost += sum1(sum2((x[i,:] - setpoint[i,:])**2))*self.env_params['r_scale'][k] #TODO: Remember to change this when custom rewards are implemented
        

      opti.minimize(cost)

      for k in range(self.N):
        opti.subject_to(x[:, k+1] == F(x[:, k], u[:, k]))

      opti.subject_to(u[0,:] >= self.env_params['a_space']['low'])
      opti.subject_to(u[0,:] <= self.env_params['a_space']['high'])
      
      if self.env_params.get('disturbances') is not None:
        for i, k in enumerate(self.env.model.info()['disturbances'], start=0):
          if k in self.env.disturbances.keys():
            opti.subject_to(u[self.env.Nu-len(self.env.model.info()['disturbances'])+i,:] == self.env.disturbances[k][t_step]) # Add disturbance to control vector
            opti.set_initial(u[self.env.Nu-len(self.env.model.info()['disturbances'])+i,:], self.env.disturbances[k][t_step])
          else:
            opti.subject_to(u[self.env.Nu-len(self.env.model.info()['disturbances'])+i,:] == self.model_info['parameters'][str(k)]) # if there is no disturbance at this timestep, use the default value
            opti.set_initial(u[self.env.Nu-len(self.env.model.info()['disturbances'])+i,:], self.model_info['parameters'][str(k)])
       
      opti.subject_to(x[:, 0] == p )
      if self.env_params.get('constraints') is not None:
        for k in self.env_params['constraints']:
          if self.env_params['cons_type'][k] == '<=':
            opti.subject_to(x[self.model_info['states'].index(k),:] <= self.env_params['constraints'][k])
          elif self.env_params['cons_type'][k] == '>=':
            opti.subject_to(x[self.model_info['states'].index(k),:] >= self.env_params['constraints'][k])
          else:
            raise ValueError('Invalid constraint type')
      
      
      SP_i = np.fromiter({k: v[t_step] for k, v in self.env_params['SP'].items()}.values(),dtype=float)
      
     
      setpoint_value = SP_i*np.ones((self.N+1,1))
      
      opti.set_value(setpoint, setpoint_value.T)
      opti.set_value(p, self.x0[:self.env.Nx])
      initial_x_values = np.zeros((self.env.Nx, self.N+1))
      initial_x_values = (self.x0[:self.env.Nx]*np.ones((self.N+1,self.env.Nx))).T  

      opti.set_initial(x, initial_x_values)

     
     
      opti.set_initial(u[0,:], self.env_params['a_space']['low']) 
     

      opts = {'qpsol':{'print_level':0}}
      
      opts["print_time"] = 0
      opts["calc_lam_p"] = False
      opts["expand"] = False
      opts["compiler"] = "shell"
      opts["qpsol"] = "qpoases"
      opts["regularity_check"] = True
      opts["qpsol_options"] = {"printLevel":"none", "enableRegularisation":True}
      opts["max_iter"] = 10
      opts["print_iteration"] = False
      opts["print_header"] = False
      opts["verbose"] = False
      opts['print_iteration'] = False
      opti.solver('sqpmethod', opts)
      M = opti.to_function('M', [p], [u[:,1]], ['p'], ['u'])
      return M
    
    def mpc(self):
      '''
      Solves a model predictive control problem (MPC) using the optimal control problem (OCP) solver.

      Returns:
      - x_opt: Optimal state trajectory
      - u_opt: Optimal control trajectory
      '''
      print('Oracle running...')
      regen_index = self.disturbance_index()
      
      M = self.ocp(t_step=0)
      F = self.integrator_gen()
      
      u_log = np.zeros((self.env.Nu, self.env.N))
      x_log = np.zeros((self.env.Nx, self.env.N))
      x = np.array(self.x0[:self.env.Nx])
      for i in range(self.env.N):
        

          if i in regen_index:
            M = self.ocp(t_step=i)          
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
          u_log[:,i] = u[0]
          x = F(x,u).full() 
      return x_log, u_log

