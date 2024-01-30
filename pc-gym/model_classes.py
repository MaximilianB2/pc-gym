from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np


# Store the model and its parameters as dataclasses
# frozen: makes the objets immutable after creation
# so parameters can not be modified at runtime
# it also makes the class hashable, as required by Equinox:
# ValueError: Non-hashable static arguments are not supported.

# kw_only: require the parameter names if they want
# to be set when the object is created

@dataclass(frozen=False, kw_only=True)
class cstr_ode:
  # Parameters
  q:float = 100 #m3/s
  V:float = 100 #m3
  rho:float = 1000 #kg/m3
  C:float = 0.239 #Joules/kg K
  deltaHr:float = -5e4 #Joules/kg K
  EA_over_R:float = 8750 #K
  k0:float = 7.2e10 #1/sec
  UA:float = 5e4 # W/K
  Ti:float = 350 #K
  caf:float = 1
  int_method:str = 'jax'

     
  def __call__(self, x, u):
    # JAX requires jnp functions and arrays hence two versions
    if self.int_method == 'jax':
      ca,T = x[0],x[1]
     
      if u.shape == (1,):
        Tc = u[0]
      else:
         Tc,self.Ti,self.caf = u[0],u[1],u[2]
      Tc = u[0]
      rA = self.k0 * jnp.exp(-self.EA_over_R/T)*ca
      dxdt = jnp.array([
          self.q/self.V*(self.caf-ca) - rA,
          self.q/self.V*(self.Ti-T) + ((-self.deltaHr)*rA)*(1/(self.rho*self.C)) + self.UA*(Tc-T)*(1/(self.rho*self.C*self.V))])
      return dxdt
    else:
      ca,T = x[0],x[1]
      if u.shape == (1,1):
          Tc = u[0]
      else:
          Tc,self.Ti,self.caf = u[0],u[1],u[2]


      rA = self.k0 * np.exp(-self.EA_over_R/T)*ca
      dxdt = [
          self.q/self.V*(self.caf-ca) - rA,
          self.q/self.V*(self.Ti-T) + ((-self.deltaHr)*rA)*(1/(self.rho*self.C)) + self.UA*(Tc-T)*(1/(self.rho*self.C*self.V))]

      return dxdt
    
  def info(self):
    # Return a dictionary with the model information
    info = {
        'parameters': self.__dict__.copy(), 
        'states': ['Ca', 'T'],
        'inputs': ['Tc'],
        'disturbances': ['Ti', 'Caf'],
    }
    info['parameters'].pop('int_method', None)  # Remove 'int_method' from the dictionary since it is not a parameter of the model
    return info
