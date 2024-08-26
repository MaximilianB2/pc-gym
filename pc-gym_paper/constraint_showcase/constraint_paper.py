import sys
sys.path.append("..\..\src\pcgym") # Add local pc-gym files to path.
import numpy as np
from stable_baselines3 import SAC
from pcgym import make_env


# Define environment
T = 26
nsteps = 60

SP = {
    'Ca': [0.86 for i in range(int(nsteps/2))] + [0.89 for i in range(int(nsteps/2))],
}

action_space = {
    'low': np.array([290]),
    'high':np.array([310]) 
}

observation_space = {
    'low' : np.array([0.7,300,0.8]),
    'high' : np.array([1,350,0.9])  
}

r_scale = {'Ca':1e3}


# Define reward to be equal to the OCP (i.e the same as the oracle)
def oracle_reward(self, x, u, con = False):
    Sp_i = 0
    cost = 0 
    R = 0.01
    if not hasattr(self, 'u_prev'):
        self.u_prev = u

    for k in self.env_params["SP"]:
        i = self.model.info()["states"].index(k)
        SP = self.SP[k]
        
        o_space_low = self.env_params["o_space"]["low"][i] 
        o_space_high = self.env_params["o_space"]["high"][i] 

        x_normalized = (x[i] - o_space_low) / (o_space_high - o_space_low)
        setpoint_normalized = (SP - o_space_low) / (o_space_high - o_space_low)

        r_scale = self.env_params.get("r_scale", {})

        cost += (np.sum(x_normalized - setpoint_normalized[self.t]) ** 2) * r_scale.get(k, 1)

        Sp_i += 1

    u_normalized = (u - self.env_params["a_space"]["low"]) / (
        self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )
    u_prev_norm = (self.u_prev - self.env_params["a_space"]["low"]) / (
        self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )
    self.u_prev = u

    # Add the control cost
    cost += np.sum(R * (u_normalized - u_prev_norm)**2)

    # Calculate normalized constraint violation
    if con:
        constraint_violation = 0
        for k, bounds in cons.items():
            i = self.model.info()["states"].index(k)
            x_value = x[i]
            lower_bound, upper_bound = bounds
            o_space_low = self.env_params["o_space"]["low"][i]
            o_space_high = self.env_params["o_space"]["high"][i]

            # Normalize the constraint bounds and current value
            x_normalized = (x_value - o_space_low) / (o_space_high - o_space_low)
            lower_normalized = (lower_bound - o_space_low) / (o_space_high - o_space_low)
            upper_normalized = (upper_bound - o_space_low) / (o_space_high - o_space_low)

            # Check which constraint is violated and calculate the normalized violation
            if cons_type[k][0] == '<=' and x_normalized > upper_normalized:
                constraint_violation += (x_normalized - upper_normalized) ** 2
                # print(constraint_violation)
            elif cons_type[k][1] == '>=' and x_normalized < lower_normalized:
                constraint_violation += (lower_normalized - x_normalized) ** 2
        # Add the normalized constraint violation to the cost
        cost += constraint_violation*500

    r = -cost
    
    try:
        return r[0]
    except Exception:
        return r



cons = {'T':[327,321]}

cons_type = {'T':['<=','>=']}

env_params_con = {
    'N': nsteps, 
    'tsim':T, 
    'SP':SP, 
    'o_space' : observation_space, 
    'a_space' : action_space,
    'x0': np.array([0.8,325,0.86]),
    'r_scale': r_scale,
    'model': 'cstr', 
    'normalise_a': True, 
    'normalise_o':True, 
    'noise':True, 
    'integration_method': 'casadi', 
    'noise_percentage':0.001, 
    'custom_reward': oracle_reward,
    'done_on_cons_vio': False,
    'constraints': cons, 
    'r_penalty': False,
    'cons_type': cons_type,
}
env_con = make_env(env_params_con)


env_params = env_params_con
env_params.pop('done_on_cons_vio')
env_params.pop('constraints')
env_params.pop('r_penalty')
env_params.pop('cons_type')

env = make_env(env_params)

# SAC_constraint = SAC("MlpPolicy", env_con, verbose=1, learning_rate=0.01).learn(1e4)
SAC_norm = SAC("MlpPolicy", env, verbose=1, learning_rate=0.01).learn(1e4)


# SAC_constraint.save('SAC_constraint.zip')
SAC_norm.save('SAC_norm.zip')

SAC_constraint = SAC.load('SAC_constraint.zip')
SAC_norm = SAC.load('SAC_norm.zip')
_, con_data = env_con.get_rollouts({'SAC':SAC_constraint,}, reps=50, oracle=True, MPC_params={'N':20,})

np.save('constraint_rollout_data.npy', con_data, allow_pickle=True)

_, norm_data = env_con.get_rollouts({'SAC':SAC_norm,}, reps=50)

np.save('norm_rollout_data.npy', norm_data, allow_pickle=True)
