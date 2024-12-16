import numpy as np


cons_type = {'T':['<=','>=']}
cons = {'T':[327,321]}
def con_reward(self, x, u, con = False):
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
        cost += constraint_violation

    r = -cost
    
    try:
        return r[0]
    except Exception:
        return r
