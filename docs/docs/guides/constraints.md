This is a user guide for the constraint function in pc-gym which will walkthrough an example of how to add a constraint to an environment.

### Constraint Definition
Firstly, a dictionary named constraints is defined. The keys in the dictionary represent the state, and the values represent the constraints on that state. In this case, the state '0' has no constraint, and the state '1' has a constraint of 332. Then a dictionary named constraint_type is defined. The keys in the dictionary represent the state, and the values represent the type of constraint on that state. In this case, the state '1' has a less than or equal to (<=) constraint. Then the environment parameters are updated with these dictionaries and two addtional boolean variables: `done_on_cons_vio` which allows the episode to end if the constraint is violated and `r_penalty` which adds a penalty to the reward function for violating the constraint. 

```py
# Constraint Definition
constraints = {
    '0': None, # No constraint on the reactor concentration of species A
    '1' : 332 # Constraint on the reactor temperature
}

# Define the constraint type
constraint_type = {'1':'<='}

# Update the environment parameters
env_params.update({
'done_on_cons_vio':False, # Done on constraint violation
'constraints': constraints, 
'cons_type': constraint_type,
'r_penalty': True  # Add a penalty for constraint violation to the reward function
})
```

### Training with a Constraint Example 