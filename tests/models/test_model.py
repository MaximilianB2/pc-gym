import pytest
import numpy as np
from pcgym import make_env
import copy
# Helper function to create environment parameters
def create_base_params(model_name, action_space, observation_space, setpoint, x0):
    return {
        "model": model_name,
        "N": 100,
        "tsim": 10,
        "integration_method": "casadi",
        "a_space": action_space,
        "o_space": observation_space,
        "SP": setpoint,
        "x0": x0,
    }

# Test configurations for each model
model_configs = {
    "cstr": {
        "action_space": {"low": np.array([0]), "high": np.array([1])},
        "observation_space": {"low": np.array([0, 0, 0]), "high": np.array([1, 1, 1])},
        "setpoint": {"T": [0.5] * 100},
        "x0": np.array([0.5, 0.5, 0.5]),
        "uncertain_params": {"k0": 0.1},  # 10% uncertainty in reaction rate constant
        "disturbances": {"Caf": np.random.uniform(0.8, 1.2, 100)},
    },
    "multistage_extraction": {
        "action_space": {"low": np.array([0, 0]), "high": np.array([1, 1])},
        "observation_space": {"low": np.array([0]*10+[0.3]), "high": np.array([1]*10+[0.4])},
        "setpoint": {"X5": [0.3] * 100},
        "x0": np.array([0.55, 0.3, 0.45, 0.25, 0.4, 0.20, 0.35, 0.15, 0.25, 0.1, 0.3]),
        "uncertain_params": {"K": 0.1},  # 10% uncertainty in partition coefficient
        "disturbances": {"X0": np.random.uniform(0.8, 1.2, 100)},
    },
    "biofilm_reactor": {
        "action_space": {"low": np.array([0]), "high": np.array([1])},
        "observation_space": {"low": np.array([0,0,0,0]*4+[0.9]), "high": np.array([10,10,10,500]*4+[1.1])},
        "setpoint": {"S2_A": [1.5] * 100},
        "x0": np.array([2,0.1,10,0.1]*4+[1]),
        "uncertain_params": {"mu_max": 0.1},  # 10% uncertainty in max growth rate
        "disturbances": {"S_in": np.random.uniform(0.8, 1.2, 100)},
    },
    "crystallization": {
        "action_space": {"low": np.array([-1]), "high": np.array([1])},
        "observation_space": {
            "low": np.array([0, 0, 0, 0, 0, 0, 0, 0.9, 14]),
            "high": np.array([1e20, 1e20, 1e20, 1e20, 0.5, 2, 20, 1.1, 16])
        },
        "setpoint": {"CV": [1] * 100, "Ln": [15] * 100},
        "x0": np.array([1478.00986666666, 22995.8230590611, 1800863.24079725, 248516167.940593, 0.15861523304, 0.5, 15, 1, 15]),
        "uncertain_params": {"kg": 0.1},  # 10% uncertainty in growth rate constant
        "disturbances": {"T": np.random.uniform(0.8, 1.2, 100)},
    },
    "four_tank": {
        "action_space": {"low": np.array([0, 0]), "high": np.array([10, 10])},
        "observation_space": {"low": np.array([0]*6), "high": np.array([0.5]*6)},
        "setpoint": {"h3": [0.5] * 100, "h4": [0.2] * 100},
        "x0": np.array([0.141, 0.112, 0.072, 0.42, 0.5, 0.2]),
        "uncertain_params": {"a1": 0.1, "a2": 0.1},  # 10% uncertainty in outlet areas
        "disturbances": {"q1": np.random.uniform(0.8, 1.2, 100)},
    },
}

@pytest.mark.parametrize("model_name", model_configs.keys())
def test_basic_functionality(model_name):
    config = model_configs[model_name]
    params = create_base_params(model_name, 
                                config["action_space"], 
                                config["observation_space"], 
                                config["setpoint"], 
                                config["x0"])
    env = make_env(params)
    
    state, _ = env.reset()
    assert state.shape == env.observation_space.shape
    
    for _ in range(10):
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        assert next_state.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        if done:
            break
@pytest.mark.parametrize("model_name", ['cstr'])
def test_uncertainty(model_name):
    config = model_configs[model_name]
    params = create_base_params(model_name, 
                                config["action_space"], 
                                config["observation_space"], 
                                config["setpoint"], 
                                config["x0"])
    uncertain_params = config["uncertain_params"]
    params.update({
        "uncertainty": True,
        "uncertainty_percentages": {
            "x0": {0: 0.1},  # 10% uncertainty in first initial state
            **uncertain_params
        },
        "uncertainty_bounds": {
            "low": np.array([0.9] * (len(uncertain_params))),
            "high": np.array([1.1] * (len(uncertain_params)))
        },
    })
    env = make_env(params)
    
    num_resets = 5
    initial_states = []
    param_values = {param: [] for param in uncertain_params}
    original_values = {}
    
    print(f"\nTesting uncertainty for model: {model_name}")
    print("Original parameter values:")
    for param in uncertain_params:
        original_value = getattr(env.model, param)
        original_values[param] = original_value
        print(f"{param}: {original_value}")
        lower_bound = original_value * 0.9
        upper_bound = original_value * 1.1
        print(f"  Expected bounds: [{lower_bound}, {upper_bound}]")
    
    for i in range(num_resets):
        state, _ = env.reset()
        initial_states.append(state)
        
        print(f"\nAfter Reset {i + 1}:")
        print(f"State: {state}")
        
        for param in uncertain_params:
            value = getattr(env.model, param)
            param_values[param].append(value)
            print(f"{param}: {value}")
    
    # Check variations in initial states
    state_variations = np.std(initial_states, axis=0)
    print("\nInitial State Variations:")
    for i, variation in enumerate(state_variations):
        print(f"State element {i}: std dev = {variation}")
    
    assert np.any(state_variations > 0), f"No variation detected in initial states for {model_name}."
    
    # Check variations in uncertain parameters
    print("\nUncertain Parameter Variations:")
    for param, values in param_values.items():
        param_variation = np.std(values)
        print(f"{param}: std dev = {param_variation}")
        print(f"Values across resets: {values}")
        assert param_variation > 0, f"No variation detected in parameter {param} for {model_name}."
    
    # Additional check: Ensure parameters change across resets
    for param, values in param_values.items():
        assert len(set(values)) > 1, f"Parameter {param} did not change across resets. Values: {values}"

    # Check that parameters are within the specified bounds
    for param, values in param_values.items():
        original_value = original_values[param]
        lower_bound = original_value * 0.9  # 10% lower
        upper_bound = original_value * 1.1  # 10% higher
        for value in values:
            assert lower_bound <= value <= upper_bound, (
                f"Parameter {param} outside bounds. "
                f"Value: {value}, Bounds: [{lower_bound}, {upper_bound}], "
                f"Original value: {original_value}"
            )

    print(f"\nUncertainty test passed for {model_name}")

@pytest.mark.parametrize("model_name", ['cstr'])
def test_constraints(model_name):
    config = model_configs[model_name]
    params = create_base_params(model_name, 
                                config["action_space"], 
                                config["observation_space"], 
                                config["setpoint"], 
                                config["x0"])
    params.update({
        "constraints": lambda x, u: np.array([x[0] - 0.8, 0.9 - x[0]]).reshape(-1,),
        "done_on_cons_vio": False,
        "r_penalty": True,
    })
    env = make_env(params)
    
    env.reset()
    constraint_violated = False
    for i in range(100):
        action = env.action_space.high  # Use max action to potentially violate constraint
        _, reward, done, _, info = env.step(action)
        if info['cons_info'][0, i, :] > 0:
            constraint_violated_0 = True
        if info['cons_info'][1, i, :] > 0:
            constraint_violated_1 = True
        if done: 
            break

        print(constraint_violated)
    assert constraint_violated_1 and constraint_violated_0, "Constraint should have been violated"
    
@pytest.mark.parametrize("model_name", ["cstr", "multistage_extraction"])   
def test_disturbances(model_name):
    config = model_configs[model_name]
    params = create_base_params(model_name, 
                                config["action_space"], 
                                config["observation_space"], 
                                config["setpoint"], 
                                config["x0"])
    disturbances = config["disturbances"]
    params.update({
        "disturbances": disturbances,
        "disturbance_bounds": {
            "low": np.array([0.8] * len(disturbances)),
            "high": np.array([1.2] * len(disturbances))
        },
    })
    env = make_env(params)
    
    state, _ = env.reset()
    assert state.shape[-1] == params["o_space"]["low"].shape[0] + len(disturbances)
    print(state)
    for _ in range(10):
        action = env.action_space.sample()
        next_state, _, _, _, _ = env.step(action)
        assert next_state.shape == state.shape
        print(state, next_state)

        assert not np.allclose(state[-len(disturbances):], next_state[-len(disturbances):])  # Disturbances should change
        state = copy.deepcopy(next_state)
@pytest.mark.parametrize("model_name", ["cstr", "multistage_extraction", "four_tank","crystallization"])   
def test_JAX_int(model_name):
    config = model_configs[model_name]
    params = create_base_params(model_name, 
                                config["action_space"], 
                                config["observation_space"], 
                                config["setpoint"], 
                                config["x0"])
    params.update({
        "integration_method":"jax"
        })
    
    env = make_env(params)
    try:
        state, _ = env.reset()
    except Exception as e:
        pytest.fail(f"env.reset() raised an exception: {e}")
    
    try:
        action = env.action_space.sample()
        env.step(action)
    except Exception as e:
        pytest.fail(f"env.step() raised an exception: {e}")

@pytest.mark.parametrize("model_name", ["cstr", "multistage_extraction", "four_tank","crystallization"])   
def test_state_and_obs_noise(model_name):
    """
    Test if with measurement noise the state and observation are different.
    """
    config = model_configs[model_name]
    params = create_base_params(model_name, 
                                config["action_space"], 
                                config["observation_space"], 
                                config["setpoint"], 
                                config["x0"])
    params.update({
        "noise": True,
        "noise_percentage":0.001,
        })
    
    env = make_env(params)
    state, _ = env.reset()
    
    action = env.action_space.sample()
    env.step(action)
    assert not np.allclose(env.state[:env.Nx_oracle], env.obs[:env.Nx_oracle])

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])