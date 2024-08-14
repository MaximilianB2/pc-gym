import pytest
import numpy as np
from pcgym import make_env
from pcgym.oracle import oracle
import time
# Helper function to create base environment parameters
base_params = {
    "N": 100,
    "tsim": 10,
    "integration_method": "casadi",
}
def create_base_env_params(model_name):
    CV_0 = np.sqrt(1800863.24079725 * 1478.00986666666/ (22995.8230590611**2) - 1)
    Ln_0 =  22995.8230590611 / ( 1478.00986666666 + 1e-6)
    model_specific_params = {
        "cstr": {
            "a_space": {"low": np.array([295]), "high": np.array([305])},
            "o_space": {"low": np.array([0.7, 300, 0.8]), "high": np.array([1, 350, 0.9])},
            "SP": {"Ca": [0.85] * 50 + [0.88] * 50},
            "x0": np.array([0.85, 330, 0.8]),
        },
        "multistage_extraction": {
            "a_space": {"low": np.array([5,10]), "high": np.array([500,1000])},
            "o_space": {"low": np.array([0]*10+[0.3]), "high": np.array([1]*10+[0.4])},
            "SP": {"X5": [0.3] * 100},
            "x0": np.array([0.55, 0.3, 0.45, 0.25, 0.4, 0.20, 0.35, 0.15, 0.25, 0.1, 0.3]),
        },
        "biofilm_reactor": {
            "a_space": {'low': np.array([0, 1, 0.05, 0.05, 0.05]),'high':np.array([10, 30, 1, 1, 1])},
            "o_space": {'low' : np.array([0,0,0,0]*4+[0]), 'high' : np.array([10,10,10,500]*4+[20])},
            "SP": {"S2_A": [12.5] * 100},
            "x0": np.array([10.,5.,5.,300.]*4+[10]),
        },
        "crystallization": {
            "a_space": {"low": np.array([-1]), "high": np.array([1])},
            "o_space": {
                "low": np.array([0, 0, 0, 0, 0, 0, 0, 0.9, 14]),
                "high": np.array([1e20, 1e20, 1e20, 1e20, 0.5, 2, 20, 1.1, 16])
            },
            "SP": {"CV": [1] * 30, "Ln": [15] * 30},
            "x0": np.array([1478.00986666666, 22995.8230590611, 1800863.24079725, 248516167.940593, 0.15861523304, CV_0, Ln_0, 1, 15]),
        },
        "four_tank": {
            "a_space": {"low": np.array([0, 0]), "high": np.array([10, 10])},
            "o_space": {"low": np.array([0]*6), "high": np.array([0.5]*6)},
            "SP": {"h3": [0.5] * 100, "h4": [0.2] * 100},
            "x0": np.array([0.141, 0.112, 0.072, 0.42, 0.5, 0.2]),
        },
    }
    
    base_params.update({"model": model_name})
    base_params.update(model_specific_params[model_name])
    
    return base_params

# Test configurations
models_to_test = ["cstr", "multistage_extraction", "biofilm_reactor", "crystallization", "four_tank"]

@pytest.mark.parametrize("model_name", models_to_test)
def test_oracle_initialization(model_name):
    env_params = create_base_env_params(model_name)
    env = make_env(env_params)
    oracle_instance = oracle(env, env_params)
    
    assert oracle_instance.N == 5, f"Default prediction horizon should be 5 for {model_name}"
    assert np.array_equal(oracle_instance.R, np.zeros((env.Nu-env.Nd_model,env.Nu-env.Nd_model))), f"Default control penalty should be 0 for {model_name}"
    assert oracle_instance.T == env_params["tsim"], f"Simulation time mismatch for {model_name}"
    assert np.allclose(oracle_instance.x0, env_params["x0"]), f"Initial state mismatch for {model_name}"

@pytest.mark.parametrize("model_name", models_to_test)
def test_oracle_mpc_setup(model_name):
    env_params = create_base_env_params(model_name)
    env = make_env(env_params)
    oracle_instance = oracle(env, env_params)
    
    mpc, simulator = oracle_instance.setup_mpc()
    
    assert mpc is not None, f"MPC controller should be initialized for {model_name}"
    assert simulator is not None, f"Simulator should be initialized for {model_name}"
    assert mpc.settings.n_horizon == oracle_instance.N, f"MPC horizon mismatch for {model_name}"

@pytest.mark.parametrize("model_name", models_to_test)
def test_oracle_mpc_execution(model_name):
    env_params = create_base_env_params(model_name)
    env = make_env(env_params)
    oracle_instance = oracle(env, env_params)
    if model_name == 'biofilm_reactor':
        oracle_instance = oracle(env, env_params, MPC_params={'N':2})
    x_log, u_log = oracle_instance.mpc()
    
    assert x_log.shape == (env.Nx_oracle, env.N), f"State log shape mismatch for {model_name}"
    assert u_log.shape == (env.Nu, env.N), f"Control input log shape mismatch for {model_name}"

@pytest.mark.parametrize("model_name", models_to_test)
def test_oracle_with_custom_mpc_params(model_name):
    env_params = create_base_env_params(model_name)
    env = make_env(env_params)
    mpc_params = {"N": 2, "R": np.eye(env.Nu - env.Nd_model)*3, "Q": np.eye(env.Nx_oracle)*3}
    oracle_instance = oracle(env, env_params, MPC_params=mpc_params)
    
    assert oracle_instance.N == 2, f"Custom prediction horizon not set correctly for {model_name}"
    assert np.array_equal(oracle_instance.R, np.eye(env.Nu-env.Nd_model)*3), f"Custom control penalty not set correctly for {model_name}"
    assert np.array_equal(oracle_instance.Q, np.eye(env.Nx_oracle)*3), f"Custom control penalty not set correctly for {model_name}"
    x_log, u_log = oracle_instance.mpc()
    assert x_log.shape == (env.Nx_oracle, env.N), f"State log shape mismatch for {model_name} with custom MPC params"
    assert u_log.shape == (env.Nu, env.N), f"Control input log shape mismatch for {model_name} with custom MPC params"

def calculate_iae(setpoint, actual):
    """Calculate Integral Absolute Error"""
    return np.sum(np.abs(setpoint - actual))

def calculate_ise(setpoint, actual):
    """Calculate Integral Squared Error"""
    return np.sum((setpoint - actual)**2)

def calculate_tv(control_inputs):
    """Calculate Total Variation of control inputs"""
    return np.sum(np.abs(np.diff(control_inputs, axis=1)))

@pytest.mark.parametrize("model_name", ["cstr", "multistage_extraction"])   
def test_oracle_disturbance_performance(model_name):
    env_params = create_base_env_params(model_name)
    
    # Add disturbances and parameter uncertainties
    if model_name == 'cstr':
        disturbance = {'Ti': np.repeat([350, 345, 350], [base_params['N']//4, base_params['N']//2, base_params['N']//4])}
        disturbance_space = {'low': np.array([320]), 'high': np.array([350])}
        env_params["disturbances"] = disturbance
        env_params["disturbance_bounds"] = disturbance_space
    else:
        disturbance = {'X0': np.repeat([0.6, 0.7, 0.8], [base_params['N']//4, base_params['N']//2, base_params['N']//4])}
        disturbance_space = {'low': np.array([0.6]), 'high': np.array([0.8])}
        env_params["disturbances"] = disturbance
        env_params["disturbance_bounds"] = disturbance_space
    env = make_env(env_params)
    oracle_instance = oracle(env, env_params)
    
    x_log, u_log = oracle_instance.mpc()
    
    # Calculate performance metrics
    setpoint_keys = list(env_params["SP"].keys())
    iae_values = []
    
    for sp_key in setpoint_keys:
        sp_index = env.model.info()["states"].index(sp_key)
        setpoint = np.array(env_params["SP"][sp_key])
        actual = x_log[sp_index, :]
        
        iae = calculate_iae(setpoint, actual)
        iae_values.append(iae)
    
    tv = calculate_tv(u_log)
    
    # Print robustness metrics
    print(f"\nRobustness metrics for {model_name}:")
    for i, sp_key in enumerate(setpoint_keys):
        print(f"IAE for {sp_key} with disturbances: {iae_values[i]:.4f}")
    print(f"Total Variation of control inputs with disturbances: {tv:.4f}")
    
    # Assert some basic robustness criteria
    assert all(iae < 2000 for iae in iae_values), f"IAE too high under disturbances for {model_name}"
    assert tv < 2000, f"Total Variation too high under disturbances for {model_name}"

@pytest.mark.parametrize("model_name", models_to_test)
def test_oracle_constraint_handling(model_name):
    env_params = create_base_env_params(model_name)
    

    constraint_configs = {
        "cstr": {"Ca": [0.5, 1]},
        "multistage_extraction": {"X5": [0, 0.33]},
        "biofilm_reactor": {"S2_A": [0.0, 14]},
        "crystallization": {"CV": [0.9, 2]},
        "four_tank": {"h3": [0, 0.55]},
    }
    if model_name == "crystallization":
        action_space_act = {'low': np.array([10]), 'high':np.array([40])}
        env_params["N"] = 30
        env_params["tsim"] = 30
        env_params['a_delta'] = True,
        env_params['a_space_act'] = action_space_act
        env_params['a_0'] = 39
    elif model_name == 'four_tank':
        env_params["tsim"] = 1000
        env_params["N"] = 60

    env_params["constraints"] = constraint_configs[model_name]
    env_params['done_on_cons_vio'] = False
    env_params['r_penalty'] = False
    env_params["cons_type"] = {k: [">=", "<="] for k in constraint_configs[model_name].keys()}
    
    env = make_env(env_params)
    oracle_instance = oracle(env, env_params,MPC_params={'N':2})
    
    x_log, u_log = oracle_instance.mpc()
    
    # Check constraint satisfaction
    constrained_state = list(env_params["constraints"].keys())[0]
    constrained_state_index = env.model.info()["states"].index(constrained_state)
    constraint_violations = np.sum(
        (x_log[constrained_state_index] < env_params["constraints"][constrained_state][0]) |
        (x_log[constrained_state_index] > env_params["constraints"][constrained_state][1])
    )
    
    print(f"\nConstraint handling metrics for {model_name}:")
    print(f"Number of constraint violations: {constraint_violations}")
    
    # Assert constraint satisfaction
    assert constraint_violations == 0, f"Constraints violated for {model_name}"

if __name__ == "__main__":
    pytest.main([__file__])
