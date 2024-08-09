import pytest
import numpy as np
from pcgym import make_env  # Adjust this import to your actual package name

@pytest.fixture
def base_env_params():
    return {
        "N": 100,
        "tsim": 10,
        "integration_method": "casadi",
    }

@pytest.fixture
def env(request):
    params = request.getfixturevalue(request.param)
    return make_env(params)

@pytest.fixture
def cstr_params(base_env_params):
    params = base_env_params.copy()
    params.update({
        "model": "cstr",
        "a_space": {"low": np.array([0]), "high": np.array([1])},
        "o_space": {"low": np.array([0, 0]), "high": np.array([1, 1])},
        "SP": {"T": [0.5] * 100},
        "x0": np.array([0.5, 0.5]),
    })
    return params

@pytest.fixture
def multistage_extraction_params(base_env_params):
    params = base_env_params.copy()
    params.update({
        "model": "multistage_extraction",
        "a_space": {"low": np.array([0, 0]), "high": np.array([1, 1])},
        "o_space": {"low": np.array([0]*10+[0.3]), "high": np.array([1]*10+[0.4])},
        "SP": {"X5": [0.3] * 100},
        "x0": np.array([0.55, 0.3, 0.45, 0.25, 0.4, 0.20, 0.35, 0.15, 0.25, 0.1,0.3]),
    })
    return params

@pytest.fixture
def biofilm_reactor_params(base_env_params):
    params = base_env_params.copy()
    params.update({
        "model": "biofilm_reactor",
        "a_space": {"low": np.array([0]), "high": np.array([1])},
        "o_space": {"low": np.array([0,0,0,0]*4+[0.9]), "high": np.array([10,10,10,500]*4+[1.1])},
        "SP": {"S2_A": [1.5] * 100},
        "x0": np.array([2,0.1,10,0.1]*4+[1]),
    })
    return params

@pytest.fixture
def crystallization_params(base_env_params):
    lbMu0 = 0  # 0.1
    ubMu0 = 1e20
    lbMu1 = 0  # 0.1
    ubMu1 = 1e20
    lbMu2 = 0
    ubMu2 = 1e20
    lbMu3 = 0
    ubMu3 = 1e20
    lbC = 0
    ubC = 0.5

    observation_space = {
    'low' : np.array([lbMu0, lbMu1, lbMu2, lbMu3, lbC, 0, 0,  0.9, 14]),
    'high' : np.array([ubMu0, ubMu1, ubMu2, ubMu3, ubC, 2, 20, 1.1, 16])  
    }

    CV_0 = np.sqrt(1800863.24079725 * 1478.00986666666/ (22995.8230590611**2) - 1)
    Ln_0 =  22995.8230590611 / ( 1478.00986666666 + 1e-6)
    params = base_env_params.copy()
    action_space = {
    'low': np.array([-1]),
    'high':np.array([1])
    }
    action_space_act = {
        'low': np.array([10]),
        'high':np.array([40])
    }   
    params.update({
        "model": "crystallization",
        "o_space": observation_space,
        "SP": {"CV": [1] * 100, "Ln": [15] * 100},
        'x0': np.array([1478.00986666666, 22995.8230590611, 1800863.24079725, 248516167.940593, 0.15861523304,CV_0, Ln_0 , 1, 15]),
        'a_delta':True,
        'a_space_act':action_space_act,
        'a_space' : action_space,
        'a_0': 39
    })
    return params

@pytest.fixture
def four_tank_params(base_env_params):
    params = base_env_params.copy()
    SP = {
        'h3': [0.5 for i in range(int(100))],
        'h4': [0.2 for i in range(int(100))]
    }

    action_space = {
        'low': np.array([0,0]),
        'high':np.array([10,10])
    }

    observation_space = {
        'low' : np.array([0,]*6),
        'high' : np.array([0.5]*6)  
    }
    params.update({
        "model": "four_tank",
        "a_space": action_space,
        "o_space": observation_space,
        "SP": SP,
        "x0": np.array([0.141, 0.112, 0.072, 0.42,SP['h3'][0],SP['h4'][0]]),
    })
    return params

# Basic tests for all models
@pytest.mark.parametrize('env', [
    'cstr_params', 'multistage_extraction_params', 'biofilm_reactor_params',
    'crystallization_params', 'four_tank_params'
], indirect=True)
def test_env_initialization(env):
    state, info = env.reset()
    assert state.shape == env.observation_space.shape
    assert isinstance(info, dict)

@pytest.mark.parametrize('env', [
    'cstr_params', 'multistage_extraction_params', 'biofilm_reactor_params',
    'crystallization_params', 'four_tank_params'
], indirect=True)
def test_env_step(env):
    env.reset()
    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    assert next_state.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

# Model-specific tests
def test_cstr_disturbances(cstr_params):
    cstr_params.update({
        "disturbances": {"Ca0": np.random.uniform(0.8, 1.2, 100)},
        "disturbance_bounds": {"low": np.array([0.8]), "high": np.array([1.2])},
    })
    env = make_env(cstr_params)
    state, _ = env.reset()
    assert state.shape[-1] == cstr_params["o_space"]["low"].shape[0] + 1  # +1 for disturbance

def test_multistage_extraction_uncertainties(multistage_extraction_params):
    multistage_extraction_params.update({
        "uncertainty": True,
        "uncertainty_percentages": {"K": 0.1},  # 10% uncertainty in K
        "uncertainty_bounds": {"low": np.array([0.9]), "high": np.array([1.1])},
    })
    env = make_env(multistage_extraction_params)
    state, _ = env.reset()
    assert state.shape[-1] == multistage_extraction_params["o_space"]["low"].shape[0] + 1  # +1 for uncertainty

def test_biofilm_reactor_constraints(biofilm_reactor_params):
    biofilm_reactor_params.update({
        "constraints": {"S": [0.8]},
        "cons_type": {"S": ["<="]},
        "done_on_cons_vio": True,
        "r_penalty": True,
    })
    env = make_env(biofilm_reactor_params)
    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        _, reward, done, _, _ = env.step(action)
        if done:
            assert reward < 0  # Check if penalty was applied
            break

def test_crystallization_custom_reward(crystallization_params):
    def custom_reward_function(env, state, action, constraint_violated):
        return -np.sum(np.abs(state[:2] - [env.SP['CV'][env.t], env.SP['Ln'][env.t]]))

    crystallization_params.update({
        "custom_reward": custom_reward_function,
    })
    env = make_env(crystallization_params)
    env.reset()
    action = env.action_space.sample()
    _, reward, _, _, _ = env.step(action)
    assert isinstance(reward, float)

def test_four_tank_custom_constraint(four_tank_params):
    def custom_constraint_function(state, action):
        return np.any(state > 0.8)  # Constraint violated if any tank level > 0.8

    four_tank_params.update({
        "custom_con": custom_constraint_function,
        "done_on_cons_vio": True,
        "r_penalty": True
    })
    env = make_env(four_tank_params)
    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        _, reward, done, _, _ = env.step(action)
        if done:
            assert reward < 0  # Check if penalty was applied
            break

# Test multiple steps for all models
@pytest.mark.parametrize('env', [
    'cstr_params', 'multistage_extraction_params', 'biofilm_reactor_params',
    'crystallization_params', 'four_tank_params'
], indirect=True)
def test_multiple_steps(env):
    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        assert next_state.shape == env.observation_space.shape
        assert isinstance(reward, float)
        if done:
            break

# Test normalization
@pytest.mark.parametrize('env', [
    'cstr_params', 'multistage_extraction_params', 'biofilm_reactor_params',
    'crystallization_params', 'four_tank_params'
], indirect=True)
def test_normalization(env):
    env.normalise_o = True
    env.normalise_a = True
    state, _ = env.reset()
    assert np.all(state >= -1) and np.all(state <= 1)
    action = env.action_space.sample()
    assert np.all(action >= -1) and np.all(action <= 1)

# Test integration methods
@pytest.mark.parametrize('env', ['cstr_params'], indirect=True)
@pytest.mark.parametrize('integration_method', ['casadi', 'jax'])
def test_integration_methods(env, integration_method):
    env.env_params['integration_method'] = integration_method
    env = make_env(env.env_params)
    env.reset()
    action = env.action_space.sample()
    next_state, _, _, _, _ = env.step(action)
    assert next_state.shape == env.observation_space.shape

@pytest.mark.parametrize('env_params', [
    'cstr_params', 'multistage_extraction_params', 'biofilm_reactor_params',
    'crystallization_params', 'four_tank_params'
])
def test_model_uncertainties(request, env_params):
    params = request.getfixturevalue(env_params)
    
    # Define uncertainties for each model
    uncertainty_configs = {
        'cstr': {'k0': 0.1},  # 10% uncertainty in reaction rate constant
        'multistage_extraction': {'K': 0.1},  # 10% uncertainty in partition coefficient
        'biofilm_reactor': {'mu_max': 0.1},  # 10% uncertainty in max growth rate
        'crystallization': {'kg': 0.1},  # 10% uncertainty in growth rate constant
        'four_tank': {'a1': 0.1, 'a2': 0.1}  # 10% uncertainty in outlet areas
    }
    
    model_name = params['model']
    uncertainties = uncertainty_configs[model_name]
    
    params.update({
        "uncertainty": True,
        "uncertainty_percentages": uncertainties,
        "uncertainty_bounds": {
            "low": np.array([0.9] * len(uncertainties)),
            "high": np.array([1.1] * len(uncertainties))
        },
    })
    
    env = make_env(params)
    state, _ = env.reset()
    
    # Check if state shape is extended by the number of uncertain parameters
    assert state.shape[-1] == params["o_space"]["low"].shape[0] + len(uncertainties)
    
    # Run a few steps to ensure the environment works with uncertainties
    for _ in range(5):
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        assert next_state.shape == state.shape
        assert isinstance(reward, float)
        if done:
            break
        
# Test multiple steps for all models with uncertainties
@pytest.mark.parametrize('env_params', [
    'cstr_params', 'multistage_extraction_params', 'biofilm_reactor_params',
    'crystallization_params', 'four_tank_params'
])
def test_multiple_steps_with_uncertainties(request, env_params):
    params = request.getfixturevalue(env_params)
    
    # Define uncertainties for each model (same as in test_model_uncertainties)
    uncertainty_configs = {
        'cstr': {'k0': 0.1},
        'multistage_extraction': {'K': 0.1},
        'biofilm_reactor': {'mu_max': 0.1},
        'crystallization': {'kg': 0.1},
        'four_tank': {'a1': 0.1, 'a2': 0.1}
    }
    
    model_name = params['model']
    uncertainties = uncertainty_configs[model_name]
    
    params.update({
        "uncertainty": True,
        "uncertainty_percentages": uncertainties,
        "uncertainty_bounds": {
            "low": np.array([0.9] * len(uncertainties)),
            "high": np.array([1.1] * len(uncertainties))
        },
    })
    
    env = make_env(params)
    state, _ = env.reset()
    
    for _ in range(10):
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        assert next_state.shape == state.shape
        assert isinstance(reward, float)
        if done:
            break
@pytest.mark.parametrize('env_params', [
    'cstr_params', 'multistage_extraction_params', 'biofilm_reactor_params',
    'crystallization_params', 'four_tank_params'
])
def test_uncertainty_application(request, env_params):
    params = request.getfixturevalue(env_params)
    
    # Define uncertainties for each model (same as before)
    uncertainty_configs = {
        'cstr': {'k0': 0.1},
        'multistage_extraction': {'K': 0.1},
        'biofilm_reactor': {'mu_max': 0.1},
        'crystallization': {'kg': 0.1},
        'four_tank': {'a1': 0.1}
    }
    
    model_name = params['model']
    uncertainties = uncertainty_configs[model_name]
    
    params.update({
        "uncertainty": True,
        "uncertainty_percentages": uncertainties,
        "uncertainty_bounds": {
            "low": np.array([0.9] * len(uncertainties)),
            "high": np.array([1.1] * len(uncertainties))
        },
    })
    
    env = make_env(params)
    original_param = getattr(env.model, str( next(iter(uncertainty_configs[model_name]))))
    # Run multiple resets to check if uncertainties are being resampled
    uncertainty_values = []
    for _ in range(10):
        state, _ = env.reset()
        uncertainty_values.append(getattr(env.model, str( next(iter(uncertainty_configs[model_name])))))
  
    # Check if uncertainties are different across resets (indicating they're being resampled)
    assert not np.allclose(uncertainty_values[0], uncertainty_values[1:])
    
    # Check if uncertainties are within the specified bounds
    uncertainty_array = np.array(uncertainty_values)

    assert np.all(uncertainty_array >= 0.9 * original_param) and np.all(uncertainty_array <= 1.1 * original_param)