import pytest
import numpy as np
from dataclasses import dataclass, field
import sys
from pcgym import make_env

@dataclass(frozen=False, kw_only=True)
class CustomModel:
    int_method: str = field(default="casadi")
    param1: float = 1.0
    param2: float = 2.0

    def __call__(self, x, u):
        # Dummy implementation
        return np.array([self.param1 * x[0] + u[0], self.param2 * x[1]])

    def info(self):
        return {
            "parameters": {
                "param1": self.param1,
                "param2": self.param2
            },
            "states": ["x1", "x2"],
            "inputs": ["u1"],
            "disturbances": []
        }

def test_make_env_custom_model():
    env_params = {
        "custom_model": CustomModel(),
        "a_space": {"low": np.array([-1]), "high": np.array([1])},
        "o_space": {"low": np.array([-1, -1]), "high": np.array([1, 1])},
        "SP": {"x2": [2] * 100},
        "N": 100,
        "tsim": 10,
        "x0": np.array([0, 0])
    }
    env = make_env(env_params)
    assert isinstance(env.model, CustomModel)
    
    obs, info = env.reset()
    assert obs.shape == (2,)
    
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    assert obs.shape == (2,)

def test_custom_model_call():
    model = CustomModel()
    x = np.array([1.0, 2.0])
    u = np.array([0.5])
    result = model(x, u)
    assert result.shape == (2,)
    assert np.isclose(result[0], 1.5)  # 1.0 * 1.0 + 0.5
    assert np.isclose(result[1], 4.0)  # 2.0 * 2.0

def test_custom_model_info():
    model = CustomModel(param1=1.5, param2=2.5)
    info = model.info()
    assert info["parameters"]["param1"] == 1.5
    assert info["parameters"]["param2"] == 2.5
    assert info["states"] == ["x1", "x2"]
    assert info["inputs"] == ["u1"]
    assert info["disturbances"] == []

def test_make_env_custom_model_integration():
    custom_model = CustomModel(param1=1.5, param2=2.5)
    env_params = {
        "custom_model": custom_model,
        "a_space": {"low": np.array([-1]), "high": np.array([1])},
        "o_space": {"low": np.array([-1, -1]), "high": np.array([1, 1])},
        "SP": {"x2": [2] * 100},
        "N": 100,
        "tsim": 10,
        "x0": np.array([1.0, 1.0])
    }
    env = make_env(env_params)
    
    obs, _ = env.reset()
    assert np.allclose(obs, np.array([1.0, 1.0]))
    
    action = np.array([0.5])
    obs, reward, done, truncated, info = env.step(action)
    print(obs)
    assert np.isclose(obs[0], 1.21578082)  # 1.5 * 1.0 + 0.5
    assert np.isclose(obs[1], 1.28403262)  # 2.5 * 1.0