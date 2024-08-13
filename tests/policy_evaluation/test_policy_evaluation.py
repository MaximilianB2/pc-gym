import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pcgym.policy_evaluation import policy_eval

@pytest.fixture
def mock_env():
    env = MagicMock()
    env.Nx = 2
    env.N = 3
    env.Nu = 1
    env.Nd = 0
    env.tsim = 10
    env.env_params = {'a_space': {'low': np.array([0]), 'high': np.array([1])}}
    env.observation_space_base.low = np.array([0, 0])
    env.observation_space_base.high = np.array([1, 1])
    env.reset.return_value = (np.array([0.5, 0.5]), {'r_init': 0})
    env.step.return_value = (np.array([0.6, 0.6]), 1, False, False, {'cons_info': np.array([[0]])})
    env.constraint_active = True
    env.n_con = 1
    env.Nx_oracle = 2
    env.model.info.return_value = {
        'states': ['s1', 's2'],
        'inputs': ['u1'],
        'disturbances': []  # Add this line
    }
    env.disturbance_active = False  # Add this line
    env.constraints = {'s1': [1], 'u1': [2]}
    env.SP = {'s1': [0.5, 0.5, 0.5]}
    return env
@pytest.fixture
def mock_make_env(mock_env):
    return MagicMock(return_value=mock_env)

@pytest.fixture
def mock_policies():
    return {'policy1': MagicMock(), 'policy2': MagicMock()}

@pytest.fixture
def pe(mock_make_env, mock_policies):
    env_params = {'param1': 1, 'param2': 2}
    return policy_eval(mock_make_env, mock_policies, 5, env_params)


def test_init(pe, mock_make_env, mock_policies):
    assert pe.make_env == mock_make_env
    assert pe.policies == mock_policies
    assert pe.n_pi == 2
    assert pe.reps == 5
    assert pe.env_params == {'param1': 1, 'param2': 2}
    assert not pe.oracle
    assert not pe.cons_viol
    assert not pe.save_fig
    assert not pe.MPC_params

def test_rollout(pe, mock_env):
    policy = MagicMock()
    policy.predict.return_value = (np.array([0.5]), None)

    # Mock the reward to match the expected shape
    pe.env.step.return_value = (np.array([0.6, 0.6]), np.array([1]), False, False, {'cons_info': np.array([[0]])})

    total_reward, s_rollout, actions, cons_info = pe.rollout(policy)

    assert len(total_reward) == 3
    assert s_rollout.shape == (2, 3)
    assert actions.shape == (1, 3)
    assert cons_info.shape == (1, 1)

def test_get_rollouts(pe, mock_env):
    # Mock the rollout method to return consistent shapes
    def mock_rollout(policy):
        return (
            np.array([1, 2, 3]),  # total_reward
            np.random.rand(2, 3),  # s_rollout
            np.random.rand(1, 3),  # actions
            np.random.rand(1, 3, 1)  # cons_info
        )

    with patch.object(pe, 'rollout', side_effect=mock_rollout):
        data = pe.get_rollouts()

    assert len(data) == 2  # Two policies
    assert 'policy1' in data
    assert 'policy2' in data
    for policy_data in data.values():
        assert 'r' in policy_data
        assert 'x' in policy_data
        assert 'u' in policy_data
        assert 'g' in policy_data
        assert policy_data['r'].shape == (1, 3, 5)  # (1, N, reps)
        assert policy_data['x'].shape == (2, 3, 5)  # (Nx, N, reps)
        assert policy_data['u'].shape == (1, 3, 5)  # (Nu, N, reps)
        assert policy_data['g'].shape == (1, 3, 1, 5)  # (n_con, N, 1, reps)

def test_oracle_reward_fn(pe):
    pe.env.custom_reward = False
    pe.env.reward_fn = MagicMock(return_value=1)
    x = np.array([[1, 2], [3, 4]])
    u = np.array([[5, 6]])

    rewards = pe.oracle_reward_fn(x, u)

    assert len(rewards) == 2
    pe.env.reward_fn.assert_called()

@pytest.mark.parametrize("reward_dist", [True, False])
def test_plot_data(pe, reward_dist):
    # Create data for all policies in self.policies
    data = {}
    for policy_name in pe.policies.keys():
        data[policy_name] = {
            'x': np.random.rand(pe.env.Nx_oracle, pe.env.N, pe.reps),
            'u': np.random.rand(pe.env.Nu, pe.env.N, pe.reps),
            'r': np.random.rand(1, pe.env.N, pe.reps),
            'g': np.random.rand(pe.env.n_con, pe.env.N, 1, pe.reps)
        }


if __name__ == '__main__':
    pytest.main()