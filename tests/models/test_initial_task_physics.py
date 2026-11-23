"""Independent balances and analytic checks for controlled process models."""

import itertools

import numpy as np
import pytest
from casadi import SX, Function, jacobian, vertcat
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq, root

from pcgym import make_env
from pcgym.models import get_model_spec


def autonomous_env(name, initial, **parameters):
    return make_env(
        {
            "model": name,
            "model_params": parameters,
            "N": 60,
            "tsim": 3,
            "x0": initial,
            "a_space": {"low": np.empty(0), "high": np.empty(0)},
            "o_space": {"low": np.full(len(initial), -np.inf), "high": np.full(len(initial), np.inf)},
            "reward_states": [],
            "normalise_a": False,
            "normalise_o": False,
        }
    )


@pytest.mark.parametrize("n", [1, 2, 5])
def test_autonomous_oscillator_matches_matrix_exponential_and_conserves_energy_momentum(n):
    initial = np.random.default_rng(7).normal(size=2 * n)
    stiffness, mass = 1.7, 2.3
    env = autonomous_env("coupled_oscillator", initial, N=n, k=stiffness, m=mass)
    trajectory = env.simulate(initial, np.empty((60, 0)))
    identity = np.eye(n)
    laplacian = 2 * identity - np.roll(identity, 1, axis=1) - np.roll(identity, -1, axis=1)
    generator = np.block([[np.zeros((n, n)), identity / mass], [-stiffness * laplacian, np.zeros((n, n))]])
    times = np.arange(61) * env.dt
    expected = np.array([expm(generator * t) @ initial for t in times])
    np.testing.assert_allclose(trajectory, expected, atol=5e-4, rtol=5e-4)
    positions, momenta = trajectory[:, :n], trajectory[:, n:]
    energy = np.sum(momenta**2, axis=1) / (2 * mass)
    energy += stiffness / 2 * np.sum((positions - np.roll(positions, 1, axis=1)) ** 2, axis=1)
    np.testing.assert_allclose(energy, energy[0], atol=1e-5, rtol=5e-4)
    np.testing.assert_allclose(momenta.sum(axis=1), initial[n:].sum(), atol=1e-10)
    np.testing.assert_allclose(
        positions.mean(axis=1), initial[:n].mean() + times * initial[n:].sum() / (n * mass), atol=1e-10
    )


@pytest.mark.parametrize("name", ["coupled_oscillator", "invariant_batch"])
def test_autonomous_models_complete_zero_input_episodes(name):
    initial = (
        np.array([1.0, 0.8, 0.0, 0.0]) if name == "invariant_batch" else np.r_[np.linspace(-1, 1, 10), np.zeros(10)]
    )
    env = autonomous_env(name, initial)
    expected = env.simulate(initial, np.empty((60, 0)))
    env.reset(seed=0)
    assert env.action_space.shape == (0,)
    for step in range(1, 61):
        obs, reward, terminated, truncated, _ = env.step(np.empty(0, dtype=np.float32))
        assert env.observation_space.contains(obs) and reward == 0
        assert not terminated and truncated == (step == 60)
        np.testing.assert_allclose(env.state, expected[step], rtol=1e-10, atol=1e-12)
    if name == "invariant_batch":
        inventories = expected @ np.array([[1, 0], [0, 1], [1, 1], [2, 1]])
        np.testing.assert_allclose(inventories, np.tile([1.0, 0.8], (61, 1)), atol=1e-10)
        assert np.all(expected >= 0)


@pytest.mark.parametrize(
    "name,parameters",
    [
        ("coupled_oscillator", {"N": 0}),
        ("coupled_oscillator", {"N": True}),
        ("coupled_oscillator", {"N": 2.5}),
        ("coupled_oscillator", {"m": 0}),
        ("coupled_oscillator", {"k": -1}),
        ("invariant_batch", {"k1f": -1}),
    ],
)
def test_autonomous_models_reject_invalid_physical_parameters(name, parameters):
    with pytest.raises(ValueError, match=name):
        get_model_spec(name).cls(int_method="casadi", **parameters)


@pytest.mark.parametrize("action", [-1.0, 0.0, 1.0])
def test_nonsmooth_trajectory_matches_linear_system_exponential(action):
    env = make_env({"model": "nonsmooth_control", "N": 60, "tsim": 5})
    states = env.simulate(env.x0, np.full((60, 1), action))
    # Augment the two-state ODE with a constant input to obtain its exact solution.
    generator = np.array([[0, 1, 0], [-2, -3, action], [0, 0, 0]], dtype=float)
    initial = np.array([0.8, 0.0, 1.0])
    expected = np.array([(expm(time * generator) @ initial)[:2] for time in np.linspace(0, 5, 61)])
    np.testing.assert_allclose(states, expected, atol=3e-5, rtol=2e-4)


def test_closed_adiabatic_cstr_conserves_chemical_plus_sensible_energy():
    env = make_env({"model": "cstr", "N": 10, "tsim": 0.05})
    states = env.simulate(env.x0, np.full((10, 1), 298.5), params={"q": 0.0, "UA": 0.0})
    heat_per_concentration = -env.model.deltaHr / (env.model.rho * env.model.C)
    energy = states[:, 1] + heat_per_concentration * states[:, 0]
    np.testing.assert_allclose(energy, energy[0], atol=1e-9, rtol=1e-10)
    assert states[-1, 0] < states[0, 0]
    assert states[-1, 1] > states[0, 1]


def test_registered_cstr_target_has_feasible_steady_coolant():
    env = make_env({"model": "cstr", "N": 60, "tsim": 26})
    model = env.model
    target = env.SP["Ca"][0]
    reaction = model.q / model.V * (model.Caf - target)
    temperature = -model.EA_over_R / np.log(reaction / (model.k0 * target))
    coolant = (
        temperature
        - (model.q * model.rho * model.C * (model.Ti - temperature) - model.deltaHr * reaction * model.V) / model.UA
    )
    assert env.env_params["a_space"]["low"][0] < coolant < env.env_params["a_space"]["high"][0]
    np.testing.assert_allclose(model(np.array([target, temperature]), np.array([coolant])), 0, atol=1e-12)


def test_closed_extraction_conserves_solute_with_unequal_phase_volumes():
    env = make_env({"model": "multistage_extraction", "N": 60, "tsim": 3})
    liquid_volume, gas_volume = 5.0, 9.0
    states = env.simulate(env.x0, np.zeros((60, 2)), params={"Vl": liquid_volume, "Vg": gas_volume})
    mass = liquid_volume * states[:, ::2].sum(axis=1) + gas_volume * states[:, 1::2].sum(axis=1)
    np.testing.assert_allclose(mass, mass[0], atol=1e-8, rtol=1e-9)
    assert np.all(states >= 0)


@pytest.mark.parametrize("nitrogen", [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
def test_biofilm_reaction_direction_and_stoichiometry(nitrogen):
    model = get_model_spec("biofilm_reactor").cls(int_method="casadi")
    state = np.tile([*nitrogen, 5.0], 4)
    derivative = model(state, np.zeros(5)).reshape(4, 4)
    # Isolate each reaction: ammonia -> nitrite, then nitrite -> nitrate.
    if nitrogen[0]:
        consumed = -derivative[:3, 0]
        np.testing.assert_allclose(derivative[:3, 1], consumed)
        np.testing.assert_allclose(derivative[:3, 2], 0)
        oxygen_per_nitrogen = 3.5
    else:
        consumed = -derivative[:3, 1]
        np.testing.assert_allclose(derivative[:3, 0], 0)
        np.testing.assert_allclose(derivative[:3, 2], consumed)
        oxygen_per_nitrogen = 1.1
    assert np.all(consumed > 0)
    np.testing.assert_allclose(derivative[:3, 3], -oxygen_per_nitrogen * consumed)


def test_biofilm_nitrogen_inventory_changes_only_through_feed_and_effluent():
    model = get_model_spec("biofilm_reactor").cls(int_method="casadi", V=7.0, Va=19.0)
    state = np.linspace(0.2, 3.2, 16)
    feed = np.array([3.0, 12.0, 0.4, 0.6, 0.8])
    derivative = model(state, feed).reshape(4, 4)
    inventory_rate = model.V * derivative[:3, :3].sum() + model.Va * derivative[3, :3].sum()
    expected = feed[0] * (feed[2:].sum() - state[12:15].sum())
    assert inventory_rate == pytest.approx(expected, abs=1e-12)


def test_closed_biofilm_conserves_nitrogen_and_stoichiometric_oxygen_inventory():
    env = make_env({"model": "biofilm_reactor", "N": 60, "tsim": 3})
    initial = np.tile([2.0, 0.1, 10.0, 5.0], 4)
    states = env.simulate(
        initial, np.tile([0.0, 12.0, 0.0, 0.0, 0.0], (60, 1)), params={"V": 7.0, "Va": 19.0, "Kla": 0.0}
    )
    stages = states.reshape(61, 4, 4)
    total_species = (stages * np.array([7.0, 7.0, 7.0, 19.0])[None, :, None]).sum(axis=1)
    nitrogen = total_species[:, :3].sum(axis=1)
    oxygen_equivalent = total_species @ np.array([0.0, 3.5, 4.6, 1.0])
    np.testing.assert_allclose(nitrogen, nitrogen[0], atol=1e-8, rtol=1e-10)
    np.testing.assert_allclose(oxygen_equivalent, oxygen_equivalent[0], atol=1e-8, rtol=1e-10)
    assert np.all(states >= 0)
    assert total_species[-1, 0] < total_species[0, 0]
    assert total_species[-1, 3] < total_species[0, 3]


@pytest.mark.parametrize("index", range(16))
def test_biofilm_cannot_consume_an_absent_species(index):
    model = get_model_spec("biofilm_reactor").cls(int_method="casadi")
    state = np.ones(16)
    state[index] = 0.0
    # No material exchange: a zero concentration cannot decrease through reaction.
    derivative = model(state, np.zeros(5))
    assert derivative[index] >= 0


def test_biofilm_oxygen_equilibrium_matches_henry_solubility_in_pressure_units():
    model = get_model_spec("biofilm_reactor").cls(int_method="casadi")
    # Independently convert the gas concentration to pressure, then use NIST's
    # kH=0.0013 mol/(kg bar) at 298.15 K and water density 1 kg/L.
    molar_mass = 31_998.8  # mg O2/mol
    oxygen_pressure = (model.O_air / molar_mass) * 0.08314462618 * 298.15
    dissolved = 0.0013 * oxygen_pressure * molar_mass
    derivative = model(np.zeros(16), np.zeros(5))
    assert derivative[15] / model.Kla == pytest.approx(dissolved)
    assert dissolved == pytest.approx(9.667932415, abs=1e-8)


def test_biofilm_open_oxygen_equivalent_inventory_has_only_external_fluxes():
    model = get_model_spec("biofilm_reactor").cls(int_method="casadi", V=7.0, Va=19.0, O_F=1.2)
    state = np.linspace(0.2, 3.2, 16)
    action = np.array([3.0, 12.0, 0.4, 0.6, 0.8])
    weights = np.array([0.0, 3.5, 4.6, 1.0])
    derivative = model(state, action).reshape(4, 4)
    inventory_rate = np.array([7.0, 7.0, 7.0, 19.0]) @ derivative @ weights
    feed = np.r_[action[2:], model.O_F]
    expected = action[0] * ((feed - state[12:]) @ weights)
    expected += model.Va * model.Kla * (model.O_air / model.m - state[15])
    assert inventory_rate == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("transfer", [0.0, 0.8])
def test_biofilm_absorber_oxygen_matches_analytic_feed_washout_and_aeration(transfer):
    env = make_env({"model": "biofilm_reactor", "N": 60, "tsim": 20})
    initial = np.tile([0.2, 0.3, 0.4, 1.7], 4)
    parameters = {"vm_1": 0.0, "vm_2": 0.0, "Kla": transfer, "O_F": 1.2, "Va": 19.0}
    trajectory = env.simulate(initial, np.tile([3.0, 0.0, 0.2, 0.3, 0.4], (60, 1)), params=parameters)
    washout = 3.0 / 19.0
    rate = transfer + washout
    equilibrium = (transfer * env.model.O_air / env.model.m + washout * 1.2) / rate
    expected = equilibrium + (initial[15] - equilibrium) * np.exp(-rate * np.arange(61) * env.dt)
    # CVODES uses its default tolerances; allow accumulated step-restart error.
    np.testing.assert_allclose(trajectory[:, 15], expected, atol=1e-6, rtol=1e-4)
    np.testing.assert_allclose(trajectory[:, :12], np.tile(initial[:12], (61, 1)), atol=1e-12)


def test_biofilm_default_target_has_feasible_steady_control_and_is_reached():
    env = make_env({"model": "biofilm_reactor", "N": 60, "tsim": 100})
    target = env.SP["S2_A"][0]

    def action(nitrite):
        return np.array([8.0, 15.5, 0.525, nitrite, 0.525])

    def residual(candidate):
        return np.r_[env.model(candidate[:16], action(candidate[16])), candidate[13] - target]

    guess = env.simulate(env.x0, np.tile(action(0.75), (60, 1)))[-1]
    solution = root(residual, np.r_[guess, 0.75], tol=1e-10)
    assert solution.success
    steady, control = solution.x[:16], action(solution.x[16])
    np.testing.assert_allclose(residual(solution.x), 0, atol=1e-10)
    assert np.all(control > env.env_params["a_space"]["low"])
    assert np.all(control < env.env_params["a_space"]["high"])
    assert np.all(steady > 0) and np.all(steady < env.observation_space_base.high[:16])
    trajectory = env.simulate(env.x0, np.tile(control, (60, 1)))
    assert trajectory[-1, 13] == pytest.approx(target, abs=1e-7)
    # A feasible target must also respond to admissible actions on either side.
    below = env.simulate(env.x0, np.tile(action(0.05), (60, 1)))[-1, 13]
    above = env.simulate(env.x0, np.tile(action(1.0), (60, 1)))[-1, 13]
    assert below < target < above


@pytest.mark.parametrize(
    "parameters",
    [
        {"V": 0},
        {"Va": -1},
        {"m": 0},
        {"K1": 0},
        {"KO_2": 0},
        {"Kla": -1},
        {"O_air": -1},
        {"O_F": -1},
        {"vm_1": np.nan},
        {"vm_2": np.inf},
        {"eq_exponent": 2},
    ],
)
def test_biofilm_rejects_unphysical_parameters(parameters):
    with pytest.raises(ValueError, match="biofilm_reactor"):
        make_env({"model": "biofilm_reactor", "N": 60, "tsim": 100, "model_params": parameters})


def test_biofilm_rejects_negative_concentrations_and_inputs():
    env = make_env({"model": "biofilm_reactor", "N": 60, "tsim": 100})
    initial = env.x0.copy()
    initial[0] = -0.1
    with pytest.raises(ValueError, match="nonnegative initial concentrations"):
        env.simulate(initial, np.ones((1, 5)))
    with pytest.raises(ValueError, match="nonnegative feed/recycle"):
        env.simulate(env.x0, np.array([[-1.0, 15.0, 0.5, 0.5, 0.5]]))
    with pytest.raises(ValueError, match="biofilm_reactor"):
        env.simulate(env.x0, np.ones((1, 5)), params={"m": 0.0})
    assert env.model.m > 0


@pytest.mark.parametrize("temperature", [320.0, 400.0])
@pytest.mark.parametrize("flow", [0.0, 1.05])
def test_polymerisation_has_no_propagation_without_monomer(temperature, flow):
    model = get_model_spec("polymerisation_reactor").cls(int_method="casadi")
    derivative = model(np.array([temperature, 0.0, 0.3]), np.array([flow, temperature, 6.0, 0.65]))
    assert derivative[0] == pytest.approx(0.0, abs=1e-12)
    assert derivative[1] == pytest.approx(flow * 6.0 / model.V)


def test_polymerisation_propagation_is_first_order_in_monomer():
    model = get_model_spec("polymerisation_reactor").cls(int_method="casadi")
    action = np.array([0.0, 340.0, 0.0, 0.0])
    single = model(np.array([340.0, 1.0, 0.3]), action)
    double = model(np.array([340.0, 2.0, 0.3]), action)
    assert single[1] < 0
    np.testing.assert_allclose(double[:2], 2 * single[:2])
    assert double[2] == single[2]


@pytest.mark.parametrize("efficiency", [0.0, 0.25, 0.5, 0.9])
def test_polymerisation_initiator_decay_does_not_depend_on_radical_efficiency(efficiency):
    model = get_model_spec("polymerisation_reactor").cls(int_method="casadi", f=efficiency)
    state = np.array([340.0, 2.0, 0.3])
    derivative = model(state, np.array([0.0, 340.0, 0.0, 0.0]))
    decomposition = model.Ad * np.exp(-model.Ed_over_R / state[0]) * state[2]
    assert derivative[2] == pytest.approx(-decomposition)


def test_polymerisation_chemical_plus_sensible_energy_balance():
    model = get_model_spec("polymerisation_reactor").cls(int_method="casadi")
    state = np.array([350.0, 4.0, 0.3])
    action = np.array([1.2, 330.0, 6.0, 0.5])
    heat_per_monomer = -model.deltaHp / (model.rho * model.cp)
    derivative = model(state, action)
    energy_rate = derivative[0] + heat_per_monomer * derivative[1]
    expected = action[0] / model.V * (action[1] + heat_per_monomer * action[2] - state[0] - heat_per_monomer * state[1])
    assert energy_rate == pytest.approx(expected, abs=1e-12)


def test_polymerisation_default_temperature_bound_covers_available_reaction_heat():
    env = make_env({"model": "polymerisation_reactor", "N": 60, "tsim": 2})
    model = env.model
    heat_per_monomer = -model.deltaHp / (model.rho * model.cp)
    maximum_input = env.env_params["a_space"]["high"]
    # E = T + (-deltaHp/rho/cp) M obeys dE/dt = F/V (E_feed - E).
    maximum_energy = max(
        env.x0[0] + heat_per_monomer * env.x0[1], maximum_input[1] + heat_per_monomer * maximum_input[2]
    )
    assert env.observation_space_base.high[0] >= maximum_energy


def test_polymerisation_registered_target_has_feasible_steady_feed():
    env = make_env({"model": "polymerisation_reactor", "N": 60, "tsim": 10})
    model = env.model
    target = env.SP["M"][0]
    _, feed_temperature, feed_monomer, feed_initiator = env.env_params["a_space"]["low"]
    heat_per_monomer = -model.deltaHp / (model.rho * model.cp)
    temperature = feed_temperature + heat_per_monomer * (feed_monomer - target)
    kp, kd, kt = [
        factor * np.exp(-activation / temperature)
        for factor, activation in [
            (model.Ap, model.Ep_over_R),
            (model.Ad, model.Ed_over_R),
            (model.At, model.Et_over_R),
        ]
    ]
    # Steady material/energy balances reduce to F (F + V kd) = constant.
    constant = model.V**2 * kp**2 * target**2 * model.f * kd * feed_initiator / (kt * (feed_monomer - target) ** 2)
    flow = (-model.V * kd + np.sqrt((model.V * kd) ** 2 + 4 * constant)) / 2
    initiator = flow * feed_initiator / (flow + model.V * kd)
    action = np.array([flow, feed_temperature, feed_monomer, feed_initiator])
    state = np.array([temperature, target, initiator])
    assert np.all(action >= env.env_params["a_space"]["low"])
    assert np.all(action <= env.env_params["a_space"]["high"])
    np.testing.assert_allclose(model(state, action), 0, atol=1e-12)
    trajectory = env.simulate(env.x0, np.tile(action, (60, 1)))
    np.testing.assert_allclose(trajectory[-1], state, atol=1e-3, rtol=1e-4)


def test_four_tank_default_box_is_invariant_at_every_boundary_corner():
    env = make_env({"model": "four_tank", "N": 60, "tsim": 1000})
    state_bounds = np.stack([env.observation_space_base.low[:4], env.observation_space_base.high[:4]])
    input_bounds = env.env_params["a_space"]
    for action in itertools.product(*zip(input_bounds["low"], input_bounds["high"])):
        for sides in itertools.product((0, 1), repeat=4):
            state = state_bounds[sides, np.arange(4)]
            derivative = env.model(state, np.asarray(action))
            # The vector field must point into the box at each lower/upper face.
            inward_derivative = derivative * (1 - 2 * np.asarray(sides))
            assert np.all(inward_derivative >= -1e-12), (state, action, derivative)


def test_four_tank_registered_targets_have_feasible_steady_pump_voltages():
    env = make_env({"model": "four_tank", "N": 60, "tsim": 1000})
    model = env.model
    h3, h4 = env.SP["h3"][0], env.SP["h4"][0]
    flow3, flow4 = model.a3 * np.sqrt(2 * model.g * h3), model.a4 * np.sqrt(2 * model.g * h4)
    action = np.array([flow4 / ((1 - model.gamma_1) * model.k1), flow3 / ((1 - model.gamma_2) * model.k2)])
    h1 = ((flow3 + model.gamma_1 * model.k1 * action[0]) / model.a1) ** 2 / (2 * model.g)
    h2 = ((flow4 + model.gamma_2 * model.k2 * action[1]) / model.a2) ** 2 / (2 * model.g)
    state = np.array([h1, h2, h3, h4])
    assert np.all(action > env.env_params["a_space"]["low"])
    assert np.all(action < env.env_params["a_space"]["high"])
    assert np.all(state > env.observation_space_base.low[:4])
    assert np.all(state < env.observation_space_base.high[:4])
    np.testing.assert_allclose(model(state, action), 0, atol=1e-12)


@pytest.mark.parametrize(
    "model,tsim,n_inputs",
    [
        ("biofilm_reactor", 100, 5),
        ("four_tank", 1000, 2),
        ("polymerisation_reactor", 2, 4),
        ("crystallization", 30, 1),
        ("distillation_column", 20, 2),
        ("cstr_series_recycle", 10, 4),
        ("multistage_extraction_reactive", 60, 2),
        ("photobioreactor", 200, 2),
        ("heat_exchanger", 10, 4),
        ("batch", 15, 1),
        ("complex_cstr", 15, 1),
        ("disease", 60, 1),
        ("hydraulic_tank", 15, 1),
        ("reactor_separator_recycle", 30, 5),
        ("surge_tank", 600, 1),
    ],
)
@pytest.mark.parametrize("normalise_o", [False, True])
def test_repaired_model_defaults_complete_bounded_action_scenarios(model, tsim, n_inputs, normalise_o):
    env = make_env({"model": model, "N": 60, "tsim": tsim, "normalise_o": normalise_o})
    sequences = {"midpoint": np.zeros((60, n_inputs), dtype=np.float32)}
    for corner in itertools.product((-1.0, 1.0), repeat=n_inputs):
        sequences[str(corner)] = np.tile(np.asarray(corner, dtype=np.float32), (60, 1))
    sequences["alternating"] = np.tile(np.where(np.arange(60) % 2, 1.0, -1.0)[:, None], (1, n_inputs)).astype(
        np.float32
    )
    for seed in range(10):
        sequences[f"random_{seed}"] = np.random.default_rng(seed).uniform(-1, 1, (60, n_inputs)).astype(np.float32)
    for label, actions in sequences.items():
        observation, _ = env.reset(seed=0)
        assert env.observation_space.contains(observation), label
        for step, action in enumerate(actions, 1):
            observation, reward, terminated, truncated, _ = env.step(action)
            assert env.observation_space.contains(observation), (label, step, observation)
            assert np.isfinite(reward)
            assert terminated == (env.finite_horizon and step == 60)
            assert truncated == (not env.finite_horizon and step == 60)


@pytest.mark.parametrize("concentration_offset", [-0.01, 0.0])
def test_crystallization_growth_only_model_stops_at_or_below_saturation(concentration_offset):
    env = make_env({"model": "crystallization", "N": 60, "tsim": 30})
    state = env.x0[:7].copy()
    kelvin = 25.0 + 273.15
    state[4] = (-686.2686 + 3.579165 * kelvin - 0.00292874 * kelvin**2) / 1000 + concentration_offset
    np.testing.assert_allclose(env.model(state, np.array([25.0])), 0, atol=1e-8)


@pytest.mark.parametrize("temperature", [10.0, 20.0, 25.0, 30.0, 40.0])
def test_crystallization_resolves_saturation_in_float32(temperature):
    env = make_env({"model": "crystallization", "N": 60, "tsim": 30})
    action = np.array([temperature], dtype=np.float32)
    state = env.x0[:7].astype(np.float32)
    kelvin = temperature + 273.15
    state[4] = (-686.2686 + 3.579165 * kelvin - 0.00292874 * kelvin**2) / 1000
    np.testing.assert_array_equal(env.model(state, action), np.zeros(7))
    state[4] = np.nextafter(state[4], np.float32(np.inf))
    rates = env.model(state, action)
    assert rates[0] > 0 and rates[3] > 0 and rates[4] < 0


def test_crystallization_rates_match_growing_particle_population_with_zero_size_births():
    env = make_env({"model": "crystallization", "N": 60, "tsim": 30})
    model = env.model
    lengths = np.array([8.0, 35.0, 120.0])
    counts = np.array([3000, 2000, 800])
    moments = np.array([counts @ lengths**order for order in range(4)])
    mean = np.average(lengths, weights=counts)
    variance = np.average((lengths - mean) ** 2, weights=counts)
    concentration, temperature = 0.14, 25.0
    kelvin = temperature + 273.15
    excess = 1000 * concentration - (-686.2686 + 3.579165 * kelvin - 0.00292874 * kelvin**2)
    births = model.ka * np.exp(model.kb / kelvin) * excess**model.kc * moments[3] ** model.kd
    speed_cm = model.kg * np.exp(model.k1 / kelvin) * excess**model.k2 * (model.a + model.b * lengths * 1e-4)
    speed_um = speed_cm * 1e4
    moment_rates = [order * (counts @ (lengths ** (order - 1) * speed_um)) for order in (1, 2, 3)]
    mass_rate = model.ro * model.alfa / 2 * (counts @ ((lengths * 1e-4) ** 2 * speed_cm))
    birth_fraction = births / counts.sum()
    mean_rate = np.average(speed_um, weights=counts) - birth_fraction * mean
    variance_rate = 2 * np.average((lengths - mean) * speed_um, weights=counts) + birth_fraction * (mean**2 - variance)
    cv = np.sqrt(variance) / mean
    cv_rate = variance_rate / (2 * np.sqrt(variance) * mean) - cv * mean_rate / mean
    state = np.array([*moments, concentration, cv, mean])
    expected = [births, *moment_rates, -mass_rate, cv_rate, mean_rate]
    for snapshot in (state, state[:5]):
        np.testing.assert_allclose(model(snapshot, np.array([temperature])), expected, rtol=1e-11)


def test_crystallization_narrow_seed_preserves_width_under_size_independent_growth():
    env = make_env({"model": "crystallization", "N": 60, "tsim": 30, "model_params": {"ka": 0, "b": 0}})
    mean, width = 20.0, 0.0001
    lengths = np.array([mean - width, mean + width])
    moments = [1000 * np.mean(lengths**order) for order in range(4)]
    initial = np.array([*moments, 0.14, width / mean, mean])
    states = env.simulate(initial, np.full((10, 1), 25.0))
    assert np.all(np.diff(states[:, 6]) > 0)
    np.testing.assert_allclose(states[:, 5] * states[:, 6], width, rtol=3e-5, atol=0)
    np.testing.assert_allclose(states[:, 2], states[:, 0] * (states[:, 6] ** 2 + width**2), rtol=1e-12)
    env.model.validate_initial_state(states[-1])


def test_crystallization_replays_published_isothermal_batch():
    """Selected measurements from de Moraes et al., doi:10.1021/acs.iecr.3c00739.s001, Table S2 (40 ± 0.01 °C)."""
    env = make_env({"model": "crystallization", "N": 60, "tsim": 30})
    times = [5.00, 11.28, 17.58, 23.88, 30.18, 36.48, 41.53]
    measured = np.array(
        [
            [1534.3, 20715, 1677000, 223640000, 0.1579],
            [6896.6, 105630, 4057200, 433250000, 0.1577],
            [20884, 328490, 11348000, 836080000, 0.1565],
            [49238, 700950, 23739000, 1501400000, 0.1548],
            [75278, 1064600, 37599000, 2091200000, 0.1519],
            [125910, 1437800, 46944000, 2592000000, 0.1500],
            [154010, 1505600, 48068000, 2900100000, 0.1492],
        ]
    )
    result = solve_ivp(
        lambda time, state: env.model(state, np.array([40.0]))[:5],
        (times[0], times[-1]),
        measured[0],
        t_eval=times,
        rtol=1e-8,
        atol=1e-10,
    )
    assert result.success
    residuals = measured - result.y.T
    r_squared = 1 - (residuals**2).sum(axis=0) / ((measured - measured.mean(axis=0)) ** 2).sum(axis=0)
    assert np.all(r_squared > 0.8), r_squared


@pytest.mark.parametrize("concentration", [0.1, 0.12])
def test_crystallization_inactive_rates_have_finite_symbolic_jacobian(concentration):
    env = make_env({"model": "crystallization", "N": 60, "tsim": 30})
    state = env.x0[:7].copy()
    state[4] = concentration
    symbolic = SX.sym("state", 7)
    derivative = vertcat(*env.model(symbolic, np.array([25.0])))
    evaluate = Function("derivative", [symbolic], [jacobian(derivative, symbolic)])
    result = np.asarray(evaluate(state))
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, 0, atol=1e-12)


def test_crystallization_inventory_and_derived_observables_along_changing_temperature():
    env = make_env({"model": "crystallization", "N": 60, "tsim": 30})
    temperatures = np.random.default_rng(0).uniform(10, 40, (60, 1))
    states = env.simulate(env.x0, temperatures)
    # The 1/6 converts the shape parameter to crystal volume; micrometres^3 -> cm^3 is 1e-12.
    inventory = states[:, 4] + env.model.ro * env.model.alfa / 6 * 1e-12 * states[:, 3]
    np.testing.assert_allclose(inventory, inventory[0], atol=1e-10, rtol=1e-8)
    expected_cv = np.sqrt(states[:, 2] * states[:, 0] / states[:, 1] ** 2 - 1)
    expected_length = states[:, 1] / states[:, 0]
    np.testing.assert_allclose(states[:, 5], expected_cv, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(states[:, 6], expected_length, rtol=1e-12, atol=1e-12)
    assert np.all(states >= 0)


def test_crystallization_current_state_can_restart_simulation_without_quality_drift():
    env = make_env({"model": "crystallization", "N": 60, "tsim": 30})
    env.reset()
    for _ in range(10):
        env.step(np.array([0.0], dtype=np.float32))
    env.model.validate_initial_state(env.state)
    continuation = env.simulate(env.state, np.array([[25.0]]))
    env.step(np.array([0.0], dtype=np.float32))
    np.testing.assert_allclose(continuation[-1], env.state[:7], atol=1e-10, rtol=1e-10)


def test_crystallization_bounded_temperature_schedule_approaches_default_quality_targets():
    env = make_env({"model": "crystallization", "N": 60, "tsim": 30})
    # Four 7.5-minute holds found by shooting the repaired nominal model.
    # This is a feasible batch-quality reference, not a learned policy or steady-state claim.
    temperatures = np.repeat([28.0767, 34.8776, 32.6475, 21.8005], 15).reshape(60, 1)
    states = env.simulate(env.x0, temperatures)
    assert abs(states[-1, 5] - env.SP["CV"][0]) < 0.02
    assert abs(states[-1, 6] - env.SP["Ln"][0]) < 0.2


def test_crystallization_moment_variation_is_visible_in_float32_observations():
    params = {"model": "crystallization", "N": 60, "tsim": 30}
    env = make_env(params)
    initial, _ = env.reset()
    assert env.observation_space.contains(initial)
    observation, *_ = env.step(np.array([-1.0], dtype=np.float32))
    assert np.all(observation[:4] != initial[:4])


@pytest.mark.parametrize("index,value", [(0, 0.0), (1, 0.0), (2, 1.0), (4, -0.1), (5, 0.5), (6, 15.0)])
def test_crystallization_rejects_invalid_or_inconsistent_initial_moments(index, value):
    initial = get_model_spec("crystallization").defaults()["x0"].copy()
    initial[index] = value
    with pytest.raises(ValueError, match="crystallization"):
        make_env({"model": "crystallization", "N": 60, "tsim": 30, "x0": initial})


def test_crystallization_rate_disturbances_control_the_advertised_rates():
    env = make_env({"model": "crystallization", "N": 60, "tsim": 30})
    model = env.model
    before = vars(model).copy()
    no_nucleation = model(env.x0, np.array([25.0, 0.0, model.kg]))
    no_growth = model(env.x0, np.array([25.0, model.ka, 0.0]))
    assert no_nucleation[0] == 0 and no_nucleation[3] > 0
    assert no_growth[0] > 0
    np.testing.assert_allclose(no_growth[1:5], 0)
    assert vars(model) == before
    assert model.info()["disturbances"] == ["ka", "kg"]


def test_crystallization_scheduled_nucleation_disturbance_matches_parameter_override():
    env = make_env(
        {
            "model": "crystallization",
            "N": 60,
            "tsim": 30,
            "disturbances": {"ka": [0.0]},
            "disturbance_bounds": {"low": np.array([0.0]), "high": np.array([2.0])},
        }
    )
    env.reset()
    env.step(np.array([0.0], dtype=np.float32))
    nominal = make_env({"model": "crystallization", "N": 60, "tsim": 30})
    expected = nominal.simulate(nominal.x0, np.array([[25.0]]), params={"ka": 0.0})
    np.testing.assert_allclose(env.state[:7], expected[1], rtol=1e-8)
    assert env.state[0] == env.x0[0]


def test_crystallization_rejects_nonexistent_heat_transfer_disturbance():
    with pytest.raises(ValueError, match="unknown names.*UA"):
        make_env(
            {
                "model": "crystallization",
                "N": 60,
                "tsim": 30,
                "disturbances": {"UA": [1.0]},
                "disturbance_bounds": {"low": np.array([0.0]), "high": np.array([2.0])},
            }
        )


def test_distillation_defaults_have_nonnegative_bottoms_and_possible_light_component_target():
    env = make_env({"model": "distillation_column", "N": 60, "tsim": 20})
    inputs = env.env_params["a_space"]
    assert inputs["low"][1] >= env.model.D
    assert env.SP["X0"][0] * env.model.D < inputs["high"][1] * env.model.X_feed


@pytest.mark.parametrize("quality", [0.0, 0.5, 1.0, 1.5])
def test_distillation_feed_quality_preserves_uniform_nonseparating_mixture(quality):
    # With alpha=1 there is no separation: uniform composition equal to the feed
    # must remain stationary regardless of its liquid/vapour split.
    model = get_model_spec("distillation_column").cls(int_method="casadi", alpha=1.0, q=quality)
    derivative = model(np.full(9, model.X_feed), np.array([3.0, 200.0]))
    np.testing.assert_allclose(derivative, 0, atol=1e-12)


def test_distillation_inventory_has_only_external_light_component_fluxes():
    model = get_model_spec("distillation_column").cls(int_method="casadi", M0=1000, M=1500, Mb=3000)
    state = np.linspace(0.8, 0.01, 9)
    action = np.array([5.0, 200.0])
    derivative = model(state, action)
    inventory_rate = derivative @ np.array([model.M0, *([model.M] * 7), model.Mb])
    expected = action[1] * model.X_feed - model.D * state[0] - (action[1] - model.D) * state[-1]
    assert inventory_rate == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "extra",
    [
        {"a_space": {"low": np.array([1.0, 80.0]), "high": np.array([10.0, 300.0])}},
        {"model_params": {"q": 0.0}},  # At minimum reflux and maximum feed the reboiler vapour flow is negative.
        {"model_params": {"D": 350.0}},
    ],
)
def test_distillation_rejects_input_ranges_with_negative_internal_flows(extra):
    with pytest.raises(ValueError, match="distillation"):
        make_env({"model": "distillation_column", "N": 60, "tsim": 20, **extra})


def test_distillation_registered_target_has_feasible_steady_feed():
    env = make_env({"model": "distillation_column", "N": 60, "tsim": 20})
    target = env.SP["X0"][0]

    def residual(values):
        return np.r_[env.model(values[:9], np.array([5.0, values[9]])), values[0] - target]

    result = root(residual, np.r_[np.linspace(0.5, 0.0, 9), 250.0])
    assert result.success
    np.testing.assert_allclose(result.fun, 0, atol=1e-9)
    assert np.all((result.x[:9] >= 0) & (result.x[:9] <= 1))
    assert env.env_params["a_space"]["low"][1] < result.x[9] < env.env_params["a_space"]["high"][1]


def test_series_recycle_registered_target_has_feasible_steady_flow():
    env = make_env({"model": "cstr_series_recycle", "N": 60, "tsim": 10})

    def residual(values):
        return np.r_[env.model(values[:4], np.array([values[4], 0.005, 305.0, 305.0])), values[2] - env.SP["C2"][0]]

    result = root(residual, [97.1, 299, 97.0, 299, 0.002])
    assert result.success
    np.testing.assert_allclose(result.fun, 0, atol=1e-8)
    assert env.env_params["a_space"]["low"][0] <= result.x[4] <= env.env_params["a_space"]["high"][0]
    assert np.all(result.x[:4] >= env.observation_space_base.low[:4])
    assert np.all(result.x[:4] <= env.observation_space_base.high[:4])


def test_series_recycle_combined_material_energy_balance():
    model = get_model_spec("cstr_series_recycle").cls(int_method="casadi")
    state = np.array([90.0, 305.0, 85.0, 310.0])
    action = np.array([0.003, 0.006, 302.0, 300.0])
    derivative = model(state, action)
    capacity = model.rho * model.cp
    weights = np.array([-model.deltaH * model.V1, capacity * model.V1, -model.deltaH * model.V2, capacity * model.V2])
    expected = action[0] * (capacity * (model.T_O - state[3]) - model.deltaH * (model.C_O - state[2]))
    expected += model.U1A1 * (action[2] - state[1]) + model.U2A2 * (action[3] - state[3])
    assert derivative @ weights == pytest.approx(expected, abs=1e-8)


def test_reactive_extraction_registered_target_has_feasible_steady_flow():
    env = make_env({"model": "multistage_extraction_reactive", "N": 60, "tsim": 60})

    def residual(values):
        return np.r_[env.model(values[:20], np.array([values[20], 100.0])), values[16] - env.SP["XA5"][0]]

    result = root(residual, np.r_[np.tile([1.97, 0.003, 1.9999, 0.0001], 5), 10.0])
    assert result.success
    np.testing.assert_allclose(result.fun, 0, atol=1e-8)
    assert env.env_params["a_space"]["low"][0] <= result.x[20] <= env.env_params["a_space"]["high"][0]
    assert np.all((result.x[:20] >= 0) & (result.x[:20] <= 2))


def test_closed_reactive_extraction_conserves_both_reactant_equivalents():
    env = make_env({"model": "multistage_extraction_reactive", "N": 60, "tsim": 60})
    initial = np.tile([1.0, 0.3, 1.0, 0.1], 5)
    states = env.simulate(initial, np.zeros((60, 2)), params={"Vl": 3.0, "Vg": 7.0})
    stages = states.reshape(61, 5, 4)
    component_a = (3 * stages[:, :, 0] + 7 * (stages[:, :, 1] + stages[:, :, 3])).sum(axis=1)
    component_b = 7 * (stages[:, :, 2] + stages[:, :, 3]).sum(axis=1)
    np.testing.assert_allclose(component_a, component_a[0], atol=1e-8, rtol=1e-9)
    np.testing.assert_allclose(component_b, component_b[0], atol=1e-8, rtol=1e-9)
    assert np.all(states >= 0)


def test_photobioreactor_target_is_reached_with_admissible_constant_feed():
    env = make_env({"model": "photobioreactor", "N": 60, "tsim": 200})
    target = env.SP["c_q"][0]

    def error(feed):
        return env.simulate(env.x0, np.tile([100.0, feed], (60, 1)))[-1, 2] - target

    feed = brentq(error, 0.0, 10.0, xtol=1e-7)
    assert 0 < feed < 10
    assert abs(error(feed)) < 1e-8
    # A percent of the tracked-state normalization span is meaningful at the attainable yield scale.
    assert env.observation_space_base.high[2] < 3 * target


def test_photobioreactor_nutrient_inventory_and_product_upper_bound():
    env = make_env({"model": "photobioreactor", "N": 60, "tsim": 200})
    actions = np.random.default_rng(3).uniform([100.0, 0.0], [400.0, 10.0], (60, 2))
    states = env.simulate(env.x0, actions)
    inventory = states[:, 1] + env.model.Y_NX * states[:, 0]
    expected = inventory[0] + np.r_[0.0, np.cumsum(actions[:, 1]) * env.dt]
    np.testing.assert_allclose(inventory, expected, atol=1e-7, rtol=1e-9)
    upper_bound = env.model.k_m * (env.tsim * (1 + 150 / env.model.Y_NX) + 10 * env.tsim**2 / (2 * env.model.Y_NX))
    assert states[-1, 2] <= upper_bound < env.observation_space_base.high[2]


def test_heat_exchanger_diameter_volume_matches_analytic_inlet_response():
    env = make_env({"model": "heat_exchanger", "N": 1, "tsim": 2})
    initial = np.full(24, 350.0)
    trajectory = env.simulate(initial, np.array([[1.0, 1.0, 360.0, 350.0]]), params={"Utm": 0.0, "Usm": 0.0})
    volume = np.pi * env.model.Dt**2 * env.model.L / 4
    expected = 360.0 - 10.0 * np.exp(-np.array([0.0, 2.0]) / (env.model.rhot * volume))
    np.testing.assert_allclose(trajectory[:, 0], expected, rtol=1e-6, atol=1e-4)


@pytest.mark.parametrize("parameters", [{"Dt": 2.0}, {"Ds": 1.5}, {"L": 0.0}, {"cpm": -1.0}])
def test_heat_exchanger_rejects_invalid_geometry_and_thermal_capacity(parameters):
    with pytest.raises(ValueError, match="heat_exchanger"):
        make_env({"model": "heat_exchanger", "N": 60, "tsim": 10, "model_params": parameters})


def test_heat_exchanger_energy_balance_with_unequal_phase_capacities():
    model = get_model_spec("heat_exchanger").cls(
        int_method="casadi", cpt=4.0, cpm=2.0, cps=3.0, rhot=10.0, rhom=20.0, rhos=30.0
    )
    state = np.linspace(350.0, 300.0, 24)
    action = np.array([2.0, 3.0, 355.0, 305.0])
    volumes = model.L * np.pi / 4 * np.array([model.Dt**2, model.Dm**2 - model.Dt**2, model.Ds**2 - model.Dm**2])
    capacities = volumes * [model.rhot * model.cpt, model.rhom * model.cpm, model.rhos * model.cps]
    energy_rate = model(state, action) @ np.tile(capacities, 8)
    external_heat = action[0] * model.cpt * (action[2] - state[21]) + action[1] * model.cps * (action[3] - state[2])
    assert energy_rate == pytest.approx(external_heat, abs=1e-10)


def test_heat_exchanger_target_has_feasible_steady_cold_inlet():
    env = make_env({"model": "heat_exchanger", "N": 60, "tsim": 10})

    def residual(values):
        return np.r_[env.model(values[:24], np.array([2.75, 2.75, 350.0, values[24]])), values[21] - env.SP["Tt8"][0]]

    result = root(residual, np.r_[np.full(24, 330.0), 310.0])
    assert result.success
    np.testing.assert_allclose(result.fun, 0, atol=1e-9)
    assert env.env_params["a_space"]["low"][3] <= result.x[24] <= env.env_params["a_space"]["high"][3]


def test_batch_conserves_stoichiometric_and_adiabatic_energy_inventories():
    env = make_env({"model": "batch", "N": 60, "tsim": 15})
    states = env.simulate(env.x0, np.full((60, 1), 300.0), params={"UA": 0.0})
    np.testing.assert_allclose(states[:, :3] @ [2, 1, 1], 2, atol=1e-10)
    # Chemical energy in A includes one A->2B event and two subsequent B->C events.
    energy = states[:, 3] + states[:, 0] + 0.375 * states[:, 1]
    np.testing.assert_allclose(energy, 301, atol=1e-9)
    assert np.all(states[:, :3] >= 0)


def test_batch_terminal_yield_has_coolant_authority_and_time_observation():
    env = make_env({"model": "batch", "N": 60, "tsim": 15})
    returns = []
    for action in (-1.0, 1.0):
        obs, _ = env.reset(seed=0)
        assert obs[-1] == 1 and env.observation_info()[-1] == ("time_remaining", "time")
        rewards = [env.step(np.array([action], dtype=np.float32))[1] for _ in range(60)]
        assert rewards[:-1] == [0.0] * 59
        assert rewards[-1] == pytest.approx(env.state[1])
        returns.append(rewards[-1])
    # At this harvest time, less cooling accelerates loss of the desired intermediate.
    assert returns[0] > returns[1] + 0.005


def test_complex_cstr_species_and_energy_balances():
    model = get_model_spec("complex_cstr").cls(int_method="casadi")
    state, action = np.array([0.7, 0.4, 0.1, 330.0]), np.array([300.0])
    derivative = model(state, action)
    assert derivative[:3] @ [2, 1, 1] == pytest.approx(model.q / model.V * (2 * model.Caf - state[:3] @ [2, 1, 1]))
    capacity = model.rho * model.C
    chemical_a = -model.deltaHr1 - 2 * model.deltaHr2
    chemical_b = -model.deltaHr2
    energy = capacity * state[3] + chemical_a * state[0] + chemical_b * state[1]
    expected = model.q / model.V * (capacity * model.Ti + chemical_a * model.Caf - energy)
    expected += model.UA / model.V * (action[0] - state[3])
    assert derivative @ [chemical_a, chemical_b, 0, capacity] == pytest.approx(expected, abs=1e-9)


def test_complex_cstr_default_product_target_has_feasible_coolant_and_is_reached():
    env = make_env({"model": "complex_cstr", "N": 60, "tsim": 15})
    result = root(lambda z: np.r_[env.model(z[:4], z[4:]), z[1] - env.SP["Cb"][0]], [0.87, 0.25, 0.002, 325, 300])
    assert result.success
    np.testing.assert_allclose(result.fun, 0, atol=1e-8)
    assert 295 < result.x[4] < 302
    states = env.simulate(env.x0, np.full((60, 1), result.x[4]))
    assert abs(states[-1, 1] - 0.25) < 0.002


def test_disease_matches_analytic_vaccination_and_recovery_without_transmission():
    env = make_env({"model": "disease", "N": 60, "tsim": 60, "model_params": {"beta": 0.0}})
    states = env.simulate(env.x0, np.full((60, 1), 0.025))
    times = np.arange(61)
    # Sixty CVODES restarts at its default tolerance accumulate small errors;
    # require absolute accuracy within 0.003 percentage points of population.
    np.testing.assert_allclose(states[:, 0], 0.99 * np.exp(-0.025 * times), rtol=0, atol=3e-5)
    np.testing.assert_allclose(states[:, 1], 0.01 * np.exp(-0.1 * times), rtol=0, atol=3e-5)
    np.testing.assert_allclose(states.sum(axis=1), 1, atol=1e-10)


def test_disease_default_cost_uses_applied_physical_vaccination_and_daily_step():
    env = make_env({"model": "disease", "N": 60, "tsim": 60})
    env.reset(seed=0)
    _, reward, *_ = env.step(np.array([0.0], dtype=np.float32))  # midpoint is 0.025/day
    assert reward == pytest.approx(-(env.state[1] + 20 * 0.025**2))
    assert env.finite_horizon
    costs = []
    for action in (-1.0, 0.0, 1.0):
        env.reset(seed=0)
        costs.append(-sum(env.step(np.array([action], dtype=np.float32))[1] for _ in range(60)))
    assert costs[1] < costs[0] and costs[1] < costs[2]


@pytest.mark.parametrize("model,x0", [("disease", [0.99, 0.1, 0.0]), ("hydraulic_tank", [-1.0, 2.5, 0.9])])
def test_population_and_volume_models_reject_unphysical_initial_states(model, x0):
    with pytest.raises(ValueError, match="requires nonnegative"):
        make_env({"model": model, "N": 60, "tsim": 15, "x0": np.array(x0)})


@pytest.mark.parametrize("model", ["disease", "reactor_separator_recycle"])
def test_population_and_composition_initial_states_accept_float32_rounding(model):
    values = get_model_spec(model).defaults()["x0"].astype(np.float32)
    env = make_env({"model": model, "N": 60, "tsim": 30, "x0": values})
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_hydraulic_tank_matches_analytic_transfer_and_feasible_target():
    env = make_env({"model": "hydraulic_tank", "N": 60, "tsim": 15})
    states = env.simulate(env.x0, np.full((60, 1), 0.3))
    np.testing.assert_allclose(states.sum(axis=1), 1.5, atol=1e-10)
    expected = 0.9 + 0.1 * np.exp(-2 * np.arange(61) * env.dt)
    np.testing.assert_allclose(states[:, 0], expected, rtol=0, atol=2e-5)
    assert abs(states[-1, 0] - env.SP["q1"][0]) < 1e-5


def test_hydraulic_tank_rejects_input_box_that_can_drain_small_inventory():
    with pytest.raises(ValueError, match="drain a tank below zero"):
        make_env({"model": "hydraulic_tank", "N": 60, "tsim": 15, "x0": np.array([0.1, 0.1, 0.9])})


def test_recycle_cannot_evaporate_absent_second_volatile_component():
    model = get_model_spec("reactor_separator_recycle").cls(int_method="casadi")
    state = np.array([10.0, 0.5, 0.0, 0.5] * 3)
    derivative = model(state, np.array([1.0, 2.0, 2.0, 1.0, 1.0]))
    assert derivative[10] == pytest.approx(0.0, abs=1e-14)


def test_recycle_component_inventories_have_only_external_fluxes_and_reaction():
    model = get_model_spec("reactor_separator_recycle").cls(int_method="casadi", A_R=7, A_M=9, A_B=11, rho=2)
    state = np.array([9, 0.5, 0.3, 0.2, 10, 0.4, 0.3, 0.3, 11, 0.1, 0.4, 0.5])
    action = np.array([1.1, 1.9, 2.1, 0.9, 1.0])
    x, dx = state.reshape(3, 4), model(state, action).reshape(3, 4)
    mass = model.rho * np.array([model.A_R, model.A_M, model.A_B])
    derivative = (mass[:, None] * (x[:, :1] * dx[:, 1:] + dx[:, :1] * x[:, 1:])).sum(axis=0)
    reaction = (
        mass[0]
        * x[0, 0]
        * np.array([-model.k_1 * x[0, 1], model.k_1 * x[0, 1] - model.k_2 * x[0, 2], model.k_2 * x[0, 2]])
    )
    np.testing.assert_allclose(
        derivative, action[0] * np.array([1, 0, 0]) - action[3] * x[2, 1:] + reaction, atol=1e-13
    )


def test_recycle_level_targets_are_reached_by_admissible_constant_flows():
    env = make_env({"model": "reactor_separator_recycle", "N": 60, "tsim": 30})
    action = np.array([1.1, 1.1 + 1 - 1 / 6, 2.1, 2.1 - 1 - 1 / 6, 1.0])
    assert np.all(action >= env.env_params["a_space"]["low"])
    assert np.all(action <= env.env_params["a_space"]["high"])
    states = env.simulate(env.x0, np.tile(action, (60, 1)))
    np.testing.assert_allclose(states[-1, [0, 4, 8]], 10, atol=1e-10)
    np.testing.assert_allclose(states.reshape(61, 3, 4)[:, :, 1:].sum(axis=2), 1, atol=1e-10)
    assert env.finite_horizon


@pytest.mark.parametrize("duration,reason", [(40, "remaining separator volatiles"), (300, "empty a vessel")])
def test_recycle_rejects_horizons_without_a_proven_positive_domain(duration, reason):
    with pytest.raises(ValueError, match=reason):
        make_env({"model": "reactor_separator_recycle", "N": 60, "tsim": duration})


def test_episode_domain_validation_is_rechecked_after_uncertainty_and_in_simulate():
    env = make_env(
        {
            "model": "hydraulic_tank",
            "N": 60,
            "tsim": 15,
            "empirical_distribution": {"D": [0.01]},
            "uncertainty_bounds": {"low": np.array([0.0]), "high": np.array([1.0])},
        }
    )
    with pytest.raises(ValueError, match="drain a tank below zero"):
        env.reset(seed=0)
    env = make_env({"model": "hydraulic_tank", "N": 60, "tsim": 15})
    with pytest.raises(ValueError, match="drain a tank below zero"):
        env.simulate([0.1, 0.1], np.full((60, 1), 0.5))
